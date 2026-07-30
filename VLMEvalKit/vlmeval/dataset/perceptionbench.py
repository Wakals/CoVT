import base64
import json
import os
import re
import warnings

import pandas as pd

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .utils.multiple_choice import report_acc
from .utils.perceptionbench import PerceptionBench_prompt, PerceptionBench_extract
from ..smp import *
from ..smp.file import get_intermediate_file_path
from ..utils import track_progress_rich

# The benchmark interleaves images into the question text with <|image_N|> markers (1-based).
IMAGE_PLACEHOLDER = re.compile(r'<\|image_(\d+)\|>')

MIME_EXT = {'jpeg': 'jpg', 'jpg': 'jpg', 'png': 'png', 'webp': 'webp', 'gif': 'gif', 'bmp': 'bmp'}

# Judge selection. The default is an API judge (gpt-4o-mini). gpt-oss-120b is the judge the
# PerceptionBench paper itself uses; it has no public API, so it is reached through a local
# OpenAI-compatible server (e.g. `vllm serve openai/gpt-oss-120b --port 8001`). It is used
# when explicitly requested via `--judge gpt-oss-120b`, and automatically as the fallback
# when no API key is available -- otherwise the run would have no way to score at all.
DEFAULT_API_JUDGE = 'gpt-4o-mini'
GPT_OSS_JUDGE = 'gpt-oss-120b'
GPT_OSS_DEFAULT_BASE = 'http://127.0.0.1:8001/v1/chat/completions'
# OpenAIWrapper asserts the key starts with 'sk-'. A locally served model needs no real
# credential, so send a dummy that satisfies that check (vLLM ignores the bearer token
# unless it was started with --api-key).
GPT_OSS_DEFAULT_KEY = 'sk-local-no-auth'


def build_perceptionbench_judge(**judge_kwargs):
    """Return (judge, resolved_name) for PerceptionBench grading.

    Override the local server with GPT_OSS_API_BASE / GPT_OSS_MODEL_NAME if it is not
    listening on the default port or is served under a different name.
    """
    model = judge_kwargs.pop('model', None) or DEFAULT_API_JUDGE
    use_gpt_oss = bool(judge_kwargs.pop('use_gpt_oss', False)) or 'gpt-oss' in str(model).lower()

    if not use_gpt_oss and not gpt_key_set():
        warnings.warn(
            f'OPENAI_API_KEY is not set, falling back to a locally served {GPT_OSS_JUDGE}. '
            f'Start it with `vllm serve openai/gpt-oss-120b --served-model-name {GPT_OSS_JUDGE} '
            '--port 8001`, or set GPT_OSS_API_BASE.'
        )
        use_gpt_oss = True

    if use_gpt_oss:
        from ..api import OpenAIWrapper
        name = os.environ.get('GPT_OSS_MODEL_NAME', GPT_OSS_JUDGE)
        api_base = os.environ.get('GPT_OSS_API_BASE', GPT_OSS_DEFAULT_BASE)
        judge = OpenAIWrapper(
            name, api_base=api_base,
            key=os.environ.get('GPT_OSS_API_KEY', GPT_OSS_DEFAULT_KEY), **judge_kwargs)
        assert judge.working(), (
            f'The local {name} judge at {api_base} is not responding. Start the server or set '
            'GPT_OSS_API_BASE.'
        )
    else:
        name = model
        judge = build_judge(model=model, **judge_kwargs)
        assert judge.working(), f'The judge model {name} is not working properly. ' + DEBUG_MESSAGE
    return judge, name


def PerceptionBench_auxeval(model, line):
    """Grade one record. The official harness runs its judge at temperature 0.3; retry with
    a bit more temperature if the verdict comes back unparseable."""
    prompt = PerceptionBench_prompt(line)
    for i in range(3):
        resp = model.generate(prompt, temperature=0.3 + 0.2 * i)
        hit, reason = PerceptionBench_extract(resp)
        if hit is not None:
            return dict(hit=hit, log=reason)
    return dict(hit=0, log='Fail to Judge')


class PerceptionBench(ImageBaseDataset):
    """PerceptionBench: 3000 open-ended questions on atomic visual perception, each with a
    short uniquely-determined answer, graded 0/1 by an LLM judge.

    Paper / data: https://huggingface.co/datasets/moonshotai/PerceptionBench

    The released file is a 1.6 GB JSONL with base64 data-URI images, so on first use it is
    converted once into the TSV + image-directory layout the rest of VLMEvalKit expects.
    """

    TYPE = 'VQA'

    DATASET_URL = {
        'PerceptionBench': (
            'https://huggingface.co/datasets/moonshotai/PerceptionBench/resolve/main/PerceptionBench.jsonl'
        ),
    }

    DATASET_MD5 = {
        'PerceptionBench': None,
    }

    def load_data(self, dataset):
        data_root = LMUDataRoot()
        tsv_path = osp.join(data_root, f'{dataset}.tsv')
        if not osp.exists(tsv_path):
            self.build_tsv(dataset, tsv_path)
        return load(tsv_path)

    def build_tsv(self, dataset, tsv_path):
        """Convert the released JSONL into a TSV plus one image file per image.

        Images are written out rather than kept as base64 so the TSV stays small and
        VLMEvalKit's >1GB LOCALIZE path never has to run.
        """
        data_root = LMUDataRoot()
        jsonl_path = osp.join(data_root, f'{dataset}.jsonl')
        if not osp.exists(jsonl_path):
            warnings.warn(f'{jsonl_path} not found, downloading (~1.6 GB) ...')
            download_file(self.DATASET_URL[dataset], jsonl_path)

        img_dir = osp.join(data_root, 'images', dataset)
        os.makedirs(img_dir, exist_ok=True)

        records = []
        for raw in open(jsonl_path, encoding='utf-8'):
            rec = json.loads(raw)
            index = rec['index']
            paths = []
            for k, uri in enumerate(rec.get('image') or []):
                uri = str(uri)
                head, _, payload = uri.partition(',')
                mime = head.split('/')[-1].split(';')[0].lower() if uri.startswith('data:') else 'jpg'
                path = osp.join(img_dir, f'{index}_{k}.{MIME_EXT.get(mime, "jpg")}')
                if not osp.exists(path):
                    with open(path, 'wb') as fh:
                        fh.write(base64.b64decode(payload))
                paths.append(path)
            answer = rec.get('answer')
            records.append({
                'index': index,
                'question': str(rec.get('problem', '') or ''),
                'answer': str(answer.get('answer') if isinstance(answer, dict) else answer),
                # `error_category` is the benchmark's atomic-capability label; report_acc
                # breaks accuracy down by the column named `category`.
                'category': rec.get('error_category'),
                'image_path': str(paths),
                'hint': str(rec.get('hint', '') or ''),
                'source_bmk': rec.get('source_bmk'),
            })

        dump(pd.DataFrame(records), tsv_path)
        warnings.warn(f'Built {tsv_path} ({len(records)} records) with images under {img_dir}')

    def build_prompt(self, line):
        """Interleave images at their <|image_N|> markers, matching the official harness:
        a marker is replaced in place by that image, and any image never referenced by a
        marker is appended after the text."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = toliststr(line['image_path']) if self.meta_only else self.dump_image(line)
        question = str(line['question'])

        msgs, last, used = [], 0, set()
        for m in IMAGE_PLACEHOLDER.finditer(question):
            n = int(m.group(1))
            if 1 <= n <= len(tgt_path):
                used.add(n - 1)
                seg = question[last:m.start()]
                if seg:
                    msgs.append(dict(type='text', value=seg))
                msgs.append(dict(type='image', value=tgt_path[n - 1]))
                last = m.end()
        if question[last:]:
            msgs.append(dict(type='text', value=question[last:]))
        for k, path in enumerate(tgt_path):
            if k not in used:
                msgs.append(dict(type='image', value=path))
        return msgs if msgs else [dict(type='text', value=question)]

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        storage = get_intermediate_file_path(eval_file, '_judge')
        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            # All answers are free-form, so an LLM judge is required -- exact matching would
            # not produce a meaningful score here.
            judge, judge_name = build_perceptionbench_judge(**judge_kwargs)
            warnings.warn(f'Grading PerceptionBench with judge: {judge_name}')

            ans_map = {} if not osp.exists(tmp_file) else load(tmp_file)
            lines = [data.iloc[i] for i in range(len(data)) if data.iloc[i]['index'] not in ans_map]
            indices = [x['index'] for x in lines]
            if len(lines):
                res = track_progress_rich(
                    PerceptionBench_auxeval, [(judge, line) for line in lines],
                    nproc=nproc, chunksize=nproc, keys=indices, save=tmp_file)
                for k, v in zip(indices, res):
                    ans_map[k] = v

            judge_results = [ans_map[x] for x in data['index']]
            data['hit'] = [x['hit'] for x in judge_results]
            data['log'] = [x['log'] for x in judge_results]
            unparsed = sum(1 for x in judge_results if x['log'] == 'Fail to Judge')
            if unparsed:
                warnings.warn(
                    f'{unparsed}/{len(data)} items got no parseable verdict and were scored 0. '
                    'They are marked "Fail to Judge" in the log column.'
                )
            dump(data, storage)

        data = load(storage)
        acc = report_acc(data)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc, score_file)
        return acc
