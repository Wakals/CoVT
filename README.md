<!-- # CoVT: Chain-of-Visual-Thought -->
<div align="center">

  <h1 style="margin: 0; font-size: 1.8em;">
    Chain-of-Visual-Thought: Teaching VLMs to See and Think Better with Continuous Visual Tokens
  </h1>

  <h4 style="margin: 15px 0; color:rgb(50, 151, 234);">
    ⭐️ CoVT enriches VLMs’ vision-centric reasoning capabilities. ⭐️
  </h4>

  [![Arixv](https://img.shields.io/badge/arxiv-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.19418)
  [![Hugging Face Collection](https://img.shields.io/badge/HuggingFace-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Wakals/covt-chain-of-visual-thought)
  [![Project Page](https://img.shields.io/badge/Project_Page-00CED1?style=for-the-badge&logoColor=white&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xMCAyMHYtNmg0djZoNXYtOGgzTDEyIDMgMiAxMmgzdjh6Ii8+PC9zdmc+)](https://wakalsprojectpage.github.io/comt-website/)

</div>


<div align="center">
  <b>
    <a href="https://wakals.github.io/" target="_blank">Yiming Qin</a><sup>1</sup>,
    <a href="https://github.com/David-BominWei" target="_blank">Bomin Wei</a><sup>2</sup>,
     <a href="https://gejiaxin.org/" target="_blank">Jiaxin Ge</a><sup>1</sup>,
      <a href="https://tech-ai.panasonic.com/en/researcher_introduction/048/" target="_blank">Konstantinos Kallidromitis</a><sup>3</sup>,<br>
       <a href="https://stephanie-fu.github.io/" target="_blank">Stephanie Fu</a><sup>1</sup>,
    <a href="https://people.eecs.berkeley.edu/~trevor/" target="_blank">Trevor Darrell</a><sup>1</sup>,
    <a href="https://people.eecs.berkeley.edu/~xdwang/" target="_blank">XuDong Wang</a><sup>1*</sup>
  </b><br>
  
  <span style="font-size: 1em; color: #555;">
    University of California, Berkeley<sup>1</sup><br>
    University of California, Los Angeles<sup>2</sup><br>
    Panasonic AI Research<sup>3</sup>
  </span>

  <p style="color: #555; font-size: 0.9em; margin-top: 8px; margin-bottom: 0;">
    *Corresponding author
  </p>

</div>

<div align="center">
  <img src="./assets/DEMO.jpg" alt="" style="width: 100%; margin: 10px 0;">
  <img src="./assets/edit_demo.jpg" alt="" style="width: 100%; margin: 10px 0;">
</div>

## 🔥 News

[2025-11-24] 🥃 Released the (training data)[https://huggingface.co/datasets/Wakals/CoVT-Dataset] and code. Give it a shot!

[2025-11-24] ⭐️ The evaluation and Gradio demo are available NOW!

[2025-11-24] 🤗 Our finetuned weights are available. [Check it here!](https://huggingface.co/collections/Wakals/covt-chain-of-visual-thought)

## 📑 Table of Contents

- [👀 Overview](#-overview)
- [🚀 Quick Start](#-quick-start)
- [🤗 Model Zoo](#-model-zoo)
- [🏖️ TODO](#-todo)
- [🪪 License](#-license)
- [📮 Contact](#-contact)

## 👀 Overview

![Teaser Image](assets/teaser.png)

> Rather than restricting VLM reasoning to a discrete language space with limited representational capacity, **CoVT** forms a visual thought chain that enables VLMs to reason in continuous visual space. By introducing *continuous visual tokens* that encode perceptual cues (e.g., segmentation, depth, instance, and edge structure), CoVT composes *chains of textual and visual thoughts* that link semantic reasoning with perceptual grounding. These visual “thought chains” bridge language and vision, enabling fine-grained understanding, spatial precision, and geometric awareness beyond the reach of text-based reasoning.


<details>
<summary><strong>💡 Abstract</strong></summary>

<br>

Vision–Language Models (VLMs) excel at reasoning in linguistic space but struggle with perceptual understanding that requires dense visual perception, *e.g.*, spatial reasoning and geometric awareness.This limitation stems from the fact that current VLMs have limited mechanisms to capture dense visual information across spatial dimensions. We introduce **Chain-of-Visual-Thought (CoVT)**, a framework that enables VLMs to reason not only in words but also through **continuous visual tokens** — compact latent representations that encode rich perceptual cues. Within a small budget of roughly **20 tokens**, CoVT distills knowledge from lightweight vision experts capturing complementary properties such as **2D appearance, 3D geometry, spatial layout, and edge structure**. During training, a VLM equipped with CoVT autoregressively predicts these visual tokens to reconstruct dense supervision signals (*e.g.*, depth, segmentation, edges, and DINO features). At inference, the model reasons directly in the continuous visual-token space, preserving efficiency while optionally decoding dense predictions for interpretability. Evaluated across more than **ten diverse perception benchmarks**, including CV-Bench, MMVP, RealWorldQA, MMStar, WorldMedQA, and HRBench, integrating CoVT into strong VLMs such as **Qwen2.5-VL** and **LLaVA** consistently improves performance by **3% to 16%**, demonstrating that compact continuous visual thinking enables more precise, grounded, and interpretable multimodal intelligence.

<br>
</details>

<details>
<summary><strong>🧩 Pipeline</strong></summary>

<br>

![Pipeline Image](assets/pipeline.png)

> **Continuous visual thinking with CoVT.** CoVT introduces compact, continuous visual tokens that encode fine-grained perceptual cues, such as object localization, spatial structure, and scene semantics, directly into VLM reasoning. These tokens ground multimodal reasoning in visual space, enabling the model to capture fine-grained relationships across vision-centric tasks (e.g., counting, depth ordering, and scene understanding) without relying on external tools. They can also be decoded into dense predictions, offering human-interpretable visualizations of the model's reasoning process.

![Method Image](assets/method.png)

> **The training pipeline of CoVT.** CoVT first generates the thinking process, containing visual thinking tokens, and then leverages these visual thoughts to condition next-token prediction and reason the final answer. To endow these tokens with perceptual meaning, we align them with lightweight vision experts (e.g., SAM, DepthAnything, PIDINet, DINO) on their respective tasks during training. Specifically: SAM uses 8 visual tokens as mask prompts; DepthAnything uses 4 tokens to reconstruct depth; PIDINet uses 4 tokens to reconstruct edges; and DINO uses 4 tokens to match patch-level features. The VLM is finetuned with LoRA and all projection layers are trainable. ***Note: During inference, dense predictions are decoded only when interpretability is desired; otherwise, reasoning occurs entirely in the latent visual space.***

<br>
</details>

<details>
<summary><strong>💫 Results</strong></summary>

<br>

![Results Image](assets/results.png)

> **Comparison of CoVT with the baseline and closed-source models.** CoVT delivers consistent improvements across all vision-centric benchmarks and further reveals that each type of visual token contributes most effectively to the tasks that align with its encoded perceptual information.

![Results Image](assets/results_llava.png)

> **Comparison between CoVT and Aurora based on LLaVA-v1.5-13B.** $^{\dag}$ indicates our reproduced results based on the provided checkpoints.

<br>
</details>

## 🚀 Quick Start!

### Evaluation

To ensure consistency and reproducibility, we use **VLMEvalKit** as the framework for evaluating models. In our repository, we have forked a copy of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). You can have a quick start by following this [instruction](docs/Eval.md).

### Gradio Demo

We provid an interactive demo built with [Gradio](https://github.com/gradio-app/gradio), showcasing a conversational interface powered by the CoVT VLM. The demo allows users to upload images, ask questions, and interact with the model in real time through a simple web UI. You can have a quick start following [here](docs/Demo.md).

![Gradio Demo Image](assets/gradio_demo.png)

### Training CoVT

Our training data is released [here](https://huggingface.co/datasets/Wakals/CoVT-Dataset)!

<details>
<summary><strong>Dataset Composition</strong></summary>

<br>

![Data Image](assets/data.png)

> CoVT dataset utilizes some subsets of LLaVA-OneVision, and merges the filtered TallyQA dataset and ADE20K-Depth from Aurora.

<br>
</details>


To reproduce our method, please follow [this instruction](docs/Train.md) and start training CoVT! 🍻

## 🤗 Model Zoo

A collection of CoVT models on Hugging Face with benchmark performance:

|Baseline| Segment | Depth | DINO | Edge | Parameters | CV-Bench | Link |
|-----|--------|--------|------|------|------------|----------|------|
|Qwen2.5-VL-7B-Instruct| ✔ |   |   |   | 7B (+1B) | 77.9    | 🤗 [HuggingFace](https://huggingface.co/Wakals/CoVT-7B-seg) |
|Qwen2.5-VL-7B-Instruct|   | ✔ |   |   | 7B (+1B) | 78.7    | 🤗 [HuggingFace](https://huggingface.co/Wakals/CoVT-7B-depth) |
|Qwen2.5-VL-7B-Instruct| ✔ | ✔ | ✔ |   | 7B (+1B) | **80.0**| 🤗 [HuggingFace](https://huggingface.co/Wakals/CoVT-7B-seg_depth_dino) |
|Qwen2.5-VL-7B-Instruct| ✔ | ✔ | ✔ | ✔ | 7B (+1B) | 79.8    | 🤗 [HuggingFace](https://huggingface.co/Wakals/CoVT-7B-seg_depth_dino_edge) |
|LLaVA-v1.5-13B        |   | ✔ |   |   | 13B (+1B)| 59.9    | 🤗 [HuggingFace](https://huggingface.co/Wakals/CoVT-LLaVA-13B-depth) |

> `+1B` denotes the parameters of the projection layer for decoding the visual thinking tokens. We don't nned these parameters during inference!

## 🏖️ TODO

- [x] Release our model weights on Hugging Face.
- [x] Release the evaluation code.
- [x] Release the Gradio demo code.
- [x] Release the dataset.
- [x] Release the training code.
- [ ] Support huggingface demo.
- [ ] Support more VLMs as the base models.

## 🪪 License

The majority of CoVT is licensed under the Apache License, however portions of the project are available under their own license terms: Qwen series, SAM, Depth Anything v2, and DINOv2 are licensed under Apache, PIDINet are licensed under their own license; If you later add other third party code, please keep this license info updated, and please let us know if that component is licensed under something other than Apache, CC-BY-NC, MIT, or CC0.

## 📮 Contact

For feedback, or collaboration opportunities, feel free to reach out!

For general questions, feel free to drop us an email at ymk4474@gmail.com or xdwang@eecs.berkeley.edu.

If you're running into code or implementation issues, the best way is to open an issue right here in the repo (highly recommended!) — chances are your question might help someone else too. 😊

## Citation

If you use this work in your research, please cite:

```
coming soon...
```
