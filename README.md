# Awesome Multimodal Benchmarks
A list of papers and open-sources on multimodal benchmarks, designed to keep pace with the expected surge in research in the coming months. If you have any additions or suggestions, please feel free to contribute.

## Bibtex

If you find our survey helpful, please consider citing our paper:

```bibtex
@article{li2024survey,
  title={A Survey on Multimodal Benchmarks: In the Era of Large AI Models},
  author={Li, Lin and Chen, Guikun and Shi, Hanrong and Xiao, Jun and Chen, Long},
  journal={arXiv preprint arXiv:2409.18142},
  year={2024}
}
```

## Table of contents

- [Understanding Benchmarks](#understanding-benchmarks)
- [Reasoning Benchmarks](#reasoning-benchmarks)
- [Generation Benchmarks](#generation-benchmarks)
- [Application Benchmarks](#application-benchmarks)

## Understanding Benchmarks:
### 1. (EQBEN) Equivariant Similarity for Vision-Language Foundation Models
**Date: 2023.03.25**

**Affiliation**: Nanyang Technological University
<details span>
<summary><b>Abstract</b></summary>
The capability to process multiple images is crucial for Large Vision-Language Models (LVLMs) to develop a more thorough and nuanced understanding of a scene. Recent multi-image LVLMs have begun to address this need. However, their evaluation has not kept pace with their development. To fill this gap, we introduce the Multimodal Multi-image Understanding (MMIU) benchmark, a comprehensive evaluation suite designed to assess LVLMs across a wide range of multi-image tasks. MMIU encompasses 7 types of multi-image relationships, 52 tasks, 77K images, and 11K meticulously curated multiple-choice questions, making it the most extensive benchmark of its kind. Our evaluation of 24 popular LVLMs, including both open-source and proprietary models, reveals significant challenges in multi-image comprehension, particularly in tasks involving spatial understanding. Even the most advanced models, such as GPT-4o, achieve only 55.7% accuracy on MMIU. Through multi-faceted analytical experiments, we identify key performance gaps and limitations, providing valuable insights for future model and data improvements. We aim for MMIU to advance the frontier of LVLM research and development, moving us toward achieving sophisticated multimodal multi-image user interactions.
</details>

  [📄 Paper](https://arxiv.org/pdf/2303.14465) | [💻 Code](https://github.com/Wangt-CN/EqBen)

### 2. (MMC4) Equivariant Similarity for Vision-Language Foundation Models
**Date: 2023.04.06**

**Affiliation**: University of California
<details span>
<summary><b>Abstract</b></summary>
In-context vision and language models like Flamingo support arbitrarily interleaved sequences of images and text as input. This format not only enables few-shot learning via interleaving independent supervised (image, text) examples, but also, more complex prompts involving interaction between images, e.g., "What do image A and image B have in common?" To support this interface, pretraining occurs over web corpora that similarly contain interleaved images+text. To date, however, large-scale data of this form have not been publicly available. We release Multimodal C4, an augmentation of the popular text-only C4 corpus with images interleaved. We use a linear assignment algorithm to place images into longer bodies of text using CLIP features, a process that we show outperforms alternatives. Multimodal C4 spans everyday topics like cooking, travel, technology, etc. A manual inspection of a random sample of documents shows that a vast majority (88%) of images are topically relevant, and that linear assignment frequently selects individual sentences specifically well-aligned with each image (80%). After filtering NSFW images, ads, etc., the resulting corpus consists of 101.2M documents with 571M images interleaved in 43B English tokens.
</details>

  [📄 Paper](https://arxiv.org/pdf/2304.06939) | [💻 Code](https://github.com/allenai/mmc4)

### n. (MMIU) MMIU: Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models 
**Date: 2024.08.05**

**Authors**: Fanqing Meng, Jin Wang, Chuanhao Li, Quanfeng Lu, Hao Tian, Jiaqi Liao, Xizhou Zhu, Jifeng Dai, Yu Qiao, Ping Luo, Kaipeng Zhang, Wenqi Shao
<details span>
<summary><b>Abstract</b></summary>
The capability to process multiple images is crucial for Large Vision-Language Models (LVLMs) to develop a more thorough and nuanced understanding of a scene. Recent multi-image LVLMs have begun to address this need. However, their evaluation has not kept pace with their development. To fill this gap, we introduce the Multimodal Multi-image Understanding (MMIU) benchmark, a comprehensive evaluation suite designed to assess LVLMs across a wide range of multi-image tasks. MMIU encompasses 7 types of multi-image relationships, 52 tasks, 77K images, and 11K meticulously curated multiple-choice questions, making it the most extensive benchmark of its kind. Our evaluation of 24 popular LVLMs, including both open-source and proprietary models, reveals significant challenges in multi-image comprehension, particularly in tasks involving spatial understanding. Even the most advanced models, such as GPT-4o, achieve only 55.7% accuracy on MMIU. Through multi-faceted analytical experiments, we identify key performance gaps and limitations, providing valuable insights for future model and data improvements. We aim for MMIU to advance the frontier of LVLM research and development, moving us toward achieving sophisticated multimodal multi-image user interactions.
</details>

  [📄 Paper](https://arxiv.org/pdf/2408.02718) | [🌐 Project Page](https://mmiu-bench.github.io/) | [💻 Code](https://github.com/OpenGVLab/MMIU)

## Reasoning Benchmarks:


## Generation Benchmarks:


## Application Benchmarks:
### 1. (MineDojo) MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge
**Date: 2022.06.17**

**Affiliation**: NVIDIA
<details span>
<summary><b>Abstract</b></summary>
Autonomous agents have made great strides in specialist domains like Atari games and Go. However, they typically learn tabula rasa in isolated environments with limited and manually conceived objectives, thus failing to generalize across a wide spectrum of tasks and capabilities. Inspired by how humans continually learn and adapt in the open world, we advocate a trinity of ingredients for building generalist agents: 1) an environment that supports a multitude of tasks and goals, 2) a large-scale database of multimodal knowledge, and 3) a flexible and scalable agent architecture. We introduce MineDojo, a new framework built on the popular Minecraft game that features a simulation suite with thousands of diverse open-ended tasks and an internet-scale knowledge base with Minecraft videos, tutorials, wiki pages, and forum discussions. Using MineDojo's data, we propose a novel agent learning algorithm that leverages large pre-trained video-language models as a learned reward function. Our agent is able to solve a variety of open-ended tasks specified in free-form language without any manually designed dense shaping reward. We open-source the simulation suite, knowledge bases, algorithm implementation, and pretrained models (this https URL) to promote research towards the goal of generally capable embodied agents.
</details>

[📄 Paper](https://arxiv.org/pdf/2206.08853) | [🌐 Project Page](https://minedojo.org/) | [💻 Code](https://github.com/MineDojo/MineDojo)
