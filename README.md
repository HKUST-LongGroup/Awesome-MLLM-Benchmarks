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

### 2. (MMC4) Multimodal C4: An Open, Billion-scale Corpus of Images Interleaved with Text
**Date: 2023.04.06**

**Affiliation**: University of California
<details span>
<summary><b>Abstract</b></summary>
In-context vision and language models like Flamingo support arbitrarily interleaved sequences of images and text as input. This format not only enables few-shot learning via interleaving independent supervised (image, text) examples, but also, more complex prompts involving interaction between images, e.g., "What do image A and image B have in common?" To support this interface, pretraining occurs over web corpora that similarly contain interleaved images+text. To date, however, large-scale data of this form have not been publicly available. We release Multimodal C4, an augmentation of the popular text-only C4 corpus with images interleaved. We use a linear assignment algorithm to place images into longer bodies of text using CLIP features, a process that we show outperforms alternatives. Multimodal C4 spans everyday topics like cooking, travel, technology, etc. A manual inspection of a random sample of documents shows that a vast majority (88%) of images are topically relevant, and that linear assignment frequently selects individual sentences specifically well-aligned with each image (80%). After filtering NSFW images, ads, etc., the resulting corpus consists of 101.2M documents with 571M images interleaved in 43B English tokens.
</details>

  [📄 Paper](https://arxiv.org/pdf/2304.06939) | [💻 Code](https://github.com/allenai/mmc4)


### 3. (OwlEval) mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality
**Date**: 2023.04.27

**Affiliation**: DAMO Academy
<details span>
<summary><b>Abstract</b></summary>
Large language models (LLMs) have demonstrated impressive zero-shot abilities
on a variety of open-ended tasks, while recent research has also explored the
use of LLMs for multi-modal generation. In this study, we introduce mPLUG-Owl,
a novel training paradigm that equips LLMs with multi-modal abilities through
modularized learning of foundation LLM, a visual knowledge module, and a visual
abstractor module. This approach can support multiple modalities and facilitate
diverse unimodal and multimodal abilities through modality collaboration. The
training paradigm of mPLUG-Owl involves a two-stage method for aligning image
and text, which learns visual knowledge with the assistance of LLM while
maintaining and even improving the generation abilities of LLM. In the first
stage, the visual knowledge module and abstractor module are trained with a
frozen LLM module to align the image and text. In the second stage,
language-only and multi-modal supervised datasets are used to jointly fine-tune
a low-rank adaption (LoRA) module on LLM and the abstractor module by freezing
the visual knowledge module. We carefully build a visually-related instruction
evaluation set OwlEval. Experimental results show that our model outperforms
existing multi-modal models, demonstrating mPLUG-Owl's impressive instruction
and visual understanding ability, multi-turn conversation ability, and
knowledge reasoning ability. Besides, we observe some unexpected and exciting
abilities such as multi-image correlation and scene text understanding, which
makes it possible to leverage it for harder real scenarios, such as vision-only
document comprehension. Our code, pre-trained model, instruction-tuned models,
and evaluation set are available at https://github.com/X-PLUG/mPLUG-Owl. The
online demo is available at https://www.modelscope.cn/studios/damo/mPLUG-Owl.
</details>

[📄 Paper](https://arxiv.org/pdf/2304.14178) | [🌐 Project Page](https://www.modelscope.cn/studios/damo/mPLUG-Owl) | [💻 Code](https://github.com/X-PLUG/mPLUG-Owl)


### 4. (OCRBench) OCRBench: On the Hidden Mystery of OCR in Large Multimodal Models
**Date**: 2023.05.13

**Affiliation**: Huazhong University of Science and Technology
<details span>
<summary><b>Abstract</b></summary>
Large models have recently played a dominant role in natural language
processing and multimodal vision-language learning. However, their
effectiveness in text-related visual tasks remains relatively unexplored. In
this paper, we conducted a comprehensive evaluation of Large Multimodal Models,
such as GPT4V and Gemini, in various text-related visual tasks including Text
Recognition, Scene Text-Centric Visual Question Answering (VQA),
Document-Oriented VQA, Key Information Extraction (KIE), and Handwritten
Mathematical Expression Recognition (HMER). To facilitate the assessment of
Optical Character Recognition (OCR) capabilities in Large Multimodal Models, we
propose OCRBench, a comprehensive evaluation benchmark. OCRBench contains 29
datasets, making it the most comprehensive OCR evaluation benchmark available.
Furthermore, our study reveals both the strengths and weaknesses of these
models, particularly in handling multilingual text, handwritten text,
non-semantic text, and mathematical expression recognition. Most importantly,
the baseline results presented in this study could provide a foundational
framework for the conception and assessment of innovative strategies targeted
at enhancing zero-shot multimodal techniques. The evaluation pipeline and
benchmark are available at https://github.com/Yuliang-Liu/MultimodalOCR.
</details>

[📄 Paper](https://arxiv.org/pdf/2305.07895) | [💻 Code](https://github.com/Yuliang-Liu/MultimodalOCR)

### 5. (GVT-Bench) What Makes for Good Visual Tokenizers for Large Language Models?
**Date**: 2023.05.20

**Affiliation**: National University of Singapore
<details span>
<summary><b>Abstract</b></summary>
We empirically investigate proper pre-training methods to build good visual
tokenizers, making Large Language Models (LLMs) powerful Multimodal Large
Language Models (MLLMs). In our benchmark, which is curated to evaluate MLLMs
visual semantic understanding and fine-grained perception capabilities, we
discussed different visual tokenizers pre-trained with dominant methods (i.e.,
DeiT, CLIP, MAE, DINO), and observe that: i) Fully/weakly supervised models
capture more semantics than self-supervised models, but the gap is narrowed by
scaling up the pre-training dataset. ii) Self-supervised models are better at
fine-grained perception, where patch-level supervision is particularly
effective. iii) Tuning the visual tokenizer leads to the loss of semantics
obtained from large-scale pretraining, which is unfavorable with relatively
small-scale instruction-tuning dataset. Given the findings, we reviewed methods
that attempted to unify semantics and fine-grained visual understanding, e.g.,
patch-level feature distillation with semantically-rich targets. We obtain an
intriguing insight mask-based strategies that were once all the rage may not be
applicable for obtaining good visual tokenizers. Based on this critical
observation, we obtain a new MLLM equipped with a tailored Good Visual
Tokenizer (GVT), which exhibits strong visual comprehension capability at
multiple scales. In particular, without introducing extra parameters and
task-specific fine-tuning, GVT achieves superior performance on visual question
answering, image captioning, and other fine-grained visual understanding tasks
such as object counting and multi-class identification.
</details>

[📄 Paper](https://arxiv.org/abs/2305.12223) | [💻 Code](https://github.com/TencentARC/GVT)

### 6. (PerceptionTest) Perception Test: A Diagnostic Benchmark for Multimodal Video Models
**Date**: 2023.05.23

**Affiliation**: DeepMind
<details span>
<summary><b>Abstract</b></summary>
We propose a novel multimodal video benchmark - the Perception Test - to
evaluate the perception and reasoning skills of pre-trained multimodal models
(e.g. Flamingo, SeViLA, or GPT-4). Compared to existing benchmarks that focus
on computational tasks (e.g. classification, detection or tracking), the
Perception Test focuses on skills (Memory, Abstraction, Physics, Semantics) and
types of reasoning (descriptive, explanatory, predictive, counterfactual)
across video, audio, and text modalities, to provide a comprehensive and
efficient evaluation tool. The benchmark probes pre-trained models for their
transfer capabilities, in a zero-shot / few-shot or limited finetuning regime.
For these purposes, the Perception Test introduces 11.6k real-world videos, 23s
average length, designed to show perceptually interesting situations, filmed by
around 100 participants worldwide. The videos are densely annotated with six
types of labels (multiple-choice and grounded video question-answers, object
and point tracks, temporal action and sound segments), enabling both language
and non-language evaluations. The fine-tuning and validation splits of the
benchmark are publicly available (CC-BY license), in addition to a challenge
server with a held-out test split. Human baseline results compared to
state-of-the-art video QA models show a substantial gap in performance (91.4%
vs 46.2%), suggesting that there is significant room for improvement in
multimodal video understanding.
  Dataset, baseline code, and challenge server are available at
https://github.com/deepmind/perception_test
</details>

[📄 Paper](https://arxiv.org/abs/2305.13786) | [💻 Code](https://github.com/deepmind/perception_test)

### 7. (CODE) Contextual Object Detection with Multimodal Large Language Models
**Date**: 2023.05.29

**Affiliation**: Nanyang Technological University
<details span>
<summary><b>Abstract</b></summary>
Recent Multimodal Large Language Models (MLLMs) are remarkable in
vision-language tasks, such as image captioning and question answering, but
lack the essential perception ability, i.e., object detection. In this work, we
address this limitation by introducing a novel research problem of contextual
object detection -- understanding visible objects within different human-AI
interactive contexts. Three representative scenarios are investigated,
including the language cloze test, visual captioning, and question answering.
Moreover, we present ContextDET, a unified multimodal model that is capable of
end-to-end differentiable modeling of visual-language contexts, so as to
locate, identify, and associate visual objects with language inputs for
human-AI interaction. Our ContextDET involves three key submodels: (i) a visual
encoder for extracting visual representations, (ii) a pre-trained LLM for
multimodal context decoding, and (iii) a visual decoder for predicting bounding
boxes given contextual object words. The new generate-then-detect framework
enables us to detect object words within human vocabulary. Extensive
experiments show the advantages of ContextDET on our proposed CODE benchmark,
open-vocabulary detection, and referring image segmentation. Github:
https://github.com/yuhangzang/ContextDET.
</details>

[📄 Paper](https://arxiv.org/abs/2305.18279) | [💻 Code](https://github.com/yuhangzang/ContextDET)

### 8. (Lvlm-ehub) LVLM-eHub: A Comprehensive Evaluation Benchmark for Large Vision-Language Models
**Date**: 2023.06.09

**Affiliation**: Shanghai AI Laborato
<details span>
<summary><b>Abstract</b></summary>
Large Vision-Language Models (LVLMs) have recently played a dominant role in
multimodal vision-language learning. Despite the great success, it lacks a
holistic evaluation of their efficacy. This paper presents a comprehensive
evaluation of publicly available large multimodal models by building a LVLM
evaluation Hub (LVLM-eHub). Our LVLM-eHub consists of $8$ representative LVLMs
such as InstructBLIP and MiniGPT-4, which are thoroughly evaluated by a
quantitative capability evaluation and an online arena platform. The former
evaluates $6$ categories of multimodal capabilities of LVLMs such as visual
question answering and embodied artificial intelligence on $47$ standard
text-related visual benchmarks, while the latter provides the user-level
evaluation of LVLMs in an open-world question-answering scenario. The study
reveals several innovative findings. First, instruction-tuned LVLM with massive
in-domain data such as InstructBLIP heavily overfits many existing tasks,
generalizing poorly in the open-world scenario. Second, instruction-tuned LVLM
with moderate instruction-following data may result in object hallucination
issues (i.e., generate objects that are inconsistent with target images in the
descriptions). It either makes the current evaluation metric such as CIDEr for
image captioning ineffective or generates wrong answers. Third, employing a
multi-turn reasoning evaluation framework can mitigate the issue of object
hallucination, shedding light on developing an effective pipeline for LVLM
evaluation. The findings provide a foundational framework for the conception
and assessment of innovative strategies aimed at enhancing zero-shot multimodal
techniques. Our LVLM-eHub will be available at
https://github.com/OpenGVLab/Multi-Modality-Arena
</details>

[📄 Paper](https://arxiv.org/abs/2306.09265) | [💻 Code](https://github.com/OpenGVLab/Multi-Modality-Arena)

### 9. (LAMM) LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark
**Date**: 2023.06.11

**Affiliation**: Shanghai AI Laboratory
<details span>
<summary><b>Abstract</b></summary>
Large language models have emerged as a promising approach towards achieving
general-purpose AI agents. The thriving open-source LLM community has greatly
accelerated the development of agents that support human-machine dialogue
interaction through natural language processing. However, human interaction
with the world extends beyond only text as a modality, and other modalities
such as vision are also crucial. Recent works on multi-modal large language
models, such as GPT-4V and Bard, have demonstrated their effectiveness in
handling visual modalities. However, the transparency of these works is limited
and insufficient to support academic research. To the best of our knowledge, we
present one of the very first open-source endeavors in the field, LAMM,
encompassing a Language-Assisted Multi-Modal instruction tuning dataset,
framework, and benchmark. Our aim is to establish LAMM as a growing ecosystem
for training and evaluating MLLMs, with a specific focus on facilitating AI
agents capable of bridging the gap between ideas and execution, thereby
enabling seamless human-AI interaction. Our main contribution is three-fold: 1)
We present a comprehensive dataset and benchmark, which cover a wide range of
vision tasks for 2D and 3D vision. Extensive experiments validate the
effectiveness of our dataset and benchmark. 2) We outline the detailed
methodology of constructing multi-modal instruction tuning datasets and
benchmarks for MLLMs, enabling rapid scaling and extension of MLLM research to
diverse domains, tasks, and modalities. 3) We provide a primary but potential
MLLM training framework optimized for modality extension. We also provide
baseline models, comprehensive experimental observations, and analysis to
accelerate future research. Our baseline model is trained within 24 A100 GPU
hours, framework supports training with V100 and RTX3090 is available thanks to
the open-source society.
</details>

[📄 Paper](https://arxiv.org/abs/2306.06687) | [🌐 Project Page](https://openlamm.github.io/) | [💻 Code](https://github.com/OpenGVLab/LAMM)

### 10. (MME) MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models
**Date**: 2023.06.13

**Affiliation**: Tencent Youtu Lab
<details span>
<summary><b>Abstract</b></summary>
Multimodal Large Language Model (MLLM) relies on the powerful LLM to perform
multimodal tasks, showing amazing emergent abilities in recent studies, such as
writing poems based on an image. However, it is difficult for these case
studies to fully reflect the performance of MLLM, lacking a comprehensive
evaluation. In this paper, we fill in this blank, presenting the first
comprehensive MLLM Evaluation benchmark MME. It measures both perception and
cognition abilities on a total of 14 subtasks. In order to avoid data leakage
that may arise from direct use of public datasets for evaluation, the
annotations of instruction-answer pairs are all manually designed. The concise
instruction design allows us to fairly compare MLLMs, instead of struggling in
prompt engineering. Besides, with such an instruction, we can also easily carry
out quantitative statistics. A total of 30 advanced MLLMs are comprehensively
evaluated on our MME, which not only suggests that existing MLLMs still have a
large room for improvement, but also reveals the potential directions for the
subsequent model optimization. The data application manner and online
leaderboards are released at
https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation.
</details>

[📄 Paper](https://arxiv.org/abs/2306.13394) | [💻 Code](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)

### 11. (Obelics) OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents
**Date**: 2023.06.16

**Affiliation**: Hugging Face
<details span>
<summary><b>Abstract</b></summary>
Large multimodal models trained on natural documents, which interleave images
and text, outperform models trained on image-text pairs on various multimodal
benchmarks. However, the datasets used to train these models have not been
released, and the collection process has not been fully specified. We introduce
the OBELICS dataset, an open web-scale filtered dataset of interleaved
image-text documents comprising 141 million web pages extracted from Common
Crawl, 353 million associated images, and 115 billion text tokens. We describe
the dataset creation process, present comprehensive filtering rules, and
provide an analysis of the dataset's content. To show the viability of OBELICS,
we train vision and language models of 9 and 80 billion parameters named
IDEFICS, and obtain competitive performance on different multimodal benchmarks.
We release our dataset, models and code.
</details>

[📄 Paper](https://arxiv.org/pdf/2306.16527) | [💻 Code](https://github.com/huggingface/OBELICS)

### 12. (Open-VQA) What Matters in Training a GPT4-Style Language Model with Multimodal Inputs?
**Date**: 2023.07.05

**Affiliation**: ByteDance Inc.
<details span>
<summary><b>Abstract</b></summary>
Recent advancements in Large Language Models (LLMs) such as GPT4 have
displayed exceptional multi-modal capabilities in following open-ended
instructions given images. However, the performance of these models heavily
relies on design choices such as network structures, training data, and
training strategies, and these choices have not been extensively discussed in
the literature, making it difficult to quantify progress in this field. To
address this issue, this paper presents a systematic and comprehensive study,
quantitatively and qualitatively, on training such models. We implement over 20
variants with controlled settings. Concretely, for network structures, we
compare different LLM backbones and model designs. For training data, we
investigate the impact of data and sampling strategies. For instructions, we
explore the influence of diversified prompts on the instruction-following
ability of the trained models. For benchmarks, we contribute the first, to our
best knowledge, comprehensive evaluation set including both image and video
tasks through crowd-sourcing. Based on our findings, we present Lynx, which
performs the most accurate multi-modal understanding while keeping the best
multi-modal generation ability compared to existing open-sourced GPT4-style
models.
</details>

[📄 Paper](https://arxiv.org/abs/2307.02469) | [🌐 Project Page](https://lynx-llm.github.io/) | [💻 Code](https://github.com/bytedance/lynx-llm)

### 13. (MMBench) MMBench: Is Your Multi-modal Model an All-around Player?
**Date**: 2023.07.06

**Affiliation**: Shanghai AI Laboratory
<details span>
<summary><b>Abstract</b></summary>
Large vision-language models (VLMs) have recently achieved remarkable
progress, exhibiting impressive multimodal perception and reasoning abilities.
However, effectively evaluating these large VLMs remains a major challenge,
hindering future development in this domain. Traditional benchmarks like VQAv2
or COCO Caption provide quantitative performance measurements but lack
fine-grained ability assessment and robust evaluation metrics. Meanwhile,
subjective benchmarks, such as OwlEval, offer comprehensive evaluations of a
model's abilities by incorporating human labor, which is not scalable and may
display significant bias. In response to these challenges, we propose MMBench,
a bilingual benchmark for assessing the multi-modal capabilities of VLMs.
MMBench methodically develops a comprehensive evaluation pipeline, primarily
comprised of the following key features: 1. MMBench is meticulously curated
with well-designed quality control schemes, surpassing existing similar
benchmarks in terms of the number and variety of evaluation questions and
abilities; 2. MMBench introduces a rigorous CircularEval strategy and
incorporates large language models to convert free-form predictions into
pre-defined choices, which helps to yield accurate evaluation results for
models with limited instruction-following capabilities. 3. MMBench incorporates
multiple-choice questions in both English and Chinese versions, enabling an
apples-to-apples comparison of VLMs' performance under a bilingual context. To
summarize, MMBench is a systematically designed objective benchmark for a
robust and holistic evaluation of vision-language models. We hope MMBench will
assist the research community in better evaluating their models and facilitate
future progress in this area. The evalutation code of MMBench has been
integrated into VLMEvalKit: https://github.com/open-compass/VLMEvalKit.
</details>

[📄 Paper](https://arxiv.org/abs/2307.06281) | [💻 Code](https://github.com/open-compass/VLMEvalKit)

### 14. (SEED-Bench) SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension
**Date**: 2023.07.30

**Affiliation**: Tencent AI Lab
<details span>
<summary><b>Abstract</b></summary>
Based on powerful Large Language Models (LLMs), recent generative Multimodal
Large Language Models (MLLMs) have gained prominence as a pivotal research
area, exhibiting remarkable capability for both comprehension and generation.
In this work, we address the evaluation of generative comprehension in MLLMs as
a preliminary step towards a comprehensive assessment of generative models, by
introducing a benchmark named SEED-Bench. SEED-Bench consists of 19K multiple
choice questions with accurate human annotations (x 6 larger than existing
benchmarks), which spans 12 evaluation dimensions including the comprehension
of both the image and video modality. We develop an advanced pipeline for
generating multiple-choice questions that target specific evaluation
dimensions, integrating both automatic filtering and manual verification
processes. Multiple-choice questions with groundtruth options derived from
human annotation enables an objective and efficient assessment of model
performance, eliminating the need for human or GPT intervention during
evaluation. We further evaluate the performance of 18 models across all 12
dimensions, covering both the spatial and temporal understanding. By revealing
the limitations of existing MLLMs through evaluation results, we aim for
SEED-Bench to provide insights for motivating future research. We will launch
and consistently maintain a leaderboard to provide a platform for the community
to assess and investigate model capability.
</details>

[📄 Paper](https://arxiv.org/abs/2307.16125) | [💻 Code](https://github.com/AILab-CVC/SEED-Bench)

### 15. (MovieChat-1k) MovieChat: From Dense Token to Sparse Memory for Long Video Understanding
**Date**: 2023.07.31

**Affiliation**: Zhejiang University
<details span>
<summary><b>Abstract</b></summary>
Recently, integrating video foundation models and large language models to
build a video understanding system can overcome the limitations of specific
pre-defined vision tasks. Yet, existing systems can only handle videos with
very few frames. For long videos, the computation complexity, memory cost, and
long-term temporal connection impose additional challenges. Taking advantage of
the Atkinson-Shiffrin memory model, with tokens in Transformers being employed
as the carriers of memory in combination with our specially designed memory
mechanism, we propose the MovieChat to overcome these challenges. MovieChat
achieves state-of-the-art performance in long video understanding, along with
the released MovieChat-1K benchmark with 1K long video and 14K manual
annotations for validation of the effectiveness of our method.
</details>

[📄 Paper](https://arxiv.org/pdf/2307.16449) | [🌐 Project Page](https://rese1f.github.io/MovieChat/) | [💻 Code](https://github.com/rese1f/MovieChat)

### 16. (MM-Vet) MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities
**Date**: 2023.08.02

**Affiliation**: National University of Singapore
<details span>
<summary><b>Abstract</b></summary>
We propose MM-Vet, an evaluation benchmark that examines large multimodal
models (LMMs) on complicated multimodal tasks. Recent LMMs have shown various
intriguing abilities, such as solving math problems written on the blackboard,
reasoning about events and celebrities in news images, and explaining visual
jokes. Rapid model advancements pose challenges to evaluation benchmark
development. Problems include: (1) How to systematically structure and evaluate
the complicated multimodal tasks; (2) How to design evaluation metrics that
work well across question and answer types; and (3) How to give model insights
beyond a simple performance ranking. To this end, we present MM-Vet, designed
based on the insight that the intriguing ability to solve complicated tasks is
often achieved by a generalist model being able to integrate different core
vision-language (VL) capabilities. MM-Vet defines 6 core VL capabilities and
examines the 16 integrations of interest derived from the capability
combination. For evaluation metrics, we propose an LLM-based evaluator for
open-ended outputs. The evaluator enables the evaluation across different
question types and answer styles, resulting in a unified scoring metric. We
evaluate representative LMMs on MM-Vet, providing insights into the
capabilities of different LMM system paradigms and models. Code and data are
available at https://github.com/yuweihao/MM-Vet.
</details>

[📄 Paper](https://arxiv.org/abs/2308.02490) | [💻 Code](https://github.com/yuweihao/MM-Vet)

### 17. (TinyLVLM) TinyLVLM-eHub: Towards Comprehensive and Efficient Evaluation for Large Vision-Language Models
**Date**: 2023.08.07

**Affiliation**: Shanghai AI Laboratory
<details span>
<summary><b>Abstract</b></summary>
Recent advancements in Large Vision-Language Models (LVLMs) have demonstrated
significant progress in tackling complex multimodal tasks. Among these
cutting-edge developments, Google's Bard stands out for its remarkable
multimodal capabilities, promoting comprehensive comprehension and reasoning
across various domains. This work presents an early and holistic evaluation of
LVLMs' multimodal abilities, with a particular focus on Bard, by proposing a
lightweight variant of LVLM-eHub, named Tiny LVLM-eHub. In comparison to the
vanilla version, Tiny LVLM-eHub possesses several appealing properties.
Firstly, it provides a systematic assessment of six categories of multimodal
capabilities, including visual perception, visual knowledge acquisition, visual
reasoning, visual commonsense, object hallucination, and embodied intelligence,
through quantitative evaluation of $42$ standard text-related visual
benchmarks. Secondly, it conducts an in-depth analysis of LVLMs' predictions
using the ChatGPT Ensemble Evaluation (CEE), which leads to a robust and
accurate evaluation and exhibits improved alignment with human evaluation
compared to the word matching approach. Thirdly, it comprises a mere $2.1$K
image-text pairs, facilitating ease of use for practitioners to evaluate their
own offline LVLMs. Through extensive experimental analysis, this study
demonstrates that Bard outperforms previous LVLMs in most multimodal
capabilities except object hallucination, to which Bard is still susceptible.
Tiny LVLM-eHub serves as a baseline evaluation for various LVLMs and encourages
innovative strategies aimed at advancing multimodal techniques. Our project is
publicly available at \url{https://github.com/OpenGVLab/Multi-Modality-Arena}.
</details>

[📄 Paper](https://arxiv.org/abs/2308.03729) | [💻 Code](https://github.com/OpenGVLab/Multi-Modality-Arena)
### 18. (TouchStone) TouchStone: Evaluating Vision-Language Models by Language Models
**Date**: 2023.08.16

**Affiliation**: Alibaba Group
<details span>
<summary><b>Abstract</b></summary>
Large vision-language models (LVLMs) have recently witnessed rapid
advancements, exhibiting a remarkable capacity for perceiving, understanding,
and processing visual information by connecting visual receptor with large
language models (LLMs). However, current assessments mainly focus on
recognizing and reasoning abilities, lacking direct evaluation of
conversational skills and neglecting visual storytelling abilities. In this
paper, we propose an evaluation method that uses strong LLMs as judges to
comprehensively evaluate the various abilities of LVLMs. Firstly, we construct
a comprehensive visual dialogue dataset TouchStone, consisting of open-world
images and questions, covering five major categories of abilities and 27
subtasks. This dataset not only covers fundamental recognition and
comprehension but also extends to literary creation. Secondly, by integrating
detailed image annotations we effectively transform the multimodal input
content into a form understandable by LLMs. This enables us to employ advanced
LLMs for directly evaluating the quality of the multimodal dialogue without
requiring human intervention. Through validation, we demonstrate that powerful
LVLMs, such as GPT-4, can effectively score dialogue quality by leveraging
their textual capabilities alone, aligning with human preferences. We hope our
work can serve as a touchstone for LVLMs' evaluation and pave the way for
building stronger LVLMs. The evaluation code is available at
https://github.com/OFA-Sys/TouchStone.
</details>

[📄 Paper](https://arxiv.org/abs/2308.16890) | [💻 Code](https://github.com/OFA-Sys/TouchStone)

### 19. (EgoSchema) EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding
**Date**: 2023.08.18

**Affiliation**: UC Berkeley
<details span>
<summary><b>Abstract</b></summary>
We introduce EgoSchema, a very long-form video question-answering dataset,
and benchmark to evaluate long video understanding capabilities of modern
vision and language systems. Derived from Ego4D, EgoSchema consists of over
5000 human curated multiple choice question answer pairs, spanning over 250
hours of real video data, covering a very broad range of natural human activity
and behavior. For each question, EgoSchema requires the correct answer to be
selected between five given options based on a three-minute-long video clip.
While some prior works have proposed video datasets with long clip lengths, we
posit that merely the length of the video clip does not truly capture the
temporal difficulty of the video task that is being considered. To remedy this,
we introduce temporal certificate sets, a general notion for capturing the
intrinsic temporal understanding length associated with a broad range of video
understanding tasks & datasets. Based on this metric, we find EgoSchema to have
intrinsic temporal lengths over 5.7x longer than the second closest dataset and
10x to 100x longer than any other video understanding dataset. Further, our
evaluation of several current state-of-the-art video and language models shows
them to be severely lacking in long-term video understanding capabilities. Even
models with several billions of parameters achieve QA accuracy less than 33%
(random is 20%) on the EgoSchema multi-choice question answering task, while
humans achieve about 76% accuracy. We posit that \name{}{}, with its long
intrinsic temporal structures and diverse complexity, would serve as a valuable
evaluation probe for developing effective long-term video understanding systems
in the future. Data and Zero-shot model evaluation code are open-sourced for
both public and commercial use under the Ego4D license at
http://egoschema.github.io
</details>

[📄 Paper](https://arxiv.org/pdf/2308.09126) | [🌐 Project Page](https://egoschema.github.io/) | [💻 Code](https://github.com/egoschema/egoschema)

### 20. (SeaEval) SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning
**Date**: 2023.09.04

**Affiliation**: A*STAR
<details span>
<summary><b>Abstract</b></summary>
We present SeaEval, a benchmark for multilingual foundation models. In
addition to characterizing how these models understand and reason with natural
language, we also investigate how well they comprehend cultural practices,
nuances, and values. Alongside standard accuracy metrics, we investigate the
brittleness of foundation models in the dimensions of semantics and
multilinguality. Our analyses span both open-sourced and closed models, leading
to empirical results across classic NLP tasks, reasoning, and cultural
comprehension. Key findings indicate (1) Most models exhibit varied behavior
when given paraphrased instructions. (2) Many models still suffer from exposure
bias (e.g., positional bias, majority label bias). (3) For questions rooted in
factual, scientific, and commonsense knowledge, consistent responses are
expected across multilingual queries that are semantically equivalent. Yet,
most models surprisingly demonstrate inconsistent performance on these queries.
(4) Multilingually-trained models have not attained "balanced multilingual"
capabilities. Our endeavors underscore the need for more generalizable semantic
representations and enhanced multilingual contextualization. SeaEval can serve
as a launchpad for more thorough investigations and evaluations for
multilingual and multicultural scenarios.
</details>

[📄 Paper](https://arxiv.org/pdf/2309.04766) | [💻 Code](https://github.com/SeaEval/SeaEval)

### 21. (Dynamic-SUPERB) Dynamic-SUPERB: Towards A Dynamic, Collaborative, and Comprehensive Instruction-Tuning Benchmark for Speech
**Date**: 2023.09.18

**Affiliation**: National Taiwan University
<details span>
<summary><b>Abstract</b></summary>
Text language models have shown remarkable zero-shot capability in
generalizing to unseen tasks when provided with well-formulated instructions.
However, existing studies in speech processing primarily focus on limited or
specific tasks. Moreover, the lack of standardized benchmarks hinders a fair
comparison across different approaches. Thus, we present Dynamic-SUPERB, a
benchmark designed for building universal speech models capable of leveraging
instruction tuning to perform multiple tasks in a zero-shot fashion. To achieve
comprehensive coverage of diverse speech tasks and harness instruction tuning,
we invite the community to collaborate and contribute, facilitating the dynamic
growth of the benchmark. To initiate, Dynamic-SUPERB features 55 evaluation
instances by combining 33 tasks and 22 datasets. This spans a broad spectrum of
dimensions, providing a comprehensive platform for evaluation. Additionally, we
propose several approaches to establish benchmark baselines. These include the
utilization of speech models, text language models, and the multimodal encoder.
Evaluation results indicate that while these baselines perform reasonably on
seen tasks, they struggle with unseen ones. We release all materials to the
public and welcome researchers to collaborate on the project, advancing
technologies in the field together.
</details>

[📄 Paper](https://arxiv.org/pdf/2309.09510) | [💻 Code](https://github.com/dynamic-superb/dynamic-superb)

### 22. (Q-Bench) Q-Bench: A Benchmark for General-Purpose Foundation Models on Low-level Vision
**Date**: 2023.09.25

**Affiliation**: Nanyang Technological University
<details span>
<summary><b>Abstract</b></summary>
The rapid evolution of Multi-modality Large Language Models (MLLMs) has
catalyzed a shift in computer vision from specialized models to general-purpose
foundation models. Nevertheless, there is still an inadequacy in assessing the
abilities of MLLMs on low-level visual perception and understanding. To address
this gap, we present Q-Bench, a holistic benchmark crafted to systematically
evaluate potential abilities of MLLMs on three realms: low-level visual
perception, low-level visual description, and overall visual quality
assessment. a) To evaluate the low-level perception ability, we construct the
LLVisionQA dataset, consisting of 2,990 diverse-sourced images, each equipped
with a human-asked question focusing on its low-level attributes. We then
measure the correctness of MLLMs on answering these questions. b) To examine
the description ability of MLLMs on low-level information, we propose the
LLDescribe dataset consisting of long expert-labelled golden low-level text
descriptions on 499 images, and a GPT-involved comparison pipeline between
outputs of MLLMs and the golden descriptions. c) Besides these two tasks, we
further measure their visual quality assessment ability to align with human
opinion scores. Specifically, we design a softmax-based strategy that enables
MLLMs to predict quantifiable quality scores, and evaluate them on various
existing image quality assessment (IQA) datasets. Our evaluation across the
three abilities confirms that MLLMs possess preliminary low-level visual
skills. However, these skills are still unstable and relatively imprecise,
indicating the need for specific enhancements on MLLMs towards these abilities.
We hope that our benchmark can encourage the research community to delve deeper
to discover and enhance these untapped potentials of MLLMs. Project Page:
https://q-future.github.io/Q-Bench.
</details>

[📄 Paper](https://arxiv.org/abs/2309.14181) | [💻 Code](https://q-future.github.io/Q-Bench)

### 23. (CHef) ChEF: A Comprehensive Evaluation Framework for Standardized Assessment of Multimodal Large Language Models
**Date**: 2023.11.06

**Affiliation**: Shanghai AI Laboratory
<details span>
<summary><b>Abstract</b></summary>
Multimodal Large Language Models (MLLMs) have shown impressive abilities in
interacting with visual content with myriad potential downstream tasks.
However, even though a list of benchmarks has been proposed, the capabilities
and limitations of MLLMs are still not comprehensively understood, due to a
lack of a standardized and holistic evaluation framework. To this end, we
present the first Comprehensive Evaluation Framework (ChEF) that can
holistically profile each MLLM and fairly compare different MLLMs. First, we
structure ChEF as four modular components, i.e., Scenario as scalable
multimodal datasets, Instruction as flexible instruction retrieving formulae,
Inferencer as reliable question answering strategies, and Metric as indicative
task-specific score functions. Based on them, ChEF facilitates versatile
evaluations in a standardized framework, and new evaluations can be built by
designing new Recipes (systematic selection of these four components). Notably,
current MLLM benchmarks can be readily summarized as recipes of ChEF. Second,
we introduce 6 new recipes to quantify competent MLLMs' desired capabilities
(or called desiderata, i.e., calibration, in-context learning, instruction
following, language performance, hallucination, and robustness) as reliable
agents that can perform real-world multimodal interactions. Third, we conduct a
large-scale evaluation of 9 prominent MLLMs on 9 scenarios and 6 desiderata.
Our evaluation summarized over 20 valuable observations concerning the
generalizability of MLLMs across various scenarios and the composite capability
of MLLMs required for multimodal interactions. We will publicly release all the
detailed implementations for further analysis, as well as an easy-to-use
modular toolkit for the integration of new recipes and models, so that ChEF can
be a growing evaluation framework for the MLLM community.
</details>

[📄 Paper](https://arxiv.org/abs/2311.02692) | [💻 Code](https://github.com/OpenGVLab/LAMM)

### 24. (MagnifierBench) OtterHD: A High-Resolution Multi-modality Model
**Date**: 2023.11.08

**Affiliation**: Nanyang Technological University
<details span>
<summary><b>Abstract</b></summary>
In this paper, we present OtterHD-8B, an innovative multimodal model evolved
from Fuyu-8B, specifically engineered to interpret high-resolution visual
inputs with granular precision. Unlike conventional models that are constrained
by fixed-size vision encoders, OtterHD-8B boasts the ability to handle flexible
input dimensions, ensuring its versatility across various inference
requirements. Alongside this model, we introduce MagnifierBench, an evaluation
framework designed to scrutinize models' ability to discern minute details and
spatial relationships of small objects. Our comparative analysis reveals that
while current leading models falter on this benchmark, OtterHD-8B, particularly
when directly processing high-resolution inputs, outperforms its counterparts
by a substantial margin. The findings illuminate the structural variances in
visual information processing among different models and the influence that the
vision encoders' pre-training resolution disparities have on model
effectiveness within such benchmarks. Our study highlights the critical role of
flexibility and high-resolution input capabilities in large multimodal models
and also exemplifies the potential inherent in the Fuyu architecture's
simplicity for handling complex visual data.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.04219) | [💻 Code](https://github.com/Luodian/Otter)

### 25. (MMMU) MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI
**Date**: 2023.11.16

**Affiliation**: IN.AI Research
<details span>
<summary><b>Abstract</b></summary>
We introduce MMMU: a new benchmark designed to evaluate multimodal models on
massive multi-discipline tasks demanding college-level subject knowledge and
deliberate reasoning. MMMU includes 11.5K meticulously collected multimodal
questions from college exams, quizzes, and textbooks, covering six core
disciplines: Art & Design, Business, Science, Health & Medicine, Humanities &
Social Science, and Tech & Engineering. These questions span 30 subjects and
183 subfields, comprising 30 highly heterogeneous image types, such as charts,
diagrams, maps, tables, music sheets, and chemical structures. Unlike existing
benchmarks, MMMU focuses on advanced perception and reasoning with
domain-specific knowledge, challenging models to perform tasks akin to those
faced by experts. The evaluation of 14 open-source LMMs as well as the
proprietary GPT-4V(ision) and Gemini highlights the substantial challenges
posed by MMMU. Even the advanced GPT-4V and Gemini Ultra only achieve
accuracies of 56% and 59% respectively, indicating significant room for
improvement. We believe MMMU will stimulate the community to build
next-generation multimodal foundation models towards expert artificial general
intelligence.
</details>

[📄 Paper](https://arxiv.org/abs/2311.16502) | [🌐 Project Page](https://mmmu-benchmark.github.io/) | [💻 Code](https://github.com/MMMU-Benchmark/MMMU)

### 26. (AutoEval-Video) AutoEval-Video: An Automatic Benchmark for Assessing Large Vision Language Models in Open-Ended Video Question Answering
**Date**: 2023.11.26

**Affiliation**: Shanghai JiaoTong University
<details span>
<summary><b>Abstract</b></summary>
We propose a novel and challenging benchmark, AutoEval-Video, to
comprehensively evaluate large vision-language models in open-ended video
question answering. The comprehensiveness of AutoEval-Video is demonstrated in
two aspects: 1) AutoEval-Video constructs open-ended video-questions across 9
skill dimensions, addressing capabilities of perception, comprehension, and
generation. 2) AutoEval-Video contains newly collected videos that cover over
40 distinct themes. To efficiently evaluate responses to the open-ended
questions, we employ an LLM-based evaluation approach, but instead of merely
providing a reference answer, we annotate unique evaluation rules for every
single instance (video-question pair). To maximize the robustness of these
rules, we develop a novel adversarial annotation mechanism. By using
instance-specific rules as prompt, GPT-4, as an automatic evaluator, can
achieve a stable evaluation accuracy of around 97.0%, comparable to the 94.9% -
97.5% accuracy of a human evaluator. Furthermore, we assess the performance of
eight large vision-language models on AutoEval-Video. Among them, GPT-4V(ision)
significantly outperforms other models, achieving an accuracy of 32.2%.
However, there is still substantial room for improvement compared to human
accuracy of 72.8%. By conducting an extensive case study, we uncover several
drawbacks of GPT-4V, such as limited temporal and dynamic comprehension, and
overly general responses. Code is available at
https://github.com/Xiuyuan-Chen/AutoEval-Video.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.14906) | [💻 Code](https://github.com/Xiuyuan-Chen/AutoEval-Video)

### 27. (SEED-Bench-2) SEED-Bench-2: Benchmarking Multimodal Large Language Models
**Date**: 2023.11.28

**Affiliation**: Tencent AI Lab
<details span>
<summary><b>Abstract</b></summary>
Multimodal large language models (MLLMs), building upon the foundation of
powerful large language models (LLMs), have recently demonstrated exceptional
capabilities in generating not only texts but also images given interleaved
multimodal inputs (acting like a combination of GPT-4V and DALL-E 3). However,
existing MLLM benchmarks remain limited to assessing only models' comprehension
ability of single image-text inputs, failing to keep up with the strides made
in MLLMs. A comprehensive benchmark is imperative for investigating the
progress and uncovering the limitations of current MLLMs. In this work, we
categorize the capabilities of MLLMs into hierarchical levels from $L_0$ to
$L_4$ based on the modalities they can accept and generate, and propose
SEED-Bench-2, a comprehensive benchmark that evaluates the
\textbf{hierarchical} capabilities of MLLMs. Specifically, SEED-Bench-2
comprises 24K multiple-choice questions with accurate human annotations, which
spans 27 dimensions, including the evaluation of both text and image
generation. Multiple-choice questions with groundtruth options derived from
human annotation enables an objective and efficient assessment of model
performance, eliminating the need for human or GPT intervention during
evaluation. We further evaluate the performance of 23 prominent open-source
MLLMs and summarize valuable observations. By revealing the limitations of
existing MLLMs through extensive evaluations, we aim for SEED-Bench-2 to
provide insights that will motivate future research towards the goal of General
Artificial Intelligence. Dataset and evaluation code are available at
\href{https://github.com/AILab-CVC/SEED-Bench}
</details>

[📄 Paper](https://arxiv.org/abs/2311.17092) | [💻 Code](https://github.com/AILab-CVC/SEED-Bench)

### 28. (Video-Bench) Video-Bench: A Comprehensive Benchmark and Toolkit for Evaluating Video-based Large Language Models
**Date**: 2023.11.29

**Affiliation**: Peking University
<details span>
<summary><b>Abstract</b></summary>
Video-based large language models (Video-LLMs) have been recently introduced,
targeting both fundamental improvements in perception and comprehension, and a
diverse range of user inquiries. In pursuit of the ultimate goal of achieving
artificial general intelligence, a truly intelligent Video-LLM model should not
only see and understand the surroundings, but also possess human-level
commonsense, and make well-informed decisions for the users. To guide the
development of such a model, the establishment of a robust and comprehensive
evaluation system becomes crucial. To this end, this paper proposes
\textit{Video-Bench}, a new comprehensive benchmark along with a toolkit
specifically designed for evaluating Video-LLMs. The benchmark comprises 10
meticulously crafted tasks, evaluating the capabilities of Video-LLMs across
three distinct levels: Video-exclusive Understanding, Prior Knowledge-based
Question-Answering, and Comprehension and Decision-making. In addition, we
introduce an automatic toolkit tailored to process model outputs for various
tasks, facilitating the calculation of metrics and generating convenient final
scores. We evaluate 8 representative Video-LLMs using \textit{Video-Bench}. The
findings reveal that current Video-LLMs still fall considerably short of
achieving human-like comprehension and analysis of real-world videos, offering
valuable insights for future research directions. The benchmark and toolkit are
available at: \url{https://github.com/PKU-YuanGroup/Video-Bench}.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.16103) | [💻 Code](https://github.com/PKU-YuanGroup/Video-Bench)

### 29. (VITATECS) VITATECS: A Diagnostic Dataset for Temporal Concept Understanding of Video-Language Models
**Date**: 2023.11.29

**Affiliation**: Peking University
<details span>
<summary><b>Abstract</b></summary>
The ability to perceive how objects change over time is a crucial ingredient
in human intelligence. However, current benchmarks cannot faithfully reflect
the temporal understanding abilities of video-language models (VidLMs) due to
the existence of static visual shortcuts. To remedy this issue, we present
VITATECS, a diagnostic VIdeo-Text dAtaset for the evaluation of TEmporal
Concept underStanding. Specifically, we first introduce a fine-grained taxonomy
of temporal concepts in natural language in order to diagnose the capability of
VidLMs to comprehend different temporal aspects. Furthermore, to disentangle
the correlation between static and temporal information, we generate
counterfactual video descriptions that differ from the original one only in the
specified temporal aspect. We employ a semi-automatic data collection framework
using large language models and human-in-the-loop annotation to obtain
high-quality counterfactual descriptions efficiently. Evaluation of
representative video-language understanding models confirms their deficiency in
temporal understanding, revealing the need for greater emphasis on the temporal
elements in video-language research.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17404) | [💻 Code](https://github.com/lscpku/VITATECS)

### 30. (MVBench) MVBench: A Comprehensive Multi-modal Video Understanding Benchmark
**Date**: 2023.11.29

**Affiliation**: Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences
<details span>
<summary><b>Abstract</b></summary>
With the rapid development of Multi-modal Large Language Models (MLLMs), a
number of diagnostic benchmarks have recently emerged to evaluate the
comprehension capabilities of these models. However, most benchmarks
predominantly assess spatial understanding in the static image tasks, while
overlooking temporal understanding in the dynamic video tasks. To alleviate
this issue, we introduce a comprehensive Multi-modal Video understanding
Benchmark, namely MVBench, which covers 20 challenging video tasks that cannot
be effectively solved with a single frame. Specifically, we first introduce a
novel static-to-dynamic method to define these temporal-related tasks. By
transforming various static tasks into dynamic ones, we enable the systematic
generation of video tasks that require a broad spectrum of temporal skills,
ranging from perception to cognition. Then, guided by the task definition, we
automatically convert public video annotations into multiple-choice QA to
evaluate each task. On one hand, such a distinct paradigm allows us to build
MVBench efficiently, without much manual intervention. On the other hand, it
guarantees evaluation fairness with ground-truth video annotations, avoiding
the biased scoring of LLMs. Moreover, we further develop a robust video MLLM
baseline, i.e., VideoChat2, by progressive multi-modal training with diverse
instruction-tuning data. The extensive results on our MVBench reveal that, the
existing MLLMs are far from satisfactory in temporal understanding, while our
VideoChat2 largely surpasses these leading models by over 15% on MVBench. All
models and data are available at https://github.com/OpenGVLab/Ask-Anything.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17005) | [💻 Code](https://github.com/OpenGVLab/Ask-Anything)

### 31. (V\*Bench) VBench: Comprehensive Benchmark Suite for Video Generative Models
**Date**: 2023.11.29

**Affiliation**: Nanyang Technological University
<details span>
<summary><b>Abstract</b></summary>
Video generation has witnessed significant advancements, yet evaluating these
models remains a challenge. A comprehensive evaluation benchmark for video
generation is indispensable for two reasons: 1) Existing metrics do not fully
align with human perceptions; 2) An ideal evaluation system should provide
insights to inform future developments of video generation. To this end, we
present VBench, a comprehensive benchmark suite that dissects "video generation
quality" into specific, hierarchical, and disentangled dimensions, each with
tailored prompts and evaluation methods. VBench has three appealing properties:
1) Comprehensive Dimensions: VBench comprises 16 dimensions in video generation
(e.g., subject identity inconsistency, motion smoothness, temporal flickering,
and spatial relationship, etc). The evaluation metrics with fine-grained levels
reveal individual models' strengths and weaknesses. 2) Human Alignment: We also
provide a dataset of human preference annotations to validate our benchmarks'
alignment with human perception, for each evaluation dimension respectively. 3)
Valuable Insights: We look into current models' ability across various
evaluation dimensions, and various content types. We also investigate the gaps
between video and image generation models. We will open-source VBench,
including all prompts, evaluation methods, generated videos, and human
preference annotations, and also include more video generation models in VBench
to drive forward the field of video generation.
</details>

[📄 Paper](https://arxiv.org/abs/2311.17982) | [🌐 Project Page](https://vchitect.github.io/VBench-project/) | [💻 Code](https://github.com/Vchitect/VBench)

### 32. (SPEC) Synthesize, Diagnose, and Optimize: Towards Fine-Grained Vision-Language Understanding
**Date**: 2023.11.30

**Affiliation**: Fudan University
<details span>
<summary><b>Abstract</b></summary>
Vision language models (VLM) have demonstrated remarkable performance across
various downstream tasks. However, understanding fine-grained visual-linguistic
concepts, such as attributes and inter-object relationships, remains a
significant challenge. While several benchmarks aim to evaluate VLMs in finer
granularity, their primary focus remains on the linguistic aspect, neglecting
the visual dimension. Here, we highlight the importance of evaluating VLMs from
both a textual and visual perspective. We introduce a progressive pipeline to
synthesize images that vary in a specific attribute while ensuring consistency
in all other aspects. Utilizing this data engine, we carefully design a
benchmark, SPEC, to diagnose the comprehension of object size, position,
existence, and count. Subsequently, we conduct a thorough evaluation of four
leading VLMs on SPEC. Surprisingly, their performance is close to random guess,
revealing significant limitations. With this in mind, we propose a simple yet
effective approach to optimize VLMs in fine-grained understanding, achieving
significant improvements on SPEC without compromising the zero-shot
performance. Results on two additional fine-grained benchmarks also show
consistent improvements, further validating the transferability of our
approach. Code and data are available at https://github.com/wjpoom/SPEC.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.00081) | [💻 Code](https://github.com/wjpoom/SPEC)

### 33. (TimeIT) TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding
**Date**: 2023.12.04

**Affiliation**: Peking University
<details span>
<summary><b>Abstract</b></summary>
This work proposes TimeChat, a time-sensitive multimodal large language model
specifically designed for long video understanding. Our model incorporates two
key architectural contributions: (1) a timestamp-aware frame encoder that binds
visual content with the timestamp of each frame, and (2) a sliding video
Q-Former that produces a video token sequence of varying lengths to accommodate
videos of various durations. Additionally, we construct an instruction-tuning
dataset, encompassing 6 tasks and a total of 125K instances, to further enhance
TimeChat's instruction-following performance. Experiment results across various
video understanding tasks, such as dense captioning, temporal grounding, and
highlight detection, demonstrate TimeChat's strong zero-shot temporal
localization and reasoning capabilities. For example, it achieves +9.2 F1 score
and +2.8 CIDEr on YouCook2, +5.8 HIT@1 on QVHighlights, and +27.5 R@1 (IoU=0.5)
on Charades-STA, compared to state-of-the-art video large language models,
holding the potential to serve as a versatile video assistant for long-form
video comprehension tasks and satisfy realistic user requirements.
</details>

[📄 Paper](https://arxiv.org/abs/2312.02051) | [💻 Code](https://github.com/RenShuhuai-Andy/TimeChat)

### 34. (M3DBench) M3DBench: Let's Instruct Large Models with Multi-modal 3D Prompts
**Date**: 2023.12.18

**Affiliation**: Fudan University
<details span>
<summary><b>Abstract</b></summary>
Recently, 3D understanding has become popular to facilitate autonomous agents
to perform further decisionmaking. However, existing 3D datasets and methods
are often limited to specific tasks. On the other hand, recent progress in
Large Language Models (LLMs) and Multimodal Language Models (MLMs) have
demonstrated exceptional general language and imagery tasking performance.
Therefore, it is interesting to unlock MLM's potential to be 3D generalist for
wider tasks. However, current MLMs' research has been less focused on 3D tasks
due to a lack of large-scale 3D instruction-following datasets. In this work,
we introduce a comprehensive 3D instructionfollowing dataset called M3DBench,
which possesses the following characteristics: 1) It supports general
multimodal instructions interleaved with text, images, 3D objects, and other
visual prompts. 2) It unifies diverse 3D tasks at both region and scene levels,
covering a variety of fundamental abilities in real-world 3D environments. 3)
It is a large-scale 3D instruction-following dataset with over 320k
instruction-response pairs. Furthermore, we establish a new benchmark for
assessing the performance of large models in understanding multi-modal 3D
prompts. Extensive experiments demonstrate the effectiveness of our dataset and
baseline, supporting general 3D-centric tasks, which can inspire future
research.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.10763) | [💻 Code](https://github.com/OpenM3D/M3DBench/)

### 35. (V*Bench) $\textit{V}^*$: Guided Visual Search as a Core Mechanism in Multimodal LLMs
**Date**: 2023.12.21

**Affiliation**: UC San Diego
<details span>
<summary><b>Abstract</b></summary>
When we look around and perform complex tasks, how we see and selectively
process what we see is crucial. However, the lack of this visual search
mechanism in current multimodal LLMs (MLLMs) hinders their ability to focus on
important visual details, especially when handling high-resolution and visually
crowded images. To address this, we introduce $\textit{V}^*$, an LLM-guided
visual search mechanism that employs the world knowledge in LLMs for efficient
visual querying. When combined with an MLLM, this mechanism enhances
collaborative reasoning, contextual understanding, and precise targeting of
specific visual elements. This integration results in a new MLLM
meta-architecture, named $\textbf{S}$how, s$\textbf{EA}$rch, and
Tel$\textbf{L}$ (SEAL). We further create $\textit{V}^*$Bench, a benchmark
specifically designed to evaluate MLLMs in their ability to process
high-resolution images and focus on visual details. Our study highlights the
necessity of incorporating visual search capabilities into multimodal systems.
The code is available https://github.com/penghao-wu/vstar.
</details>

[📄 Paper](https://arxiv.org/html/2312.14135v1) | [💻 Code](https://github.com/penghao-wu/vstar)

### 36. (Mementos) Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences
**Date**: 2024.01.10

**Affiliation**: University of Maryland
<details span>
<summary><b>Abstract</b></summary>
Multimodal Large Language Models (MLLMs) have demonstrated proficiency in
handling a variety of visual-language tasks. However, current MLLM benchmarks
are predominantly designed to evaluate reasoning based on static information
about a single image, and the ability of modern MLLMs to extrapolate from image
sequences, which is essential for understanding our ever-changing world, has
been less investigated. To address this challenge, this paper introduces
Mementos, a new benchmark designed to assess MLLMs' sequential image reasoning
abilities. Mementos features 4,761 diverse image sequences with varying
lengths. We also employ a GPT-4 assisted method to evaluate MLLM reasoning
performance. Through a careful evaluation of nine recent MLLMs on Mementos,
including GPT-4V and Gemini, we find that they struggle to accurately describe
dynamic information about given image sequences, often leading to
hallucinations/misrepresentations of objects and their corresponding behaviors.
Our quantitative analysis and case studies identify three key factors impacting
MLLMs' sequential image reasoning: the correlation between object and
behavioral hallucinations, the influence of cooccurring behaviors, and the
compounding impact of behavioral hallucinations. Our dataset is available at
https://github.com/umd-huang-lab/Mementos.
</details>

[📄 Paper](https://arxiv.org/pdf/2401.10529) | [💻 Code](https://github.com/umd-huang-lab/Mementos)

### 37. (MMVP) Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs
**Date**: 2024.01.11

**Affiliation**: NewYorkUniversity
<details span>
<summary><b>Abstract</b></summary>
Is vision good enough for language? Recent advancements in multimodal models
primarily stem from the powerful reasoning abilities of large language models
(LLMs). However, the visual component typically depends only on the
instance-level contrastive language-image pre-training (CLIP). Our research
reveals that the visual capabilities in recent multimodal LLMs (MLLMs) still
exhibit systematic shortcomings. To understand the roots of these errors, we
explore the gap between the visual embedding space of CLIP and vision-only
self-supervised learning. We identify ''CLIP-blind pairs'' - images that CLIP
perceives as similar despite their clear visual differences. With these pairs,
we construct the Multimodal Visual Patterns (MMVP) benchmark. MMVP exposes
areas where state-of-the-art systems, including GPT-4V, struggle with
straightforward questions across nine basic visual patterns, often providing
incorrect answers and hallucinated explanations. We further evaluate various
CLIP-based vision-and-language models and found a notable correlation between
visual patterns that challenge CLIP models and those problematic for multimodal
LLMs. As an initial effort to address these issues, we propose a Mixture of
Features (MoF) approach, demonstrating that integrating vision self-supervised
learning features with MLLMs can significantly enhance their visual grounding
capabilities. Together, our research suggests visual representation learning
remains an open challenge, and accurate visual grounding is crucial for future
successful multimodal systems.
</details>

[📄 Paper](https://arxiv.org/abs/2401.06209) | [🌐 Project Page](https://tsb0601.github.io/mmvp_blog/) | [💻 Code](https://github.com/tsb0601/MMVP)

### 38. (MM-SAP) MM-SAP: A Comprehensive Benchmark for Assessing Self-Awareness of Multimodal Large Language Models in Perception
**Date**: 2024.01.15

**Affiliation**: Shanghai Jiao Tong University
<details span>
<summary><b>Abstract</b></summary>
Recent advancements in Multimodal Large Language Models (MLLMs) have
demonstrated exceptional capabilities in visual perception and understanding.
However, these models also suffer from hallucinations, which limit their
reliability as AI systems. We believe that these hallucinations are partially
due to the models' struggle with understanding what they can and cannot
perceive from images, a capability we refer to as self-awareness in perception.
Despite its importance, this aspect of MLLMs has been overlooked in prior
studies. In this paper, we aim to define and evaluate the self-awareness of
MLLMs in perception. To do this, we first introduce the knowledge quadrant in
perception, which helps define what MLLMs know and do not know about images.
Using this framework, we propose a novel benchmark, the Self-Awareness in
Perception for MLLMs (MM-SAP), specifically designed to assess this capability.
We apply MM-SAP to a variety of popular MLLMs, offering a comprehensive
analysis of their self-awareness and providing detailed insights. The
experiment results reveal that current MLLMs possess limited self-awareness
capabilities, pointing to a crucial area for future advancement in the
development of trustworthy MLLMs. Code and data are available at
https://github.com/YHWmz/MM-SAP.
</details>

[📄 Paper](https://arxiv.org/pdf/2401.07529) | [💻 Code](https://github.com/YHWmz/MM-SAP)

### 39. (AesBench) AesBench: An Expert Benchmark for Multimodal Large Language Models on Image Aesthetics Perception
**Date**: 2024.01.16

**Affiliation**: Xidian University
<details span>
<summary><b>Abstract</b></summary>
With collective endeavors, multimodal large language models (MLLMs) are
undergoing a flourishing development. However, their performances on image
aesthetics perception remain indeterminate, which is highly desired in
real-world applications. An obvious obstacle lies in the absence of a specific
benchmark to evaluate the effectiveness of MLLMs on aesthetic perception. This
blind groping may impede the further development of more advanced MLLMs with
aesthetic perception capacity. To address this dilemma, we propose AesBench, an
expert benchmark aiming to comprehensively evaluate the aesthetic perception
capacities of MLLMs through elaborate design across dual facets. (1) We
construct an Expert-labeled Aesthetics Perception Database (EAPD), which
features diversified image contents and high-quality annotations provided by
professional aesthetic experts. (2) We propose a set of integrative criteria to
measure the aesthetic perception abilities of MLLMs from four perspectives,
including Perception (AesP), Empathy (AesE), Assessment (AesA) and
Interpretation (AesI). Extensive experimental results underscore that the
current MLLMs only possess rudimentary aesthetic perception ability, and there
is still a significant gap between MLLMs and humans. We hope this work can
inspire the community to engage in deeper explorations on the aesthetic
potentials of MLLMs. Source data will be available at
https://github.com/yipoh/AesBench.
</details>

[📄 Paper](https://arxiv.org/abs/2401.08276) | [💻 Code](https://github.com/yipoh/AesBench)

### 40. (Q-Bench+) Q-Bench+: A Benchmark for Multi-modal Foundation Models on Low-level Vision from Single Images to Pairs
**Date**: 2024.02.11

**Affiliation**: Shanghai Jiao Tong University
<details span>
<summary><b>Abstract</b></summary>
The rapid development of Multi-modality Large Language Models (MLLMs) has
navigated a paradigm shift in computer vision, moving towards versatile
foundational models. However, evaluating MLLMs in low-level visual perception
and understanding remains a yet-to-explore domain. To this end, we design
benchmark settings to emulate human language responses related to low-level
vision: the low-level visual perception (A1) via visual question answering
related to low-level attributes (e.g. clarity, lighting); and the low-level
visual description (A2), on evaluating MLLMs for low-level text descriptions.
Furthermore, given that pairwise comparison can better avoid ambiguity of
responses and has been adopted by many human experiments, we further extend the
low-level perception-related question-answering and description evaluations of
MLLMs from single images to image pairs. Specifically, for perception (A1), we
carry out the LLVisionQA+ dataset, comprising 2,990 single images and 1,999
image pairs each accompanied by an open-ended question about its low-level
features; for description (A2), we propose the LLDescribe+ dataset, evaluating
MLLMs for low-level descriptions on 499 single images and 450 pairs.
Additionally, we evaluate MLLMs on assessment (A3) ability, i.e. predicting
score, by employing a softmax-based approach to enable all MLLMs to generate
quantifiable quality ratings, tested against human opinions in 7 image quality
assessment (IQA) datasets. With 24 MLLMs under evaluation, we demonstrate that
several MLLMs have decent low-level visual competencies on single images, but
only GPT-4V exhibits higher accuracy on pairwise comparisons than single image
evaluations (like humans). We hope that our benchmark will motivate further
research into uncovering and enhancing these nascent capabilities of MLLMs.
Datasets will be available at https://github.com/Q-Future/Q-Bench.
</details>

[📄 Paper](https://arxiv.org/abs/2402.07116) | [💻 Code](https://github.com/Q-Future/Q-Bench)

### 41. (AIR-Bench) AIR-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension
**Date**: 2024.02.12

**Affiliation**: Zhejiang University
<details span>
<summary><b>Abstract</b></summary>
Recently, instruction-following audio-language models have received broad
attention for human-audio interaction. However, the absence of benchmarks
capable of evaluating audio-centric interaction capabilities has impeded
advancements in this field. Previous models primarily focus on assessing
different fundamental tasks, such as Automatic Speech Recognition (ASR), and
lack an assessment of the open-ended generative capabilities centered around
audio. Thus, it is challenging to track the progression in the Large
Audio-Language Models (LALMs) domain and to provide guidance for future
improvement. In this paper, we introduce AIR-Bench (\textbf{A}udio
\textbf{I}nst\textbf{R}uction \textbf{Bench}mark), the first benchmark designed
to evaluate the ability of LALMs to understand various types of audio signals
(including human speech, natural sounds, and music), and furthermore, to
interact with humans in the textual format. AIR-Bench encompasses two
dimensions: \textit{foundation} and \textit{chat} benchmarks. The former
consists of 19 tasks with approximately 19k single-choice questions, intending
to inspect the basic single-task ability of LALMs. The latter one contains 2k
instances of open-ended question-and-answer data, directly assessing the
comprehension of the model on complex audio and its capacity to follow
instructions. Both benchmarks require the model to generate hypotheses
directly. We design a unified framework that leverages advanced language
models, such as GPT-4, to evaluate the scores of generated hypotheses given the
meta-information of the audio. Experimental results demonstrate a high level of
consistency between GPT-4-based evaluation and human evaluation. By revealing
the limitations of existing LALMs through evaluation results, AIR-Bench can
provide insights into the direction of future research.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.07729) | [💻 Code](https://github.com/OFA-Sys/AIR-Bench)

### 42. (MCUB) Model Composition for Multimodal Large Language Models
**Date**: 2024.02.20

**Affiliation**: Tsinghua University
<details span>
<summary><b>Abstract</b></summary>
Recent developments in Multimodal Large Language Models (MLLMs) have shown
rapid progress, moving towards the goal of creating versatile MLLMs that
understand inputs from various modalities. However, existing methods typically
rely on joint training with paired multimodal instruction data, which is
resource-intensive and challenging to extend to new modalities. In this paper,
we propose a new paradigm through the model composition of existing MLLMs to
create a new model that retains the modal understanding capabilities of each
original model. Our basic implementation, NaiveMC, demonstrates the
effectiveness of this paradigm by reusing modality encoders and merging LLM
parameters. Furthermore, we introduce DAMC to address parameter interference
and mismatch issues during the merging process, thereby enhancing the model
performance. To facilitate research in this area, we propose MCUB, a benchmark
for assessing ability of MLLMs to understand inputs from diverse modalities.
Experiments on this benchmark and four other multimodal understanding tasks
show significant improvements over baselines, proving that model composition
can create a versatile model capable of processing inputs from multiple
modalities.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.12750) | [💻 Code](https://github.com/THUNLP-MT/ModelCompose)

### 43. (CODIS) CODIS: Benchmarking Context-Dependent Visual Comprehension for Multimodal Large Language Models
**Date**: 2024.02.21

**Affiliation**: Tsinghua University
<details span>
<summary><b>Abstract</b></summary>
Multimodal large language models (MLLMs) have demonstrated promising results
in a variety of tasks that combine vision and language. As these models become
more integral to research and applications, conducting comprehensive
evaluations of their capabilities has grown increasingly important. However,
most existing benchmarks fail to consider that, in certain situations, images
need to be interpreted within a broader context. In this work, we introduce a
new benchmark, named as CODIS, designed to assess the ability of models to use
context provided in free-form text to enhance visual comprehension. Our
findings indicate that MLLMs consistently fall short of human performance on
this benchmark. Further analysis confirms that these models struggle to
effectively extract and utilize contextual information to improve their
understanding of images. This underscores the pressing need to enhance the
ability of MLLMs to comprehend visuals in a context-dependent manner. View our
project website at https://thunlp-mt.github.io/CODIS.
</details>

[📄 Paper](https://arxiv.org/abs/2402.13607) | [💻 Code](https://thunlp-mt.github.io/CODIS)

### 44. (OSCaR) OSCaR: Object State Captioning and State Change Representation
**Date**: 2024.02.27

**Affiliation**: University of Rochester
<details span>
<summary><b>Abstract</b></summary>
The capability of intelligent models to extrapolate and comprehend changes in
object states is a crucial yet demanding aspect of AI research, particularly
through the lens of human interaction in real-world settings. This task
involves describing complex visual environments, identifying active objects,
and interpreting their changes as conveyed through language. Traditional
methods, which isolate object captioning and state change detection, offer a
limited view of dynamic environments. Moreover, relying on a small set of
symbolic words to represent changes has restricted the expressiveness of the
language. To address these challenges, in this paper, we introduce the Object
State Captioning and State Change Representation (OSCaR) dataset and benchmark.
OSCaR consists of 14,084 annotated video segments with nearly 1,000 unique
objects from various egocentric video collections. It sets a new testbed for
evaluating multimodal large language models (MLLMs). Our experiments
demonstrate that while MLLMs show some skill, they lack a full understanding of
object state changes. The benchmark includes a fine-tuned model that, despite
initial capabilities, requires significant improvements in accuracy and
generalization ability for effective understanding of these changes. Our code
and dataset are available at https://github.com/nguyennm1024/OSCaR.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.17128) | [💻 Code](https://github.com/nguyennm1024/OSCaR)

### 45. (TempCompass) TempCompass: Do Video LLMs Really Understand Videos?
**Date**: 2024.03.01

**Affiliation**: Peking University
<details span>
<summary><b>Abstract</b></summary>
Recently, there is a surge in interest surrounding video large language
models (Video LLMs). However, existing benchmarks fail to provide a
comprehensive feedback on the temporal perception ability of Video LLMs. On the
one hand, most of them are unable to distinguish between different temporal
aspects (e.g., speed, direction) and thus cannot reflect the nuanced
performance on these specific aspects. On the other hand, they are limited in
the diversity of task formats (e.g., only multi-choice QA), which hinders the
understanding of how temporal perception performance may vary across different
types of tasks. Motivated by these two problems, we propose the
\textbf{TempCompass} benchmark, which introduces a diversity of temporal
aspects and task formats. To collect high-quality test data, we devise two
novel strategies: (1) In video collection, we construct conflicting videos that
share the same static content but differ in a specific temporal aspect, which
prevents Video LLMs from leveraging single-frame bias or language priors. (2)
To collect the task instructions, we propose a paradigm where humans first
annotate meta-information for a video and then an LLM generates the
instruction. We also design an LLM-based approach to automatically and
accurately evaluate the responses from Video LLMs. Based on TempCompass, we
comprehensively evaluate 8 state-of-the-art (SOTA) Video LLMs and 3 Image LLMs,
and reveal the discerning fact that these models exhibit notably poor temporal
perception ability. Our data will be available at
https://github.com/llyx97/TempCompass.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.00476) | [💻 Code](https://github.com/llyx97/TempCompass)

### 46. (Henna) Peacock: A Family of Arabic Multimodal Large Language Models and Benchmarks
**Date**: 2024.03.02

**Affiliation**: The University of British Columbia & Invertible AI
<details span>
<summary><b>Abstract</b></summary>
Multimodal large language models (MLLMs) have proven effective in a wide
range of tasks requiring complex reasoning and linguistic comprehension.
However, due to a lack of high-quality multimodal resources in languages other
than English, success of MLLMs remains relatively limited to English-based
settings. This poses significant challenges in developing comparable models for
other languages, including even those with large speaker populations such as
Arabic. To alleviate this challenge, we introduce a comprehensive family of
Arabic MLLMs, dubbed \textit{Peacock}, with strong vision and language
capabilities. Through comprehensive qualitative and quantitative analysis, we
demonstrate the solid performance of our models on various visual reasoning
tasks and further show their emerging dialectal potential. Additionally, we
introduce ~\textit{Henna}, a new benchmark specifically designed for assessing
MLLMs on aspects related to Arabic culture, setting the first stone for
culturally-aware Arabic MLLMs.The GitHub repository for the \textit{Peacock}
project is available at \url{https://github.com/UBC-NLP/peacock}.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.01031) | [💻 Code](https://github.com/UBC-NLP/peacock)

### 47. (VL-ICLBench) VL-ICL Bench: The Devil in the Details of Benchmarking Multimodal In-Context Learning
**Date**: 2024.03.19

**Affiliation**: University of Edinburgh
<details span>
<summary><b>Abstract</b></summary>
Large language models (LLMs) famously exhibit emergent in-context learning
(ICL) -- the ability to rapidly adapt to new tasks using few-shot examples
provided as a prompt, without updating the model's weights. Built on top of
LLMs, vision large language models (VLLMs) have advanced significantly in areas
such as recognition, reasoning, and grounding. However, investigations into
\emph{multimodal ICL} have predominantly focused on few-shot visual question
answering (VQA), and image captioning, which we will show neither exploit the
strengths of ICL, nor test its limitations. The broader capabilities and
limitations of multimodal ICL remain under-explored. In this study, we
introduce a comprehensive benchmark VL-ICL Bench for multimodal in-context
learning, encompassing a broad spectrum of tasks that involve both images and
text as inputs and outputs, and different types of challenges, from {perception
to reasoning and long context length}. We evaluate the abilities of
state-of-the-art VLLMs against this benchmark suite, revealing their diverse
strengths and weaknesses, and showing that even the most advanced models, such
as GPT-4, find the tasks challenging. By highlighting a range of new ICL tasks,
and the associated strengths and limitations of existing models, we hope that
our dataset will inspire future work on enhancing the in-context learning
capabilities of VLLMs, as well as inspire new applications that leverage VLLM
ICL. The code and dataset are available at https://github.com/ys-zong/VL-ICL.
</details>

[📄 Paper](https://arxiv.org/abs/2403.13164) | [💻 Code](https://github.com/ys-zong/VL-ICL)

### 48. (P^2GB) Plug-and-Play Grounding of Reasoning in Multimodal Large Language Models
**Date**: 2024.03.28

**Affiliation**: Peking University
<details span>
<summary><b>Abstract</b></summary>
The rise of Multimodal Large Language Models (MLLMs), renowned for their
advanced instruction-following and reasoning capabilities, has significantly
propelled the field of visual reasoning. However, due to limitations in their
image tokenization processes, most MLLMs struggle to capture fine details of
text and objects in images, especially in high-resolution samples. To overcome
this limitation, we introduce P2G, a novel framework for plug-and-play
grounding in MLLMs. P2G utilizes the tool-usage potential of MLLMs to employ
expert agents for on-the-fly grounding of reasoning into critical visual and
textual elements in images, thereby enabling deliberate reasoning through
multimodal prompting. Additionally, we develop P2GB, a benchmark designed to
evaluate MLLMs' proficiency in understanding inter-object relationships and
textual content in challenging high-resolution images. Extensive experiments on
visual reasoning tasks demonstrate the superiority of P2G, achieving
performance comparable to GPT-4V on P2GB with a 7B backbone. Our work
underscores the potential of grounding reasoning with external agents in MLLMs,
presenting a promising alternative to mere model scaling.
</details>

[📄 Paper](https://arxiv.org/abs/2403.19322) 

### 49. (MMStar) Are We on the Right Way for Evaluating Large Vision-Language Models?
**Date**: 2024.03.29

**Affiliation**: University of Science and Technology of China
<details span>
<summary><b>Abstract</b></summary>
Large vision-language models (LVLMs) have recently achieved rapid progress,
sparking numerous studies to evaluate their multi-modal capabilities. However,
we dig into current evaluation works and identify two primary issues: 1) Visual
content is unnecessary for many samples. The answers can be directly inferred
from the questions and options, or the world knowledge embedded in LLMs. This
phenomenon is prevalent across current benchmarks. For instance, GeminiPro
achieves 42.9% on the MMMU benchmark without any visual input, and outperforms
the random choice baseline across six benchmarks over 24% on average. 2)
Unintentional data leakage exists in LLM and LVLM training. LLM and LVLM could
still answer some visual-necessary questions without visual content, indicating
the memorizing of these samples within large-scale training data. For example,
Sphinx-X-MoE gets 43.6% on MMMU without accessing images, surpassing its LLM
backbone with 17.9%. Both problems lead to misjudgments of actual multi-modal
gains and potentially misguide the study of LVLM. To this end, we present
MMStar, an elite vision-indispensable multi-modal benchmark comprising 1,500
samples meticulously selected by humans. MMStar benchmarks 6 core capabilities
and 18 detailed axes, aiming to evaluate LVLMs' multi-modal capacities with
carefully balanced and purified samples. These samples are first roughly
selected from current benchmarks with an automated pipeline, human review is
then involved to ensure each curated sample exhibits visual dependency, minimal
data leakage, and requires advanced multi-modal capabilities. Moreover, two
metrics are developed to measure data leakage and actual performance gain in
multi-modal training. We evaluate 16 leading LVLMs on MMStar to assess their
multi-modal capabilities, and on 7 benchmarks with the proposed metrics to
investigate their data leakage and actual multi-modal gain.
</details>

[📄 Paper](https://arxiv.org/abs/2403.20330) | [🌐 Project Page](https://mmstar-benchmark.github.io/) | [💻 Code](https://github.com/MMStar-Benchmark/MMStar)

### 50. (MDVP-Bench) Draw-and-Understand: Leveraging Visual Prompts to Enable MLLMs to Comprehend What You Want
**Date**: 2024.03.29

**Affiliation**: ShanghaiAILaboratory
<details span>
<summary><b>Abstract</b></summary>
The interaction between humans and artificial intelligence (AI) is a crucial
factor that reflects the effectiveness of multimodal large language models
(MLLMs). However, current MLLMs primarily focus on image-level comprehension
and limit interaction to textual instructions, thereby constraining their
flexibility in usage and depth of response. In this paper, we introduce the
Draw-and-Understand project: a new model, a multi-domain dataset, and a
challenging benchmark for visual prompting. Specifically, we propose SPHINX-V,
a new end-to-end trained Multimodal Large Language Model (MLLM) that connects a
vision encoder, a visual prompt encoder and an LLM for various visual prompts
(points, bounding boxes, and free-form shape) and language understanding. To
advance visual prompting research for MLLMs, we introduce MDVP-Data and
MDVP-Bench. MDVP-Data features a multi-domain dataset containing 1.6M unique
image-visual prompt-text instruction-following samples, including natural
images, document images, OCR images, mobile screenshots, web screenshots, and
multi-panel images. Furthermore, we present MDVP-Bench, a comprehensive and
challenging benchmark to assess a model's capability in understanding visual
prompting instructions. Our experiments demonstrate SPHINX-V's impressive
multimodal interaction capabilities through visual prompting, revealing
significant improvements in detailed pixel-level description and
question-answering abilities.
</details>

[📄 Paper](https://arxiv.org/abs/2403.20271) | [🌐 Project Page](https://draw-and-understand.github.io/) | [💻 Code](https://github.com/AFeng-x/Draw-and-Understand)

### 51. (FABA-Bench) Facial Affective Behavior Analysis with Instruction Tuning
**Date**: 2024.04.07

**Affiliation**: Michigan State University
<details span>
<summary><b>Abstract</b></summary>
Facial affective behavior analysis (FABA) is crucial for understanding human
mental states from images. However, traditional approaches primarily deploy
models to discriminate among discrete emotion categories, and lack the fine
granularity and reasoning capability for complex facial behaviors. The advent
of Multi-modal Large Language Models (MLLMs) has been proven successful in
general visual understanding tasks. However, directly harnessing MLLMs for FABA
is challenging due to the scarcity of datasets and benchmarks, neglecting
facial prior knowledge, and low training efficiency. To address these
challenges, we introduce (i) an instruction-following dataset for two FABA
tasks, e.g., emotion and action unit recognition, (ii) a benchmark FABA-Bench
with a new metric considering both recognition and generation ability, and
(iii) a new MLLM "EmoLA" as a strong baseline to the community. Our initiative
on the dataset and benchmarks reveal the nature and rationale of facial
affective behaviors, i.e., fine-grained facial movement, interpretability, and
reasoning. Moreover, to build an effective and efficient FABA MLLM, we
introduce a facial prior expert module with face structure knowledge and a
low-rank adaptation module into pre-trained MLLM. We conduct extensive
experiments on FABA-Bench and four commonly-used FABA datasets. The results
demonstrate that the proposed facial prior expert can boost the performance and
EmoLA achieves the best results on our FABA-Bench. On commonly-used FABA
datasets, EmoLA is competitive rivaling task-specific state-of-the-art models.
</details>

[📄 Paper](https://arxiv.org/abs/2404.05052) | [🌐 Project Page](https://johnx69.github.io/FABA/) | [💻 Code](https://github.com/JackYFL/EmoLA)

### 52. (LaVy-Bench) LaVy: Vietnamese Multimodal Large Language Model
**Date**: 2024.04.11

**Affiliation**: Hanoi University of Science and Technology
<details span>
<summary><b>Abstract</b></summary>
Large Language Models (LLMs) and Multimodal Large language models (MLLMs)
have taken the world by storm with impressive abilities in complex reasoning
and linguistic comprehension. Meanwhile there are plethora of works related to
Vietnamese Large Language Models, the lack of high-quality resources in
multimodality limits the progress of Vietnamese MLLMs. In this paper, we
pioneer in address this by introducing LaVy, a state-of-the-art Vietnamese
MLLM, and we also introduce LaVy-Bench benchmark designated for evaluating
MLLMs's understanding on Vietnamese visual language tasks. Our project is
public at https://github.com/baochi0212/LaVy
</details>

[📄 Paper](https://arxiv.org/pdf/2404.07922) | [💻 Code](https://github.com/baochi0212/LaVy)

### 53. (BLINK) BLINK: Multimodal Large Language Models Can See but Not Perceive
**Date**: 2024.04.12

**Affiliation**: University of Pennsylvania
<details span>
<summary><b>Abstract</b></summary>
We introduce Blink, a new benchmark for multimodal language models (LLMs)
that focuses on core visual perception abilities not found in other
evaluations. Most of the Blink tasks can be solved by humans "within a blink"
(e.g., relative depth estimation, visual correspondence, forensics detection,
and multi-view reasoning). However, we find these perception-demanding tasks
cast significant challenges for current multimodal LLMs because they resist
mediation through natural language. Blink reformats 14 classic computer vision
tasks into 3,807 multiple-choice questions, paired with single or multiple
images and visual prompting. While humans get 95.70% accuracy on average, Blink
is surprisingly challenging for existing multimodal LLMs: even the
best-performing GPT-4V and Gemini achieve accuracies of 51.26% and 45.72%, only
13.17% and 7.63% higher than random guessing, indicating that such perception
abilities have not "emerged" yet in recent multimodal LLMs. Our analysis also
highlights that specialist CV models could solve these problems much better,
suggesting potential pathways for future improvements. We believe Blink will
stimulate the community to help multimodal LLMs catch up with human-level
visual perception.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.12390) | [🌐 Project Page](https://zeyofu.github.io/blink/) | [💻 Code](https://github.com/zeyofu/BLINK_Benchmark)

### 54. (UNIAA) UNIAA: A Unified Multi-modal Image Aesthetic Assessment Baseline and Benchmark
**Date**: 2024.04.15

**Affiliation**: Peking University
<details span>
<summary><b>Abstract</b></summary>
As an alternative to expensive expert evaluation, Image Aesthetic Assessment
(IAA) stands out as a crucial task in computer vision. However, traditional IAA
methods are typically constrained to a single data source or task, restricting
the universality and broader application. In this work, to better align with
human aesthetics, we propose a Unified Multi-modal Image Aesthetic Assessment
(UNIAA) framework, including a Multi-modal Large Language Model (MLLM) named
UNIAA-LLaVA and a comprehensive benchmark named UNIAA-Bench. We choose MLLMs
with both visual perception and language ability for IAA and establish a
low-cost paradigm for transforming the existing datasets into unified and
high-quality visual instruction tuning data, from which the UNIAA-LLaVA is
trained. To further evaluate the IAA capability of MLLMs, we construct the
UNIAA-Bench, which consists of three aesthetic levels: Perception, Description,
and Assessment. Extensive experiments validate the effectiveness and
rationality of UNIAA. UNIAA-LLaVA achieves competitive performance on all
levels of UNIAA-Bench, compared with existing MLLMs. Specifically, our model
performs better than GPT-4V in aesthetic perception and even approaches the
junior-level human. We find MLLMs have great potential in IAA, yet there
remains plenty of room for further improvement. The UNIAA-LLaVA and UNIAA-Bench
will be released.
</details>

[📄 Paper](https://arxiv.org/abs/2404.09619)

### 55. (SEED-Bench-2-Plus) SEED-Bench-2-Plus: Benchmarking Multimodal Large Language Models with Text-Rich Visual Comprehension
**Date**: 2024.04.16

**Affiliation**: Tencent AI Lab
<details span>
<summary><b>Abstract</b></summary>
Comprehending text-rich visual content is paramount for the practical
application of Multimodal Large Language Models (MLLMs), since text-rich
scenarios are ubiquitous in the real world, which are characterized by the
presence of extensive texts embedded within images. Recently, the advent of
MLLMs with impressive versatility has raised the bar for what we can expect
from MLLMs. However, their proficiency in text-rich scenarios has yet to be
comprehensively and objectively assessed, since current MLLM benchmarks
primarily focus on evaluating general visual comprehension. In this work, we
introduce SEED-Bench-2-Plus, a benchmark specifically designed for evaluating
\textbf{text-rich visual comprehension} of MLLMs. Our benchmark comprises 2.3K
multiple-choice questions with precise human annotations, spanning three broad
categories: Charts, Maps, and Webs, each of which covers a wide spectrum of
text-rich scenarios in the real world. These categories, due to their inherent
complexity and diversity, effectively simulate real-world text-rich
environments. We further conduct a thorough evaluation involving 34 prominent
MLLMs (including GPT-4V, Gemini-Pro-Vision and Claude-3-Opus) and emphasize the
current limitations of MLLMs in text-rich visual comprehension. We hope that
our work can serve as a valuable addition to existing MLLM benchmarks,
providing insightful observations and inspiring further research in the area of
text-rich visual comprehension with MLLMs. The dataset and evaluation code can
be accessed at https://github.com/AILab-CVC/SEED-Bench.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.16790) | [💻 Code](https://github.com/AILab-CVC/SEED-Bench)

### 56. (MileBENCH) MileBench: Benchmarking MLLMs in Long Context
**Date**: 2024.04.18

**Affiliation**: The Chinese University of Hong Kong
<details span>
<summary><b>Abstract</b></summary>
Despite the advancements and impressive performance of Multimodal Large
Language Models (MLLMs) on benchmarks, their effectiveness in real-world,
long-context, and multi-image tasks is unclear due to the benchmarks' limited
scope. Existing benchmarks often focus on single-image and short-text samples,
and when assessing multi-image tasks, they either limit the image count or
focus on specific task (e.g time-series captioning), potentially obscuring the
performance challenges of MLLMs. To address these limitations, we introduce
MileBench, a pioneering benchmark designed to test the MultImodal Long-contExt
capabilities of MLLMs. This benchmark comprises not only multimodal long
contexts, but also multiple tasks requiring both comprehension and generation.
We establish two distinct evaluation sets, diagnostic and realistic, to
systematically assess MLLMs' long-context adaptation capacity and their ability
to complete tasks in long-context scenarios. Our experimental results, obtained
from testing 22 models, revealed that while the closed-source GPT-4o
outperforms others, most open-source MLLMs struggle in long-context situations.
Interestingly, the performance gap tends to widen with an increase in the
number of images. We strongly encourage an intensification of research efforts
towards enhancing MLLMs' long-context capabilities, especially in scenarios
involving multiple images.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.18532) | [🌐 Project Page](https://milebench.github.io/) | [💻 Code](https://github.com/MileBench/MileBench)

### 57. (ImplicitAVE) ImplicitAVE: An Open-Source Dataset and Multimodal LLMs Benchmark for Implicit Attribute Value Extraction
**Date**: 2024.04.24

**Affiliation**: University of Illinois Chicago
<details span>
<summary><b>Abstract</b></summary>
Existing datasets for attribute value extraction (AVE) predominantly focus on
explicit attribute values while neglecting the implicit ones, lack product
images, are often not publicly available, and lack an in-depth human inspection
across diverse domains. To address these limitations, we present ImplicitAVE,
the first, publicly available multimodal dataset for implicit attribute value
extraction. ImplicitAVE, sourced from the MAVE dataset, is carefully curated
and expanded to include implicit AVE and multimodality, resulting in a refined
dataset of 68k training and 1.6k testing data across five domains. We also
explore the application of multimodal large language models (MLLMs) to implicit
AVE, establishing a comprehensive benchmark for MLLMs on the ImplicitAVE
dataset. Six recent MLLMs with eleven variants are evaluated across diverse
settings, revealing that implicit value extraction remains a challenging task
for MLLMs. The contributions of this work include the development and release
of ImplicitAVE, and the exploration and benchmarking of various MLLMs for
implicit AVE, providing valuable insights and potential future research
directions. Dataset and code are available at
https://github.com/HenryPengZou/ImplicitAVE
</details>

[📄 Paper](https://arxiv.org/abs/2404.15592) | [💻 Code](https://github.com/HenryPengZou/ImplicitAVE)

### 58. (MMT-Bench) MMT-Bench: A Comprehensive Multimodal Benchmark for Evaluating Large Vision-Language Models Towards Multitask AGI
**Date**: 2024.04.25

**Affiliation**: Shanghai AI Laboratory
<details span>
<summary><b>Abstract</b></summary>
Large Vision-Language Models (LVLMs) show significant strides in
general-purpose multimodal applications such as visual dialogue and embodied
navigation. However, existing multimodal evaluation benchmarks cover a limited
number of multimodal tasks testing rudimentary capabilities, falling short in
tracking LVLM development. In this study, we present MMT-Bench, a comprehensive
benchmark designed to assess LVLMs across massive multimodal tasks requiring
expert knowledge and deliberate visual recognition, localization, reasoning,
and planning. MMT-Bench comprises $31,325$ meticulously curated multi-choice
visual questions from various multimodal scenarios such as vehicle driving and
embodied navigation, covering $32$ core meta-tasks and $162$ subtasks in
multimodal understanding. Due to its extensive task coverage, MMT-Bench enables
the evaluation of LVLMs using a task map, facilitating the discovery of in- and
out-of-domain tasks. Evaluation results involving $30$ LVLMs such as the
proprietary GPT-4V, GeminiProVision, and open-sourced InternVL-Chat, underscore
the significant challenges posed by MMT-Bench. We anticipate that MMT-Bench
will inspire the community to develop next-generation multimodal foundation
models aimed at achieving general-purpose multimodal intelligence.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.16006) | [🌐 Project Page](https://mmt-bench.github.io/) | [💻 Code](https://github.com/OpenGVLab/MMT-Bench)

### 59. (WorldNet) WorldGPT: Empowering LLM as Multimodal World Model
**Date**: 2024.04.28

**Affiliation**: Zhejiang University
<details span>
<summary><b>Abstract</b></summary>
World models are progressively being employed across diverse fields,
extending from basic environment simulation to complex scenario construction.
However, existing models are mainly trained on domain-specific states and
actions, and confined to single-modality state representations. In this paper,
We introduce WorldGPT, a generalist world model built upon Multimodal Large
Language Model (MLLM). WorldGPT acquires an understanding of world dynamics
through analyzing millions of videos across various domains. To further enhance
WorldGPT's capability in specialized scenarios and long-term tasks, we have
integrated it with a novel cognitive architecture that combines memory
offloading, knowledge retrieval, and context reflection. As for evaluation, we
build WorldNet, a multimodal state transition prediction benchmark encompassing
varied real-life scenarios. Conducting evaluations on WorldNet directly
demonstrates WorldGPT's capability to accurately model state transition
patterns, affirming its effectiveness in understanding and predicting the
dynamics of complex scenarios. We further explore WorldGPT's emerging potential
in serving as a world simulator, helping multimodal agents generalize to
unfamiliar domains through efficiently synthesising multimodal instruction
instances which are proved to be as reliable as authentic data for fine-tuning
purposes. The project is available on
\url{https://github.com/DCDmllm/WorldGPT}.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.18202) | [💻 Code](https://github.com/DCDmllm/WorldGPT)

### 60. (MTVQA) MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering
**Date**: 2024.05.11

**Affiliation**: ByteDance Inc.
<details span>
<summary><b>Abstract</b></summary>
Text-Centric Visual Question Answering (TEC-VQA) in its proper format not
only facilitates human-machine interaction in text-centric visual environments
but also serves as a de facto gold proxy to evaluate AI models in the domain of
text-centric scene understanding. Nonetheless, most existing TEC-VQA benchmarks
have focused on high-resource languages like English and Chinese. Despite
pioneering works to expand multilingual QA pairs in non-text-centric VQA
datasets through translation engines, the translation-based protocol encounters
a substantial "visual-textual misalignment" problem when applied to TEC-VQA.
Specifically, it prioritizes the text in question-answer pairs while
disregarding the visual text present in images. Moreover, it fails to address
complexities related to nuanced meaning, contextual distortion, language bias,
and question-type diversity. In this work, we tackle multilingual TEC-VQA by
introducing MTVQA, the first benchmark featuring high-quality human expert
annotations across 9 diverse languages, consisting of 6,778 question-answer
pairs across 2,116 images. Further, by comprehensively evaluating numerous
state-of-the-art Multimodal Large Language Models (MLLMs), including GPT-4o,
GPT-4V, Claude3, and Gemini, on the MTVQA dataset, it is evident that there is
still a large room for performance improvement, underscoring the value of
MTVQA. Additionally, we supply multilingual training data within the MTVQA
dataset, demonstrating that straightforward fine-tuning with this data can
substantially enhance multilingual TEC-VQA performance. We aspire that MTVQA
will offer the research community fresh insights and stimulate further
exploration in multilingual visual text comprehension. The project homepage is
available at https://bytedance.github.io/MTVQA/.
</details>

[📄 Paper](https://arxiv.org/pdf/2405.11985) | [🌐 Project Page](https://bytedance.github.io/MTVQA/) | [💻 Code](https://github.com/bytedance/MTVQA)

### 61. (3DCoMPaT-GRIN) Kestrel: Point Grounding Multimodal LLM for Part-Aware 3D Vision-Language Understanding
**Date**: 2024.05.18

**Affiliation**: King Abdullah University of Science and Technology
<details span>
<summary><b>Abstract</b></summary>
While 3D MLLMs have achieved significant progress, they are restricted to
object and scene understanding and struggle to understand 3D spatial structures
at the part level. In this paper, we introduce Kestrel, representing a novel
approach that empowers 3D MLLMs with part-aware understanding, enabling better
interpretation and segmentation grounding of 3D objects at the part level.
Despite its significance, the current landscape lacks tasks and datasets that
endow and assess this capability. Therefore, we propose two novel tasks: (1)
Part-Aware Point Grounding, the model is tasked with directly predicting a
part-level segmentation mask based on user instructions, and (2) Part-Aware
Point Grounded Captioning, the model provides a detailed caption that includes
part-level descriptions and their corresponding masks. To support learning and
evaluating for these tasks, we introduce 3DCoMPaT Grounded Instructions Dataset
(3DCoMPaT-GRIN). 3DCoMPaT-GRIN Vanilla, comprising 789k part-aware point
cloud-instruction-segmentation mask triplets, is used to evaluate MLLMs'
ability of part-aware segmentation grounding. 3DCoMPaT-GRIN Grounded Caption,
containing 107k part-aware point cloud-instruction-grounded caption triplets,
assesses both MLLMs' part-aware language comprehension and segmentation
grounding capabilities. Our introduced tasks, dataset, and Kestrel represent a
preliminary effort to bridge the gap between human cognition and 3D MLLMs,
i.e., the ability to perceive and engage with the environment at both global
and part levels. Extensive experiments on the 3DCoMPaT-GRIN show that Kestrel
can generate user-specified segmentation masks, a capability not present in any
existing 3D MLLM. Kestrel thus established a benchmark for evaluating the
part-aware language comprehension and segmentation grounding of 3D objects.
Project page at https://feielysia.github.io/Kestrel.github.io/
</details>

[📄 Paper](https://arxiv.org/pdf/2405.18937) | [🌐 Project Page](https://feielysia.github.io/Kestrel.github.io/)

### 62. (MMUBench) Single Image Unlearning: Efficient Machine Unlearning in Multimodal Large Language Models
**Date**: 2024.05.21

**Affiliation**: Southeast University
<details span>
<summary><b>Abstract</b></summary>
Machine unlearning empowers individuals with the `right to be forgotten' by
removing their private or sensitive information encoded in machine learning
models. However, it remains uncertain whether MU can be effectively applied to
Multimodal Large Language Models (MLLMs), particularly in scenarios of
forgetting the leaked visual data of concepts. To overcome the challenge, we
propose an efficient method, Single Image Unlearning (SIU), to unlearn the
visual recognition of a concept by fine-tuning a single associated image for
few steps. SIU consists of two key aspects: (i) Constructing Multifaceted
fine-tuning data. We introduce four targets, based on which we construct
fine-tuning data for the concepts to be forgotten; (ii) Jointly training loss.
To synchronously forget the visual recognition of concepts and preserve the
utility of MLLMs, we fine-tune MLLMs through a novel Dual Masked KL-divergence
Loss combined with Cross Entropy loss. Alongside our method, we establish
MMUBench, a new benchmark for MU in MLLMs and introduce a collection of metrics
for its evaluation. Experimental results on MMUBench show that SIU completely
surpasses the performance of existing methods. Furthermore, we surprisingly
find that SIU can avoid invasive membership inference attacks and jailbreak
attacks. To the best of our knowledge, we are the first to explore MU in MLLMs.
We will release the code and benchmark in the near future.
</details>

[📄 Paper](https://arxiv.org/pdf/2405.12523)

### 63. (Video-MME) Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis
**Date**: 2024.05.31

**Affiliation**: USTC
<details span>
<summary><b>Abstract</b></summary>
In the quest for artificial general intelligence, Multi-modal Large Language
Models (MLLMs) have emerged as a focal point in recent advancements. However,
the predominant focus remains on developing their capabilities in static image
understanding. The potential of MLLMs in processing sequential visual data is
still insufficiently explored, highlighting the absence of a comprehensive,
high-quality assessment of their performance. In this paper, we introduce
Video-MME, the first-ever full-spectrum, Multi-Modal Evaluation benchmark of
MLLMs in Video analysis. Our work distinguishes from existing benchmarks
through four key features: 1) Diversity in video types, spanning 6 primary
visual domains with 30 subfields to ensure broad scenario generalizability; 2)
Duration in temporal dimension, encompassing both short-, medium-, and
long-term videos, ranging from 11 seconds to 1 hour, for robust contextual
dynamics; 3) Breadth in data modalities, integrating multi-modal inputs besides
video frames, including subtitles and audios, to unveil the all-round
capabilities of MLLMs; 4) Quality in annotations, utilizing rigorous manual
labeling by expert annotators to facilitate precise and reliable model
assessment. 900 videos with a total of 254 hours are manually selected and
annotated by repeatedly viewing all the video content, resulting in 2,700
question-answer pairs. With Video-MME, we extensively evaluate various
state-of-the-art MLLMs, including GPT-4 series and Gemini 1.5 Pro, as well as
open-source image models like InternVL-Chat-V1.5 and video models like
LLaVA-NeXT-Video. Our experiments reveal that Gemini 1.5 Pro is the
best-performing commercial model, significantly outperforming the open-source
models. Our dataset along with these findings underscores the need for further
improvements in handling longer sequences and multi-modal data. Project Page:
https://video-mme.github.io
</details>

[📄 Paper](https://arxiv.org/pdf/2405.21075) | [🌐 Project Page](https://video-mme.github.io) | [💻 Code](https://github.com/BradyFU/Video-MME)

### 64. (MMMB) Parrot: Multilingual Visual Instruction Tuning
**Date**: 2024.06.02

**Affiliation**: Nanjing University
<details span>
<summary><b>Abstract</b></summary>
The rapid development of Multimodal Large Language Models (MLLMs) like GPT-4V
has marked a significant step towards artificial general intelligence. Existing
methods mainly focus on aligning vision encoders with LLMs through supervised
fine-tuning (SFT) to endow LLMs with multimodal abilities, making MLLMs'
inherent ability to react to multiple languages progressively deteriorate as
the training process evolves. We empirically find that the imbalanced SFT
datasets, primarily composed of English-centric image-text pairs, lead to
significantly reduced performance in non-English languages. This is due to the
failure of aligning the vision encoder and LLM with multilingual tokens during
the SFT process. In this paper, we introduce Parrot, a novel method that
utilizes textual guidance to drive visual token alignment at the language
level. Parrot makes the visual tokens condition on diverse language inputs and
uses Mixture-of-Experts (MoE) to promote the alignment of multilingual tokens.
Specifically, to enhance non-English visual tokens alignment, we compute the
cross-attention using the initial visual features and textual embeddings, the
result of which is then fed into the MoE router to select the most relevant
experts. The selected experts subsequently convert the initial visual tokens
into language-specific visual tokens. Moreover, considering the current lack of
benchmarks for evaluating multilingual capabilities within the field, we
collect and make available a Massive Multilingual Multimodal Benchmark which
includes 6 languages, 15 categories, and 12,000 questions, named as MMMB. Our
method not only demonstrates state-of-the-art performance on multilingual
MMBench and MMMB, but also excels across a broad range of multimodal tasks.
Both the source code and the training dataset of Parrot will be made publicly
available. Code is available at: https://github.com/AIDC-AI/Parrot.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.02539) | [💻 Code](https://github.com/AIDC-AI/Parrot)

### 65. (IIT) Wings: Learning Multimodal LLMs without Text-only Forgetting
**Date**: 2024.06.03

**Affiliation**: Nanjing University
<details span>
<summary><b>Abstract</b></summary>
Multimodal large language models (MLLMs), initiated with a trained LLM, first
align images with text and then fine-tune on multimodal mixed inputs. However,
the MLLM catastrophically forgets the text-only instructions, which do not
include images and can be addressed within the initial LLM. In this paper, we
present Wings, a novel MLLM that excels in both text-only dialogues and
multimodal comprehension. Analyzing MLLM attention in multimodal instructions
reveals that text-only forgetting is related to the attention shifts from
pre-image to post-image text. From that, we construct extra modules that act as
the boosted learner to compensate for the attention shift. The complementary
visual and textual learners, like "wings" on either side, are connected in
parallel within each layer's attention block. Initially, image and text inputs
are aligned with visual learners operating alongside the main attention,
balancing focus on visual elements. Textual learners are later collaboratively
integrated with attention-based routing to blend the outputs of the visual and
textual learners. We design the Low-Rank Residual Attention (LoRRA) to
guarantee high efficiency for learners. Our experimental results demonstrate
that Wings outperforms equally-scaled MLLMs in both text-only and visual
question-answering tasks. On a newly constructed Interleaved Image-Text (IIT)
benchmark, Wings exhibits superior performance from text-only-rich to
multimodal-rich question-answering tasks.
</details>

[📄 Paper](https://arxiv.org/abs/2406.03496)

### 66. (MUIE) Recognizing Everything from All Modalities at Once: Grounded Multimodal Universal Information Extraction
**Date**: 2024.06.03

**Affiliation**: Harbin Institute of Technology (Shenzhen)
<details span>
<summary><b>Abstract</b></summary>
In the field of information extraction (IE), tasks across a wide range of
modalities and their combinations have been traditionally studied in isolation,
leaving a gap in deeply recognizing and analyzing cross-modal information. To
address this, this work for the first time introduces the concept of grounded
Multimodal Universal Information Extraction (MUIE), providing a unified task
framework to analyze any IE tasks over various modalities, along with their
fine-grained groundings. To tackle MUIE, we tailor a multimodal large language
model (MLLM), Reamo, capable of extracting and grounding information from all
modalities, i.e., recognizing everything from all modalities at once. Reamo is
updated via varied tuning strategies, equipping it with powerful capabilities
for information recognition and fine-grained multimodal grounding. To address
the absence of a suitable benchmark for grounded MUIE, we curate a
high-quality, diverse, and challenging test set, which encompasses IE tasks
across 9 common modality combinations with the corresponding multimodal
groundings. The extensive comparison of Reamo with existing MLLMs integrated
into pipeline approaches demonstrates its advantages across all evaluation
dimensions, establishing a strong benchmark for the follow-up research. Our
resources are publicly released at https://haofei.vip/MUIE.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.03701) | [🌐 Project Page](https://haofei.vip/MUIE) | [💻 Code](https://github.com/scofield7419/MUIE-REAMO)

### 67. (MLVU) MLVU: A Comprehensive Benchmark for Multi-Task Long Video Understanding
**Date**: 2024.06.04

**Affiliation**: Beijing Academy of Artificial Intelligence
<details span>
<summary><b>Abstract</b></summary>
The evaluation of Long Video Understanding (LVU) performance poses an
important but challenging research problem. Despite previous efforts, the
existing video understanding benchmarks are severely constrained by several
issues, especially the insufficient lengths of videos, a lack of diversity in
video types and evaluation tasks, and the inappropriateness for evaluating LVU
performances. To address the above problems, we propose a new benchmark, called
MLVU (Multi-task Long Video Understanding Benchmark), for the comprehensive and
in-depth evaluation of LVU. MLVU presents the following critical values: 1) The
substantial and flexible extension of video lengths, which enables the
benchmark to evaluate LVU performance across a wide range of durations. 2) The
inclusion of various video genres, e.g., movies, surveillance footage,
egocentric videos, cartoons, game videos, etc., which reflects the models' LVU
performances in different scenarios. 3) The development of diversified
evaluation tasks, which enables a comprehensive examination of MLLMs' key
abilities in long-video understanding. The empirical study with 20 latest MLLMs
reveals significant room for improvement in today's technique, as all existing
methods struggle with most of the evaluation tasks and exhibit severe
performance degradation when handling longer videos. Additionally, it suggests
that factors such as context length, image-understanding quality, and the
choice of LLM backbone can play critical roles in future advancements. We
anticipate that MLVU will advance the research of long video understanding by
providing a comprehensive and in-depth analysis of MLLMs.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.04264) | [💻 Code](https://github.com/JUNJIE99/MLVU)

### 68. (II-Bench) II-Bench: An Image Implication Understanding Benchmark for Multimodal Large Language Models
**Date**: 2024.06.05

**Affiliation**: Shenzhen Institute of Advanced Technology, CAS
<details span>
<summary><b>Abstract</b></summary>
The rapid advancements in the development of multimodal large language models
(MLLMs) have consistently led to new breakthroughs on various benchmarks. In
response, numerous challenging and comprehensive benchmarks have been proposed
to more accurately assess the capabilities of MLLMs. However, there is a dearth
of exploration of the higher-order perceptual capabilities of MLLMs. To fill
this gap, we propose the Image Implication understanding Benchmark, II-Bench,
which aims to evaluate the model's higher-order perception of images. Through
extensive experiments on II-Bench across multiple MLLMs, we have made
significant findings. Initially, a substantial gap is observed between the
performance of MLLMs and humans on II-Bench. The pinnacle accuracy of MLLMs
attains 74.8%, whereas human accuracy averages 90%, peaking at an impressive
98%. Subsequently, MLLMs perform worse on abstract and complex images,
suggesting limitations in their ability to understand high-level semantics and
capture image details. Finally, it is observed that most models exhibit
enhanced accuracy when image sentiment polarity hints are incorporated into the
prompts. This observation underscores a notable deficiency in their inherent
understanding of image sentiment. We believe that II-Bench will inspire the
community to develop the next generation of MLLMs, advancing the journey
towards expert artificial general intelligence (AGI). II-Bench is publicly
available at https://huggingface.co/datasets/m-a-p/II-Bench.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.05862) | [🌐 Project Page](https://ii-bench.github.io/) | [💻 Code](https://github.com/II-Bench/II-Bench)

### 69. (CVQA) CVQA: Culturally-diverse Multilingual Visual Question Answering Benchmark
**Date**: 2024.06.05

**Affiliation**: MBZUAI
<details span>
<summary><b>Abstract</b></summary>
Visual Question Answering (VQA) is an important task in multimodal AI, and it
is often used to test the ability of vision-language models to understand and
reason on knowledge present in both visual and textual data. However, most of
the current VQA models use datasets that are primarily focused on English and a
few major world languages, with images that are typically Western-centric.
While recent efforts have tried to increase the number of languages covered on
VQA datasets, they still lack diversity in low-resource languages. More
importantly, although these datasets often extend their linguistic range via
translation or some other approaches, they usually keep images the same,
resulting in narrow cultural representation. To address these limitations, we
construct CVQA, a new Culturally-diverse multilingual Visual Question Answering
benchmark, designed to cover a rich set of languages and cultures, where we
engage native speakers and cultural experts in the data collection process. As
a result, CVQA includes culturally-driven images and questions from across 28
countries on four continents, covering 26 languages with 11 scripts, providing
a total of 9k questions. We then benchmark several Multimodal Large Language
Models (MLLMs) on CVQA, and show that the dataset is challenging for the
current state-of-the-art models. This benchmark can serve as a probing
evaluation suite for assessing the cultural capability and bias of multimodal
models and hopefully encourage more research efforts toward increasing cultural
awareness and linguistic diversity in this field.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.05967) | [🌐 Project Page](https://cvqa-benchmark.org/)

### 70. (M3GIA) M3GIA: A Cognition Inspired Multilingual and Multimodal General Intelligence Ability Benchmark
**Date**: 2024.06.05

**Affiliation**: Westlake University
<details span>
<summary><b>Abstract</b></summary>
As recent multi-modality large language models (MLLMs) have shown formidable
proficiency on various complex tasks, there has been increasing attention on
debating whether these models could eventually mirror human intelligence.
However, existing benchmarks mainly focus on evaluating solely on task
performance, such as the accuracy of identifying the attribute of an object.
Combining well-developed cognitive science to understand the intelligence of
MLLMs beyond superficial achievements remains largely unexplored. To this end,
we introduce the first cognitive-driven multi-lingual and multi-modal benchmark
to evaluate the general intelligence ability of MLLMs, dubbed M3GIA.
Specifically, we identify five key cognitive factors based on the
well-recognized Cattell-Horn-Carrol (CHC) model of intelligence and propose a
novel evaluation metric. In addition, since most MLLMs are trained to perform
in different languages, a natural question arises: is language a key factor
influencing the cognitive ability of MLLMs? As such, we go beyond English to
encompass other languages based on their popularity, including Chinese, French,
Spanish, Portuguese and Korean, to construct our M3GIA. We make sure all the
data relevant to the cultural backgrounds are collected from their native
context to avoid English-centric bias. We collected a significant corpus of
data from human participants, revealing that the most advanced MLLM reaches the
lower boundary of human intelligence in English. Yet, there remains a
pronounced disparity in the other five languages assessed. We also reveals an
interesting winner takes all phenomenon that are aligned with the discovery in
cognitive studies. Our benchmark will be open-sourced, with the aspiration of
facilitating the enhancement of cognitive capabilities in MLLMs.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.05343)

### 71. (MM-NIAH) Needle In A Multimodal Haystack
**Date**: 2024.06.07

**Affiliation**: Shanghai AI Laboratory
<details span>
<summary><b>Abstract</b></summary>
With the rapid advancement of multimodal large language models (MLLMs), their
evaluation has become increasingly comprehensive. However, understanding long
multimodal content, as a foundational ability for real-world applications,
remains underexplored. In this work, we present Needle In A Multimodal Haystack
(MM-NIAH), the first benchmark specifically designed to systematically evaluate
the capability of existing MLLMs to comprehend long multimodal documents. Our
benchmark includes three types of evaluation tasks: multimodal retrieval,
counting, and reasoning. In each task, the model is required to answer the
questions according to different key information scattered throughout the given
multimodal document. Evaluating the leading MLLMs on MM-NIAH, we observe that
existing models still have significant room for improvement on these tasks,
especially on vision-centric evaluation. We hope this work can provide a
platform for further research on long multimodal document comprehension and
contribute to the advancement of MLLMs. Code and benchmark are released at
https://github.com/OpenGVLab/MM-NIAH.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.07230) | [💻 Code](https://github.com/OpenGVLab/MM-NIAH)

### 72. (IT) Image Textualization: An Automatic Framework for Creating Accurate and Detailed Image Descriptions
**Date**: 2024.06.07

**Affiliation**: The Hong Kong University of Science and Technology
<details span>
<summary><b>Abstract</b></summary>
Image description datasets play a crucial role in the advancement of various
applications such as image understanding, text-to-image generation, and
text-image retrieval. Currently, image description datasets primarily originate
from two sources. One source is the scraping of image-text pairs from the web.
Despite their abundance, these descriptions are often of low quality and noisy.
Another is through human labeling. Datasets such as COCO are generally very
short and lack details. Although detailed image descriptions can be annotated
by humans, the high annotation cost limits the feasibility. These limitations
underscore the need for more efficient and scalable methods to generate
accurate and detailed image descriptions. In this paper, we propose an
innovative framework termed Image Textualization (IT), which automatically
produces high-quality image descriptions by leveraging existing multi-modal
large language models (MLLMs) and multiple vision expert models in a
collaborative manner, which maximally convert the visual information into text.
To address the current lack of benchmarks for detailed descriptions, we propose
several benchmarks for comprehensive evaluation, which verifies the quality of
image descriptions created by our framework. Furthermore, we show that
LLaVA-7B, benefiting from training on IT-curated descriptions, acquire improved
capability to generate richer image descriptions, substantially increasing the
length and detail of their output with less hallucination.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.07502) | [💻 Code](https://github.com/sterzhang/image-textualization)

### 73. (VideoNIAH) Needle In A Video Haystack: A Scalable Synthetic Framework for Benchmarking Video MLLMs
**Date**: 2024.06.09

**Affiliation**: Chinese Academy of Sciences
<details span>
<summary><b>Abstract</b></summary>
Video understanding is a crucial next step for multimodal large language
models (MLLMs). To probe specific aspects of video understanding ability,
existing video benchmarks typically require careful video selection based on
the target capability, along with laborious annotation of query-response pairs
to match the specific video content. This process is both challenging and
resource-intensive. In this paper, we propose VideoNIAH (Video Needle In A
Haystack), a benchmark construction framework through synthetic video
generation. VideoNIAH decouples test video content from their query-responses
by inserting unrelated image/text 'needles' into original videos. It generates
annotations solely from these needles, ensuring diversity in video sources and
a variety of query-responses. Additionally, by inserting multiple needles,
VideoNIAH rigorously evaluates the temporal understanding capabilities of
models. We utilized VideoNIAH to compile a video benchmark VNBench, including
tasks such as retrieval, ordering, and counting. VNBench can efficiently
evaluate the fine-grained understanding ability and spatio-temporal modeling
ability of a video model, while also supporting the long-context evaluation.
Additionally, we evaluated recent video-centric multimodal large language
models (MLLMs), both open-source and proprietary, providing a comprehensive
analysis. We found that although proprietary models have significant advantages
over open-source models, all existing video models still perform poorly on
long-distance dependency tasks. VideoNIAH is a simple yet highly scalable
benchmark construction framework, and we believe it will inspire future video
benchmark works. The code and data are available at
https://github.com/joez17/VideoNIAH.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.09367) | [💻 Code](https://github.com/joez17/VideoNIAH)

### 74. (VEGA) VEGA: Learning Interleaved Image-Text Comprehension in Vision-Language Large Models
**Date**: 2024.06.10

**Affiliation**: Xiamen University
<details span>
<summary><b>Abstract</b></summary>
The swift progress of Multi-modal Large Models (MLLMs) has showcased their
impressive ability to tackle tasks blending vision and language. Yet, most
current models and benchmarks cater to scenarios with a narrow scope of visual
and textual contexts. These models often fall short when faced with complex
comprehension tasks, which involve navigating through a plethora of irrelevant
and potentially misleading information in both text and image forms. To bridge
this gap, we introduce a new, more demanding task known as Interleaved
Image-Text Comprehension (IITC). This task challenges models to discern and
disregard superfluous elements in both images and text to accurately answer
questions and to follow intricate instructions to pinpoint the relevant image.
In support of this task, we further craft a new VEGA dataset, tailored for the
IITC task on scientific content, and devised a subtask, Image-Text Association
(ITA), to refine image-text correlation skills. Our evaluation of four leading
closed-source models, as well as various open-source models using VEGA,
underscores the rigorous nature of IITC. Even the most advanced models, such as
Gemini-1.5-pro and GPT4V, only achieved modest success. By employing a
multi-task, multi-scale post-training strategy, we have set a robust baseline
for MLLMs on the IITC task, attaining an $85.8\%$ accuracy rate in image
association and a $0.508$ Rouge score. These results validate the effectiveness
of our dataset in improving MLLMs capabilities for nuanced image-text
comprehension.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.10228) | [🌐 Project Page](https://zhourax.github.io/VEGA/) | [💻 Code](https://github.com/zhourax/VEGA)

### 75. (LLaNA) LLaNA: Large Language and NeRF Assistant
**Date**: 2024.06.11

**Affiliation**: University of Bologna
<details span>
<summary><b>Abstract</b></summary>
Multimodal Large Language Models (MLLMs) have demonstrated an excellent
understanding of images and 3D data. However, both modalities have shortcomings
in holistically capturing the appearance and geometry of objects. Meanwhile,
Neural Radiance Fields (NeRFs), which encode information within the weights of
a simple Multi-Layer Perceptron (MLP), have emerged as an increasingly
widespread modality that simultaneously encodes the geometry and photorealistic
appearance of objects. This paper investigates the feasibility and
effectiveness of ingesting NeRF into MLLM. We create LLaNA, the first
general-purpose NeRF-language assistant capable of performing new tasks such as
NeRF captioning and Q\&A. Notably, our method directly processes the weights of
the NeRF's MLP to extract information about the represented objects without the
need to render images or materialize 3D data structures. Moreover, we build a
dataset of NeRFs with text annotations for various NeRF-language tasks with no
human intervention. Based on this dataset, we develop a benchmark to evaluate
the NeRF understanding capability of our method. Results show that processing
NeRF weights performs favourably against extracting 2D or 3D representations
from NeRFs.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.11840) | [🌐 Project Page](https://andreamaduzzi.github.io/llana/)

### 76. (MMNeedle) Multimodal Needle in a Haystack: Benchmarking Long-Context Capability of Multimodal Large Language Models
**Date**: 2024.06.11

**Affiliation**: Rutgers University
<details span>
<summary><b>Abstract</b></summary>
Multimodal Large Language Models (MLLMs) have shown significant promise in
various applications, leading to broad interest from researchers and
practitioners alike. However, a comprehensive evaluation of their long-context
capabilities remains underexplored. To address these gaps, we introduce the
MultiModal Needle-in-a-haystack (MMNeedle) benchmark, specifically designed to
assess the long-context capabilities of MLLMs. Besides multi-image input, we
employ image stitching to further increase the input context length, and
develop a protocol to automatically generate labels for sub-image level
retrieval. Essentially, MMNeedle evaluates MLLMs by stress-testing their
capability to locate a target sub-image (needle) within a set of images
(haystack) based on textual instructions and descriptions of image contents.
This setup necessitates an advanced understanding of extensive visual contexts
and effective information retrieval within long-context image inputs. With this
benchmark, we evaluate state-of-the-art MLLMs, encompassing both API-based and
open-source models. The findings reveal that GPT-4o consistently surpasses
other models in long-context scenarios, but suffers from hallucination problems
in negative samples, i.e., when needles are not in the haystacks. Our
comprehensive long-context evaluation of MLLMs also sheds lights on the
considerable performance gap between API-based and open-source models. All the
code, data, and instructions required to reproduce the main results are
available at https://github.com/Wang-ML-Lab/multimodal-needle-in-a-haystack.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.11230) | [💻 Code](https://github.com/Wang-ML-Lab/multimodal-needle-in-a-haystack)

### 77. (MuirBench) MuirBench: A Comprehensive Benchmark for Robust Multi-image Understanding
**Date**: 2024.06.13

**Affiliation**: USC
<details span>
<summary><b>Abstract</b></summary>
We introduce MuirBench, a comprehensive benchmark that focuses on robust
multi-image understanding capabilities of multimodal LLMs. MuirBench consists
of 12 diverse multi-image tasks (e.g., scene understanding, ordering) that
involve 10 categories of multi-image relations (e.g., multiview, temporal
relations). Comprising 11,264 images and 2,600 multiple-choice questions,
MuirBench is created in a pairwise manner, where each standard instance is
paired with an unanswerable variant that has minimal semantic differences, in
order for a reliable assessment. Evaluated upon 20 recent multi-modal LLMs, our
results reveal that even the best-performing models like GPT-4o and Gemini Pro
find it challenging to solve MuirBench, achieving 68.0% and 49.3% in accuracy.
Open-source multimodal LLMs trained on single images can hardly generalize to
multi-image questions, hovering below 33.3% in accuracy. These results
highlight the importance of MuirBench in encouraging the community to develop
multimodal LLMs that can look beyond a single image, suggesting potential
pathways for future improvements.
</details>

[📄 Paper](https://arxiv.org/abs/2406.09411) | [🌐 Project Page](https://muirbench.github.io/) | [💻 Code](https://github.com/muirbench/MuirBench)

### 78. (ADLMCQ) LLAVIDAL: Benchmarking Large Language Vision Models for Daily Activities of Living
**Date**: 2024.06.14

**Affiliation**: UNC Charlotte
<details span>
<summary><b>Abstract</b></summary>
Large Language Vision Models (LLVMs) have demonstrated effectiveness in
processing internet videos, yet they struggle with the visually perplexing
dynamics present in Activities of Daily Living (ADL) due to limited pertinent
datasets and models tailored to relevant cues. To this end, we propose a
framework for curating ADL multiview datasets to fine-tune LLVMs, resulting in
the creation of ADL-X, comprising 100K RGB video-instruction pairs, language
descriptions, 3D skeletons, and action-conditioned object trajectories. We
introduce LLAVIDAL, an LLVM capable of incorporating 3D poses and relevant
object trajectories to understand the intricate spatiotemporal relationships
within ADLs. Furthermore, we present a novel benchmark, ADLMCQ, for quantifying
LLVM effectiveness in ADL scenarios. When trained on ADL-X, LLAVIDAL
consistently achieves state-of-the-art performance across all ADL evaluation
metrics. Qualitative analysis reveals LLAVIDAL's temporal reasoning
capabilities in understanding ADL. The link to the dataset is provided at:
https://adl-x.github.io/
</details>

[📄 Paper](https://arxiv.org/pdf/2406.09390) | [🌐 Project Page](https://adl-x.github.io/) | [💻 Code](https://github.com/ADL-X/LLAVIDAL)

### 79. (Event-Bench) Towards Event-oriented Long Video Understanding
**Date**: 2024.06.14

**Affiliation**: Renmin University of China
<details span>
<summary><b>Abstract</b></summary>
With the rapid development of video Multimodal Large Language Models (MLLMs),
numerous benchmarks have been proposed to assess their video understanding
capability. However, due to the lack of rich events in the videos, these
datasets may suffer from the short-cut bias that the answers can be deduced
from a few frames, without the need to watch the entire video. To address this
issue, we introduce Event-Bench, an event-oriented long video understanding
benchmark built on existing datasets and human annotations. Event-Bench
includes six event-related tasks and 2,190 test instances to comprehensively
evaluate video event understanding ability. Additionally, we propose Video
Instruction Merging~(VIM), a cost-effective method that enhances video MLLMs
using merged, event-intensive video instructions, addressing the scarcity of
human-annotated, event-intensive data. Extensive experiments show that the
best-performing model, GPT-4o, achieves an overall accuracy of 53.33,
significantly outperforming the best open-source model by 41.42%. Leveraging an
effective instruction synthesis method and an adaptive model architecture, VIM
surpasses both state-of-the-art open-source models and GPT-4V on the
Event-Bench. All code, data, and models are publicly available at
https://github.com/RUCAIBox/Event-Bench.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.14129) | [💻 Code](https://github.com/RUCAIBox/Event-Bench)

### 80. (CV-Bench) Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs
**Date**: 2024.06.16

**Affiliation**: New York University
<details span>
<summary><b>Abstract</b></summary>
We introduce Cambrian-1, a family of multimodal LLMs (MLLMs) designed with a
vision-centric approach. While stronger language models can enhance multimodal
capabilities, the design choices for vision components are often insufficiently
explored and disconnected from visual representation learning research. This
gap hinders accurate sensory grounding in real-world scenarios. Our study uses
LLMs and visual instruction tuning as an interface to evaluate various visual
representations, offering new insights into different models and architectures
-- self-supervised, strongly supervised, or combinations thereof -- based on
experiments with over 20 vision encoders. We critically examine existing MLLM
benchmarks, addressing the difficulties involved in consolidating and
interpreting results from various tasks, and introduce a new vision-centric
benchmark, CV-Bench. To further improve visual grounding, we propose the
Spatial Vision Aggregator (SVA), a dynamic and spatially-aware connector that
integrates high-resolution vision features with LLMs while reducing the number
of tokens. Additionally, we discuss the curation of high-quality visual
instruction-tuning data from publicly available sources, emphasizing the
importance of data source balancing and distribution ratio. Collectively,
Cambrian-1 not only achieves state-of-the-art performance but also serves as a
comprehensive, open cookbook for instruction-tuned MLLMs. We provide model
weights, code, supporting tools, datasets, and detailed instruction-tuning and
evaluation recipes. We hope our release will inspire and accelerate
advancements in multimodal systems and visual representation learning.
</details>

[📄 Paper](https://arxiv.org/abs/2406.16860) | [🌐 Project Page](https://cambrian-mllm.github.io/) | [💻 Code](https://github.com/cambrian-mllm/cambrian)

### 81. (EmoBench) EmoLLM: Multimodal Emotional Understanding Meets Large Language Models
**Date**: 2024.06.16

**Affiliation**: Wuhan University
<details span>
<summary><b>Abstract</b></summary>
Multi-modal large language models (MLLMs) have achieved remarkable
performance on objective multimodal perception tasks, but their ability to
interpret subjective, emotionally nuanced multimodal content remains largely
unexplored. Thus, it impedes their ability to effectively understand and react
to the intricate emotions expressed by humans through multimodal media. To
bridge this gap, we introduce EmoBench, the first comprehensive benchmark
designed specifically to evaluate the emotional capabilities of MLLMs across
five popular emotional tasks, using a diverse dataset of 287k images and videos
paired with corresponding textual instructions. Meanwhile, we propose EmoLLM, a
novel model for multimodal emotional understanding, incorporating with two core
techniques. 1) Multi-perspective Visual Projection, it captures diverse
emotional cues from visual data from multiple perspectives. 2) EmoPrompt, it
guides MLLMs to reason about emotions in the correct direction. Experimental
results demonstrate that EmoLLM significantly elevates multimodal emotional
understanding performance, with an average improvement of 12.1% across multiple
foundation models on EmoBench. Our work contributes to the advancement of MLLMs
by facilitating a deeper and more nuanced comprehension of intricate human
emotions, paving the way for the development of artificial emotional
intelligence capabilities with wide-ranging applications in areas such as
human-computer interaction, mental health support, and empathetic AI systems.
Code, data, and model will be released.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.16442) | [💻 Code](https://github.com/yan9qu/EmoLLM)

### 82. (DenseFusion-1M) DenseFusion-1M: Merging Vision Experts for Comprehensive Multimodal Perception
**Date**: 2024.07.08

**Affiliation**: Peking University
<details span>
<summary><b>Abstract</b></summary>
Existing Multimodal Large Language Models (MLLMs) increasingly emphasize
complex understanding of various visual elements, including multiple objects,
text information, and spatial relations. Their development for comprehensive
visual perception hinges on the availability of high-quality image-text
datasets that offer diverse visual elements and throughout image descriptions.
However, the scarcity of such hyper-detailed datasets currently hinders
progress within the MLLM community. The bottleneck stems from the limited
perceptual capabilities of current caption engines, which fall short in
providing complete and accurate annotations. To facilitate the cutting-edge
research of MLLMs on comprehensive vision perception, we thereby propose
Perceptual Fusion, using a low-budget but highly effective caption engine for
complete and accurate image descriptions. Specifically, Perceptual Fusion
integrates diverse perception experts as image priors to provide explicit
information on visual elements and adopts an efficient MLLM as a centric pivot
to mimic advanced MLLMs' perception abilities. We carefully select 1M highly
representative images from uncurated LAION dataset and generate dense
descriptions using our engine, dubbed DenseFusion-1M. Extensive experiments
validate that our engine outperforms its counterparts, where the resulting
dataset significantly improves the perception and cognition abilities of
existing MLLMs across diverse vision-language benchmarks, especially with
high-resolution images as inputs. The dataset and code are publicly available
at https://github.com/baaivision/DenseFusion.
</details>

[📄 Paper](https://arxiv.org/pdf/2407.08303) | [💻 Code](https://github.com/baaivision/DenseFusion)

### 83. (MuChoMusic) MuChoMusic: Evaluating Music Understanding in Multimodal Audio-Language Models
**Date**: 2024.08.02

**Affiliation**: Universitat Pompeu Fabra
<details span>
<summary><b>Abstract</b></summary>
Multimodal models that jointly process audio and language hold great promise
in audio understanding and are increasingly being adopted in the music domain.
By allowing users to query via text and obtain information about a given audio
input, these models have the potential to enable a variety of music
understanding tasks via language-based interfaces. However, their evaluation
poses considerable challenges, and it remains unclear how to effectively assess
their ability to correctly interpret music-related inputs with current methods.
Motivated by this, we introduce MuChoMusic, a benchmark for evaluating music
understanding in multimodal language models focused on audio. MuChoMusic
comprises 1,187 multiple-choice questions, all validated by human annotators,
on 644 music tracks sourced from two publicly available music datasets, and
covering a wide variety of genres. Questions in the benchmark are crafted to
assess knowledge and reasoning abilities across several dimensions that cover
fundamental musical concepts and their relation to cultural and functional
contexts. Through the holistic analysis afforded by the benchmark, we evaluate
five open-source models and identify several pitfalls, including an
over-reliance on the language modality, pointing to a need for better
multimodal integration. Data and code are open-sourced.
</details>

[📄 Paper](https://arxiv.org/pdf/2408.01337) | [🌐 Project Page](https://mulab-mir.github.io/muchomusic/) | [💻 Code](https://github.com/mulab-mir/muchomusic)

### 84. (UniBench) UniBench: Visual Reasoning Requires Rethinking Vision-Language Beyond Scaling
**Date**: 2024.08.04

**Affiliation**: MetaFAIR
<details span>
<summary><b>Abstract</b></summary>
Significant research efforts have been made to scale and improve
vision-language model (VLM) training approaches. Yet, with an ever-growing
number of benchmarks, researchers are tasked with the heavy burden of
implementing each protocol, bearing a non-trivial computational cost, and
making sense of how all these benchmarks translate into meaningful axes of
progress. To facilitate a systematic evaluation of VLM progress, we introduce
UniBench: a unified implementation of 50+ VLM benchmarks spanning a
comprehensive range of carefully categorized capabilities from object
recognition to spatial awareness, counting, and much more. We showcase the
utility of UniBench for measuring progress by evaluating nearly 60 publicly
available vision-language models, trained on scales of up to 12.8B samples. We
find that while scaling training data or model size can boost many
vision-language model capabilities, scaling offers little benefit for reasoning
or relations. Surprisingly, we also discover today's best VLMs struggle on
simple digit recognition and counting tasks, e.g. MNIST, which much simpler
networks can solve. Where scale falls short, we find that more precise
interventions, such as data quality or tailored-learning objectives offer more
promise. For practitioners, we also offer guidance on selecting a suitable VLM
for a given application. Finally, we release an easy-to-run UniBench code-base
with the full set of 50+ benchmarks and comparisons across 59 models as well as
a distilled, representative set of benchmarks that runs in 5 minutes on a
single GPU.
</details>

[📄 Paper](https://arxiv.org/abs/2408.04810) | [💻 Code](https://github.com/facebookresearch/unibench)


### 85. (MMIU) MMIU: Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models 
**Date**: 2024.08.05

**Affiliation**: Shanghai AI Laboratory
<details span>
<summary><b>Abstract</b></summary>
The capability to process multiple images is crucial for Large Vision-Language Models (LVLMs) to develop a more thorough and nuanced understanding of a scene. Recent multi-image LVLMs have begun to address this need. However, their evaluation has not kept pace with their development. To fill this gap, we introduce the Multimodal Multi-image Understanding (MMIU) benchmark, a comprehensive evaluation suite designed to assess LVLMs across a wide range of multi-image tasks. MMIU encompasses 7 types of multi-image relationships, 52 tasks, 77K images, and 11K meticulously curated multiple-choice questions, making it the most extensive benchmark of its kind. Our evaluation of 24 popular LVLMs, including both open-source and proprietary models, reveals significant challenges in multi-image comprehension, particularly in tasks involving spatial understanding. Even the most advanced models, such as GPT-4o, achieve only 55.7% accuracy on MMIU. Through multi-faceted analytical experiments, we identify key performance gaps and limitations, providing valuable insights for future model and data improvements. We aim for MMIU to advance the frontier of LVLM research and development, moving us toward achieving sophisticated multimodal multi-image user interactions.
</details>

  [📄 Paper](https://arxiv.org/pdf/2408.02718) | [🌐 Project Page](https://mmiu-bench.github.io/) | [💻 Code](https://github.com/OpenGVLab/MMIU)

## Reasoning Benchmarks:

### 1. (DUDE) Document Understanding Dataset and Evaluation (DUDE)
**Date**: 2023.05.15

**Affiliation**: KU Leuven
<details span>
<summary><b>Abstract</b></summary>
We call on the Document AI (DocAI) community to reevaluate current
methodologies and embrace the challenge of creating more practically-oriented
benchmarks. Document Understanding Dataset and Evaluation (DUDE) seeks to
remediate the halted research progress in understanding visually-rich documents
(VRDs). We present a new dataset with novelties related to types of questions,
answers, and document layouts based on multi-industry, multi-domain, and
multi-page VRDs of various origins, and dates. Moreover, we are pushing the
boundaries of current methods by creating multi-task and multi-domain
evaluation setups that more accurately simulate real-world situations where
powerful generalization and adaptation under low-resource settings are desired.
DUDE aims to set a new standard as a more practical, long-standing benchmark
for the community, and we hope that it will lead to future extensions and
contributions that address real-world challenges. Finally, our work illustrates
the importance of finding more efficient ways to model language, images, and
layout in DocAI.
</details>

[📄 Paper](https://arxiv.org/abs/2305.08455)

### 2. (M3Exam) M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models
**Date**: 2023.06.08

**Affiliation**: DAMO Academy
<details span>
<summary><b>Abstract</b></summary>
Despite the existence of various benchmarks for evaluating natural language
processing models, we argue that human exams are a more suitable means of
evaluating general intelligence for large language models (LLMs), as they
inherently demand a much wider range of abilities such as language
understanding, domain knowledge, and problem-solving skills. To this end, we
introduce M3Exam, a novel benchmark sourced from real and official human exam
questions for evaluating LLMs in a multilingual, multimodal, and multilevel
context. M3Exam exhibits three unique characteristics: (1) multilingualism,
encompassing questions from multiple countries that require strong multilingual
proficiency and cultural knowledge; (2) multimodality, accounting for the
multimodal nature of many exam questions to test the model's multimodal
understanding capability; and (3) multilevel structure, featuring exams from
three critical educational periods to comprehensively assess a model's
proficiency at different levels. In total, M3Exam contains 12,317 questions in
9 diverse languages with three educational levels, where about 23\% of the
questions require processing images for successful solving. We assess the
performance of top-performing LLMs on M3Exam and find that current models,
including GPT-4, still struggle with multilingual text, particularly in
low-resource and non-Latin script languages. Multimodal LLMs also perform
poorly with complex multimodal questions. We believe that M3Exam can be a
valuable resource for comprehensively evaluating LLMs by examining their
multilingual and multimodal abilities and tracking their development. Data and
evaluation code is available at \url{https://github.com/DAMO-NLP-SG/M3Exam}.
</details>

[📄 Paper](https://arxiv.org/abs/2306.05179) | [💻 Code](https://github.com/DAMO-NLP-SG/M3Exam)

### 3. (SciGraphQA) SciGraphQA: A Large-Scale Synthetic Multi-Turn Question-Answering Dataset for Scientific Graphs
**Date**: 2023.08.07

**Affiliation**: Tifin
<details span>
<summary><b>Abstract</b></summary>
In this work, we present SciGraphQA, a synthetic multi-turn question-answer
dataset related to academic graphs. SciGraphQA is 13 times larger than
ChartVQA, the previously largest chart-visual question-answering dataset. It is
also the largest open-sourced chart VQA dataset with non-synthetic charts. To
build our dataset, we selected 290,000 Computer Science or Machine Learning
ArXiv papers published between 2010 and 2020, and then used Palm-2 to generate
295K samples of open-vocabulary multi-turn question-answering dialogues about
the graphs. As context, we provided the text-only Palm-2 with paper title,
abstract, paragraph mentioning the graph, and rich text contextual data from
the graph itself, obtaining dialogues with an average 2.23 question-answer
turns for each graph. We asked GPT-4 to assess the matching quality of our
question-answer turns given the paper's context, obtaining an average rating of
8.7/10 on our 3K test set. We evaluated the 0-shot capability of the most
popular MLLM models such as LLaVa, mPLUGowl, BLIP-2, and openFlamingo's on our
dataset, finding LLaVA-13B being the most performant with a CIDEr score of
0.08. We further enriched the question prompts for LLAVA by including the
serialized data tables extracted from the graphs using the DePlot model,
boosting LLaVA's 0-shot CIDEr to 0.15. To verify the validity of our dataset,
we also fine-tuned LLaVa using our dataset, reaching a substantially higher
CIDEr score of 0.26. We anticipate further accuracy improvement by including
segmentation mask tokens and leveraging larger LLM backbones coupled with
emergent prompting techniques. Our code and data are open-sourced.
</details>

[📄 Paper](https://arxiv.org/abs/2308.03349)

### 4. (MathVista (MathV)) MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts
**Date**: 2023.10.02

**Affiliation**: UCLA
<details span>
<summary><b>Abstract</b></summary>
Large Language Models (LLMs) and Large Multimodal Models (LMMs) exhibit
impressive problem-solving skills in many tasks and domains, but their ability
in mathematical reasoning in visual contexts has not been systematically
studied. To bridge this gap, we present MathVista, a benchmark designed to
combine challenges from diverse mathematical and visual tasks. It consists of
6,141 examples, derived from 28 existing multimodal datasets involving
mathematics and 3 newly created datasets (i.e., IQTest, FunctionQA, and
PaperQA). Completing these tasks requires fine-grained, deep visual
understanding and compositional reasoning, which all state-of-the-art
foundation models find challenging. With MathVista, we have conducted a
comprehensive, quantitative evaluation of 12 prominent foundation models. The
best-performing GPT-4V model achieves an overall accuracy of 49.9%,
substantially outperforming Bard, the second-best performer, by 15.1%. Our
in-depth analysis reveals that the superiority of GPT-4V is mainly attributed
to its enhanced visual perception and mathematical reasoning. However, GPT-4V
still falls short of human performance by 10.4%, as it often struggles to
understand complex figures and perform rigorous reasoning. This significant gap
underscores the critical role that MathVista will play in the development of
general-purpose AI agents capable of tackling mathematically intensive and
visually rich real-world tasks. We further explore the new ability of
self-verification, the application of self-consistency, and the interactive
chatbot capabilities of GPT-4V, highlighting its promising potential for future
research. The project is available at https://mathvista.github.io/.
</details>

[📄 Paper](https://arxiv.org/abs/2310.02255) | [🌐 Project Page](https://mathvista.github.io/) | [💻 Code](https://github.com/lupantech/MathVista)

### 5. (MMEdit) Can We Edit Multimodal Large Language Models?
**Date**: 2023.10.08

**Affiliation**: Zhejiang University
<details span>
<summary><b>Abstract</b></summary>
In this paper, we focus on editing Multimodal Large Language Models (MLLMs).
Compared to editing single-modal LLMs, multimodal model editing is more
challenging, which demands a higher level of scrutiny and careful consideration
in the editing process. To facilitate research in this area, we construct a new
benchmark, dubbed MMEdit, for editing multimodal LLMs and establishing a suite
of innovative metrics for evaluation. We conduct comprehensive experiments
involving various model editing baselines and analyze the impact of editing
different components for multimodal LLMs. Empirically, we notice that previous
baselines can implement editing multimodal LLMs to some extent, but the effect
is still barely satisfactory, indicating the potential difficulty of this task.
We hope that our work can provide the NLP community with insights. Code and
dataset are available in https://github.com/zjunlp/EasyEdit.
</details>

[📄 Paper](https://arxiv.org/abs/2310.08475) | [💻 Code](https://github.com/zjunlp/EasyEdit)

### 6. (What’sUp) What's "up" with vision-language models? Investigating their struggle with spatial reasoning
**Date**: 2023.10.30

**Affiliation**: University of California
<details span>
<summary><b>Abstract</b></summary>
Recent vision-language (VL) models are powerful, but can they reliably
distinguish "right" from "left"? We curate three new corpora to quantify model
comprehension of such basic spatial relations. These tests isolate spatial
reasoning more precisely than existing datasets like VQAv2, e.g., our What'sUp
benchmark contains sets of photographs varying only the spatial relations of
objects, keeping their identity fixed (see Figure 1: models must comprehend not
only the usual case of a dog under a table, but also, the same dog on top of
the same table). We evaluate 18 VL models, finding that all perform poorly,
e.g., BLIP finetuned on VQAv2, which nears human parity on VQAv2, achieves 56%
accuracy on our benchmarks vs. humans at 99%. We conclude by studying causes of
this surprising behavior, finding: 1) that popular vision-language pretraining
corpora like LAION-2B contain little reliable data for learning spatial
relationships; and 2) that basic modeling interventions like up-weighting
preposition-containing instances or fine-tuning on our corpora are not
sufficient to address the challenges our benchmarks pose. We are hopeful that
these corpora will facilitate further research, and we release our data and
code at https://github.com/amitakamath/whatsup_vlms.
</details>

[📄 Paper](https://arxiv.org/abs/2310.19785) | [💻 Code](https://github.com/amitakamath/whatsup_vlms)

### 7. (ViLMA) ViLMA: A Zero-Shot Benchmark for Linguistic and Temporal Grounding in Video-Language Models
**Date**: 2023.11.13

**Affiliation**: Koç University, KUIS AI Center
<details span>
<summary><b>Abstract</b></summary>
With the ever-increasing popularity of pretrained Video-Language Models
(VidLMs), there is a pressing need to develop robust evaluation methodologies
that delve deeper into their visio-linguistic capabilities. To address this
challenge, we present ViLMA (Video Language Model Assessment), a task-agnostic
benchmark that places the assessment of fine-grained capabilities of these
models on a firm footing. Task-based evaluations, while valuable, fail to
capture the complexities and specific temporal aspects of moving images that
VidLMs need to process. Through carefully curated counterfactuals, ViLMA offers
a controlled evaluation suite that sheds light on the true potential of these
models, as well as their performance gaps compared to human-level
understanding. ViLMA also includes proficiency tests, which assess basic
capabilities deemed essential to solving the main counterfactual tests. We show
that current VidLMs' grounding abilities are no better than those of
vision-language models which use static images. This is especially striking
once the performance on proficiency tests is factored in. Our benchmark serves
as a catalyst for future research on VidLMs, helping to highlight areas that
still need to be explored.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.07022) | [🌐 Project Page](https://cyberiada.github.io/ViLMA/) | [💻 Code](https://github.com/ilkerkesen/ViLMA)

### 8. (MMC-Benchmark) MMC: Advancing Multimodal Chart Understanding with Large-scale Instruction Tuning
**Date**: 2023.11.15

**Affiliation**: University of Maryland, College Park
<details span>
<summary><b>Abstract</b></summary>
With the rapid development of large language models (LLMs) and their
integration into large multimodal models (LMMs), there has been impressive
progress in zero-shot completion of user-oriented vision-language tasks.
However, a gap remains in the domain of chart image understanding due to the
distinct abstract components in charts. To address this, we introduce a
large-scale MultiModal Chart Instruction (\textbf{MMC-Instruction}) dataset
comprising 600k instances supporting diverse tasks and chart types. Leveraging
this data, we develop MultiModal Chart Assistant (\textbf{MMCA}), an LMM that
achieves state-of-the-art performance on existing chart QA benchmarks.
Recognizing the need for a comprehensive evaluation of LMM chart understanding,
we also propose a MultiModal Chart Benchmark (\textbf{MMC-Benchmark}), a
comprehensive human-annotated benchmark with nine distinct tasks evaluating
reasoning capabilities over charts. Extensive experiments on MMC-Benchmark
reveal the limitations of existing LMMs on correctly interpreting charts, even
for the most recent GPT-4V model. Our work provides an instruction-tuning
methodology and benchmark to advance multimodal understanding of charts. Code
and data are available at https://github.com/FuxiaoLiu/MMC.
</details>

[📄 Paper](https://arxiv.org/abs/2311.10774) | [💻 Code](https://github.com/FuxiaoLiu/MMC)

### 9. (ChartingNewTerritories) Charting New Territories: Exploring the Geographic and Geospatial Capabilities of Multimodal LLMs
**Date**: 2023.11.25

**Affiliation**: University of Cambridge
<details span>
<summary><b>Abstract</b></summary>
Multimodal large language models (MLLMs) have shown remarkable capabilities
across a broad range of tasks but their knowledge and abilities in the
geographic and geospatial domains are yet to be explored, despite potential
wide-ranging benefits to navigation, environmental research, urban development,
and disaster response. We conduct a series of experiments exploring various
vision capabilities of MLLMs within these domains, particularly focusing on the
frontier model GPT-4V, and benchmark its performance against open-source
counterparts. Our methodology involves challenging these models with a
small-scale geographic benchmark consisting of a suite of visual tasks, testing
their abilities across a spectrum of complexity. The analysis uncovers not only
where such models excel, including instances where they outperform humans, but
also where they falter, providing a balanced view of their capabilities in the
geographic domain. To enable the comparison and evaluation of future models,
our benchmark will be publicly released.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.14656) | [💻 Code](https://github.com/jonathan-roberts1/charting-new-territories)

### 10. (ChartBench) ChartBench: A Benchmark for Complex Visual Reasoning in Charts
**Date**: 2023.12.26

**Affiliation**: International Digital Economy Academy (IDEA)
<details span>
<summary><b>Abstract</b></summary>
Multimodal Large Language Models (MLLMs) have shown impressive capabilities
in image understanding and generation. However, current benchmarks fail to
accurately evaluate the chart comprehension of MLLMs due to limited chart types
and inappropriate metrics. To address this, we propose ChartBench, a
comprehensive benchmark designed to assess chart comprehension and data
reliability through complex visual reasoning. ChartBench includes 42
categories, 66.6k charts, and 600k question-answer pairs. Notably, many charts
lack data point annotations, which requires MLLMs to derive values similar to
human understanding by leveraging inherent chart elements such as color,
legends, and coordinate systems. We also design an enhanced evaluation metric,
Acc+, to evaluate MLLMs without extensive manual or costly LLM-based
evaluations. Furthermore, we propose two baselines based on the chain of
thought and supervised fine-tuning to improve model performance on unannotated
charts. Extensive experimental evaluations of 18 open-sourced and 3 proprietary
MLLMs reveal their limitations in chart comprehension and offer valuable
insights for further research. Code and dataset are publicly available at
https://chartbench.github.io.
</details>

[📄 Paper](https://arxiv.org/abs/2312.15915) | [🌐 Project Page](https://chartbench.github.io/) | [💻 Code](https://github.com/IDEA-FinAI/ChartBench)

### 11. (CMMMU) CMMMU: A Chinese Massive Multi-discipline Multimodal Understanding Benchmark
**Date**: 2024.01.22

**Affiliation**: Hong Kong University of Science and Technology
<details span>
<summary><b>Abstract</b></summary>
As the capabilities of large multimodal models (LMMs) continue to advance,
evaluating the performance of LMMs emerges as an increasing need. Additionally,
there is an even larger gap in evaluating the advanced knowledge and reasoning
abilities of LMMs in non-English contexts such as Chinese. We introduce CMMMU,
a new Chinese Massive Multi-discipline Multimodal Understanding benchmark
designed to evaluate LMMs on tasks demanding college-level subject knowledge
and deliberate reasoning in a Chinese context. CMMMU is inspired by and
strictly follows the annotation and analysis pattern of MMMU. CMMMU includes
12k manually collected multimodal questions from college exams, quizzes, and
textbooks, covering six core disciplines: Art & Design, Business, Science,
Health & Medicine, Humanities & Social Science, and Tech & Engineering, like
its companion, MMMU. These questions span 30 subjects and comprise 39 highly
heterogeneous image types, such as charts, diagrams, maps, tables, music
sheets, and chemical structures. CMMMU focuses on complex perception and
reasoning with domain-specific knowledge in the Chinese context. We evaluate 11
open-source LLMs and one proprietary GPT-4V(ision). Even GPT-4V only achieves
accuracies of 42%, indicating a large space for improvement. CMMMU will boost
the community to build the next-generation LMMs towards expert artificial
intelligence and promote the democratization of LMMs by providing diverse
language contexts.
</details>

[📄 Paper](https://arxiv.org/abs/2401.11944) | [🌐 Project Page](https://cmmmu-benchmark.github.io/) | [💻 Code](https://github.com/CMMMU-Benchmark/CMMMU)

### 12. (CMMU) CMMU: A Benchmark for Chinese Multi-modal Multi-type Question Understanding and Reasoning
**Date**: 2024.01.25

**Affiliation**: Beijing Academy of Artificial Intelligence
<details span>
<summary><b>Abstract</b></summary>
Multi-modal large language models(MLLMs) have achieved remarkable progress
and demonstrated powerful knowledge comprehension and reasoning abilities.
However, the mastery of domain-specific knowledge, which is essential for
evaluating the intelligence of MLLMs, continues to be a challenge. Current
multi-modal benchmarks for domain-specific knowledge concentrate on
multiple-choice questions and are predominantly available in English, which
imposes limitations on the comprehensiveness of the evaluation. To this end, we
introduce CMMU, a novel benchmark for multi-modal and multi-type question
understanding and reasoning in Chinese. CMMU consists of 3,603 questions in 7
subjects, covering knowledge from primary to high school. The questions can be
categorized into 3 types: multiple-choice, multiple-response, and
fill-in-the-blank, bringing greater challenges to MLLMs. In addition, we
propose an evaluation strategy called Positional Error Variance for assessing
multiple-choice questions. The strategy aims to perform a quantitative analysis
of position bias. We evaluate seven open-source MLLMs along with GPT4-V,
Gemini-Pro, and Qwen-VL-Plus. The results demonstrate that CMMU poses a
significant challenge to the recent MLLMs. The data and code are available at
https://github.com/FlagOpen/CMMU.
</details>

[📄 Paper](https://arxiv.org/abs/2401.14011) | [💻 Code](https://github.com/FlagOpen/CMMU)

### 13. (MULTI) MULTI: Multimodal Understanding Leaderboard with Text and Images
**Date**: 2024.02.03

**Affiliation**: Shanghai Jiao Tong University
<details span>
<summary><b>Abstract</b></summary>
Rapid progress in multimodal large language models (MLLMs) highlights the
need to introduce challenging yet realistic benchmarks to the academic
community, while existing benchmarks primarily focus on understanding simple
natural images and short context. In this paper, we present MULTI as a
cutting-edge benchmark for evaluating MLLMs on understanding complex tables and
images, and reasoning with long context. MULTI provides multimodal inputs and
requires responses that are either precise or open-ended, reflecting real-life
examination styles. MULTI includes over 18,000 questions and challenges MLLMs
with a variety of tasks, ranging from formula derivation to image detail
analysis and cross-modality reasoning. We also introduce MULTI-Elite, a
500-question selected hard subset, and MULTI-Extend, with more than 4,500
external knowledge context pieces. Our evaluation indicates significant
potential for MLLM advancement, with GPT-4V achieving a 63.7% accuracy rate on
MULTI, in contrast to other MLLMs scoring between 28.5% and 55.3%. MULTI serves
not only as a robust evaluation platform but also paves the way for the
development of expert-level AI.
</details>

[📄 Paper](https://arxiv.org/abs/2402.03173) | [🌐 Project Page](https://opendfm.github.io/MULTI-Benchmark/) | [💻 Code](https://github.com/OpenDFM/MULTI-Benchmark)

### 14. (SceMQA) SceMQA: A Scientific College Entrance Level Multimodal Question Answering Benchmark
**Date**: 2024.02.06

**Affiliation**: University of Notre Dam
<details span>
<summary><b>Abstract</b></summary>
The paper introduces SceMQA, a novel benchmark for scientific multimodal
question answering at the college entrance level. It addresses a critical
educational phase often overlooked in existing benchmarks, spanning high school
to pre-college levels. SceMQA focuses on core science subjects including
Mathematics, Physics, Chemistry, and Biology. It features a blend of
multiple-choice and free-response formats, ensuring a comprehensive evaluation
of AI models' abilities. Additionally, our benchmark provides specific
knowledge points for each problem and detailed explanations for each answer.
SceMQA also uniquely presents problems with identical contexts but varied
questions to facilitate a more thorough and accurate assessment of reasoning
capabilities. In the experiment, we evaluate both open-source and close-source
state-of-the-art Multimodal Large Language Models (MLLMs), across various
experimental settings. The results show that further research and development
are needed in developing more capable MLLM, as highlighted by only 50% to 60%
accuracy achieved by the strongest models. Our benchmark and analysis will be
available at https://scemqa.github.io/
</details>

[📄 Paper](https://arxiv.org/abs/2402.05138) | [🌐 Project Page](https://scemqa.github.io/) | [💻 Code](https://github.com/SceMQA/SceMQA)

### 15. (MIKE) MIKE: A New Benchmark for Fine-grained Multimodal Entity Knowledge Editing
**Date**: 2024.02.18

**Affiliation**: UCAS
<details span>
<summary><b>Abstract</b></summary>
Multimodal knowledge editing represents a critical advancement in enhancing
the capabilities of Multimodal Large Language Models (MLLMs). Despite its
potential, current benchmarks predominantly focus on coarse-grained knowledge,
leaving the intricacies of fine-grained (FG) multimodal entity knowledge
largely unexplored. This gap presents a notable challenge, as FG entity
recognition is pivotal for the practical deployment and effectiveness of MLLMs
in diverse real-world scenarios. To bridge this gap, we introduce MIKE, a
comprehensive benchmark and dataset specifically designed for the FG multimodal
entity knowledge editing. MIKE encompasses a suite of tasks tailored to assess
different perspectives, including Vanilla Name Answering, Entity-Level Caption,
and Complex-Scenario Recognition. In addition, a new form of knowledge editing,
Multi-step Editing, is introduced to evaluate the editing efficiency. Through
our extensive evaluations, we demonstrate that the current state-of-the-art
methods face significant challenges in tackling our proposed benchmark,
underscoring the complexity of FG knowledge editing in MLLMs. Our findings
spotlight the urgent need for novel approaches in this domain, setting a clear
agenda for future research and development efforts within the community.
</details>

[📄 Paper](https://arxiv.org/abs/2402.14835)

### 16. (ChartX) ChartX & ChartVLM: A Versatile Benchmark and Foundation Model for Complicated Chart Reasoning
**Date**: 2024.02.19

**Affiliation**: Shanghai Artificial Intelligence Laboratory
<details span>
<summary><b>Abstract</b></summary>
Recently, many versatile Multi-modal Large Language Models (MLLMs) have
emerged continuously. However, their capacity to query information depicted in
visual charts and engage in reasoning based on the queried contents remains
under-explored. In this paper, to comprehensively and rigorously benchmark the
ability of the off-the-shelf MLLMs in the chart domain, we construct ChartX, a
multi-modal evaluation set covering 18 chart types, 7 chart tasks, 22
disciplinary topics, and high-quality chart data. Besides, we develop ChartVLM
to offer a new perspective on handling multi-modal tasks that strongly depend
on interpretable patterns, such as reasoning tasks in the field of charts or
geometric images. We evaluate the chart-related ability of mainstream MLLMs and
our ChartVLM on the proposed ChartX evaluation set. Extensive experiments
demonstrate that ChartVLM surpasses both versatile and chart-related large
models, achieving results comparable to GPT-4V. We believe that our study can
pave the way for further exploration in creating a more comprehensive chart
evaluation set and developing more interpretable multi-modal models. Both
ChartX and ChartVLM are available at:
https://github.com/UniModal4Reasoning/ChartVLM
</details>

[📄 Paper](https://arxiv.org/abs/2402.12185) | [💻 Code](https://github.com/UniModal4Reasoning/ChartVLM)

### 17. (Math-Vision) Measuring Multimodal Mathematical Reasoning with MATH-Vision Dataset
**Date**: 2024.02.22

**Affiliation**: The Chinese University of Hong Kong
<details span>
<summary><b>Abstract</b></summary>
Recent advancements in Large Multimodal Models (LMMs) have shown promising
results in mathematical reasoning within visual contexts, with models
approaching human-level performance on existing benchmarks such as MathVista.
However, we observe significant limitations in the diversity of questions and
breadth of subjects covered by these benchmarks. To address this issue, we
present the MATH-Vision (MATH-V) dataset, a meticulously curated collection of
3,040 high-quality mathematical problems with visual contexts sourced from real
math competitions. Spanning 16 distinct mathematical disciplines and graded
across 5 levels of difficulty, our dataset provides a comprehensive and diverse
set of challenges for evaluating the mathematical reasoning abilities of LMMs.
Through extensive experimentation, we unveil a notable performance gap between
current LMMs and human performance on MATH-V, underscoring the imperative for
further advancements in LMMs. Moreover, our detailed categorization allows for
a thorough error analysis of LMMs, offering valuable insights to guide future
research and development. The project is available at
https://mathvision-cuhk.github.io
</details>

[📄 Paper](https://arxiv.org/abs/2402.14804) | [🌐 Project Page](https://mathvision-cuhk.github.io) | [💻 Code](https://github.com/mathvision-cuhk/MATH-V)

### 18. (CRPE) The All-Seeing Project V2: Towards General Relation Comprehension of the Open World
**Date**: 2024.02.29

**Affiliation**: Shanghai AI Laboratory
<details span>
<summary><b>Abstract</b></summary>
We present the All-Seeing Project V2: a new model and dataset designed for
understanding object relations in images. Specifically, we propose the
All-Seeing Model V2 (ASMv2) that integrates the formulation of text generation,
object localization, and relation comprehension into a relation conversation
(ReC) task. Leveraging this unified task, our model excels not only in
perceiving and recognizing all objects within the image but also in grasping
the intricate relation graph between them, diminishing the relation
hallucination often encountered by Multi-modal Large Language Models (MLLMs).
To facilitate training and evaluation of MLLMs in relation understanding, we
created the first high-quality ReC dataset ({AS-V2) which is aligned with the
format of standard instruction tuning data. In addition, we design a new
benchmark, termed Circular-based Relation Probing Evaluation (CRPE) for
comprehensively evaluating the relation comprehension capabilities of MLLMs.
Notably, our ASMv2 achieves an overall accuracy of 52.04 on this relation-aware
benchmark, surpassing the 43.14 of LLaVA-1.5 by a large margin. We hope that
our work can inspire more future research and contribute to the evolution
towards artificial general intelligence. Our project is released at
https://github.com/OpenGVLab/all-seeing.
</details>

[📄 Paper](https://arxiv.org/abs/2402.19474) | [💻 Code](https://github.com/OpenGVLab/all-seeing)

### 19. (NPHardEval4V) NPHardEval4V: A Dynamic Reasoning Benchmark of Multimodal Large Language Models
**Date**: 2024.03.04

**Affiliation**: University of Michigan
<details span>
<summary><b>Abstract</b></summary>
Understanding the reasoning capabilities of Multimodal Large Language Models
(MLLMs) is an important area of research. In this study, we introduce a dynamic
benchmark, NPHardEval4V, aimed at addressing the existing gaps in evaluating
the pure reasoning abilities of MLLMs. Our benchmark aims to provide a venue to
disentangle the effect of various factors such as image recognition and
instruction following, from the overall performance of the models, allowing us
to focus solely on evaluating their reasoning abilities. It is built by
converting textual description of questions from NPHardEval to image
representations. Our findings reveal significant discrepancies in reasoning
abilities across different models and highlight the relatively weak performance
of MLLMs compared to LLMs in terms of reasoning. We also investigate the impact
of different prompting styles, including visual, text, and combined visual and
text prompts, on the reasoning abilities of MLLMs, demonstrating the different
impacts of multimodal inputs in model performance. Unlike traditional
benchmarks, which focus primarily on static evaluations, our benchmark will be
updated monthly to prevent overfitting and ensure a more authentic and
fine-grained evaluation of the models. We believe that this benchmark can aid
in understanding and guide the further development of reasoning abilities in
MLLMs. The benchmark dataset and code are available at
https://github.com/lizhouf/NPHardEval4V
</details>

[📄 Paper](https://arxiv.org/abs/2403.01777) | [💻 Code](https://github.com/lizhouf/NPHardEval4V)

### 20. (VLKEB) VLKEB: A Large Vision-Language Model Knowledge Editing Benchmark
**Date**: 2024.03.12

**Affiliation**: University of Chinese Academy of Sciences
<details span>
<summary><b>Abstract</b></summary>
Recently, knowledge editing on large language models (LLMs) has received
considerable attention. Compared to this, editing Large Vision-Language Models
(LVLMs) faces extra challenges from diverse data modalities and complicated
model components, and data for LVLMs editing are limited. The existing LVLM
editing benchmark, which comprises three metrics (Reliability, Locality, and
Generality), falls short in the quality of synthesized evaluation images and
cannot assess whether models apply edited knowledge in relevant content.
Therefore, we employ more reliable data collection methods to construct a new
Large $\textbf{V}$ision-$\textbf{L}$anguage Model $\textbf{K}$nowledge
$\textbf{E}$diting $\textbf{B}$enchmark, $\textbf{VLKEB}$, and extend the
Portability metric for more comprehensive evaluation. Leveraging a multi-modal
knowledge graph, our image data are bound with knowledge entities. This can be
further used to extract entity-related knowledge, which constitutes the base of
editing data. We conduct experiments of different editing methods on five
LVLMs, and thoroughly analyze how do they impact the models. The results reveal
strengths and deficiencies of these methods and hopefully provide insights for
future research. The codes and dataset are available at:
$\href{https://github.com/VLKEB/VLKEB}{\text{https://github.com/VLKEB/VLKEB}}$.
</details>

[📄 Paper](https://arxiv.org/abs/2403.07350) | [💻 Code](https://github.com/VLKEB/VLKEB)

### 21. (MathVerse) MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?
**Date**: 2024.03.21

**Affiliation**: CUHK MMLab
<details span>
<summary><b>Abstract</b></summary>
The remarkable progress of Multi-modal Large Language Models (MLLMs) has
garnered unparalleled attention, due to their superior performance in visual
contexts. However, their capabilities in visual math problem-solving remain
insufficiently evaluated and understood. We investigate current benchmarks to
incorporate excessive visual content within textual questions, which
potentially assist MLLMs in deducing answers without truly interpreting the
input diagrams. To this end, we introduce MathVerse, an all-around visual math
benchmark designed for an equitable and in-depth evaluation of MLLMs. We
meticulously collect 2,612 high-quality, multi-subject math problems with
diagrams from publicly available sources. Each problem is then transformed by
human annotators into six distinct versions, each offering varying degrees of
information content in multi-modality, contributing to 15K test samples in
total. This approach allows MathVerse to comprehensively assess whether and how
much MLLMs can truly understand the visual diagrams for mathematical reasoning.
In addition, we propose a Chain-of-Thought (CoT) evaluation strategy for a
fine-grained assessment of the output answers. Rather than naively judging True
or False, we employ GPT-4(V) to adaptively extract crucial reasoning steps, and
then score each step with detailed error analysis, which can reveal the
intermediate CoT reasoning quality by MLLMs. We hope the MathVerse benchmark
may provide unique insights to guide the future development of MLLMs. Project
page: https://mathverse-cuhk.github.io
</details>

[📄 Paper](https://arxiv.org/abs/2403.14624v2) | [🌐 Project Page](https://mathverse-cuhk.github.io) | [💻 Code](https://github.com/ZrrSkywalker/MathVerse)

### 22. (Visual CoT) Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning
**Date**: 2024.03.25

**Affiliation**: The Chinese University of HongKong
<details span>
<summary><b>Abstract</b></summary>
Multi-Modal Large Language Models (MLLMs) have demonstrated impressive
performance in various VQA tasks. However, they often lack interpretability and
struggle with complex visual inputs, especially when the resolution of the
input image is high or when the interested region that could provide key
information for answering the question is small. To address these challenges,
we collect and introduce the large-scale Visual CoT dataset comprising 438k
question-answer pairs, annotated with intermediate bounding boxes highlighting
key regions essential for answering the questions. Additionally, about 98k
pairs of them are annotated with detailed reasoning steps. Importantly, we
propose a multi-turn processing pipeline that dynamically focuses on visual
inputs and provides interpretable thoughts. We also introduce the related
benchmark to evaluate the MLLMs in scenarios requiring specific local region
identification. Extensive experiments demonstrate the effectiveness of our
framework and shed light on better inference strategies. The Visual CoT
dataset, benchmark, and pre-trained models are released to foster further
research in this direction.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.16999) | [💻 Code](https://github.com/deepcs233/Visual-CoT)

### 23. (VisualWebBench) VisualWebBench: How Far Have Multimodal LLMs Evolved in Web Page Understanding and Grounding?
**Date**: 2024.04.09

**Affiliation**: Carnegie Mellon University
<details span>
<summary><b>Abstract</b></summary>
Multimodal Large Language models (MLLMs) have shown promise in web-related
tasks, but evaluating their performance in the web domain remains a challenge
due to the lack of comprehensive benchmarks. Existing benchmarks are either
designed for general multimodal tasks, failing to capture the unique
characteristics of web pages, or focus on end-to-end web agent tasks, unable to
measure fine-grained abilities such as OCR, understanding, and grounding. In
this paper, we introduce \bench{}, a multimodal benchmark designed to assess
the capabilities of MLLMs across a variety of web tasks. \bench{} consists of
seven tasks, and comprises 1.5K human-curated instances from 139 real websites,
covering 87 sub-domains. We evaluate 14 open-source MLLMs, Gemini Pro, Claude-3
series, and GPT-4V(ision) on \bench{}, revealing significant challenges and
performance gaps. Further analysis highlights the limitations of current MLLMs,
including inadequate grounding in text-rich environments and subpar performance
with low-resolution image inputs. We believe \bench{} will serve as a valuable
resource for the research community and contribute to the creation of more
powerful and versatile MLLMs for web-related applications.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.05955) | [🌐 Project Page](https://visualwebbench.github.io/) | [💻 Code](https://github.com/VisualWebBench/VisualWebBench)

### 24. (CFMM) Eyes Can Deceive: Benchmarking Counterfactual Reasoning Abilities of Multi-modal Large Language Models
**Date**: 2024.04.19

**Affiliation**: Fudan University
<details span>
<summary><b>Abstract</b></summary>
Counterfactual reasoning, as a crucial manifestation of human intelligence,
refers to making presuppositions based on established facts and extrapolating
potential outcomes. Existing multimodal large language models (MLLMs) have
exhibited impressive cognitive and reasoning capabilities, which have been
examined across a wide range of Visual Question Answering (VQA) benchmarks.
Nevertheless, how will existing MLLMs perform when faced with counterfactual
questions? To answer this question, we first curate a novel
\textbf{C}ounter\textbf{F}actual \textbf{M}ulti\textbf{M}odal reasoning
benchmark, abbreviated as \textbf{CFMM}, to systematically assess the
counterfactual reasoning capabilities of MLLMs. Our CFMM comprises six
challenging tasks, each including hundreds of carefully human-labeled and
GPT-generated counterfactual questions, to evaluate MLLM's counterfactual
reasoning capabilities across diverse aspects. Through experiments,
interestingly, we find that existing MLLMs prefer to believe what they see, but
ignore the counterfactual presuppositions presented in the question, thereby
leading to inaccurate responses. Furthermore, we evaluate a wide range of
prevalent MLLMs on our proposed CFMM. The significant gap between their
performance on our CFMM and that on several VQA benchmarks indicates that there
is still considerable room for improvement in existing MLLMs toward approaching
human-level intelligence. On the other hand, through boosting MLLMs
performances on our CFMM in the future, potential avenues toward developing
MLLMs with advanced intelligence can be explored.
</details>

[📄 Paper](https://arxiv.org/abs/2404.12966)

### 25. (TableVQA-Bench) TableVQA-Bench: A Visual Question Answering Benchmark on Multiple Table Domains
**Date**: 2024.04.19

**Affiliation**: NAVER Cloud AI
<details span>
<summary><b>Abstract</b></summary>
In this paper, we establish a benchmark for table visual question answering,
referred to as the TableVQA-Bench, derived from pre-existing table
question-answering (QA) and table structure recognition datasets. It is
important to note that existing datasets have not incorporated images or QA
pairs, which are two crucial components of TableVQA. As such, the primary
objective of this paper is to obtain these necessary components. Specifically,
images are sourced either through the application of a \textit{stylesheet} or
by employing the proposed table rendering system. QA pairs are generated by
exploiting the large language model (LLM) where the input is a text-formatted
table. Ultimately, the completed TableVQA-Bench comprises 1,500 QA pairs. We
comprehensively compare the performance of various multi-modal large language
models (MLLMs) on TableVQA-Bench. GPT-4V achieves the highest accuracy among
commercial and open-sourced MLLMs from our experiments. Moreover, we discover
that the number of vision queries plays a significant role in TableVQA
performance. To further analyze the capabilities of MLLMs in comparison to
their LLM backbones, we investigate by presenting image-formatted tables to
MLLMs and text-formatted tables to LLMs, respectively. Our findings suggest
that processing visual inputs is more challenging than text inputs, as
evidenced by the lower performance of MLLMs, despite generally requiring higher
computational costs than LLMs. The proposed TableVQA-Bench and evaluation codes
are available at
\href{https://github.com/naver-ai/tablevqabench}{https://github.com/naver-ai/tablevqabench}.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.19205) | [💻 Code](https://github.com/naver-ai/tablevqabench)

### 26. (MARVEL) MARVEL: Multidimensional Abstraction and Reasoning through Visual Evaluation and Learning
**Date**: 2024.04.21

**Affiliation**: University of Southern California
<details span>
<summary><b>Abstract</b></summary>
While multi-modal large language models (MLLMs) have shown significant
progress on many popular visual reasoning benchmarks, whether they possess
abstract visual reasoning abilities remains an open question. Similar to the
Sudoku puzzles, abstract visual reasoning (AVR) problems require finding
high-level patterns (e.g., repetition constraints) that control the input
shapes (e.g., digits) in a specific task configuration (e.g., matrix). However,
existing AVR benchmarks only considered a limited set of patterns (addition,
conjunction), input shapes (rectangle, square), and task configurations (3 by 3
matrices). To evaluate MLLMs' reasoning abilities comprehensively, we introduce
MARVEL, a multidimensional AVR benchmark with 770 puzzles composed of six core
knowledge patterns, geometric and abstract shapes, and five different task
configurations. To inspect whether the model accuracy is grounded in perception
and reasoning, MARVEL complements the general AVR question with perception
questions in a hierarchical evaluation framework. We conduct comprehensive
experiments on MARVEL with nine representative MLLMs in zero-shot and few-shot
settings. Our experiments reveal that all models show near-random performance
on the AVR question, with significant performance gaps (40%) compared to humans
across all patterns and task configurations. Further analysis of perception
questions reveals that MLLMs struggle to comprehend the visual features
(near-random performance) and even count the panels in the puzzle ( <45%),
hindering their ability for abstract reasoning. We release our entire code and
dataset.
</details>

[📄 Paper](https://arxiv.org/abs/2404.13591) | [💻 Code](https://github.com/1171-jpg/MARVEL_AVR)

### 27. (SOK-Bench) SOK-Bench: A Situated Video Reasoning Benchmark with Aligned Open-World Knowledge
**Date**: 2024.05.09

**Affiliation**: The University of Hong Kong
<details span>
<summary><b>Abstract</b></summary>
Learning commonsense reasoning from visual contexts and scenes in real-world
is a crucial step toward advanced artificial intelligence. However, existing
video reasoning benchmarks are still inadequate since they were mainly designed
for factual or situated reasoning and rarely involve broader knowledge in the
real world. Our work aims to delve deeper into reasoning evaluations,
specifically within dynamic, open-world, and structured context knowledge. We
propose a new benchmark (SOK-Bench), consisting of 44K questions and 10K
situations with instance-level annotations depicted in the videos. The
reasoning process is required to understand and apply situated knowledge and
general knowledge for problem-solving. To create such a dataset, we propose an
automatic and scalable generation method to generate question-answer pairs,
knowledge graphs, and rationales by instructing the combinations of LLMs and
MLLMs. Concretely, we first extract observable situated entities, relations,
and processes from videos for situated knowledge and then extend to open-world
knowledge beyond the visible content. The task generation is facilitated
through multiple dialogues as iterations and subsequently corrected and refined
by our designed self-promptings and demonstrations. With a corpus of both
explicit situated facts and implicit commonsense, we generate associated
question-answer pairs and reasoning processes, finally followed by manual
reviews for quality assurance. We evaluated recent mainstream large
vision-language models on the benchmark and found several insightful
conclusions. For more information, please refer to our benchmark at
www.bobbywu.com/SOKBench.
</details>

[📄 Paper](https://arxiv.org/pdf/2405.09713) | [🌐 Project Page](https://bobbywu.com/SOKBench/) | [💻 Code](https://github.com/csbobby/SOK-Bench)

### 28. (EvalQABench) LOVA3: Learning to Visual Question Answering, Asking and Assessment
**Date**: 2024.05.14

**Affiliation**: National University of Singapore
<details span>
<summary><b>Abstract</b></summary>
Question answering, asking, and assessment are three innate human traits
crucial for understanding the world and acquiring knowledge. By enhancing these
capabilities, humans can more effectively utilize data, leading to better
comprehension and learning outcomes. However, current Multimodal Large Language
Models (MLLMs) primarily focus on question answering, often neglecting the full
potential of questioning and assessment skills. In this study, we introduce
LOVA3, an innovative framework named ``Learning tO Visual Question Answering,
Asking and Assessment,'' designed to equip MLLMs with these additional
capabilities. Our approach involves the creation of two supplementary training
tasks GenQA and EvalQA, aiming at fostering the skills of asking and assessing
questions in the context of images. To develop the questioning ability, we
compile a comprehensive set of multimodal foundational tasks. For assessment,
we introduce a new benchmark called EvalQABench, comprising 64,000 training
samples (split evenly between positive and negative samples) and 5,000 testing
samples. We posit that enhancing MLLMs with the capabilities to answer, ask,
and assess questions will improve their multimodal comprehension and lead to
better performance. We validate our hypothesis by training an MLLM using the
LOVA3 framework and testing it on 10 multimodal benchmarks. The results
demonstrate consistent performance improvements, thereby confirming the
efficacy of our approach.
</details>

[📄 Paper](https://arxiv.org/pdf/2405.14974) | [💻 Code](https://github.com/showlab/LOVA3)

### 29. (SpatialRGPT-Bench) SpatialRGPT: Grounded Spatial Reasoning in Vision Language Model
**Date**: 2024.06.03

**Affiliation**: UC San Diego
<details span>
<summary><b>Abstract</b></summary>
Vision Language Models (VLMs) have demonstrated remarkable performance in 2D
vision and language tasks. However, their ability to reason about spatial
arrangements remains limited. In this work, we introduce Spatial Region GPT
(SpatialRGPT) to enhance VLMs' spatial perception and reasoning capabilities.
SpatialRGPT advances VLMs' spatial understanding through two key innovations:
(1) a data curation pipeline that enables effective learning of regional
representation from 3D scene graphs, and (2) a flexible plugin module for
integrating depth information into the visual encoder of existing VLMs. During
inference, when provided with user-specified region proposals, SpatialRGPT can
accurately perceive their relative directions and distances. Additionally, we
propose SpatialRGBT-Bench, a benchmark with ground-truth 3D annotations
encompassing indoor, outdoor, and simulated environments, for evaluating 3D
spatial cognition in VLMs. Our results demonstrate that SpatialRGPT
significantly enhances performance in spatial reasoning tasks, both with and
without local region prompts. The model also exhibits strong generalization
capabilities, effectively reasoning about complex spatial relations and
functioning as a region-aware dense reward annotator for robotic tasks. Code,
dataset, and benchmark will be released at
https://www.anjiecheng.me/SpatialRGPT
</details>

[📄 Paper](https://arxiv.org/abs/2406.01584) | [🌐 Project Page](https://www.anjiecheng.me/SpatialRGPT)

### 30. (MMTab) Multimodal Table Understanding
**Date**: 2024.06.08

**Affiliation**: Chinese Academy of Sciences
<details span>
<summary><b>Abstract</b></summary>
Although great progress has been made by previous table understanding methods
including recent approaches based on large language models (LLMs), they rely
heavily on the premise that given tables must be converted into a certain text
sequence (such as Markdown or HTML) to serve as model input. However, it is
difficult to access such high-quality textual table representations in some
real-world scenarios, and table images are much more accessible. Therefore, how
to directly understand tables using intuitive visual information is a crucial
and urgent challenge for developing more practical applications. In this paper,
we propose a new problem, multimodal table understanding, where the model needs
to generate correct responses to various table-related requests based on the
given table image. To facilitate both the model training and evaluation, we
construct a large-scale dataset named MMTab, which covers a wide spectrum of
table images, instructions and tasks. On this basis, we develop Table-LLaVA, a
generalist tabular multimodal large language model (MLLM), which significantly
outperforms recent open-source MLLM baselines on 23 benchmarks under held-in
and held-out settings. The code and data is available at this
https://github.com/SpursGoZmy/Table-LLaVA
</details>

[📄 Paper](https://arxiv.org/pdf/2406.08100) | [💻 Code](https://github.com/SpursGoZmy/Table-LLaVA)

### 31. (MMWorld) MMWorld: Towards Multi-discipline Multi-faceted World Model Evaluation in Videos
**Date**: 2024.06.08

**Affiliation**: UC Santa Cruz
<details span>
<summary><b>Abstract</b></summary>
Multimodal Language Language Models (MLLMs) demonstrate the emerging
abilities of "world models" -- interpreting and reasoning about complex
real-world dynamics. To assess these abilities, we posit videos are the ideal
medium, as they encapsulate rich representations of real-world dynamics and
causalities. To this end, we introduce MMWorld, a new benchmark for
multi-discipline, multi-faceted multimodal video understanding. MMWorld
distinguishes itself from previous video understanding benchmarks with two
unique advantages: (1) multi-discipline, covering various disciplines that
often require domain expertise for comprehensive understanding; (2)
multi-faceted reasoning, including explanation, counterfactual thinking, future
prediction, etc. MMWorld consists of a human-annotated dataset to evaluate
MLLMs with questions about the whole videos and a synthetic dataset to analyze
MLLMs within a single modality of perception. Together, MMWorld encompasses
1,910 videos across seven broad disciplines and 69 subdisciplines, complete
with 6,627 question-answer pairs and associated captions. The evaluation
includes 2 proprietary and 10 open-source MLLMs, which struggle on MMWorld
(e.g., GPT-4V performs the best with only 52.3\% accuracy), showing large room
for improvement. Further ablation studies reveal other interesting findings
such as models' different skill sets from humans. We hope MMWorld can serve as
an essential step towards world model evaluation in videos.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.08407) | [🌐 Project Page](https://mmworld-bench.github.io/) | [💻 Code](https://github.com/eric-ai-lab/MMWorld)

### 32. (MMRel) MMRel: A Relation Understanding Dataset and Benchmark in the MLLM Era
**Date**: 2024.06.09

**Affiliation**: Nanyang Technological University
<details span>
<summary><b>Abstract</b></summary>
Despite the recent advancements in Multi-modal Large Language Models (MLLMs),
understanding inter-object relations, i.e., interactions or associations
between distinct objects, remains a major challenge for such models. This issue
significantly hinders their advanced reasoning capabilities and is primarily
due to the lack of large-scale, high-quality, and diverse multi-modal data
essential for training and evaluating MLLMs. In this paper, we provide a
taxonomy of inter-object relations and introduce Multi-Modal Relation
Understanding (MMRel), a comprehensive dataset designed to bridge this gap by
providing large-scale, high-quality and diverse data for studying inter-object
relations with MLLMs. MMRel features three distinctive attributes: (i) It
includes over 15K question-answer pairs, which are sourced from three distinct
domains, ensuring large scale and high diversity; (ii) It contains a subset
featuring highly unusual relations, on which MLLMs often fail due to
hallucinations, thus are very challenging; (iii) It provides manually verified
high-quality labels for inter-object relations. Thanks to these features, MMRel
is ideal for evaluating MLLMs on relation understanding, as well as being used
to fine-tune MLLMs to enhance relation understanding and even benefit overall
performance in various vision-language tasks. Extensive experiments on various
popular MLLMs validate the effectiveness of MMRel. Both MMRel dataset and the
complete labeling scripts have been made publicly available.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.09121) | [💻 Code](https://github.com/niejiahao1998/MMRel)

### 33. (VCog-Bench) What is the Visual Cognition Gap between Humans and Multimodal LLMs?
**Date**: 2024.06.10

**Affiliation**: University of Illinois Urbana-Champaign
<details span>
<summary><b>Abstract</b></summary>
Recently, Multimodal Large Language Models (MLLMs) have shown great promise
in language-guided perceptual tasks such as recognition, segmentation, and
object detection. However, their effectiveness in addressing visual cognition
problems that require high-level reasoning is not well-established. One such
challenge is abstract visual reasoning (AVR) -- the cognitive ability to
discern relationships among patterns in a set of images and extrapolate to
predict subsequent patterns. This skill is crucial during the early
neurodevelopmental stages of children. Inspired by the AVR tasks in Raven's
Progressive Matrices (RPM) and Wechsler Intelligence Scale for Children (WISC),
we propose a new dataset MaRs-VQA and a new benchmark VCog-Bench containing
three datasets to evaluate the zero-shot AVR capability of MLLMs and compare
their performance with existing human intelligent investigation. Our
comparative experiments with different open-source and closed-source MLLMs on
the VCog-Bench revealed a gap between MLLMs and human intelligence,
highlighting the visual cognitive limitations of current MLLMs. We believe that
the public release of VCog-Bench, consisting of MaRs-VQA, and the inference
pipeline will drive progress toward the next generation of MLLMs with
human-like visual cognition abilities.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.10424) | [💻 Code](https://github.com/IrohXu/VCog-Bench)

### 34. (GSR-Bench) GSR-BENCH: A Benchmark for Grounded Spatial Reasoning Evaluation via Multimodal LLMs
**Date**: 2024.06.13

**Affiliation**: George Mason University
<details span>
<summary><b>Abstract</b></summary>
The ability to understand and reason about spatial relationships between
objects in images is an important component of visual reasoning. This skill
rests on the ability to recognize and localize objects of interest and
determine their spatial relation. Early vision and language models (VLMs) have
been shown to struggle to recognize spatial relations. We extend the previously
released What'sUp dataset and propose a novel comprehensive evaluation for
spatial relationship understanding that highlights the strengths and weaknesses
of 27 different models. In addition to the VLMs evaluated in What'sUp, our
extensive evaluation encompasses 3 classes of Multimodal LLMs (MLLMs) that vary
in their parameter sizes (ranging from 7B to 110B), training/instruction-tuning
methods, and visual resolution to benchmark their performances and scrutinize
the scaling laws in this task.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.13246)

### 35. (MC-MKE) MC-MKE: A Fine-Grained Multimodal Knowledge Editing Benchmark Emphasizing Modality Consistency
**Date**: 2024.06.13

**Affiliation**: Peking University
<details span>
<summary><b>Abstract</b></summary>
Multimodal large language models (MLLMs) are prone to non-factual or outdated
knowledge issues, which can manifest as misreading and misrecognition errors
due to the complexity of multimodal knowledge. Previous benchmarks have not
systematically analyzed the performance of editing methods in correcting these
two error types. To better represent and correct these errors, we decompose
multimodal knowledge into its visual and textual components. Different error
types correspond to different editing formats, which edits distinct part of the
multimodal knowledge. We present MC-MKE, a fine-grained Multimodal Knowledge
Editing benchmark emphasizing Modality Consistency. Our benchmark facilitates
independent correction of misreading and misrecognition errors by editing the
corresponding knowledge component. We evaluate three multimodal knowledge
editing methods on MC-MKE, revealing their limitations, particularly in terms
of modality consistency. Our work highlights the challenges posed by multimodal
knowledge editing and motivates further research in developing effective
techniques for this task.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.13219)

### 36. (MathV360K) Math-LLaVA: Bootstrapping Mathematical Reasoning for Multimodal Large Language Models
**Date**: 2024.06.17

**Affiliation**: University of Electronic Science and Technology of China
<details span>
<summary><b>Abstract</b></summary>
Large language models (LLMs) have demonstrated impressive reasoning
capabilities, particularly in textual mathematical problem-solving. However,
existing open-source image instruction fine-tuning datasets, containing limited
question-answer pairs per image, do not fully exploit visual information to
enhance the multimodal mathematical reasoning capabilities of Multimodal LLMs
(MLLMs). To bridge this gap, we address the lack of high-quality, diverse
multimodal mathematical datasets by collecting 40K high-quality images with
question-answer pairs from 24 existing datasets and synthesizing 320K new
pairs, creating the MathV360K dataset, which enhances both the breadth and
depth of multimodal mathematical questions. We introduce Math-LLaVA, a
LLaVA-1.5-based model fine-tuned with MathV360K. This novel approach
significantly improves the multimodal mathematical reasoning capabilities of
LLaVA-1.5, achieving a 19-point increase and comparable performance to GPT-4V
on MathVista's minitest split. Furthermore, Math-LLaVA demonstrates enhanced
generalizability, showing substantial improvements on the MMMU benchmark. Our
research highlights the importance of dataset diversity and synthesis in
advancing MLLMs' mathematical reasoning abilities. The code and data are
available at: \url{https://github.com/HZQ950419/Math-LLaVA}.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.17294) | [💻 Code](https://github.com/HZQ950419/Math-LLaVA)

### 37. (CharXiv) CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs
**Date**: 2024.06.18

**Affiliation**: Princeton Language and Intelligence (PLI), Princeton University
<details span>
<summary><b>Abstract</b></summary>
Chart understanding plays a pivotal role when applying Multimodal Large
Language Models (MLLMs) to real-world tasks such as analyzing scientific papers
or financial reports. However, existing datasets often focus on oversimplified
and homogeneous charts with template-based questions, leading to an
over-optimistic measure of progress. We demonstrate that although open-source
models can appear to outperform strong proprietary models on these benchmarks,
a simple stress test with slightly different charts or questions can
deteriorate performance by up to 34.5%. In this work, we propose CharXiv, a
comprehensive evaluation suite involving 2,323 natural, challenging, and
diverse charts from arXiv papers. CharXiv includes two types of questions: 1)
descriptive questions about examining basic chart elements and 2) reasoning
questions that require synthesizing information across complex visual elements
in the chart. To ensure quality, all charts and questions are handpicked,
curated, and verified by human experts. Our results reveal a substantial,
previously underestimated gap between the reasoning skills of the strongest
proprietary model (i.e., GPT-4o), which achieves 47.1% accuracy, and the
strongest open-source model (i.e., InternVL Chat V1.5), which achieves 29.2%.
All models lag far behind human performance of 80.5%, underscoring weaknesses
in the chart understanding capabilities of existing MLLMs. We hope CharXiv
facilitates future research on MLLM chart understanding by providing a more
realistic and faithful measure of progress. Project page and leaderboard:
https://charxiv.github.io/
</details>

[📄 Paper](https://arxiv.org/pdf/2406.18521) | [🌐 Project Page](https://charxiv.github.io/) | [💻 Code](https://github.com/princeton-nlp/CharXiv)

### 38. (REXTIME) ReXTime: A Benchmark Suite for Reasoning-Across-Time in Videos
**Date**: 2024.06.19

**Affiliation**: National Taiwan University
<details span>
<summary><b>Abstract</b></summary>
We introduce ReXTime, a benchmark designed to rigorously test AI models'
ability to perform temporal reasoning within video events. Specifically,
ReXTime focuses on reasoning across time, i.e. human-like understanding when
the question and its corresponding answer occur in different video segments.
This form of reasoning, requiring advanced understanding of cause-and-effect
relationships across video segments, poses significant challenges to even the
frontier multimodal large language models. To facilitate this evaluation, we
develop an automated pipeline for generating temporal reasoning question-answer
pairs, significantly reducing the need for labor-intensive manual annotations.
Our benchmark includes 921 carefully vetted validation samples and 2,143 test
samples, each manually curated for accuracy and relevance. Evaluation results
show that while frontier large language models outperform academic models, they
still lag behind human performance by a significant 14.3% accuracy gap.
Additionally, our pipeline creates a training dataset of 9,695 machine
generated samples without manual effort, which empirical studies suggest can
enhance the across-time reasoning via fine-tuning.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.19392) | [🌐 Project Page](https://rextime.github.io/) | [💻 Code](https://github.com/ReXTime/ReXTime)

### 39. (ScanReason) ScanReason: Empowering 3D Visual Grounding with Reasoning Capabilities
**Date**: 2024.07.01

**Affiliation**: The University of Hong Kong
<details span>
<summary><b>Abstract</b></summary>
Although great progress has been made in 3D visual grounding, current models
still rely on explicit textual descriptions for grounding and lack the ability
to reason human intentions from implicit instructions. We propose a new task
called 3D reasoning grounding and introduce a new benchmark ScanReason which
provides over 10K question-answer-location pairs from five reasoning types that
require the synerization of reasoning and grounding. We further design our
approach, ReGround3D, composed of the visual-centric reasoning module empowered
by Multi-modal Large Language Model (MLLM) and the 3D grounding module to
obtain accurate object locations by looking back to the enhanced geometry and
fine-grained details from the 3D scenes. A chain-of-grounding mechanism is
proposed to further boost the performance with interleaved reasoning and
grounding steps during inference. Extensive experiments on the proposed
benchmark validate the effectiveness of our proposed approach.
</details>

[📄 Paper](https://arxiv.org/abs/2407.01525) | [🌐 Project Page](https://zcmax.github.io/projects/ScanReason/) | [💻 Code](https://github.com/ZCMax/ScanReason)

### 40. (MindBench) MindBench: A Comprehensive Benchmark for Mind Map Structure Recognition and Analysis
**Date**: 2024.07.03

**Affiliation**: Meituan
<details span>
<summary><b>Abstract</b></summary>
Multimodal Large Language Models (MLLM) have made significant progress in the
field of document analysis. Despite this, existing benchmarks typically focus
only on extracting text and simple layout information, neglecting the complex
interactions between elements in structured documents such as mind maps and
flowcharts. To address this issue, we introduce the new benchmark named
MindBench, which not only includes meticulously constructed bilingual authentic
or synthetic images, detailed annotations, evaluation metrics and baseline
models, but also specifically designs five types of structured understanding
and parsing tasks. These tasks include full parsing, partial parsing,
position-related parsing, structured Visual Question Answering (VQA), and
position-related VQA, covering key areas such as text recognition, spatial
awareness, relationship discernment, and structured parsing. Extensive
experimental results demonstrate the substantial potential and significant room
for improvement in current models' ability to handle structured document
information. We anticipate that the launch of MindBench will significantly
advance research and application development in structured document analysis
technology. MindBench is available at:
https://miasanlei.github.io/MindBench.github.io/.
</details>

[📄 Paper](https://arxiv.org/abs/2407.02842) | [🌐 Project Page](https://miasanlei.github.io/MindBench.github.io/) | [💻 Code](https://github.com/MiaSanLei/MindBench)

### 41. (MMSci) MMSci: A Multimodal Multi-Discipline Dataset for PhD-Level Scientific Comprehension
**Date**: 2024.07.04

**Affiliation**: University of California
<details span>
<summary><b>Abstract</b></summary>
The rapid advancement of Large Language Models (LLMs) and Large Multimodal
Models (LMMs) has heightened the demand for AI-based scientific assistants
capable of understanding scientific articles and figures. Despite progress,
there remains a significant gap in evaluating models' comprehension of
professional, graduate-level, and even PhD-level scientific content. Current
datasets and benchmarks primarily focus on relatively simple scientific tasks
and figures, lacking comprehensive assessments across diverse advanced
scientific disciplines. To bridge this gap, we collected a multimodal,
multidisciplinary dataset from open-access scientific articles published in
Nature Communications journals. This dataset spans 72 scientific disciplines,
ensuring both diversity and quality. We created benchmarks with various tasks
and settings to comprehensively evaluate LMMs' capabilities in understanding
scientific figures and content. Our evaluation revealed that these tasks are
highly challenging: many open-source models struggled significantly, and even
GPT-4V and GPT-4o faced difficulties. We also explored using our dataset as
training resources by constructing visual instruction-following data, enabling
the 7B LLaVA model to achieve performance comparable to GPT-4V/o on our
benchmark. Additionally, we investigated the use of our interleaved article
texts and figure images for pre-training LMMs, resulting in improvements on the
material generation task. The source dataset, including articles, figures,
constructed benchmarks, and visual instruction-following data, is open-sourced.
</details>

[📄 Paper](https://arxiv.org/pdf/2407.04903) | [💻 Code](https://github.com/Leezekun/MMSci)

### 42. (LogicVista) LogicVista: Multimodal LLM Logical Reasoning Benchmark in Visual Contexts
**Date**: 2024.07.04

**Affiliation**: University of California
<details span>
<summary><b>Abstract</b></summary>
We propose LogicVista, an evaluation benchmark that assesses the integrated
logical reasoning capabilities of multimodal large language models (MLLMs) in
Visual contexts. Recent advancements in MLLMs have demonstrated various
fascinating abilities, from crafting poetry based on an image to performing
mathematical reasoning. However, there is still a lack of systematic evaluation
of MLLMs' proficiency in logical reasoning tasks, which are essential for
activities like navigation and puzzle-solving. Thus we evaluate general logical
cognition abilities across 5 logical reasoning tasks encompassing 9 different
capabilities, using a sample of 448 multiple-choice questions. Each question is
annotated with the correct answer and the human-written reasoning behind the
selection, enabling both open-ended and multiple-choice evaluation. A total of
8 MLLMs are comprehensively evaluated using LogicVista. Code and Data Available
at https://github.com/Yijia-Xiao/LogicVista.
</details>

[📄 Paper](https://arxiv.org/html/2407.04973v1) | [💻 Code](https://github.com/Yijia-Xiao/LogicVista)

### 43. (VideoCoT) VideoCoT: A Video Chain-of-Thought Dataset with Active Annotation Tool
**Date**: 2024.07.05

**Affiliation**: South China University of Technology
<details span>
<summary><b>Abstract</b></summary>
Multimodal large language models (MLLMs) are flourishing, but mainly focus on
images with less attention than videos, especially in sub-fields such as prompt
engineering, video chain-of-thought (CoT), and instruction tuning on videos.
Therefore, we try to explore the collection of CoT datasets in videos to lead
to video OpenQA and improve the reasoning ability of MLLMs. Unfortunately,
making such video CoT datasets is not an easy task. Given that human annotation
is too cumbersome and expensive, while machine-generated is not reliable due to
the hallucination issue, we develop an automatic annotation tool that combines
machine and human experts, under the active learning paradigm. Active learning
is an interactive strategy between the model and human experts, in this way,
the workload of human labeling can be reduced and the quality of the dataset
can be guaranteed. With the help of the automatic annotation tool, we strive to
contribute three datasets, namely VideoCoT, TopicQA, TopicCoT. Furthermore, we
propose a simple but effective benchmark based on the collected datasets, which
exploits CoT to maximize the complex reasoning capabilities of MLLMs. Extensive
experiments demonstrate the effectiveness our solution.
</details>

[📄 Paper](https://arxiv.org/pdf/2407.05355) 

### 44. (SPIQA) SPIQA: A Dataset for Multimodal Question Answering on Scientific Papers
**Date**: 2024.07.09

**Affiliation**: Google Research
<details span>
<summary><b>Abstract</b></summary>
Seeking answers to questions within long scientific research articles is a
crucial area of study that aids readers in quickly addressing their inquiries.
However, existing question-answering (QA) datasets based on scientific papers
are limited in scale and focus solely on textual content. To address this
limitation, we introduce SPIQA (Scientific Paper Image Question Answering), the
first large-scale QA dataset specifically designed to interpret complex figures
and tables within the context of scientific research articles across various
domains of computer science. Leveraging the breadth of expertise and ability of
multimodal large language models (MLLMs) to understand figures, we employ
automatic and manual curation to create the dataset. We craft an
information-seeking task involving multiple images that cover a wide variety of
plots, charts, tables, schematic diagrams, and result visualizations. SPIQA
comprises 270K questions divided into training, validation, and three different
evaluation splits. Through extensive experiments with 12 prominent foundational
models, we evaluate the ability of current multimodal systems to comprehend the
nuanced aspects of research articles. Additionally, we propose a
Chain-of-Thought (CoT) evaluation strategy with in-context retrieval that
allows fine-grained, step-by-step assessment and improves model performance. We
further explore the upper bounds of performance enhancement with additional
textual information, highlighting its promising potential for future research
and the dataset's impact on revolutionizing how we interact with scientific
literature.
</details>

[📄 Paper](https://arxiv.org/pdf/2407.09413) | [💻 Code](https://github.com/google/spiqa)

### 45. (MATHCHECK-GEO) Is Your Model Really A Good Math Reasoner? Evaluating Mathematical Reasoning with Checklist
**Date**: 2024.07.11

**Affiliation**: Xi’an Jiaotong-liverpool University
<details span>
<summary><b>Abstract</b></summary>
Exceptional mathematical reasoning ability is one of the key features that
demonstrate the power of large language models (LLMs). How to comprehensively
define and evaluate the mathematical abilities of LLMs, and even reflect the
user experience in real-world scenarios, has emerged as a critical issue.
Current benchmarks predominantly concentrate on problem-solving capabilities,
which presents a substantial risk of model overfitting and fails to accurately
represent genuine mathematical reasoning abilities. In this paper, we argue
that if a model really understands a problem, it should be robustly and readily
applied across a diverse array of tasks. Motivated by this, we introduce
MATHCHECK, a well-designed checklist for testing task generalization and
reasoning robustness, as well as an automatic tool to generate checklists
efficiently. MATHCHECK includes multiple mathematical reasoning tasks and
robustness test types to facilitate a comprehensive evaluation of both
mathematical reasoning ability and behavior testing. Utilizing MATHCHECK, we
develop MATHCHECK-GSM and MATHCHECK-GEO to assess mathematical textual
reasoning and multi-modal reasoning capabilities, respectively, serving as
upgraded versions of benchmarks including GSM8k, GeoQA, UniGeo, and Geometry3K.
We adopt MATHCHECK-GSM and MATHCHECK-GEO to evaluate over 20 LLMs and 11 MLLMs,
assessing their comprehensive mathematical reasoning abilities. Our results
demonstrate that while frontier LLMs like GPT-4o continue to excel in various
abilities on the checklist, many other model families exhibit a significant
decline. Further experiments indicate that, compared to traditional math
benchmarks, MATHCHECK better reflects true mathematical abilities and
represents mathematical intelligence more linearly, thereby supporting our
design. On our MATHCHECK, we can easily conduct detailed behavior analysis to
deeply investigate models.
</details>

[📄 Paper](https://arxiv.org/abs/2407.08733) | [🌐 Project Page](https://mathcheck.github.io/) | [💻 Code](https://github.com/PremiLab-Math/MathCheck)

### 46. (CHOPINLLM) On Pre-training of Multimodal Language Models Customized for Chart Understanding
**Date**: 2024.07.20

**Affiliation**: University of British Columbia
<details span>
<summary><b>Abstract</b></summary>
Recent studies customizing Multimodal Large Language Models (MLLMs) for
domain-specific tasks have yielded promising results, especially in the field
of scientific chart comprehension. These studies generally utilize visual
instruction tuning with specialized datasets to enhance question and answer
(QA) accuracy within the chart domain. However, they often neglect the
fundamental discrepancy between natural image-caption pre-training data and
digital chart image-QA data, particularly in the models' capacity to extract
underlying numeric values from charts. This paper tackles this oversight by
exploring the training processes necessary to improve MLLMs' comprehension of
charts. We present three key findings: (1) Incorporating raw data values in
alignment pre-training markedly improves comprehension of chart data. (2)
Replacing images with their textual representation randomly during end-to-end
fine-tuning transfer the language reasoning capability to chart interpretation
skills. (3) Requiring the model to first extract the underlying chart data and
then answer the question in the fine-tuning can further improve the accuracy.
Consequently, we introduce CHOPINLLM, an MLLM tailored for in-depth chart
comprehension. CHOPINLLM effectively interprets various types of charts,
including unannotated ones, while maintaining robust reasoning abilities.
Furthermore, we establish a new benchmark to evaluate MLLMs' understanding of
different chart types across various comprehension levels. Experimental results
show that CHOPINLLM exhibits strong performance in understanding both annotated
and unannotated charts across a wide range of types.
</details>

[📄 Paper](https://arxiv.org/pdf/2407.14506)

### 47. (CompBench) CompBench: A Comparative Reasoning Benchmark for Multimodal LLMs
**Date**: 2024.07.23

**Affiliation**: The Ohio State University
<details span>
<summary><b>Abstract</b></summary>
The ability to compare objects, scenes, or situations is crucial for
effective decision-making and problem-solving in everyday life. For instance,
comparing the freshness of apples enables better choices during grocery
shopping, while comparing sofa designs helps optimize the aesthetics of our
living space. Despite its significance, the comparative capability is largely
unexplored in artificial general intelligence (AGI). In this paper, we
introduce CompBench, a benchmark designed to evaluate the comparative reasoning
capability of multimodal large language models (MLLMs). CompBench mines and
pairs images through visually oriented questions covering eight dimensions of
relative comparison: visual attribute, existence, state, emotion, temporality,
spatiality, quantity, and quality. We curate a collection of around 40K image
pairs using metadata from diverse vision datasets and CLIP similarity scores.
These image pairs span a broad array of visual domains, including animals,
fashion, sports, and both outdoor and indoor scenes. The questions are
carefully crafted to discern relative characteristics between two images and
are labeled by human annotators for accuracy and relevance. We use CompBench to
evaluate recent MLLMs, including GPT-4V(ision), Gemini-Pro, and LLaVA-1.6. Our
results reveal notable shortcomings in their comparative abilities. We believe
CompBench not only sheds light on these limitations but also establishes a
solid foundation for future enhancements in the comparative capability of
MLLMs.
</details>

[📄 Paper](https://arxiv.org/pdf/2407.16837) | [🌐 Project Page](https://compbench.github.io/) | [💻 Code](https://github.com/RaptorMai/CompBench)



## Generation Benchmarks:

### 1. (LLaVA-Bench) Visual Instruction Tuning
**Date**: 2023.04.17

**Affiliation**: University of Wisconsin–Madison
<details span>
<summary><b>Abstract</b></summary>
Instruction tuning large language models (LLMs) using machine-generated
instruction-following data has improved zero-shot capabilities on new tasks,
but the idea is less explored in the multimodal field. In this paper, we
present the first attempt to use language-only GPT-4 to generate multimodal
language-image instruction-following data. By instruction tuning on such
generated data, we introduce LLaVA: Large Language and Vision Assistant, an
end-to-end trained large multimodal model that connects a vision encoder and
LLM for general-purpose visual and language understanding.Our early experiments
show that LLaVA demonstrates impressive multimodel chat abilities, sometimes
exhibiting the behaviors of multimodal GPT-4 on unseen images/instructions, and
yields a 85.1% relative score compared with GPT-4 on a synthetic multimodal
instruction-following dataset. When fine-tuned on Science QA, the synergy of
LLaVA and GPT-4 achieves a new state-of-the-art accuracy of 92.53%. We make
GPT-4 generated visual instruction tuning data, our model and code base
publicly available.
</details>

[📄 Paper](https://arxiv.org/abs/2304.08485) | [🌐 Project Page](https://llava-vl.github.io/) | [💻 Code](https://github.com/haotian-liu/LLaVA)
 
### 2. (POPE) Evaluating Object Hallucination in Large Vision-Language Models
**Date**: 2023.05.10

**Affiliation**: Renmin University of China
<details span>
<summary><b>Abstract</b></summary>
Inspired by the superior language abilities of large language models (LLM),
large vision-language models (LVLM) have been recently explored by integrating
powerful LLMs for improving the performance on complex multimodal tasks.
Despite the promising progress on LVLMs, we find that LVLMs suffer from the
hallucination problem, i.e. they tend to generate objects that are inconsistent
with the target images in the descriptions. To investigate it, this work
presents the first systematic study on object hallucination of LVLMs. We
conduct the evaluation experiments on several representative LVLMs, and show
that they mostly suffer from severe object hallucination issue. We further
discuss that the visual instructions may influence the hallucination, and find
that: objects that frequently occur in the visual instructions or co-occur with
the image objects, are obviously prone to be hallucinated by LVLMs. Besides, we
find that existing evaluation methods might be affected by the input
instructions and generation styles of LVLMs. Thus, we further design an
improved evaluation method for object hallucination by proposing a
polling-based query method called POPE. Experiment results demonstrate that our
POPE can evaluate the object hallucination in a more stable and flexible way.
Our codes and data are publicly available at https://github.com/RUCAIBox/POPE.
</details>

[📄 Paper](https://arxiv.org/pdf/2305.10355) | [💻 Code](https://github.com/RUCAIBox/POPE)

### 3. (LRV-Instruction) Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning
**Date**: 2023.06.14

**Affiliation**: University of Maryland
<details span>
<summary><b>Abstract</b></summary>
Despite the promising progress in multi-modal tasks, current large
multi-modal models (LMMs) are prone to hallucinating inconsistent descriptions
with respect to the associated image and human instructions. This paper
addresses this issue by introducing the first large and diverse visual
instruction tuning dataset, named Large-scale Robust Visual (LRV)-Instruction.
Our dataset comprises 400k visual instructions generated by GPT4, covering 16
vision-and-language tasks with open-ended instructions and answers. Unlike
existing studies that primarily focus on positive instruction samples, we
design LRV-Instruction to include both positive and negative instructions for
more robust visual instruction tuning. Our negative instructions are designed
at three semantic levels: (i) Nonexistent Object Manipulation, (ii) Existent
Object Manipulation and (iii) Knowledge Manipulation. To efficiently measure
the hallucination generated by LMMs, we propose GPT4-Assisted Visual
Instruction Evaluation (GAVIE), a stable approach to evaluate visual
instruction tuning like human experts. GAVIE does not require human-annotated
groundtruth answers and can adapt to diverse instruction formats. We conduct
comprehensive experiments to investigate the hallucination of LMMs. Our results
demonstrate existing LMMs exhibit significant hallucinations when presented
with our negative instructions, particularly Existent Object and Knowledge
Manipulation instructions. Moreover, we successfully mitigate hallucination by
finetuning MiniGPT4 and mPLUG-Owl on LRV-Instruction while improving
performance on several public datasets compared to state-of-the-art methods.
Additionally, we observed that a balanced ratio of positive and negative
instances in the training data leads to a more robust model. Code and data are
available at https://github.com/FuxiaoLiu/LRV-Instruction.
</details>

[📄 Paper](https://arxiv.org/abs/2306.14565) | [💻 Code](https://github.com/FuxiaoLiu/LRV-Instruction)

### 4. (StorySalon) Intelligent Grimm -- Open-ended Visual Storytelling via Latent Diffusion Models
**Date**: 2023.07.01

**Affiliation**: Shanghai Jiao Tong University
<details span>
<summary><b>Abstract</b></summary>
Generative models have recently exhibited exceptional capabilities in
text-to-image generation, but still struggle to generate image sequences
coherently. In this work, we focus on a novel, yet challenging task of
generating a coherent image sequence based on a given storyline, denoted as
open-ended visual storytelling. We make the following three contributions: (i)
to fulfill the task of visual storytelling, we propose a learning-based
auto-regressive image generation model, termed as StoryGen, with a novel
vision-language context module, that enables to generate the current frame by
conditioning on the corresponding text prompt and preceding image-caption
pairs; (ii) to address the data shortage of visual storytelling, we collect
paired image-text sequences by sourcing from online videos and open-source
E-books, establishing processing pipeline for constructing a large-scale
dataset with diverse characters, storylines, and artistic styles, named
StorySalon; (iii) Quantitative experiments and human evaluations have validated
the superiority of our StoryGen, where we show StoryGen can generalize to
unseen characters without any optimization, and generate image sequences with
coherent content and consistent character. Code, dataset, and models are
available at https://haoningwu3639.github.io/StoryGen_Webpage/
</details>

[📄 Paper](https://arxiv.org/abs/2306.00973) | [🌐 Project Page](https://haoningwu3639.github.io/StoryGen_Webpage) | [💻 Code](https://github.com/haoningwu3639/StoryGen)

### 5. (M-HalDetect) Detecting and Preventing Hallucinations in Large Vision Language Models
**Date**: 2023.08.06

**Affiliation**: Scale AI
<details span>
<summary><b>Abstract</b></summary>
Instruction tuned Large Vision Language Models (LVLMs) have significantly
advanced in generalizing across a diverse set of multi-modal tasks, especially
for Visual Question Answering (VQA). However, generating detailed responses
that are visually grounded is still a challenging task for these models. We
find that even the current state-of-the-art LVLMs (InstructBLIP) still contain
a staggering 30 percent of the hallucinatory text in the form of non-existent
objects, unfaithful descriptions, and inaccurate relationships. To address
this, we introduce M-HalDetect, a (M)ultimodal (Hal)lucination (Detect)ion
Dataset that can be used to train and benchmark models for hallucination
detection and prevention. M-HalDetect consists of 16k fine-grained annotations
on VQA examples, making it the first comprehensive multi-modal hallucination
detection dataset for detailed image descriptions. Unlike previous work that
only consider object hallucination, we additionally annotate both entity
descriptions and relationships that are unfaithful. To demonstrate the
potential of this dataset for hallucination prevention, we optimize
InstructBLIP through our novel Fine-grained Direct Preference Optimization
(FDPO). We also train fine-grained multi-modal reward models from InstructBLIP
and evaluate their effectiveness with best-of-n rejection sampling. We perform
human evaluation on both FDPO and rejection sampling, and find that they reduce
hallucination rates in InstructBLIP by 41% and 55% respectively. We also find
that our reward model generalizes to other multi-modal models, reducing
hallucinations in LLaVA and mPLUG-OWL by 15% and 57% respectively, and has
strong correlation with human evaluated accuracy scores.
</details>

[📄 Paper](https://arxiv.org/abs/2308.06394) | [💻 Code](https://github.com/hendryx-scale/mhal-detect)

### 6. (DEMON) Fine-tuning Multimodal LLMs to Follow Zero-shot Demonstrative Instructions
**Date**: 2023.08.08

**Affiliation**: Zhejiang University
<details span>
<summary><b>Abstract</b></summary>
Recent advancements in Multimodal Large Language Models (MLLMs) have been
utilizing Visual Prompt Generators (VPGs) to convert visual features into
tokens that LLMs can recognize. This is achieved by training the VPGs on
millions of image-caption pairs, where the VPG-generated tokens of images are
fed into a frozen LLM to generate the corresponding captions. However, this
image-captioning based training objective inherently biases the VPG to
concentrate solely on the primary visual contents sufficient for caption
generation, often neglecting other visual details. This shortcoming results in
MLLMs' underperformance in comprehending demonstrative instructions consisting
of multiple, interleaved, and multimodal instructions that demonstrate the
required context to complete a task. To address this issue, we introduce a
generic and lightweight Visual Prompt Generator Complete module (VPG-C), which
can infer and complete the missing details essential for comprehending
demonstrative instructions. Further, we propose a synthetic discriminative
training strategy to fine-tune VPG-C, eliminating the need for supervised
demonstrative instructions. As for evaluation, we build DEMON, a comprehensive
benchmark for demonstrative instruction understanding. Synthetically trained
with the proposed strategy, VPG-C achieves significantly stronger zero-shot
performance across all tasks of DEMON. Further evaluation on the MME and
OwlEval benchmarks also demonstrate the superiority of VPG-C. Our benchmark,
code, and pre-trained models are available at
https://github.com/DCDmllm/Cheetah.
</details>

[📄 Paper](https://arxiv.org/pdf/2308.04152) | [💻 Code](https://github.com/DCDmllm/Cheetah)

### 7. (VisIT-Bench) VisIT-Bench: A Benchmark for Vision-Language Instruction Following Inspired by Real-World Use
**Date**: 2023.08.12

**Affiliation**: Hebrew University
<details span>
<summary><b>Abstract</b></summary>
We introduce VisIT-Bench (Visual InsTruction Benchmark), a benchmark for
evaluation of instruction-following vision-language models for real-world use.
Our starting point is curating 70 'instruction families' that we envision
instruction tuned vision-language models should be able to address. Extending
beyond evaluations like VQAv2 and COCO, tasks range from basic recognition to
game playing and creative generation. Following curation, our dataset comprises
592 test queries, each with a human-authored instruction-conditioned caption.
These descriptions surface instruction-specific factors, e.g., for an
instruction asking about the accessibility of a storefront for wheelchair
users, the instruction-conditioned caption describes ramps/potential obstacles.
These descriptions enable 1) collecting human-verified reference outputs for
each instance; and 2) automatic evaluation of candidate multimodal generations
using a text-only LLM, aligning with human judgment. We quantify quality gaps
between models and references using both human and automatic evaluations; e.g.,
the top-performing instruction-following model wins against the GPT-4 reference
in just 27% of the comparison. VisIT-Bench is dynamic to participate,
practitioners simply submit their model's response on the project website;
Data, code and leaderboard is available at visit-bench.github.io.
</details>

[📄 Paper](https://arxiv.org/pdf/2308.06595) | [🌐 Project Page](visit-bench.github.io) | [💻 Code](https://github.com/mlfoundations/VisIT-Bench/)

### 8. (HaELM) Evaluation and Analysis of Hallucination in Large Vision-Language Models
**Date**: 2023.08.29

**Affiliation**: Beijing Jiaotong University
<details span>
<summary><b>Abstract</b></summary>
Large Vision-Language Models (LVLMs) have recently achieved remarkable
success. However, LVLMs are still plagued by the hallucination problem, which
limits the practicality in many scenarios. Hallucination refers to the
information of LVLMs' responses that does not exist in the visual input, which
poses potential risks of substantial consequences. There has been limited work
studying hallucination evaluation in LVLMs. In this paper, we propose
Hallucination Evaluation based on Large Language Models (HaELM), an LLM-based
hallucination evaluation framework. HaELM achieves an approximate 95%
performance comparable to ChatGPT and has additional advantages including low
cost, reproducibility, privacy preservation and local deployment. Leveraging
the HaELM, we evaluate the hallucination in current LVLMs. Furthermore, we
analyze the factors contributing to hallucination in LVLMs and offer helpful
suggestions to mitigate the hallucination problem. Our training data and human
annotation hallucination data will be made public soon.
</details>

[📄 Paper](https://arxiv.org/pdf/2308.15126) | [💻 Code](https://github.com/junyangwang0410/HaELM)

### 9. (MMHAL-BENCH) Aligning Large Multimodal Models with Factually Augmented RLHF
**Date**: 2023.09.25

**Affiliation**: UCBerkeley
<details span>
<summary><b>Abstract</b></summary>
Large Multimodal Models (LMM) are built across modalities and the
misalignment between two modalities can result in "hallucination", generating
textual outputs that are not grounded by the multimodal information in context.
To address the multimodal misalignment issue, we adapt the Reinforcement
Learning from Human Feedback (RLHF) from the text domain to the task of
vision-language alignment, where human annotators are asked to compare two
responses and pinpoint the more hallucinated one, and the vision-language model
is trained to maximize the simulated human rewards. We propose a new alignment
algorithm called Factually Augmented RLHF that augments the reward model with
additional factual information such as image captions and ground-truth
multi-choice options, which alleviates the reward hacking phenomenon in RLHF
and further improves the performance. We also enhance the GPT-4-generated
training data (for vision instruction tuning) with previously available
human-written image-text pairs to improve the general capabilities of our
model. To evaluate the proposed approach in real-world scenarios, we develop a
new evaluation benchmark MMHAL-BENCH with a special focus on penalizing
hallucinations. As the first LMM trained with RLHF, our approach achieves
remarkable improvement on the LLaVA-Bench dataset with the 94% performance
level of the text-only GPT-4 (while previous best methods can only achieve the
87% level), and an improvement by 60% on MMHAL-BENCH over other baselines. We
opensource our code, model, data at https://llava-rlhf.github.io.
</details>

[📄 Paper](https://arxiv.org/abs/2309.14525) | [🌐 Project Page](https://llava-rlhf.github.io) | [💻 Code](https://github.com/llava-rlhf/LLaVA-RLHF)

### 10. (NOPE) Negative Object Presence Evaluation (NOPE) to Measure Object Hallucination in Vision-Language Models
**Date**: 2023.10.09

**Affiliation**: Hong Kong University of Science and Technology
<details span>
<summary><b>Abstract</b></summary>
Object hallucination poses a significant challenge in vision-language (VL)
models, often leading to the generation of nonsensical or unfaithful responses
with non-existent objects. However, the absence of a general measurement for
evaluating object hallucination in VL models has hindered our understanding and
ability to mitigate this issue. In this work, we present NOPE (Negative Object
Presence Evaluation), a novel benchmark designed to assess object hallucination
in VL models through visual question answering (VQA). We propose a
cost-effective and scalable approach utilizing large language models to
generate 29.5k synthetic negative pronoun (NegP) data of high quality for NOPE.
We extensively investigate the performance of 10 state-of-the-art VL models in
discerning the non-existence of objects in visual questions, where the ground
truth answers are denoted as NegP (e.g., "none"). Additionally, we evaluate
their standard performance on visual questions on 9 other VQA datasets. Through
our experiments, we demonstrate that no VL model is immune to the vulnerability
of object hallucination, as all models achieve accuracy below 10\% on NegP.
Furthermore, we uncover that lexically diverse visual questions, question types
with large scopes, and scene-relevant objects capitalize the risk of object
hallucination in VL models.
</details>

[📄 Paper](https://arxiv.org/abs/2310.05338)

### 11. (OpenLEAF) OpenLEAF: Open-Domain Interleaved Image-Text Generation and Evaluation
**Date**: 2023.10.11

**Affiliation**: University of Rochester
<details span>
<summary><b>Abstract</b></summary>
This work investigates a challenging task named open-domain interleaved
image-text generation, which generates interleaved texts and images following
an input query. We propose a new interleaved generation framework based on
prompting large-language models (LLMs) and pre-trained text-to-image (T2I)
models, namely OpenLEAF. In OpenLEAF, the LLM generates textual descriptions,
coordinates T2I models, creates visual prompts for generating images, and
incorporates global contexts into the T2I models. This global context improves
the entity and style consistencies of images in the interleaved generation. For
model assessment, we first propose to use large multi-modal models (LMMs) to
evaluate the entity and style consistencies of open-domain interleaved
image-text sequences. According to the LMM evaluation on our constructed
evaluation set, the proposed interleaved generation framework can generate
high-quality image-text content for various domains and applications, such as
how-to question answering, storytelling, graphical story rewriting, and
webpage/poster generation tasks. Moreover, we validate the effectiveness of the
proposed LMM evaluation technique with human assessment. We hope our proposed
framework, benchmark, and LMM evaluation could help establish the intriguing
interleaved image-text generation task.
</details>

[📄 Paper](https://arxiv.org/abs/2310.07749)

### 12. (HallusionBench) HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models
**Date**: 2023.10.23

**Affiliation**: Beijing Academy of Artificial Intelligence
<details span>
<summary><b>Abstract</b></summary>
We introduce HallusionBench, a comprehensive benchmark designed for the
evaluation of image-context reasoning. This benchmark presents significant
challenges to advanced large visual-language models (LVLMs), such as
GPT-4V(Vision), Gemini Pro Vision, Claude 3, and LLaVA-1.5, by emphasizing
nuanced understanding and interpretation of visual data. The benchmark
comprises 346 images paired with 1129 questions, all meticulously crafted by
human experts. We introduce a novel structure for these visual questions
designed to establish control groups. This structure enables us to conduct a
quantitative analysis of the models' response tendencies, logical consistency,
and various failure modes. In our evaluation on HallusionBench, we benchmarked
15 different models, highlighting a 31.42% question-pair accuracy achieved by
the state-of-the-art GPT-4V. Notably, all other evaluated models achieve
accuracy below 16%. Moreover, our analysis not only highlights the observed
failure modes, including language hallucination and visual illusion, but also
deepens an understanding of these pitfalls. Our comprehensive case studies
within HallusionBench shed light on the challenges of hallucination and
illusion in LVLMs. Based on these insights, we suggest potential pathways for
their future improvement. The benchmark and codebase can be accessed at
https://github.com/tianyi-lab/HallusionBench.
</details>

[📄 Paper](https://arxiv.org/pdf/2310.14566) | [💻 Code](https://github.com/tianyi-lab/HallusionBench)

### 13. (Bingo) Holistic Analysis of Hallucination in GPT-4V(ision): Bias and Interference Challenges
**Date**: 2023.11.03

**Affiliation**: UNC-Chapel Hill
<details span>
<summary><b>Abstract</b></summary>
While GPT-4V(ision) impressively models both visual and textual information
simultaneously, it's hallucination behavior has not been systematically
assessed. To bridge this gap, we introduce a new benchmark, namely, the Bias
and Interference Challenges in Visual Language Models (Bingo). This benchmark
is designed to evaluate and shed light on the two common types of
hallucinations in visual language models: bias and interference. Here, bias
refers to the model's tendency to hallucinate certain types of responses,
possibly due to imbalance in its training data. Interference pertains to
scenarios where the judgment of GPT-4V(ision) can be disrupted due to how the
text prompt is phrased or how the input image is presented. We identify a
notable regional bias, whereby GPT-4V(ision) is better at interpreting Western
images or images with English writing compared to images from other countries
or containing text in other languages. Moreover, GPT-4V(ision) is vulnerable to
leading questions and is often confused when interpreting multiple images
together. Popular mitigation approaches, such as self-correction and
chain-of-thought reasoning, are not effective in resolving these challenges. We
also identified similar biases and interference vulnerabilities with LLaVA and
Bard. Our results characterize the hallucination challenges in GPT-4V(ision)
and state-of-the-art visual-language models, and highlight the need for new
solutions. The Bingo benchmark is available at https://github.com/gzcch/Bingo.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.03287) | [💻 Code](https://github.com/gzcch/Bingo)

### 14. (AMBER) AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation
**Date**: 2023.11.13

**Affiliation**: Beijing Jiaotong University
<details span>
<summary><b>Abstract</b></summary>
Despite making significant progress in multi-modal tasks, current Multi-modal
Large Language Models (MLLMs) encounter the significant challenge of
hallucinations, which may lead to harmful consequences. Therefore, evaluating
MLLMs' hallucinations is becoming increasingly important in model improvement
and practical application deployment. Previous works are limited in high
evaluation costs (e.g., relying on humans or advanced LLMs) and insufficient
evaluation dimensions (e.g., types of tasks and hallucinations). In this paper,
we propose an LLM-free multi-dimensional benchmark AMBER, which can be used to
evaluate both generative task and discriminative task including existence,
attribute and relation hallucination. Based on AMBER, we design a low-cost and
efficient evaluation pipeline. Additionally, we conduct a comprehensive
evaluation and detailed analysis of mainstream MLLMs including GPT-4V(ision),
and also give guideline suggestions for mitigating hallucinations. The data and
code of AMBER are available at https://github.com/junyangwang0410/AMBER.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.07397) | [💻 Code](https://github.com/junyangwang0410/AMBER)

### 15. (MM-SafetyBench) MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models
**Date**: 2023.11.29

**Affiliation**: Shanghai AI Laboratory
<details span>
<summary><b>Abstract</b></summary>
The security concerns surrounding Large Language Models (LLMs) have been
extensively explored, yet the safety of Multimodal Large Language Models
(MLLMs) remains understudied. In this paper, we observe that Multimodal Large
Language Models (MLLMs) can be easily compromised by query-relevant images, as
if the text query itself were malicious. To address this, we introduce
MM-SafetyBench, a comprehensive framework designed for conducting
safety-critical evaluations of MLLMs against such image-based manipulations. We
have compiled a dataset comprising 13 scenarios, resulting in a total of 5,040
text-image pairs. Our analysis across 12 state-of-the-art models reveals that
MLLMs are susceptible to breaches instigated by our approach, even when the
equipped LLMs have been safety-aligned. In response, we propose a
straightforward yet effective prompting strategy to enhance the resilience of
MLLMs against these types of attacks. Our work underscores the need for a
concerted effort to strengthen and enhance the safety measures of open-source
MLLMs against potential malicious exploits. The resource is available at
https://github.com/isXinLiu/MM-SafetyBench
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17600) | [💻 Code](https://github.com/isXinLiu/MM-SafetyBench)

### 16. (BenchLMM) BenchLMM: Benchmarking Cross-style Visual Capability of Large Multimodal Models
**Date**: 2023.12.05

**Affiliation**: Nanyang Technological University
<details span>
<summary><b>Abstract</b></summary>
Large Multimodal Models (LMMs) such as GPT-4V and LLaVA have shown remarkable
capabilities in visual reasoning with common image styles. However, their
robustness against diverse style shifts, crucial for practical applications,
remains largely unexplored. In this paper, we propose a new benchmark,
BenchLMM, to assess the robustness of LMMs against three different styles:
artistic image style, imaging sensor style, and application style, where each
style has five sub-styles. Utilizing BenchLMM, we comprehensively evaluate
state-of-the-art LMMs and reveal: 1) LMMs generally suffer performance
degradation when working with other styles; 2) An LMM performs better than
another model in common style does not guarantee its superior performance in
other styles; 3) LMMs' reasoning capability can be enhanced by prompting LMMs
to predict the style first, based on which we propose a versatile and
training-free method for improving LMMs; 4) An intelligent LMM is expected to
interpret the causes of its errors when facing stylistic variations. We hope
that our benchmark and analysis can shed new light on developing more
intelligent and versatile LMMs.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.02896) | [💻 Code](https://github.com/AIFEG/BenchLMM)

### 17. (MMCBench) Benchmarking Large Multimodal Models against Common Corruptions
**Date**: 2024.01.11

**Affiliation**: UIUC
<details span>
<summary><b>Abstract</b></summary>
This technical report aims to fill a deficiency in the assessment of large
multimodal models (LMMs) by specifically examining the self-consistency of
their outputs when subjected to common corruptions. We investigate the
cross-modal interactions between text, image, and speech, encompassing four
essential generation tasks: text-to-image, image-to-text, text-to-speech, and
speech-to-text. We create a comprehensive benchmark, named MMCBench, that
covers more than 100 popular LMMs (totally over 150 model checkpoints). A
thorough evaluation under common corruptions is critical for practical
deployment and facilitates a better understanding of the reliability of
cutting-edge LMMs. The benchmarking code is available at
https://github.com/sail-sg/MMCBench
</details>

[📄 Paper](https://arxiv.org/pdf/2401.11943) | [💻 Code](https://github.com/sail-sg/MMCBench)

### 18. (RTVLM) Red Teaming Visual Language Models
**Date**: 2024.01.24

**Affiliation**: The University of Hong Kong
<details span>
<summary><b>Abstract</b></summary>
VLMs (Vision-Language Models) extend the capabilities of LLMs (Large Language
Models) to accept multimodal inputs. Since it has been verified that LLMs can
be induced to generate harmful or inaccurate content through specific test
cases (termed as Red Teaming), how VLMs perform in similar scenarios,
especially with their combination of textual and visual inputs, remains a
question. To explore this problem, we present a novel red teaming dataset
RTVLM, which encompasses 10 subtasks (e.g., image misleading, multi-modal
jail-breaking, face fairness, etc) under 4 primary aspects (faithfulness,
privacy, safety, fairness). Our RTVLM is the first red-teaming dataset to
benchmark current VLMs in terms of these 4 different aspects. Detailed analysis
shows that 10 prominent open-sourced VLMs struggle with the red teaming in
different degrees and have up to 31% performance gap with GPT-4V. Additionally,
we simply apply red teaming alignment to LLaVA-v1.5 with Supervised Fine-tuning
(SFT) using RTVLM, and this bolsters the models' performance with 10% in RTVLM
test set, 13% in MM-Hal, and without noticeable decline in MM-Bench,
overpassing other LLaVA-based models with regular alignment data. This reveals
that current open-sourced VLMs still lack red teaming alignment. Our code and
datasets will be open-source.
</details>

[📄 Paper](https://arxiv.org/pdf/2401.12915) | [🌐 Project Page](https://huggingface.co/datasets/MMInstruction/RedTeamingVLM)

### 19. (MHaluBench) Unified Hallucination Detection for Multimodal Large Language Models
**Date**: 2024.02.05

**Affiliation**: Zhejiang University
<details span>
<summary><b>Abstract</b></summary>
Despite significant strides in multimodal tasks, Multimodal Large Language
Models (MLLMs) are plagued by the critical issue of hallucination. The reliable
detection of such hallucinations in MLLMs has, therefore, become a vital aspect
of model evaluation and the safeguarding of practical application deployment.
Prior research in this domain has been constrained by a narrow focus on
singular tasks, an inadequate range of hallucination categories addressed, and
a lack of detailed granularity. In response to these challenges, our work
expands the investigative horizons of hallucination detection. We present a
novel meta-evaluation benchmark, MHaluBench, meticulously crafted to facilitate
the evaluation of advancements in hallucination detection methods.
Additionally, we unveil a novel unified multimodal hallucination detection
framework, UNIHD, which leverages a suite of auxiliary tools to validate the
occurrence of hallucinations robustly. We demonstrate the effectiveness of
UNIHD through meticulous evaluation and comprehensive analysis. We also provide
strategic insights on the application of specific tools for addressing various
categories of hallucinations.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.03190) | [🌐 Project Page](https://www.zjukg.org/project/EasyDetect/) | [💻 Code](https://github.com/OpenKG-ORG/EasyDetect)

### 20. (CorrelationQA) The Instinctive Bias: Spurious Images lead to Illusion in MLLMs
**Date**: 2024.02.06

**Affiliation**: The Hong Kong University of Science and Technology
<details span>
<summary><b>Abstract</b></summary>
Large language models (LLMs) have recently experienced remarkable progress,
where the advent of multi-modal large language models (MLLMs) has endowed LLMs
with visual capabilities, leading to impressive performances in various
multi-modal tasks. However, those powerful MLLMs such as GPT-4V still fail
spectacularly when presented with certain image and text inputs. In this paper,
we identify a typical class of inputs that baffles MLLMs, which consist of
images that are highly relevant but inconsistent with answers, causing MLLMs to
suffer from visual illusion. To quantify the effect, we propose CorrelationQA,
the first benchmark that assesses the visual illusion level given spurious
images. This benchmark contains 7,308 text-image pairs across 13 categories.
Based on the proposed CorrelationQA, we conduct a thorough analysis on 9
mainstream MLLMs, illustrating that they universally suffer from this
instinctive bias to varying degrees. We hope that our curated benchmark and
evaluation results aid in better assessments of the MLLMs' robustness in the
presence of misleading images. The code and datasets are available at
https://github.com/MasaiahHan/CorrelationQA.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.03757) | [💻 Code](https://github.com/MasaiahHan/CorrelationQA)

### 21. (SHIELD) SHIELD : An Evaluation Benchmark for Face Spoofing and Forgery Detection with Multimodal Large Language Models
**Date**: 2024.02.07

**Affiliation**: Shanghai Jiao Tong University
<details span>
<summary><b>Abstract</b></summary>
Multimodal large language models (MLLMs) have demonstrated remarkable
problem-solving capabilities in various vision fields (e.g., generic object
recognition and grounding) based on strong visual semantic representation and
language reasoning ability. However, whether MLLMs are sensitive to subtle
visual spoof/forged clues and how they perform in the domain of face attack
detection (e.g., face spoofing and forgery detection) is still unexplored. In
this paper, we introduce a new benchmark, namely SHIELD, to evaluate the
ability of MLLMs on face spoofing and forgery detection. Specifically, we
design true/false and multiple-choice questions to evaluate multimodal face
data in these two face security tasks. For the face anti-spoofing task, we
evaluate three different modalities (i.e., RGB, infrared, depth) under four
types of presentation attacks (i.e., print attack, replay attack, rigid mask,
paper mask). For the face forgery detection task, we evaluate GAN-based and
diffusion-based data with both visual and acoustic modalities. Each question is
subjected to both zero-shot and few-shot tests under standard and chain of
thought (COT) settings. The results indicate that MLLMs hold substantial
potential in the face security domain, offering advantages over traditional
specific models in terms of interpretability, multimodal flexible reasoning,
and joint face spoof and forgery detection. Additionally, we develop a novel
Multi-Attribute Chain of Thought (MA-COT) paradigm for describing and judging
various task-specific and task-irrelevant attributes of face images, which
provides rich task-related knowledge for subtle spoof/forged clue mining.
Extensive experiments in separate face anti-spoofing, separate face forgery
detection, and joint detection tasks demonstrate the effectiveness of the
proposed MA-COT. The project is available at
https$:$//github.com/laiyingxin2/SHIELD
</details>

[📄 Paper](https://arxiv.org/pdf/2402.04178) | [💻 Code](https://github.com/laiyingxin2/SHIELD)

### 22. (VQAv2-IDK) Visually Dehallucinative Instruction Generation: Know What You Don't Know
**Date**: 2024.02.15

**Affiliation**: NCSOFT Corporation
<details span>
<summary><b>Abstract</b></summary>
"When did the emperor Napoleon invented iPhone?" Such hallucination-inducing
question is well known challenge in generative language modeling. In this
study, we present an innovative concept of visual hallucination, referred to as
"I Know (IK)" hallucination, to address scenarios where "I Don't Know" is the
desired response. To effectively tackle this issue, we propose the VQAv2-IDK
benchmark, the subset of VQAv2 comprising unanswerable image-question pairs as
determined by human annotators. Stepping further, we present the visually
dehallucinative instruction generation method for IK hallucination and
introduce the IDK-Instructions visual instruction database. Our experiments
show that current methods struggle with IK hallucination. Yet, our approach
effectively reduces these hallucinations, proving its versatility across
different frameworks and datasets.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.09717)

### 23. (MAD-Bench) How Easy is It to Fool Your Multimodal LLMs? An Empirical Analysis on Deceptive Prompts
**Date**: 2024.02.20

**Affiliation**: Apple
<details span>
<summary><b>Abstract</b></summary>
The remarkable advancements in Multimodal Large Language Models (MLLMs) have
not rendered them immune to challenges, particularly in the context of handling
deceptive information in prompts, thus producing hallucinated responses under
such conditions. To quantitatively assess this vulnerability, we present
MAD-Bench, a carefully curated benchmark that contains 1000 test samples
divided into 5 categories, such as non-existent objects, count of objects, and
spatial relationship. We provide a comprehensive analysis of popular MLLMs,
ranging from GPT-4v, Reka, Gemini-Pro, to open-sourced models, such as
LLaVA-NeXT and MiniCPM-Llama3. Empirically, we observe significant performance
gaps between GPT-4o and other models; and previous robust instruction-tuned
models are not effective on this new benchmark. While GPT-4o achieves 82.82%
accuracy on MAD-Bench, the accuracy of any other model in our experiments
ranges from 9% to 50%. We further propose a remedy that adds an additional
paragraph to the deceptive prompts to encourage models to think twice before
answering the question. Surprisingly, this simple method can even double the
accuracy; however, the absolute numbers are still too low to be satisfactory.
We hope MAD-Bench can serve as a valuable benchmark to stimulate further
research to enhance model resilience against deceptive prompts.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.13220)

### 24. (VHTest) Visual Hallucinations of Multi-modal Large Language Models
**Date**: 2024.02.22

**Affiliation**: University of Science & Technology of China
<details span>
<summary><b>Abstract</b></summary>
Visual hallucination (VH) means that a multi-modal LLM (MLLM) imagines
incorrect details about an image in visual question answering. Existing studies
find VH instances only in existing image datasets, which results in biased
understanding of MLLMs' performance under VH due to limited diversity of such
VH instances. In this work, we propose a tool called VHTest to generate a
diverse set of VH instances. Specifically, VHTest finds some initial VH
instances in existing image datasets (e.g., COCO), generates a text description
for each VH mode, and uses a text-to-image generative model (e.g., DALL-E-3) to
generate VH images based on the text descriptions. We collect a benchmark
dataset with 1,200 VH instances in 8 VH modes using VHTest. We find that
existing MLLMs such as GPT-4V, LLaVA-1.5, and MiniGPT-v2 hallucinate for a
large fraction of the instances in our benchmark. Moreover, we find that
fine-tuning an MLLM using our benchmark dataset reduces its likelihood to
hallucinate without sacrificing its performance on other benchmarks. Our
benchmarks are publicly available: https://github.com/wenhuang2000/VHTest.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.14683) | [💻 Code](https://github.com/wenhuang2000/VHTest)

### 25. (MMECeption) GenCeption: Evaluate Multimodal LLMs with Unlabeled Unimodal Data
**Date**: 2024.02.23

**Affiliation**: EQT Motherbrain
<details span>
<summary><b>Abstract</b></summary>
Multimodal Large Language Models (MLLMs) are typically assessed using
expensive annotated multimodal benchmarks, which often lag behind the rapidly
evolving demands of MLLM evaluation. This paper outlines and validates
GenCeption, a novel, annotation-free evaluation method that requires only
unimodal data to measure inter-modality semantic coherence and inversely
assesses MLLMs' tendency to hallucinate. This approach eliminates the need for
costly data annotation, minimizes the risk of training data contamination,
results in slower benchmark saturation, and avoids the illusion of emerging
abilities. Inspired by the DrawCeption game, GenCeption begins with a
non-textual sample and proceeds through iterative description and generation
steps. The semantic drift across iterations is quantified using the GC@T
metric. Based on the GenCeption method, we establish the MMECeption benchmark
for evaluating Vision LLMs (VLLMs), and compare performance of several popular
VLLMs and human annotators. Our empirical results validate GenCeption's
effectiveness, demonstrating strong correlations with established VLLM
benchmarks. VLLMs still significantly lack behind human performance and
struggle especially with text-intensive tasks.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.14973) | [💻 Code](https://github.com/llcresearch/GenCeption)

### 26. (CoIN) CoIN: A Benchmark of Continual Instruction tuNing for Multimodel Large Language Model
**Date**: 2024.03.13

**Affiliation**: University of Electronic Science and Technology of China
<details span>
<summary><b>Abstract</b></summary>
Instruction tuning represents a prevalent strategy employed by Multimodal
Large Language Models (MLLMs) to align with human instructions and adapt to new
tasks. Nevertheless, MLLMs encounter the challenge of adapting to users'
evolving knowledge and demands. Therefore, how to retain existing skills while
acquiring new knowledge needs to be investigated. In this paper, we present a
comprehensive benchmark, namely Continual Instruction tuNing (CoIN), to assess
existing MLLMs in the sequential instruction tuning paradigm. CoIN comprises 10
commonly used datasets spanning 8 task categories, ensuring a diverse range of
instructions and tasks. Besides, the trained model is evaluated from two
aspects: Instruction Following and General Knowledge, which assess the
alignment with human intention and knowledge preserved for reasoning,
respectively. Experiments on CoIN demonstrate that current powerful MLLMs still
suffer catastrophic forgetting, and the failure in intention alignment assumes
the main responsibility, instead of the knowledge forgetting. To this end, we
introduce MoELoRA to MLLMs which is effective to retain the previous
instruction alignment. Experimental results consistently illustrate the
forgetting decreased from this method on CoIN.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.08350) | [💻 Code](https://github.com/zackschen/CoIN)

### 27. (JailBreakV-28K) JailBreakV-28K: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks
**Date**: 2024.04.04

**Affiliation**: The Ohio State University
<details span>
<summary><b>Abstract</b></summary>
With the rapid advancements in Multimodal Large Language Models (MLLMs),
securing these models against malicious inputs while aligning them with human
values has emerged as a critical challenge. In this paper, we investigate an
important and unexplored question of whether techniques that successfully
jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking
MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering
benchmark designed to assess the transferability of LLM jailbreak techniques to
MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak
attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed
in this paper, we generate 20, 000 text-based jailbreak prompts using advanced
jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from
recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test
cases across a spectrum of adversarial scenarios. Our evaluation of 10
open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks
transferred from LLMs, highlighting a critical vulnerability in MLLMs that
stems from their text-processing capabilities. Our findings underscore the
urgent need for future research to address alignment vulnerabilities in MLLMs
from both textual and visual inputs.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.03027) | [🌐 Project Page](https://eddyluo1232.github.io/JailBreakV28K/) | [💻 Code](https://github.com/EddyLuo1232/JailBreakV_28K)

### 28. (Plot2Code) Plot2Code: A Comprehensive Benchmark for Evaluating Multi-modal Large Language Models in Code Generation from Scientific Plots
**Date**: 2024.05.07

**Affiliation**: The University of Hong Kong
<details span>
<summary><b>Abstract</b></summary>
The remarkable progress of Multi-modal Large Language Models (MLLMs) has
attracted significant attention due to their superior performance in visual
contexts. However, their capabilities in turning visual figure to executable
code, have not been evaluated thoroughly. To address this, we introduce
Plot2Code, a comprehensive visual coding benchmark designed for a fair and
in-depth assessment of MLLMs. We carefully collect 132 manually selected
high-quality matplotlib plots across six plot types from publicly available
matplotlib galleries. For each plot, we carefully offer its source code, and an
descriptive instruction summarized by GPT-4. This approach enables Plot2Code to
extensively evaluate MLLMs' code capabilities across various input modalities.
Furthermore, we propose three automatic evaluation metrics, including code pass
rate, text-match ratio, and GPT-4V overall rating, for a fine-grained
assessment of the output code and rendered images. Instead of simply judging
pass or fail, we employ GPT-4V to make an overall judgement between the
generated and reference images, which has been shown to be consistent with
human evaluation. The evaluation results, which include analyses of 14 MLLMs
such as the proprietary GPT-4V, Gemini-Pro, and the open-sourced Mini-Gemini,
highlight the substantial challenges presented by Plot2Code. With Plot2Code, we
reveal that most existing MLLMs struggle with visual coding for text-dense
plots, heavily relying on textual instruction. We hope that the evaluation
results from Plot2Code on visual coding will guide the future development of
MLLMs. All data involved with Plot2Code are available at
https://huggingface.co/datasets/TencentARC/Plot2Code.
</details>

[📄 Paper](https://arxiv.org/pdf/2405.07990) | [💻 Code](https://github.com/TencentARC/Plot2Code)

### 29. (THRONE) THRONE: An Object-based Hallucination Benchmark for the Free-form Generations of Large Vision-Language Models
**Date**: 2024.05.08

**Affiliation**: University of Oxford
<details span>
<summary><b>Abstract</b></summary>
Mitigating hallucinations in large vision-language models (LVLMs) remains an
open problem. Recent benchmarks do not address hallucinations in open-ended
free-form responses, which we term "Type I hallucinations". Instead, they focus
on hallucinations responding to very specific question formats -- typically a
multiple-choice response regarding a particular object or attribute -- which we
term "Type II hallucinations". Additionally, such benchmarks often require
external API calls to models which are subject to change. In practice, we
observe that a reduction in Type II hallucinations does not lead to a reduction
in Type I hallucinations but rather that the two forms of hallucinations are
often anti-correlated. To address this, we propose THRONE, a novel object-based
automatic framework for quantitatively evaluating Type I hallucinations in LVLM
free-form outputs. We use public language models (LMs) to identify
hallucinations in LVLM responses and compute informative metrics. By evaluating
a large selection of recent LVLMs using public datasets, we show that an
improvement in existing metrics do not lead to a reduction in Type I
hallucinations, and that established benchmarks for measuring Type I
hallucinations are incomplete. Finally, we provide a simple and effective data
augmentation method to reduce Type I and Type II hallucinations as a strong
baseline.
</details>

[📄 Paper](https://arxiv.org/abs/2405.05256v1)

### 30. (MRHal-Bench) Automated Multi-level Preference for MLLMs
**Date**: 2024.05.11

**Affiliation**: Baidu Inc.
<details span>
<summary><b>Abstract</b></summary>
Current multimodal Large Language Models (MLLMs) suffer from
``hallucination'', occasionally generating responses that are not grounded in
the input images. To tackle this challenge, one promising path is to utilize
reinforcement learning from human feedback (RLHF), which steers MLLMs towards
learning superior responses while avoiding inferior ones. We rethink the common
practice of using binary preferences (i.e., superior, inferior), and find that
adopting multi-level preferences (e.g., superior, medium, inferior) is better
for two benefits: 1) It narrows the gap between adjacent levels, thereby
encouraging MLLMs to discern subtle differences. 2) It further integrates
cross-level comparisons (beyond adjacent-level comparisons), thus providing a
broader range of comparisons with hallucination examples. To verify our
viewpoint, we present the Automated Multi-level Preference (AMP) framework for
MLLMs. To facilitate this framework, we first develop an automated dataset
generation pipeline that provides high-quality multi-level preference datasets
without any human annotators. Furthermore, we design the Multi-level Direct
Preference Optimization (MDPO) algorithm to robustly conduct complex
multi-level preference learning. Additionally, we propose a new hallucination
benchmark, MRHal-Bench. Extensive experiments across public hallucination and
general benchmarks, as well as our MRHal-Bench, demonstrate the effectiveness
of our proposed method. Code is available at https://github.com/takomc/amp.
</details>

[📄 Paper](https://arxiv.org/pdf/2405.11165) | [💻 Code](https://github.com/takomc/amp)

### 31. (MetaToken) MetaToken: Detecting Hallucination in Image Descriptions by Meta Classification
**Date**: 2024.05.29

**Affiliation**: Volkswagen AG
<details span>
<summary><b>Abstract</b></summary>
Large Vision Language Models (LVLMs) have shown remarkable capabilities in
multimodal tasks like visual question answering or image captioning. However,
inconsistencies between the visual information and the generated text, a
phenomenon referred to as hallucinations, remain an unsolved problem with
regard to the trustworthiness of LVLMs. To address this problem, recent works
proposed to incorporate computationally costly Large (Vision) Language Models
in order to detect hallucinations on a sentence- or subsentence-level. In this
work, we introduce MetaToken, a lightweight binary classifier to detect
hallucinations on the token-level at negligible cost. Based on a statistical
analysis, we reveal key factors of hallucinations in LVLMs which have been
overseen in previous works. MetaToken can be applied to any open-source LVLM
without any knowledge about ground truth data providing a reliable detection of
hallucinations. We evaluate our method on four state-of-the-art LVLMs
demonstrating the effectiveness of our approach.
</details>

[📄 Paper](https://arxiv.org/abs/2405.19186v1)

### 32. (MultiTrust) Benchmarking Trustworthiness of Multimodal Large Language Models: A Comprehensive Study
**Date**: 2024.06.07

**Affiliation**: Tsinghua University
<details span>
<summary><b>Abstract</b></summary>
Despite the superior capabilities of Multimodal Large Language Models (MLLMs)
across diverse tasks, they still face significant trustworthiness challenges.
Yet, current literature on the assessment of trustworthy MLLMs remains limited,
lacking a holistic evaluation to offer thorough insights into future
improvements. In this work, we establish MultiTrust, the first comprehensive
and unified benchmark on the trustworthiness of MLLMs across five primary
aspects: truthfulness, safety, robustness, fairness, and privacy. Our benchmark
employs a rigorous evaluation strategy that addresses both multimodal risks and
cross-modal impacts, encompassing 32 diverse tasks with self-curated datasets.
Extensive experiments with 21 modern MLLMs reveal some previously unexplored
trustworthiness issues and risks, highlighting the complexities introduced by
the multimodality and underscoring the necessity for advanced methodologies to
enhance their reliability. For instance, typical proprietary models still
struggle with the perception of visually confusing images and are vulnerable to
multimodal jailbreaking and adversarial attacks; MLLMs are more inclined to
disclose privacy in text and reveal ideological and cultural biases even when
paired with irrelevant images in inference, indicating that the multimodality
amplifies the internal risks from base LLMs. Additionally, we release a
scalable toolbox for standardized trustworthiness research, aiming to
facilitate future advancements in this important field. Code and resources are
publicly available at: https://multi-trust.github.io/.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.07057) | [🌐 Project Page](https://multi-trust.github.io/) | [💻 Code](https://github.com/thu-ml/MMTrustEval)

### 33. (MLLMGuard) MLLMGuard: A Multi-dimensional Safety Evaluation Suite for Multimodal Large Language Models
**Date**: 2024.06.07

**Affiliation**: Tsinghua University
<details span>
<summary><b>Abstract</b></summary>
Powered by remarkable advancements in Large Language Models (LLMs),
Multimodal Large Language Models (MLLMs) demonstrate impressive capabilities in
manifold tasks. However, the practical application scenarios of MLLMs are
intricate, exposing them to potential malicious instructions and thereby posing
safety risks. While current benchmarks do incorporate certain safety
considerations, they often lack comprehensive coverage and fail to exhibit the
necessary rigor and robustness. For instance, the common practice of employing
GPT-4V as both the evaluator and a model to be evaluated lacks credibility, as
it tends to exhibit a bias toward its own responses. In this paper, we present
MLLMGuard, a multidimensional safety evaluation suite for MLLMs, including a
bilingual image-text evaluation dataset, inference utilities, and a lightweight
evaluator. MLLMGuard's assessment comprehensively covers two languages (English
and Chinese) and five important safety dimensions (Privacy, Bias, Toxicity,
Truthfulness, and Legality), each with corresponding rich subtasks. Focusing on
these dimensions, our evaluation dataset is primarily sourced from platforms
such as social media, and it integrates text-based and image-based red teaming
techniques with meticulous annotation by human experts. This can prevent
inaccurate evaluation caused by data leakage when using open-source datasets
and ensures the quality and challenging nature of our benchmark. Additionally,
a fully automated lightweight evaluator termed GuardRank is developed, which
achieves significantly higher evaluation accuracy than GPT-4. Our evaluation
results across 13 advanced models indicate that MLLMs still have a substantial
journey ahead before they can be considered safe and responsible.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.07594) | [💻 Code](https://github.com/Carol-gutianle/MLLMGuard)

### 34. (CoMM) CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation
**Date**: 2024.06.10

**Affiliation**: The Hong Kong University of Science and Technology
<details span>
<summary><b>Abstract</b></summary>
Interleaved image-text generation has emerged as a crucial multimodal task,
aiming at creating sequences of interleaved visual and textual content given a
query. Despite notable advancements in recent multimodal large language models
(MLLMs), generating integrated image-text sequences that exhibit narrative
coherence and entity and style consistency remains challenging due to poor
training data quality. To address this gap, we introduce CoMM, a high-quality
Coherent interleaved image-text MultiModal dataset designed to enhance the
coherence, consistency, and alignment of generated multimodal content.
Initially, CoMM harnesses raw data from diverse sources, focusing on
instructional content and visual storytelling, establishing a foundation for
coherent and consistent content. To further refine the data quality, we devise
a multi-perspective filter strategy that leverages advanced pre-trained models
to ensure the development of sentences, consistency of inserted images, and
semantic alignment between them. Various quality evaluation metrics are
designed to prove the high quality of the filtered dataset. Meanwhile,
extensive few-shot experiments on various downstream tasks demonstrate CoMM's
effectiveness in significantly enhancing the in-context learning capabilities
of MLLMs. Moreover, we propose four new tasks to evaluate MLLMs' interleaved
generation abilities, supported by a comprehensive evaluation framework. We
believe CoMM opens a new avenue for advanced MLLMs with superior multimodal
in-context learning and understanding ability.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.10462) | [💻 Code](https://github.com/HKUST-LongGroup/CoMM)

### 35. (MMR) Seeing Clearly, Answering Incorrectly: A Multimodal Robustness Benchmark for Evaluating MLLMs on Leading Questions
**Date**: 2024.06.10

**Affiliation**: Beijing Academy of Artificial Intelligence
<details span>
<summary><b>Abstract</b></summary>
Multimodal Large Language Models (MLLMs) have exhibited impressive
capabilities in visual understanding and reasoning, providing sightly
reasonable answers, such as image descriptions. This has spurred extensive
research on the evaluation of MLLMs. Most evaluation benchmarks assume that
incorrect answers indicate a lack of understanding of the visual content.
However, our findings reveal that, in many cases, MLLMs answer questions
incorrectly despite correctly understanding the visual content. This suggests
that incorrect answers do not necessarily imply a lack of comprehension but may
instead result from lacking robustness to leading questions. To comprehensively
measure MLLMs' understanding capability and robustness to leading questions, we
introduce a MultiModal Robustness benchmark (MMR). MMR contains paired positive
and negative questions across 12 categories, meticulously annotated by humans.
We evaluate 18 leading MLLMs on the MMB benchmark, revealing that MLLMs suffer
from fragility to leading questions despite understanding the visual content.
To enhance MLLMs' understanding capability and robustness, we further present a
training set with paired positive and negative visual question-answer samples.
Experiments verify that MLLMs' robustness can be significantly enhanced by
tuning on this new training set. The benchmark, training set, and code can be
found at https://github.com/BAAI-DCAI/Multimodal-Robustness-Benchmark.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.10638) | [💻 Code](https://github.com/BAAI-DCAI/Multimodal-Robustness-Benchmark)

### 36. (MTruthfulQA) Towards Truthful Multilingual Large Language Models: Benchmarking and Alignment Strategies
**Date**: 2024.06.14

**Affiliation**: Microsoft STC Asian
<details span>
<summary><b>Abstract</b></summary>
In the era of large language models (LLMs), building multilingual large
language models (MLLMs) that can serve users worldwide holds great
significance. However, existing research seldom focuses on the truthfulness of
MLLMs. Meanwhile, contemporary multilingual aligning technologies struggle to
balance massive languages and often exhibit serious truthfulness gaps across
different languages, especially those that differ greatly from English. In our
work, we construct a benchmark for truthfulness evaluation in multilingual
scenarios and explore the ways to align facts across languages to enhance the
truthfulness of MLLMs. Furthermore, we propose Fact-aware Multilingual
Selective Synergy (FaMSS) to optimize the data allocation across a large number
of languages and different data types. Experimental results demonstrate that
our approach can effectively reduce the multilingual representation disparity
and enhance the multilingual capabilities of LLMs.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.14434)

### 37. (Med-HallMark) Detecting and Evaluating Medical Hallucinations in Large Vision Language Models
**Date**: 2024.06.14

**Affiliation**: Fudan University
<details span>
<summary><b>Abstract</b></summary>
Large Vision Language Models (LVLMs) are increasingly integral to healthcare
applications, including medical visual question answering and imaging report
generation. While these models inherit the robust capabilities of foundational
Large Language Models (LLMs), they also inherit susceptibility to
hallucinations-a significant concern in high-stakes medical contexts where the
margin for error is minimal. However, currently, there are no dedicated methods
or benchmarks for hallucination detection and evaluation in the medical field.
To bridge this gap, we introduce Med-HallMark, the first benchmark specifically
designed for hallucination detection and evaluation within the medical
multimodal domain. This benchmark provides multi-tasking hallucination support,
multifaceted hallucination data, and hierarchical hallucination categorization.
Furthermore, we propose the MediHall Score, a new medical evaluative metric
designed to assess LVLMs' hallucinations through a hierarchical scoring system
that considers the severity and type of hallucination, thereby enabling a
granular assessment of potential clinical impacts. We also present
MediHallDetector, a novel Medical LVLM engineered for precise hallucination
detection, which employs multitask training for hallucination detection.
Through extensive experimental evaluations, we establish baselines for popular
LVLMs using our benchmark. The findings indicate that MediHall Score provides a
more nuanced understanding of hallucination impacts compared to traditional
metrics and demonstrate the enhanced performance of MediHallDetector. We hope
this work can significantly improve the reliability of LVLMs in medical
applications. All resources of this work will be released soon.
</details>

[📄 Paper](https://arxiv.org/abs/2406.10185v1)

### 38. (AutoHallusion) AUTOHALLUSION: Automatic Generation of Hallucination Benchmarks for Vision-Language Models
**Date**: 2024.06.16

**Affiliation**: University of Maryland
<details span>
<summary><b>Abstract</b></summary>
Large vision-language models (LVLMs) hallucinate: certain context cues in an
image may trigger the language module's overconfident and incorrect reasoning
on abnormal or hypothetical objects. Though a few benchmarks have been
developed to investigate LVLM hallucinations, they mainly rely on hand-crafted
corner cases whose fail patterns may hardly generalize, and finetuning on them
could undermine their validity. These motivate us to develop the first
automatic benchmark generation approach, AUTOHALLUSION, that harnesses a few
principal strategies to create diverse hallucination examples. It probes the
language modules in LVLMs for context cues and uses them to synthesize images
by: (1) adding objects abnormal to the context cues; (2) for two co-occurring
objects, keeping one and excluding the other; or (3) removing objects closely
tied to the context cues. It then generates image-based questions whose
ground-truth answers contradict the language module's prior. A model has to
overcome contextual biases and distractions to reach correct answers, while
incorrect or inconsistent answers indicate hallucinations. AUTOHALLUSION
enables us to create new benchmarks at the minimum cost and thus overcomes the
fragility of hand-crafted benchmarks. It also reveals common failure patterns
and reasons, providing key insights to detect, avoid, or control
hallucinations. Comprehensive evaluations of top-tier LVLMs, e.g.,
GPT-4V(ision), Gemini Pro Vision, Claude 3, and LLaVA-1.5, show a 97.7% and
98.7% success rate of hallucination induction on synthetic and real-world
datasets of AUTOHALLUSION, paving the way for a long battle against
hallucinations.
</details>

[📄 Paper](https://arxiv.org/abs/2406.10900v1)

### 39. (VideoHallucer) VideoHallucer: Evaluating Intrinsic and Extrinsic Hallucinations in Large Video-Language Models
**Date**: 2024.06.16

**Affiliation**: Beijing Institute for General Artificial Intelligence
<details span>
<summary><b>Abstract</b></summary>
Recent advancements in Multimodal Large Language Models (MLLMs) have extended
their capabilities to video understanding. Yet, these models are often plagued
by "hallucinations", where irrelevant or nonsensical content is generated,
deviating from the actual video context. This work introduces VideoHallucer,
the first comprehensive benchmark for hallucination detection in large
video-language models (LVLMs). VideoHallucer categorizes hallucinations into
two main types: intrinsic and extrinsic, offering further subcategories for
detailed analysis, including object-relation, temporal, semantic detail,
extrinsic factual, and extrinsic non-factual hallucinations. We adopt an
adversarial binary VideoQA method for comprehensive evaluation, where pairs of
basic and hallucinated questions are crafted strategically. By evaluating
eleven LVLMs on VideoHallucer, we reveal that i) the majority of current models
exhibit significant issues with hallucinations; ii) while scaling datasets and
parameters improves models' ability to detect basic visual cues and
counterfactuals, it provides limited benefit for detecting extrinsic factual
hallucinations; iii) existing models are more adept at detecting facts than
identifying hallucinations. As a byproduct, these analyses further instruct the
development of our self-PEP framework, achieving an average of 5.38%
improvement in hallucination resistance across all model architectures.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.16338) | [🌐 Project Page](https://videohallucer.github.io/) | [💻 Code](https://github.com/patrick-tssn/VideoHallucer)

### 40. (MM-SpuBench) MM-SpuBench: Towards Better Understanding of Spurious Biases in Multimodal LLMs
**Date**: 2024.06.17

**Affiliation**: University of Virginia
<details span>
<summary><b>Abstract</b></summary>
Spurious bias, a tendency to use spurious correlations between non-essential
input attributes and target variables for predictions, has revealed a severe
robustness pitfall in deep learning models trained on single modality data.
Multimodal Large Language Models (MLLMs), which integrate both vision and
language models, have demonstrated strong capability in joint vision-language
understanding. However, whether spurious biases are prevalent in MLLMs remains
under-explored. We mitigate this gap by analyzing the spurious biases in a
multimodal setting, uncovering the specific test data patterns that can
manifest this problem when biases in the vision model cascade into the
alignment between visual and text tokens in MLLMs. To better understand this
problem, we introduce MM-SpuBench, a comprehensive visual question-answering
(VQA) benchmark designed to evaluate MLLMs' reliance on nine distinct
categories of spurious correlations from five open-source image datasets. The
VQA dataset is built from human-understandable concept information
(attributes). Leveraging this benchmark, we conduct a thorough evaluation of
current state-of-the-art MLLMs. Our findings illuminate the persistence of the
reliance on spurious correlations from these models and underscore the urge for
new methodologies to mitigate spurious biases. To support the MLLM robustness
research, we release our VQA benchmark at
https://huggingface.co/datasets/mmbench/MM-SpuBench.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.17126) | [🌐 Project Page](https://huggingface.co/datasets/mmbench/MM-SpuBench)

### 41. (MOSSBench) MOSSBench: Is Your Multimodal Language Model Oversensitive to Safe Queries?
**Date**: 2024.06.17

**Affiliation**: University of California
<details span>
<summary><b>Abstract</b></summary>
Humans are prone to cognitive distortions -- biased thinking patterns that
lead to exaggerated responses to specific stimuli, albeit in very different
contexts. This paper demonstrates that advanced Multimodal Large Language
Models (MLLMs) exhibit similar tendencies. While these models are designed to
respond queries under safety mechanism, they sometimes reject harmless queries
in the presence of certain visual stimuli, disregarding the benign nature of
their contexts. As the initial step in investigating this behavior, we identify
three types of stimuli that trigger the oversensitivity of existing MLLMs:
Exaggerated Risk, Negated Harm, and Counterintuitive Interpretation. To
systematically evaluate MLLMs' oversensitivity to these stimuli, we propose the
Multimodal OverSenSitivity Benchmark (MOSSBench). This toolkit consists of 300
manually collected benign multimodal queries, cross-verified by third-party
reviewers (AMT). Empirical studies using MOSSBench on 20 MLLMs reveal several
insights: (1). Oversensitivity is prevalent among SOTA MLLMs, with refusal
rates reaching up to 76% for harmless queries. (2). Safer models are more
oversensitive: increasing safety may inadvertently raise caution and
conservatism in the model's responses. (3). Different types of stimuli tend to
cause errors at specific stages -- perception, intent reasoning, and safety
judgement -- in the response process of MLLMs. These findings highlight the
need for refined safety mechanisms that balance caution with contextually
appropriate responses, improving the reliability of MLLMs in real-world
applications. We make our project available at
https://turningpoint-ai.github.io/MOSSBench/.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.17806) | [🌐 Project Page](https://turningpoint-ai.github.io/MOSSBench/) | [💻 Code](https://github.com/xirui-li/MOSSBench)

### 42. (MFC-Bench) MFC-Bench: Benchmarking Multimodal Fact-Checking with Large Vision-Language Models
**Date**: 2024.06.17

**Affiliation**: Beijing University of Posts and Telecommunications
<details span>
<summary><b>Abstract</b></summary>
Large vision-language models (LVLMs) have significantly improved multimodal
reasoning tasks, such as visual question answering and image captioning. These
models embed multimodal facts within their parameters, rather than relying on
external knowledge bases to store factual information explicitly. However, the
content discerned by LVLMs may deviate from actual facts due to inherent bias
or incorrect inference. To address this issue, we introduce MFC-Bench, a
rigorous and comprehensive benchmark designed to evaluate the factual accuracy
of LVLMs across three tasks: Manipulation, Out-of-Context, and Veracity
Classification. Through our evaluation on MFC-Bench, we benchmarked 12 diverse
and representative LVLMs, uncovering that current models still fall short in
multimodal fact-checking and demonstrate insensitivity to various forms of
manipulated content. We hope that MFC-Bench could raise attention to the
trustworthy artificial intelligence potentially assisted by LVLMs in the
future. The MFC-Bench and accompanying resources are publicly accessible at
https://github.com/wskbest/MFC-Bench, contributing to ongoing research in the
multimodal fact-checking field.
</details>

[📄 Paper](https://arxiv.org/abs/2406.11288) | [💻 Code](https://github.com/wskbest/MFC-Bench)

### 43. (VGA) VGA: Vision GUI Assistant -- Minimizing Hallucinations through Image-Centric Fine-Tuning
**Date**: 2024.06.20

**Affiliation**: East China Normal University
<details span>
<summary><b>Abstract</b></summary>
Recent advances in Large Vision-Language Models (LVLMs) have significantly
improve performance in image comprehension tasks, such as formatted charts and
rich-content images. Yet, Graphical User Interface (GUI) pose a greater
challenge due to their structured format and detailed textual information.
Existing LVLMs often overly depend on internal knowledge and neglect image
content, resulting in hallucinations and incorrect responses in GUI
comprehension. To address these issues, we introduce VGA, a fine-tuned model
designed for comprehensive GUI understanding. Our model aims to enhance the
interpretation of visual data of GUI and reduce hallucinations. We first
construct a Vision Question Answering (VQA) dataset of 63.8k high-quality
examples with our propose Referent Method, which ensures the model's responses
are highly depend on visual content within the image. We then design a
two-stage fine-tuning method called Foundation and Advanced Comprehension (FAC)
to enhance both the model's ability to extract information from image content
and alignment with human intent. Experiments show that our approach enhances
the model's ability to extract information from images and achieves
state-of-the-art results in GUI understanding tasks. Our dataset and
fine-tuning script will be released soon.
</details>

[📄 Paper](https://arxiv.org/abs/2406.14056)

### 44. (Web2Code) Web2Code: A Large-scale Webpage-to-Code Dataset and Evaluation Framework for Multimodal LLMs
**Date**: 2024.06.20

**Affiliation**: MBZUAI
<details span>
<summary><b>Abstract</b></summary>
Multimodal large language models (MLLMs) have shown impressive success across
modalities such as image, video, and audio in a variety of understanding and
generation tasks. However, current MLLMs are surprisingly poor at understanding
webpage screenshots and generating their corresponding HTML code. To address
this problem, we propose Web2Code, a benchmark consisting of a new large-scale
webpage-to-code dataset for instruction tuning and an evaluation framework for
the webpage understanding and HTML code translation abilities of MLLMs. For
dataset construction, we leverage pretrained LLMs to enhance existing
webpage-to-code datasets as well as generate a diverse pool of new webpages
rendered into images. Specifically, the inputs are webpage images and
instructions, while the responses are the webpage's HTML code. We further
include diverse natural language QA pairs about the webpage content in the
responses to enable a more comprehensive understanding of the web content. To
evaluate model performance in these tasks, we develop an evaluation framework
for testing MLLMs' abilities in webpage understanding and web-to-code
generation. Extensive experiments show that our proposed dataset is beneficial
not only to our proposed tasks but also in the general visual domain, while
previous datasets result in worse performance. We hope our work will contribute
to the development of general MLLMs suitable for web-based content generation
and task automation. Our data and code will be available at
https://github.com/MBZUAI-LLM/web2code.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.20098) | [💻 Code](https://github.com/MBZUAI-LLM/web2code)

### 45. (HQH) Evaluating the Quality of Hallucination Benchmarks for Large Vision-Language Models
**Date**: 2024.06.24

**Affiliation**: Chinese Academy of Sciences
<details span>
<summary><b>Abstract</b></summary>
Despite the rapid progress and outstanding performance of Large
Vision-Language Models (LVLMs) in recent years, LVLMs have been plagued by the
issue of hallucination, i.e., LVLMs tend to generate responses that are
inconsistent with the corresponding visual inputs. To evaluate the degree of
hallucination in LVLMs, previous works have proposed a series of benchmarks
featuring different types of tasks and evaluation metrics. However, we find
that the quality of the existing hallucination benchmarks varies, with some
suffering from problems, e.g., inconsistent evaluation results under repeated
tests, and misalignment with human evaluation. To this end, we propose a
Hallucination benchmark Quality Measurement framework (HQM), which leverages
various indicators to assess the reliability and validity of existing
hallucination benchmarks separately. Specifically, for reliability we explore
test-retest reliability and parallel-forms reliability, while for validity we
examine criterion validity and coverage of hallucination types. Furthermore,
based on the results of our quality measurement, we construct a High-Quality
Hallucination Benchmark (HQH) for LVLMs. We conduct an extensive evaluation of
over 10 representative LVLMs, including GPT-4o and Gemini-Vision-Pro, to
provide an in-depth analysis of the hallucination issues in existing models.
Our benchmark is publicly available at https://github.com/HQHBench/HQHBench.
</details>

[📄 Paper](https://arxiv.org/abs/2406.17115) | [💻 Code](https://github.com/HQHBench/HQHBench)

### 46. (MIA-Bench) MIA-Bench: Towards Better Instruction Following Evaluation of Multimodal LLMs
**Date**: 2024.07.01

**Affiliation**: Apple
<details span>
<summary><b>Abstract</b></summary>
We introduce MIA-Bench, a new benchmark designed to evaluate multimodal large
language models (MLLMs) on their ability to strictly adhere to complex
instructions. Our benchmark comprises a diverse set of 400 image-prompt pairs,
each crafted to challenge the models' compliance with layered instructions in
generating accurate responses that satisfy specific requested patterns.
Evaluation results from a wide array of state-of-the-art MLLMs reveal
significant variations in performance, highlighting areas for improvement in
instruction fidelity. Additionally, we create extra training data and explore
supervised fine-tuning to enhance the models' ability to strictly follow
instructions without compromising performance on other tasks. We hope this
benchmark not only serves as a tool for measuring MLLM adherence to
instructions, but also guides future developments in MLLM training methods.
</details>

[📄 Paper](https://arxiv.org/pdf/2407.01509)

### 47. (StoryStream) SEED-Story: Multimodal Long Story Generation with Large Language Model
**Date**: 2024.07.08

**Affiliation**: HKUST(GZ)
<details span>
<summary><b>Abstract</b></summary>
With the remarkable advancements in image generation and open-form text
generation, the creation of interleaved image-text content has become an
increasingly intriguing field. Multimodal story generation, characterized by
producing narrative texts and vivid images in an interleaved manner, has
emerged as a valuable and practical task with broad applications. However, this
task poses significant challenges, as it necessitates the comprehension of the
complex interplay between texts and images, and the ability to generate long
sequences of coherent, contextually relevant texts and visuals. In this work,
we propose SEED-Story, a novel method that leverages a Multimodal Large
Language Model (MLLM) to generate extended multimodal stories. Our model, built
upon the powerful comprehension capability of MLLM, predicts text tokens as
well as visual tokens, which are subsequently processed with an adapted visual
de-tokenizer to produce images with consistent characters and styles. We
further propose multimodal attention sink mechanism to enable the generation of
stories with up to 25 sequences (only 10 for training) in a highly efficient
autoregressive manner. Additionally, we present a large-scale and
high-resolution dataset named StoryStream for training our model and
quantitatively evaluating the task of multimodal story generation in various
aspects.
</details>

[📄 Paper](https://arxiv.org/pdf/2407.08683) | [💻 Code](https://github.com/TencentARC/SEED-Story)

### 48. (ROPE) Multi-Object Hallucination in Vision-Language Models
**Date**: 2024.07.08

**Affiliation**: University of Michigan
<details span>
<summary><b>Abstract</b></summary>
Large vision language models (LVLMs) often suffer from object hallucination,
producing objects not present in the given images. While current benchmarks for
object hallucination primarily concentrate on the presence of a single object
class rather than individual entities, this work systematically investigates
multi-object hallucination, examining how models misperceive (e.g., invent
nonexistent objects or become distracted) when tasked with focusing on multiple
objects simultaneously. We introduce Recognition-based Object Probing
Evaluation (ROPE), an automated evaluation protocol that considers the
distribution of object classes within a single image during testing and uses
visual referring prompts to eliminate ambiguity. With comprehensive empirical
studies and analysis of potential factors leading to multi-object
hallucination, we found that (1) LVLMs suffer more hallucinations when focusing
on multiple objects compared to a single object. (2) The tested object class
distribution affects hallucination behaviors, indicating that LVLMs may follow
shortcuts and spurious correlations.(3) Hallucinatory behaviors are influenced
by data-specific factors, salience and frequency, and model intrinsic
behaviors. We hope to enable LVLMs to recognize and reason about multiple
objects that often occur in realistic visual scenes, provide insights, and
quantify our progress towards mitigating the issues.
</details>

[📄 Paper](https://arxiv.org/abs/2407.06192v1) | [🌐 Project Page](https://multi-object-hallucination.github.io/) | [💻 Code](https://github.com/sled-group/moh)

### 49. (BEAF) BEAF: Observing BEfore-AFter Changes to Evaluate Hallucination in Vision-language Models
**Date**: 2024.07.18

**Affiliation**: POSTECH
<details span>
<summary><b>Abstract</b></summary>
Vision language models (VLMs) perceive the world through a combination of a
visual encoder and a large language model (LLM). The visual encoder,
pre-trained on large-scale vision-text datasets, provides zero-shot
generalization to visual data, and the LLM endows its high reasoning ability to
VLMs. It leads VLMs to achieve high performance on wide benchmarks without
fine-tuning, exhibiting zero or few-shot capability. However, recent studies
show that VLMs are vulnerable to hallucination. This undesirable behavior
degrades reliability and credibility, thereby making users unable to fully
trust the output from VLMs. To enhance trustworthiness and better tackle the
hallucination of VLMs, we curate a new evaluation dataset, called the
BEfore-AFter hallucination dataset (BEAF), and introduce new metrics: True
Understanding (TU), IGnorance (IG), StuBbornness (SB), and InDecision (ID).
Unlike prior works that focus only on constructing questions and answers, the
key idea of our benchmark is to manipulate visual scene information by image
editing models and to design the metrics based on scene changes. This allows us
to clearly assess whether VLMs correctly understand a given scene by observing
the ability to perceive changes. We also visualize image-wise object
relationship by virtue of our two-axis view: vision and text. Upon evaluating
VLMs with our dataset, we observed that our metrics reveal different aspects of
VLM hallucination that have not been reported before. Project page:
\url{https://beafbench.github.io/}
</details>

[📄 Paper](https://arxiv.org/html/2407.13442v1) | [🌐 Project Page](https://beafbench.github.io/) | [💻 Code](https://github.com/postech-ami/BEAF?tab=readme-ov-file)

### 50. (HaloQuest) HaloQuest: A Visual Hallucination Dataset for Advancing Multimodal Reasoning
**Date**: 2024.07.22

**Affiliation**: Columbia University
<details span>
<summary><b>Abstract</b></summary>
Hallucination has been a major problem for large language models and remains
a critical challenge when it comes to multimodality in which vision-language
models (VLMs) have to deal with not just textual but also visual inputs.
Despite rapid progress in VLMs, resources for evaluating and addressing
multimodal hallucination are limited and mostly focused on evaluation. This
work introduces HaloQuest, a novel visual question answering dataset that
captures various aspects of multimodal hallucination such as false premises,
insufficient contexts, and visual challenges. A novel idea from HaloQuest is to
leverage synthetic images, apart from real ones, to enable dataset creation at
scale. With over 7.7K examples spanning across a wide variety of categories,
HaloQuest was designed to be both a challenging benchmark for VLMs and a
fine-tuning dataset for advancing multimodal reasoning. Our experiments reveal
that current models struggle with HaloQuest, with all open-source VLMs
achieving below 36% accuracy. On the other hand, fine-tuning on HaloQuest
significantly reduces hallucination rates while preserving performance on
standard reasoning tasks. Our results discover that benchmarking with generated
images is highly correlated (r=0.97) with real images. Last but not least, we
propose a novel Auto-Eval mechanism that is highly correlated with human raters
(r=0.99) for evaluating VLMs. In sum, this work makes concrete strides towards
understanding, evaluating, and mitigating hallucination in VLMs, serving as an
important step towards more reliable multimodal AI systems in the future.
</details>

[📄 Paper](https://arxiv.org/abs/2407.15680) | [💻 Code](https://github.com/google/haloquest)

### 51. (Hallu-PI) Hallu-PI: Evaluating Hallucination in Multi-modal Large Language Models within Perturbed Inputs
**Date**: 2024.08.02

**Affiliation**: Nanjing University
<details span>
<summary><b>Abstract</b></summary>
Multi-modal Large Language Models (MLLMs) have demonstrated remarkable
performance on various visual-language understanding and generation tasks.
However, MLLMs occasionally generate content inconsistent with the given
images, which is known as "hallucination". Prior works primarily center on
evaluating hallucination using standard, unperturbed benchmarks, which overlook
the prevalent occurrence of perturbed inputs in real-world scenarios-such as
image cropping or blurring-that are critical for a comprehensive assessment of
MLLMs' hallucination. In this paper, to bridge this gap, we propose Hallu-PI,
the first benchmark designed to evaluate Hallucination in MLLMs within
Perturbed Inputs. Specifically, Hallu-PI consists of seven perturbed scenarios,
containing 1,260 perturbed images from 11 object types. Each image is
accompanied by detailed annotations, which include fine-grained hallucination
types, such as existence, attribute, and relation. We equip these annotations
with a rich set of questions, making Hallu-PI suitable for both discriminative
and generative tasks. Extensive experiments on 12 mainstream MLLMs, such as
GPT-4V and Gemini-Pro Vision, demonstrate that these models exhibit significant
hallucinations on Hallu-PI, which is not observed in unperturbed scenarios.
Furthermore, our research reveals a severe bias in MLLMs' ability to handle
different types of hallucinations. We also design two baselines specifically
for perturbed scenarios, namely Perturbed-Reminder and Perturbed-ICL. We hope
that our study will bring researchers' attention to the limitations of MLLMs
when dealing with perturbed inputs, and spur further investigations to address
this issue. Our code and datasets are publicly available at
https://github.com/NJUNLP/Hallu-PI.
</details>

[📄 Paper](https://arxiv.org/abs/2408.01355) | [💻 Code](https://github.com/NJUNLP/Hallu-PI)

### 52. (Reefknot) Reefknot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models
**Date**: 2024.08.18

**Affiliation**: HongKong University of Science and Technology(Guangzhou)
<details span>
<summary><b>Abstract</b></summary>
Hallucination issues persistently plagued current multimodal large language
models (MLLMs). While existing research primarily focuses on object-level or
attribute-level hallucinations, sidelining the more sophisticated relation
hallucinations that necessitate advanced reasoning abilities from MLLMs.
Besides, recent benchmarks regarding relation hallucinations lack in-depth
evaluation and effective mitigation. Moreover, their datasets are typically
derived from a systematic annotation process, which could introduce inherent
biases due to the predefined process. To handle the aforementioned challenges,
we introduce Reefknot, a comprehensive benchmark specifically targeting
relation hallucinations, consisting of over 20,000 samples derived from
real-world scenarios. Specifically, we first provide a systematic definition of
relation hallucinations, integrating perspectives from perceptive and cognitive
domains. Furthermore, we construct the relation-based corpus utilizing the
representative scene graph dataset Visual Genome (VG), from which semantic
triplets follow real-world distributions. Our comparative evaluation across
three distinct tasks revealed a substantial shortcoming in the capabilities of
current MLLMs to mitigate relation hallucinations. Finally, we advance a novel
confidence-based mitigation strategy tailored to tackle the relation
hallucinations problem. Across three datasets, including Reefknot, we observed
an average reduction of 9.75% in the hallucination rate. We believe our paper
sheds valuable insights into achieving trustworthy multimodal intelligence. Our
dataset and code will be released upon paper acceptance.
</details>

[📄 Paper](https://arxiv.org/abs/2408.09429)



## Application Benchmarks:
### 1. (MineDojo) MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge
**Date**: 2022.06.17

**Affiliation**: NVIDIA
<details span>
<summary><b>Abstract</b></summary>
Autonomous agents have made great strides in specialist domains like Atari games and Go. However, they typically learn tabula rasa in isolated environments with limited and manually conceived objectives, thus failing to generalize across a wide spectrum of tasks and capabilities. Inspired by how humans continually learn and adapt in the open world, we advocate a trinity of ingredients for building generalist agents: 1) an environment that supports a multitude of tasks and goals, 2) a large-scale database of multimodal knowledge, and 3) a flexible and scalable agent architecture. We introduce MineDojo, a new framework built on the popular Minecraft game that features a simulation suite with thousands of diverse open-ended tasks and an internet-scale knowledge base with Minecraft videos, tutorials, wiki pages, and forum discussions. Using MineDojo's data, we propose a novel agent learning algorithm that leverages large pre-trained video-language models as a learned reward function. Our agent is able to solve a variety of open-ended tasks specified in free-form language without any manually designed dense shaping reward. We open-source the simulation suite, knowledge bases, algorithm implementation, and pretrained models (this https URL) to promote research towards the goal of generally capable embodied agents.
</details>

[📄 Paper](https://arxiv.org/pdf/2206.08853) | [🌐 Project Page](https://minedojo.org/) | [💻 Code](https://github.com/MineDojo/MineDojo)


### 2. (NuScenes-QA) NuScenes-QA: A Multi-modal Visual Question Answering Benchmark for Autonomous Driving Scenario
**Date**: 2023.05.24

**Affiliation**: Fudan University
<details span>
<summary><b>Abstract</b></summary>
We introduce a novel visual question answering (VQA) task in the context of
autonomous driving, aiming to answer natural language questions based on
street-view clues. Compared to traditional VQA tasks, VQA in autonomous driving
scenario presents more challenges. Firstly, the raw visual data are
multi-modal, including images and point clouds captured by camera and LiDAR,
respectively. Secondly, the data are multi-frame due to the continuous,
real-time acquisition. Thirdly, the outdoor scenes exhibit both moving
foreground and static background. Existing VQA benchmarks fail to adequately
address these complexities. To bridge this gap, we propose NuScenes-QA, the
first benchmark for VQA in the autonomous driving scenario, encompassing 34K
visual scenes and 460K question-answer pairs. Specifically, we leverage
existing 3D detection annotations to generate scene graphs and design question
templates manually. Subsequently, the question-answer pairs are generated
programmatically based on these templates. Comprehensive statistics prove that
our NuScenes-QA is a balanced large-scale benchmark with diverse question
formats. Built upon it, we develop a series of baselines that employ advanced
3D detection and VQA techniques. Our extensive experiments highlight the
challenges posed by this new task. Codes and dataset are available at
https://github.com/qiantianwen/NuScenes-QA.
</details>

[📄 Paper](https://arxiv.org/pdf/2305.14836) | [💻 Code](https://github.com/qiantianwen/NuScenes-QA)

### 3. (MIND2WEB) Mind2Web: Towards a Generalist Agent for the Web
**Date**: 2023.06.09

**Affiliation**: The Ohio State University
<details span>
<summary><b>Abstract</b></summary>
We introduce Mind2Web, the first dataset for developing and evaluating
generalist agents for the web that can follow language instructions to complete
complex tasks on any website. Existing datasets for web agents either use
simulated websites or only cover a limited set of websites and tasks, thus not
suitable for generalist web agents. With over 2,000 open-ended tasks collected
from 137 websites spanning 31 domains and crowdsourced action sequences for the
tasks, Mind2Web provides three necessary ingredients for building generalist
web agents: 1) diverse domains, websites, and tasks, 2) use of real-world
websites instead of simulated and simplified ones, and 3) a broad spectrum of
user interaction patterns. Based on Mind2Web, we conduct an initial exploration
of using large language models (LLMs) for building generalist web agents. While
the raw HTML of real-world websites are often too large to be fed to LLMs, we
show that first filtering it with a small LM significantly improves the
effectiveness and efficiency of LLMs. Our solution demonstrates a decent level
of performance, even on websites or entire domains the model has never seen
before, but there is still a substantial room to improve towards truly
generalizable agents. We open-source our dataset, model implementation, and
trained models (https://osu-nlp-group.github.io/Mind2Web) to facilitate further
research on building a generalist agent for the web.
</details>

[📄 Paper](https://arxiv.org/pdf/2306.06070) | [🌐 Project Page](https://osu-nlp-group.github.io/Mind2Web) | [💻 Code](https://github.com/OSU-NLP-Group/Mind2Web)

### 4. (OpenEQA) OpenEQA: Embodied Question Answering in the Era of Foundation Models
**Date**: 2023.06.19

**Affiliation**: Georgia Tech
<details span>
<summary><b>Abstract</b></summary>
We present a modern formulation of Embodied Question Answering (EQA) as the task of understanding an environment well enough to answer questions about it in natural language. An agent can achieve such an understanding by either drawing upon episodic memory, exemplified by agents on smart glasses, or by actively exploring the environment, as in the case of mobile robots. We accompany our formulation with OpenEQA – the first open-vocabulary benchmark dataset for EQA supporting both episodic memory and active exploration use cases. OpenEQA contains over 1600 high-quality human generated questions drawn from over 180 real-world environments. In addition to the dataset, we also provide an automatic LLM-powered evaluation protocol that has excellent correlation with human judgement. Using this dataset and evaluation protocol, we evaluate several state-of-the-art foundation models including GPT-4V, and find that they significantly lag behind human-level performance. Consequently, OpenEQA stands out as a straightforward, measurable, and practically relevant benchmark that poses a considerable challenge to current generation of foundation models. We hope this inspires and stimulates future research at the intersection of Embodied AI, conversational agents, and world models.
</details>

[📄 Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Majumdar_OpenEQA_Embodied_Question_Answering_in_the_Era_of_Foundation_Models_CVPR_2024_paper.pdf) | [🌐 Project Page](https://open-eqa.github.io/) | [💻 Code](https://github.com/facebookresearch/open-eqa)

### 5. (AITW) Android in the Wild: A Large-Scale Dataset for Android Device Control
**Date**: 2023.07.19

**Affiliation**: Google Research
<details span>
<summary><b>Abstract</b></summary>
There is a growing interest in device-control systems that can interpret
human natural language instructions and execute them on a digital device by
directly controlling its user interface. We present a dataset for
device-control research, Android in the Wild (AITW), which is orders of
magnitude larger than current datasets. The dataset contains human
demonstrations of device interactions, including the screens and actions, and
corresponding natural language instructions. It consists of 715k episodes
spanning 30k unique instructions, four versions of Android (v10-13),and eight
device types (Pixel 2 XL to Pixel 6) with varying screen resolutions. It
contains multi-step tasks that require semantic understanding of language and
visual context. This dataset poses a new challenge: actions available through
the user interface must be inferred from their visual appearance. And, instead
of simple UI element-based actions, the action space consists of precise
gestures (e.g., horizontal scrolls to operate carousel widgets). We organize
our dataset to encourage robustness analysis of device-control systems, i.e.,
how well a system performs in the presence of new task descriptions, new
applications, or new platform versions. We develop two agents and report
performance across the dataset. The dataset is available at
https://github.com/google-research/google-research/tree/master/android_in_the_wild.
</details>

[📄 Paper](https://arxiv.org/abs/2307.10088) | [💻 Code](https://github.com/google-research/google-research/tree/master/android_in_the_wild)

### 6. (WebArena) WebArena: A Realistic Web Environment for Building Autonomous Agents
**Date**: 2023.07.25

**Affiliation**: Carnegie Mellon University
<details span>
<summary><b>Abstract</b></summary>
With advances in generative AI, there is now potential for autonomous agents
to manage daily tasks via natural language commands. However, current agents
are primarily created and tested in simplified synthetic environments, leading
to a disconnect with real-world scenarios. In this paper, we build an
environment for language-guided agents that is highly realistic and
reproducible. Specifically, we focus on agents that perform tasks on the web,
and create an environment with fully functional websites from four common
domains: e-commerce, social forum discussions, collaborative software
development, and content management. Our environment is enriched with tools
(e.g., a map) and external knowledge bases (e.g., user manuals) to encourage
human-like task-solving. Building upon our environment, we release a set of
benchmark tasks focusing on evaluating the functional correctness of task
completions. The tasks in our benchmark are diverse, long-horizon, and designed
to emulate tasks that humans routinely perform on the internet. We experiment
with several baseline agents, integrating recent techniques such as reasoning
before acting. The results demonstrate that solving complex tasks is
challenging: our best GPT-4-based agent only achieves an end-to-end task
success rate of 14.41%, significantly lower than the human performance of
78.24%. These results highlight the need for further development of robust
agents, that current state-of-the-art large language models are far from
perfect performance in these real-life tasks, and that WebArena can be used to
measure such progress.
</details>

[📄 Paper](https://arxiv.org/abs/2307.13854) | [🌐 Project Page](https://webarena.dev/) | [💻 Code](https://github.com/web-arena-x/webarena)

### 7. (PCA-EVAL) Towards End-to-End Embodied Decision Making via Multi-modal Large Language Model: Explorations with GPT4-Vision and Beyond
**Date**: 2023.10.03

**Affiliation**: Peking University
<details span>
<summary><b>Abstract</b></summary>
In this study, we explore the potential of Multimodal Large Language Models
(MLLMs) in improving embodied decision-making processes for agents. While Large
Language Models (LLMs) have been widely used due to their advanced reasoning
skills and vast world knowledge, MLLMs like GPT4-Vision offer enhanced visual
understanding and reasoning capabilities. We investigate whether
state-of-the-art MLLMs can handle embodied decision-making in an end-to-end
manner and whether collaborations between LLMs and MLLMs can enhance
decision-making. To address these questions, we introduce a new benchmark
called PCA-EVAL, which evaluates embodied decision-making from the perspectives
of Perception, Cognition, and Action. Additionally, we propose HOLMES, a
multi-agent cooperation framework that allows LLMs to leverage MLLMs and APIs
to gather multimodal information for informed decision-making. We compare
end-to-end embodied decision-making and HOLMES on our benchmark and find that
the GPT4-Vision model demonstrates strong end-to-end embodied decision-making
abilities, outperforming GPT4-HOLMES in terms of average decision accuracy
(+3%). However, this performance is exclusive to the latest GPT4-Vision model,
surpassing the open-source state-of-the-art MLLM by 26%. Our results indicate
that powerful MLLMs like GPT4-Vision hold promise for decision-making in
embodied agents, offering new avenues for MLLM research. Code and data are open
at https://github.com/pkunlp-icler/PCA-EVAL/.
</details>

[📄 Paper](https://arxiv.org/pdf/2310.02071) | [💻 Code](https://github.com/pkunlp-icler/PCA-EVAL/)

### 8. (RoboVQA) RoboVQA: Multimodal Long-Horizon Reasoning for Robotics
**Date**: 2023.11.01

**Affiliation**: Google DeepMind
<details span>
<summary><b>Abstract</b></summary>
We present a scalable, bottom-up and intrinsically diverse data collection
scheme that can be used for high-level reasoning with long and medium horizons
and that has 2.2x higher throughput compared to traditional narrow top-down
step-by-step collection. We collect realistic data by performing any user
requests within the entirety of 3 office buildings and using multiple robot and
human embodiments. With this data, we show that models trained on all
embodiments perform better than ones trained on the robot data only, even when
evaluated solely on robot episodes. We find that for a fixed collection budget
it is beneficial to take advantage of cheaper human collection along with robot
collection. We release a large and highly diverse (29,520 unique instructions)
dataset dubbed RoboVQA containing 829,502 (video, text) pairs for
robotics-focused visual question answering. We also demonstrate how evaluating
real robot experiments with an intervention mechanism enables performing tasks
to completion, making it deployable with human oversight even if imperfect
while also providing a single performance metric. We demonstrate a single
video-conditioned model named RoboVQA-VideoCoCa trained on our dataset that is
capable of performing a variety of grounded high-level reasoning tasks in broad
realistic settings with a cognitive intervention rate 46% lower than the
zero-shot state of the art visual language model (VLM) baseline and is able to
guide real robots through long-horizon tasks. The performance gap with
zero-shot state-of-the-art models indicates that a lot of grounded data remains
to be collected for real-world deployment, emphasizing the critical need for
scalable data collection approaches. Finally, we show that video VLMs
significantly outperform single-image VLMs with an average error rate reduction
of 19% across all VQA tasks. Data and videos available at
https://robovqa.github.io
</details>

[📄 Paper](https://arxiv.org/abs/2311.00899) | [🌐 Project Page](https://robovqa.github.io) | [💻 Code](https://github.com/google-deepmind/robovqa/tree/main)

### 9. (EgoPlan-Bench) EgoPlan-Bench: Benchmarking Multimodal Large Language Models for Human-Level Planning
**Date**: 2023.12.11

**Affiliation**: Tencent AI Lab
<details span>
<summary><b>Abstract</b></summary>
The pursuit of artificial general intelligence (AGI) has been accelerated by
Multimodal Large Language Models (MLLMs), which exhibit superior reasoning,
generalization capabilities, and proficiency in processing multimodal inputs. A
crucial milestone in the evolution of AGI is the attainment of human-level
planning, a fundamental ability for making informed decisions in complex
environments, and solving a wide range of real-world problems. Despite the
impressive advancements in MLLMs, a question remains: How far are current MLLMs
from achieving human-level planning? To shed light on this question, we
introduce EgoPlan-Bench, a comprehensive benchmark to evaluate the planning
abilities of MLLMs in real-world scenarios from an egocentric perspective,
mirroring human perception. EgoPlan-Bench emphasizes the evaluation of planning
capabilities of MLLMs, featuring realistic tasks, diverse action plans, and
intricate visual observations. Our rigorous evaluation of a wide range of MLLMs
reveals that EgoPlan-Bench poses significant challenges, highlighting a
substantial scope for improvement in MLLMs to achieve human-level task
planning. To facilitate this advancement, we further present EgoPlan-IT, a
specialized instruction-tuning dataset that effectively enhances model
performance on EgoPlan-Bench. We have made all codes, data, and a maintained
benchmark leaderboard available to advance future research.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.06722) | [🌐 Project Page](https://chenyi99.github.io/ego_plan/) | [💻 Code](https://github.com/ChenYi99/EgoPlan)

### 10. (DriveLM-DATA) DriveLM: Driving with Graph Visual Question Answering
**Date**: 2023.12.21

**Affiliation**: Shanghai AI Lab
<details span>
<summary><b>Abstract</b></summary>
We study how vision-language models (VLMs) trained on web-scale data can be
integrated into end-to-end driving systems to boost generalization and enable
interactivity with human users. While recent approaches adapt VLMs to driving
via single-round visual question answering (VQA), human drivers reason about
decisions in multiple steps. Starting from the localization of key objects,
humans estimate object interactions before taking actions. The key insight is
that with our proposed task, Graph VQA, where we model graph-structured
reasoning through perception, prediction and planning question-answer pairs, we
obtain a suitable proxy task to mimic the human reasoning process. We
instantiate datasets (DriveLM-Data) built upon nuScenes and CARLA, and propose
a VLM-based baseline approach (DriveLM-Agent) for jointly performing Graph VQA
and end-to-end driving. The experiments demonstrate that Graph VQA provides a
simple, principled framework for reasoning about a driving scene, and
DriveLM-Data provides a challenging benchmark for this task. Our DriveLM-Agent
baseline performs end-to-end autonomous driving competitively in comparison to
state-of-the-art driving-specific architectures. Notably, its benefits are
pronounced when it is evaluated zero-shot on unseen objects or sensor
configurations. We hope this work can be the starting point to shed new light
on how to apply VLMs for autonomous driving. To facilitate future research, all
code, data, and models are available to the public.
</details>

[📄 Paper](https://arxiv.org/abs/2312.14150) | [💻 Code](https://github.com/OpenDriveLab/DriveLM)

### 11. (TransportationGames) TransportationGames: Benchmarking Transportation Knowledge of (Multimodal) Large Language Models
**Date**: 2024.01.09

**Affiliation**: Beijing Jiaotong University
<details span>
<summary><b>Abstract</b></summary>
Large language models (LLMs) and multimodal large language models (MLLMs)
have shown excellent general capabilities, even exhibiting adaptability in many
professional domains such as law, economics, transportation, and medicine.
Currently, many domain-specific benchmarks have been proposed to verify the
performance of (M)LLMs in specific fields. Among various domains,
transportation plays a crucial role in modern society as it impacts the
economy, the environment, and the quality of life for billions of people.
However, it is unclear how much traffic knowledge (M)LLMs possess and whether
they can reliably perform transportation-related tasks. To address this gap, we
propose TransportationGames, a carefully designed and thorough evaluation
benchmark for assessing (M)LLMs in the transportation domain. By
comprehensively considering the applications in real-world scenarios and
referring to the first three levels in Bloom's Taxonomy, we test the
performance of various (M)LLMs in memorizing, understanding, and applying
transportation knowledge by the selected tasks. The experimental results show
that although some models perform well in some tasks, there is still much room
for improvement overall. We hope the release of TransportationGames can serve
as a foundation for future research, thereby accelerating the implementation
and application of (M)LLMs in the transportation domain.
</details>

[📄 Paper](https://arxiv.org/pdf/2401.04471) | [🌐 Project Page](https://transportation.games/#/HomePage)

### 12. (VisualWebArena) VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks
**Date**: 2024.01.24

**Affiliation**: Carnegie Mellon University
<details span>
<summary><b>Abstract</b></summary>
Autonomous agents capable of planning, reasoning, and executing actions on
the web offer a promising avenue for automating computer tasks. However, the
majority of existing benchmarks primarily focus on text-based agents,
neglecting many natural tasks that require visual information to effectively
solve. Given that most computer interfaces cater to human perception, visual
information often augments textual data in ways that text-only models struggle
to harness effectively. To bridge this gap, we introduce VisualWebArena, a
benchmark designed to assess the performance of multimodal web agents on
realistic \textit{visually grounded tasks}. VisualWebArena comprises of a set
of diverse and complex web-based tasks that evaluate various capabilities of
autonomous multimodal agents. To perform on this benchmark, agents need to
accurately process image-text inputs, interpret natural language instructions,
and execute actions on websites to accomplish user-defined objectives. We
conduct an extensive evaluation of state-of-the-art LLM-based autonomous
agents, including several multimodal models. Through extensive quantitative and
qualitative analysis, we identify several limitations of text-only LLM agents,
and reveal gaps in the capabilities of state-of-the-art multimodal language
agents. VisualWebArena provides a framework for evaluating multimodal
autonomous language agents, and offers insights towards building stronger
autonomous agents for the web. Our code, baseline models, and data is publicly
available at https://jykoh.com/vwa.
</details>

[📄 Paper](https://arxiv.org/abs/2401.13649) | [🌐 Project Page](https://jykoh.com/vwa) | [💻 Code](https://github.com/web-arena-x/visualwebarena)

### 13. (Mobile-Eval) Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception
**Date**: 2024.01.29

**Affiliation**: Beijing Jiaotong University
<details span>
<summary><b>Abstract</b></summary>
Mobile device agent based on Multimodal Large Language Models (MLLM) is
becoming a popular application. In this paper, we introduce Mobile-Agent, an
autonomous multi-modal mobile device agent. Mobile-Agent first leverages visual
perception tools to accurately identify and locate both the visual and textual
elements within the app's front-end interface. Based on the perceived vision
context, it then autonomously plans and decomposes the complex operation task,
and navigates the mobile Apps through operations step by step. Different from
previous solutions that rely on XML files of Apps or mobile system metadata,
Mobile-Agent allows for greater adaptability across diverse mobile operating
environments in a vision-centric way, thereby eliminating the necessity for
system-specific customizations. To assess the performance of Mobile-Agent, we
introduced Mobile-Eval, a benchmark for evaluating mobile device operations.
Based on Mobile-Eval, we conducted a comprehensive evaluation of Mobile-Agent.
The experimental results indicate that Mobile-Agent achieved remarkable
accuracy and completion rates. Even with challenging instructions, such as
multi-app operations, Mobile-Agent can still complete the requirements. Code
and model will be open-sourced at https://github.com/X-PLUG/MobileAgent.
</details>

[📄 Paper](https://arxiv.org/pdf/2401.16158) | [💻 Code](https://github.com/X-PLUG/MobileAgent)

### 14. (LHRS-Bench) LHRS-Bot: Empowering Remote Sensing with VGI-Enhanced Large Multimodal Language Model
**Date**: 2024.02.04

**Affiliation**: Nanjing University
<details span>
<summary><b>Abstract</b></summary>
The revolutionary capabilities of large language models (LLMs) have paved the
way for multimodal large language models (MLLMs) and fostered diverse
applications across various specialized domains. In the remote sensing (RS)
field, however, the diverse geographical landscapes and varied objects in RS
imagery are not adequately considered in recent MLLM endeavors. To bridge this
gap, we construct a large-scale RS image-text dataset, LHRS-Align, and an
informative RS-specific instruction dataset, LHRS-Instruct, leveraging the
extensive volunteered geographic information (VGI) and globally available RS
images. Building on this foundation, we introduce LHRS-Bot, an MLLM tailored
for RS image understanding through a novel multi-level vision-language
alignment strategy and a curriculum learning method. Additionally, we introduce
LHRS-Bench, a benchmark for thoroughly evaluating MLLMs' abilities in RS image
understanding. Comprehensive experiments demonstrate that LHRS-Bot exhibits a
profound understanding of RS images and the ability to perform nuanced
reasoning within the RS domain.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.02544) | [💻 Code](https://github.com/NJU-LHRS/LHRS-Bot)

### 15. (Asclepius) Asclepius: A Spectrum Evaluation Benchmark for Medical Multi-Modal Large Language Models
**Date**: 2024.02.17

**Affiliation**: The Chinese University of Hong Kong
<details span>
<summary><b>Abstract</b></summary>
The significant breakthroughs of Medical Multi-Modal Large Language Models
(Med-MLLMs) renovate modern healthcare with robust information synthesis and
medical decision support. However, these models are often evaluated on
benchmarks that are unsuitable for the Med-MLLMs due to the intricate nature of
the real-world diagnostic frameworks, which encompass diverse medical
specialties and involve complex clinical decisions. Moreover, these benchmarks
are susceptible to data leakage, since Med-MLLMs are trained on large
assemblies of publicly available data. Thus, an isolated and clinically
representative benchmark is highly desirable for credible Med-MLLMs evaluation.
To this end, we introduce Asclepius, a novel Med-MLLM benchmark that rigorously
and comprehensively assesses model capability in terms of: distinct medical
specialties (cardiovascular, gastroenterology, etc.) and different diagnostic
capacities (perception, disease analysis, etc.). Grounded in 3 proposed core
principles, Asclepius ensures a comprehensive evaluation by encompassing 15
medical specialties, stratifying into 3 main categories and 8 sub-categories of
clinical tasks, and exempting from train-validate contamination. We further
provide an in-depth analysis of 6 Med-MLLMs and compare them with 5 human
specialists, providing insights into their competencies and limitations in
various medical contexts. Our work not only advances the understanding of
Med-MLLMs' capabilities but also sets a precedent for future evaluations and
the safe deployment of these models in clinical environments. We launch and
maintain a leaderboard for community assessment of Med-MLLM capabilities
(https://asclepius-med.github.io/).
</details>

[📄 Paper](https://arxiv.org/pdf/2402.11217) | [🌐 Project Page](https://asclepius-med.github.io/)

### 16. (MM-Soc) MM-Soc: Benchmarking Multimodal Large Language Models in Social Media Platforms
**Date**: 2024.02.21

**Affiliation**: Georgia Institute of Technology
<details span>
<summary><b>Abstract</b></summary>
Social media platforms are hubs for multimodal information exchange,
encompassing text, images, and videos, making it challenging for machines to
comprehend the information or emotions associated with interactions in online
spaces. Multimodal Large Language Models (MLLMs) have emerged as a promising
solution to these challenges, yet they struggle to accurately interpret human
emotions and complex content such as misinformation. This paper introduces
MM-Soc, a comprehensive benchmark designed to evaluate MLLMs' understanding of
multimodal social media content. MM-Soc compiles prominent multimodal datasets
and incorporates a novel large-scale YouTube tagging dataset, targeting a range
of tasks from misinformation detection, hate speech detection, and social
context generation. Through our exhaustive evaluation on ten size-variants of
four open-source MLLMs, we have identified significant performance disparities,
highlighting the need for advancements in models' social understanding
capabilities. Our analysis reveals that, in a zero-shot setting, various types
of MLLMs generally exhibit difficulties in handling social media tasks.
However, MLLMs demonstrate performance improvements post fine-tuning,
suggesting potential pathways for improvement. Our code and data are available
at https://github.com/claws-lab/MMSoc.git.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.14154) | [💻 Code](https://github.com/claws-lab/MMSoc)

### 17. (M3D-Bench) M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models
**Date**: 2024.03.31

**Affiliation**: Beijing Academy of Artificial Intelligence
<details span>
<summary><b>Abstract</b></summary>
Medical image analysis is essential to clinical diagnosis and treatment,
which is increasingly supported by multi-modal large language models (MLLMs).
However, previous research has primarily focused on 2D medical images, leaving
3D images under-explored, despite their richer spatial information. This paper
aims to advance 3D medical image analysis with MLLMs. To this end, we present a
large-scale 3D multi-modal medical dataset, M3D-Data, comprising 120K
image-text pairs and 662K instruction-response pairs specifically tailored for
various 3D medical tasks, such as image-text retrieval, report generation,
visual question answering, positioning, and segmentation. Additionally, we
propose M3D-LaMed, a versatile multi-modal large language model for 3D medical
image analysis. Furthermore, we introduce a new 3D multi-modal medical
benchmark, M3D-Bench, which facilitates automatic evaluation across eight
tasks. Through comprehensive evaluation, our method proves to be a robust model
for 3D medical image analysis, outperforming existing solutions. All code,
data, and models are publicly available at: https://github.com/BAAI-DCAI/M3D.
</details>

[📄 Paper](https://arxiv.org/abs/2404.00578) | [💻 Code](https://github.com/BAAI-DCAI/M3D)

### 18. (Ferret-UI) Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs
**Date**: 2024.04.09

**Affiliation**: Apple
<details span>
<summary><b>Abstract</b></summary>
Recent advancements in multimodal large language models (MLLMs) have been
noteworthy, yet, these general-domain MLLMs often fall short in their ability
to comprehend and interact effectively with user interface (UI) screens. In
this paper, we present Ferret-UI, a new MLLM tailored for enhanced
understanding of mobile UI screens, equipped with referring, grounding, and
reasoning capabilities. Given that UI screens typically exhibit a more
elongated aspect ratio and contain smaller objects of interest (e.g., icons,
texts) than natural images, we incorporate "any resolution" on top of Ferret to
magnify details and leverage enhanced visual features. Specifically, each
screen is divided into 2 sub-images based on the original aspect ratio (i.e.,
horizontal division for portrait screens and vertical division for landscape
screens). Both sub-images are encoded separately before being sent to LLMs. We
meticulously gather training samples from an extensive range of elementary UI
tasks, such as icon recognition, find text, and widget listing. These samples
are formatted for instruction-following with region annotations to facilitate
precise referring and grounding. To augment the model's reasoning ability, we
further compile a dataset for advanced tasks, including detailed description,
perception/interaction conversations, and function inference. After training on
the curated datasets, Ferret-UI exhibits outstanding comprehension of UI
screens and the capability to execute open-ended instructions. For model
evaluation, we establish a comprehensive benchmark encompassing all the
aforementioned tasks. Ferret-UI excels not only beyond most open-source UI
MLLMs, but also surpasses GPT-4V on all the elementary UI tasks.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.05719)

### 19. (DesignProbe) DesignProbe: A Graphic Design Benchmark for Multimodal Large Language Models
**Date**: 2024.04.11

**Affiliation**: Massachusetts Institute of Technology
<details span>
<summary><b>Abstract</b></summary>
A well-executed graphic design typically achieves harmony in two levels, from
the fine-grained design elements (color, font and layout) to the overall
design. This complexity makes the comprehension of graphic design challenging,
for it needs the capability to both recognize the design elements and
understand the design. With the rapid development of Multimodal Large Language
Models (MLLMs), we establish the DesignProbe, a benchmark to investigate the
capability of MLLMs in design. Our benchmark includes eight tasks in total,
across both the fine-grained element level and the overall design level. At
design element level, we consider both the attribute recognition and semantic
understanding tasks. At overall design level, we include style and metaphor. 9
MLLMs are tested and we apply GPT-4 as evaluator. Besides, further experiments
indicates that refining prompts can enhance the performance of MLLMs. We first
rewrite the prompts by different LLMs and found increased performances appear
in those who self-refined by their own LLMs. We then add extra task knowledge
in two different ways (text descriptions and image examples), finding that
adding images boost much more performance over texts.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.14801)

### 20. (DesignQA) DesignQA: A Multimodal Benchmark for Evaluating Large Language Models' Understanding of Engineering Documentation
**Date**: 2024.04.11

**Affiliation**: Massachusetts Institute of Technology
<details span>
<summary><b>Abstract</b></summary>
This research introduces DesignQA, a novel benchmark aimed at evaluating the
proficiency of multimodal large language models (MLLMs) in comprehending and
applying engineering requirements in technical documentation. Developed with a
focus on real-world engineering challenges, DesignQA uniquely combines
multimodal data-including textual design requirements, CAD images, and
engineering drawings-derived from the Formula SAE student competition.
Different from many existing MLLM benchmarks, DesignQA contains
document-grounded visual questions where the input image and input document
come from different sources. The benchmark features automatic evaluation
metrics and is divided into segments-Rule Comprehension, Rule Compliance, and
Rule Extraction-based on tasks that engineers perform when designing according
to requirements. We evaluate state-of-the-art models (at the time of writing)
like GPT-4o, GPT-4, Claude-Opus, Gemini-1.0, and LLaVA-1.5 against the
benchmark, and our study uncovers the existing gaps in MLLMs' abilities to
interpret complex engineering documentation. The MLLMs tested, while promising,
struggle to reliably retrieve relevant rules from the Formula SAE
documentation, face challenges in recognizing technical components in CAD
images, and encounter difficulty in analyzing engineering drawings. These
findings underscore the need for multimodal models that can better handle the
multifaceted questions characteristic of design according to technical
documentation. This benchmark sets a foundation for future advancements in
AI-supported engineering design processes. DesignQA is publicly available at:
https://github.com/anniedoris/design_qa/.
</details>

[📄 Paper](https://arxiv.org/abs/2404.07917) | [💻 Code](https://github.com/anniedoris/design_qa/)

### 21. (QB-Poster) PosterLLaVa: Constructing a Unified Multi-modal Layout Generator with LLM
**Date**: 2024.06.02

**Affiliation**: Hong Kong Polytechnic University
<details span>
<summary><b>Abstract</b></summary>
Layout generation is the keystone in achieving automated graphic design,
requiring arranging the position and size of various multi-modal design
elements in a visually pleasing and constraint-following manner. Previous
approaches are either inefficient for large-scale applications or lack
flexibility for varying design requirements. Our research introduces a unified
framework for automated graphic layout generation, leveraging the multi-modal
large language model (MLLM) to accommodate diverse design tasks. In contrast,
our data-driven method employs structured text (JSON format) and visual
instruction tuning to generate layouts under specific visual and textual
constraints, including user-defined natural language specifications. We
conducted extensive experiments and achieved state-of-the-art (SOTA)
performance on public multi-modal layout generation benchmarks, demonstrating
the effectiveness of our method. Moreover, recognizing existing datasets'
limitations in capturing the complexity of real-world graphic designs, we
propose two new datasets for much more challenging tasks (user-constrained
generation and complicated poster), further validating our model's utility in
real-life settings. Marking by its superior accessibility and adaptability,
this approach further automates large-scale graphic design tasks. The code and
datasets will be publicly available on
https://github.com/posterllava/PosterLLaVA.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.02884) | [💻 Code](https://github.com/posterllava/PosterLLaVA)

### 22. (MMRo) MMRo: Are Multimodal LLMs Eligible as the Brain for In-Home Robotics?
**Date**: 2024.06.19

**Affiliation**: Midea group
<details span>
<summary><b>Abstract</b></summary>
It is fundamentally challenging for robots to serve as useful assistants in
human environments because this requires addressing a spectrum of sub-problems
across robotics, including perception, language understanding, reasoning, and
planning. The recent advancements in Multimodal Large Language Models (MLLMs)
have demonstrated their exceptional abilities in solving complex mathematical
problems, mastering commonsense and abstract reasoning. This has led to the
recent utilization of MLLMs as the brain in robotic systems, enabling these
models to conduct high-level planning prior to triggering low-level control
actions for task execution. However, it remains uncertain whether existing
MLLMs are reliable in serving the brain role of robots. In this study, we
introduce the first benchmark for evaluating Multimodal LLM for Robotic (MMRo)
benchmark, which tests the capability of MLLMs for robot applications.
Specifically, we identify four essential capabilities perception, task
planning, visual reasoning, and safety measurement that MLLMs must possess to
qualify as the robot's central processing unit. We have developed several
scenarios for each capability, resulting in a total of 14 metrics for
evaluation. We present experimental results for various MLLMs, including both
commercial and open-source models, to assess the performance of existing
systems. Our findings indicate that no single model excels in all areas,
suggesting that current MLLMs are not yet trustworthy enough to serve as the
cognitive core for robots. Our data can be found in
https://mm-robobench.github.io/.
</details>

[📄 Paper](https://arxiv.org/pdf/2406.19693) | [🌐 Project Page](https://mm-robobench.github.io/)

### 23. (PubMedVision) HuatuoGPT-Vision, Towards Injecting Medical Visual Knowledge into Multimodal LLMs at Scale
**Date**: 2024.06.19

**Affiliation**: Shenzhen Research Institute of Big Data
<details span>
<summary><b>Abstract</b></summary>
The rapid development of multimodal large language models (MLLMs), such as
GPT-4V, has led to significant advancements. However, these models still face
challenges in medical multimodal capabilities due to limitations in the
quantity and quality of medical vision-text data, stemming from data privacy
concerns and high annotation costs. While pioneering approaches utilize
PubMed's large-scale, de-identified medical image-text pairs to address these
limitations, they still fall short due to inherent data noise. To tackle this,
we refined medical image-text pairs from PubMed and employed MLLMs (GPT-4V) in
an 'unblinded' capacity to denoise and reformat the data, resulting in the
creation of the PubMedVision dataset with 1.3 million medical VQA samples. Our
validation demonstrates that: (1) PubMedVision can significantly enhance the
medical multimodal capabilities of current MLLMs, showing significant
improvement in benchmarks including the MMMU Health & Medicine track; (2)
manual checks by medical experts and empirical results validate the superior
data quality of our dataset compared to other data construction methods. Using
PubMedVision, we train a 34B medical MLLM HuatuoGPT-Vision, which shows
superior performance in medical multimodal scenarios among open-source MLLMs.
</details>

[📄 Paper](https://arxiv.org/abs/2406.19280) | [💻 Code](https://github.com/FreedomIntelligence/HuatuoGPT-Vision)

### 24. (SPR) Read Anywhere Pointed: Layout-aware GUI Screen Reading with Tree-of-Lens Grounding
**Date**: 2024.06.27

**Affiliation**: University of California
<details span>
<summary><b>Abstract</b></summary>
Graphical User Interfaces (GUIs) are central to our interaction with digital
devices. Recently, growing efforts have been made to build models for various
GUI understanding tasks. However, these efforts largely overlook an important
GUI-referring task: screen reading based on user-indicated points, which we
name the Screen Point-and-Read (SPR) task. This task is predominantly handled
by rigid accessible screen reading tools, in great need of new models driven by
advancements in Multimodal Large Language Models (MLLMs). In this paper, we
propose a Tree-of-Lens (ToL) agent, utilizing a novel ToL grounding mechanism,
to address the SPR task. Based on the input point coordinate and the
corresponding GUI screenshot, our ToL agent constructs a Hierarchical Layout
Tree. Based on the tree, our ToL agent not only comprehends the content of the
indicated area but also articulates the layout and spatial relationships
between elements. Such layout information is crucial for accurately
interpreting information on the screen, distinguishing our ToL agent from other
screen reading tools. We also thoroughly evaluate the ToL agent against other
baselines on a newly proposed SPR benchmark, which includes GUIs from mobile,
web, and operating systems. Last but not least, we test the ToL agent on mobile
GUI navigation tasks, demonstrating its utility in identifying incorrect
actions along the path of agent execution trajectories. Code and data:
screen-point-and-read.github.io
</details>

[📄 Paper](https://arxiv.org/abs/2406.19263) | [🌐 Project Page](screen-point-and-read.github.io) | [💻 Code](https://github.com/eric-ai-lab/Screen-Point-and-Read)
 
### 25. (CRAB) CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents
**Date**: 2024.07.02

**Affiliation**: KAUST
<details span>
<summary><b>Abstract</b></summary>
The development of autonomous agents increasingly relies on Multimodal
Language Models (MLMs) to perform tasks described in natural language with GUI
environments, such as websites, desktop computers, or mobile phones. Existing
benchmarks for MLM agents in interactive environments are limited by their
focus on a single environment, lack of detailed and generalized evaluation
methods, and the complexities of constructing tasks and evaluators. To overcome
these limitations, we introduce Crab, the first agent benchmark framework
designed to support cross-environment tasks, incorporating a graph-based
fine-grained evaluation method and an efficient mechanism for task and
evaluator construction. Our framework supports multiple devices and can be
easily extended to any environment with a Python interface. Leveraging Crab, we
developed a cross-platform Crab Benchmark-v0 comprising 100 tasks in computer
desktop and mobile phone environments. We evaluated four advanced MLMs using
different single and multi-agent system configurations on this benchmark. The
experimental results demonstrate that the single agent with GPT-4o achieves the
best completion ratio of 35.26%. All framework code, agent code, and task
datasets are publicly available at https://github.com/camel-ai/crab.
</details>

[📄 Paper](https://arxiv.org/pdf/2407.01511) | [💻 Code](https://github.com/camel-ai/crab)

### 26. (GMAI-MMBench) GMAI-MMBench: A Comprehensive Multimodal Evaluation Benchmark Towards General Medical AI
**Date**: 2024.08.06

**Affiliation**: Shanghai AI Laboratory
<details span>
<summary><b>Abstract</b></summary>
Large Vision-Language Models (LVLMs) are capable of handling diverse data
types such as imaging, text, and physiological signals, and can be applied in
various fields. In the medical field, LVLMs have a high potential to offer
substantial assistance for diagnosis and treatment. Before that, it is crucial
to develop benchmarks to evaluate LVLMs' effectiveness in various medical
applications. Current benchmarks are often built upon specific academic
literature, mainly focusing on a single domain, and lacking varying perceptual
granularities. Thus, they face specific challenges, including limited clinical
relevance, incomplete evaluations, and insufficient guidance for interactive
LVLMs. To address these limitations, we developed the GMAI-MMBench, the most
comprehensive general medical AI benchmark with well-categorized data structure
and multi-perceptual granularity to date. It is constructed from 284 datasets
across 38 medical image modalities, 18 clinical-related tasks, 18 departments,
and 4 perceptual granularities in a Visual Question Answering (VQA) format.
Additionally, we implemented a lexical tree structure that allows users to
customize evaluation tasks, accommodating various assessment needs and
substantially supporting medical AI research and applications. We evaluated 50
LVLMs, and the results show that even the advanced GPT-4o only achieves an
accuracy of 53.96%, indicating significant room for improvement. Moreover, we
identified five key insufficiencies in current cutting-edge LVLMs that need to
be addressed to advance the development of better medical applications. We
believe that GMAI-MMBench will stimulate the community to build the next
generation of LVLMs toward GMAI.
</details>

[📄 Paper](https://arxiv.org/pdf/2408.03361) | [🌐 Project Page](https://uni-medical.github.io/GMAI-MMBench.github.io/) | [💻 Code](https://github.com/uni-medical/GMAI-MMBench)

### 27. (VisualAgentBench) VisualAgentBench: Towards Large Multimodal Models as Visual Foundation Agents
**Date**: 2024.08.13

**Affiliation**: Tsinghua University
<details span>
<summary><b>Abstract</b></summary>
Large Multimodal Models (LMMs) have ushered in a new era in artificial
intelligence, merging capabilities in both language and vision to form highly
capable Visual Foundation Agents. These agents are postulated to excel across a
myriad of tasks, potentially approaching general artificial intelligence.
However, existing benchmarks fail to sufficiently challenge or showcase the
full potential of LMMs in complex, real-world environments. To address this
gap, we introduce VisualAgentBench (VAB), a comprehensive and pioneering
benchmark specifically designed to train and evaluate LMMs as visual foundation
agents across diverse scenarios, including Embodied, Graphical User Interface,
and Visual Design, with tasks formulated to probe the depth of LMMs'
understanding and interaction capabilities. Through rigorous testing across
nine proprietary LMM APIs and eight open models, we demonstrate the
considerable yet still developing agent capabilities of these models.
Additionally, VAB constructs a trajectory training set constructed through
hybrid methods including Program-based Solvers, LMM Agent Bootstrapping, and
Human Demonstrations, promoting substantial performance improvements in LMMs
through behavior cloning. Our work not only aims to benchmark existing models
but also provides a solid foundation for future development into visual
foundation agents. Code, train \& test data, and part of fine-tuned open LMMs
are available at \url{https://github.com/THUDM/VisualAgentBench}.
</details>

[📄 Paper](https://arxiv.org/pdf/2408.06327v1) | [💻 Code](https://github.com/THUDM/VisualAgentBench)
