# Hi-CoT: Hierarchical Chain-of-Thought

[**Hierarchical Chain-of-Thought: Enhancing LLM Reasoning Performance and Efficiency**](https://arxiv.org/pdf/2604.00130)

## Overview

Hi-CoT is a structured reasoning prompting paradigm that improves large language model (LLM) performance on complex, multi-step mathematical reasoning tasks. Unlike conventional Chain-of-Thought (CoT) prompting, which produces flat, unstructured reasoning chains, Hi-CoT decomposes reasoning into hierarchical substeps by alternating between **instructional planning** (`<|instruction|>`) and **step-by-step execution** (`<|execution|>`).

This repository provides evaluation code for benchmarking Hi-CoT and baseline prompting strategies across standard mathematical reasoning benchmarks.


---

## Prompting Strategies

The following prompting strategies are supported via the `--template` flag:

| Template | Description |
|---|---|
| `hicot` | Hierarchical CoT with strict `<\|instruction\|>` / `<\|execution\|>` structure |
| `hicot_wo_structure` | Hierarchical CoT without enforced step labeling |
| `cot` | Standard Chain-of-Thought |
| `ps` | Plan-and-Solve prompting |
| `standard` | Direct answer, no reasoning |

---

## Supported Models

- `Qwen/Qwen3-*` family
- `deepseek-ai/DeepSeek-R1-Distill-*` family

---

## Evaluation Benchmarks

- AIME24
- AMC
- MATH500
- Minerva
- OlympiadBench

---

## Installation

```bash
pip install vllm transformers fire numpy datasets
```

Ensure your evaluation data is available at `./data/evaluation_suite` (a Hugging Face `DatasetDict` saved to disk with the benchmark names as keys).

---

## Usage

```bash
python eval.py \
  --model_name Qwen/Qwen3-1.7B \
  --template hicot \
  --tasks aime amc math minerva olympiad_bench \
  --temperature 0 \
  --max_tokens 4096 \
  --n_samples 1
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `Qwen/Qwen3-1.7B` | HuggingFace model identifier |
| `--template` | `qwen_math` | Prompting strategy (see above) |
| `--tasks` | all benchmarks | List of benchmarks to evaluate |
| `--temperature` | `0` | Sampling temperature |
| `--max_tokens` | `4096` | Max output tokens |
| `--max_model_len` | `4096` | Max model context length (set `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` for longer) |
| `--n_samples` | `1` | Number of samples per problem |
| `--max_test` | `999999` | Max problems per benchmark |
| `--gpu_memory_utilization` | `0.98` | vLLM GPU memory fraction |
| `--save` | `True` | Save summary JSON to `model_eval_outputs/` |

---

## Output

Results are saved to `model_eval_outputs/` as a JSON summary file with the naming convention:

```
summary_<model>_template_<template>_temp<T>_topp<p>_n<n>.json
```

The summary includes per-benchmark accuracy, average response length, and hierarchy format compliance statistics (fraction of responses that strictly follow the `<|instruction|>/<|execution|>` format, and their corresponding accuracy).

---

## Acknowledgements

This codebase is adapted from the [OAT](https://github.com/sail-sg/oat) repository. We thank the OAT authors for open-sourcing their work, which served as the foundation for our evaluation pipeline.

---

## Citation

If you use this code, please cite:

```bibtex
@article{huang2026hierarchical,
  title={Hierarchical Chain-of-Thought Prompting: Enhancing LLM Reasoning Performance and Efficiency},
  author={Huang, Xingshuai and Li, Derek and Nikpour, Bahareh and Omidi, Parsa},
  journal={arXiv preprint arXiv:2604.00130},
  year={2026}
}
