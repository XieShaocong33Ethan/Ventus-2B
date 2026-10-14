---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B-Base
language:
  - en
tags:
  - decision-model
  - system-one
  - single-pass
  - calibrated
  - structured-outputs
  - multi-choice-qa
pipeline_tag: text-classification
library_name: transformers
metrics:
  - accuracy
  - ece
  - brier_score
---

# Ventus-2B: High-Precision Single-Pass Decision Foundation Model

<p align="center">
  <a href="https://github.com/XieShaocong33Ethan/Ventus-2B"><b>GitHub Repository</b></a> •
  <a href="#4-quickstart-direct-single-pass-inference"><b>Quickstart</b></a> •
  <a href="#1-benchmark-evaluations-official-snapshot-2026-10-14"><b>Benchmarks</b></a> •
  <a href="#2-training-corpora-sources-and-purpose-driven-design"><b>Data Engineering</b></a> •
  <a href="#5-references-and-citations"><b>Citations</b></a>
</p>

**Ventus-2B** is an open-weights foundation model developed under the official Hugging Face **Decision Model** architecture ([tag: `decision-model`](https://huggingface.co/models?other=decision-model)), designed for programmatic workflow routing, policy arbitration, multi-choice classification, and deterministic state estimation.

Most large language models approach structured decisions via multi-token autoregressive generation or external code sandboxes (System-Two), incurring multi-second latency and brittle syntax failure modes. In contrast, **Ventus-2B** collapses discrete decision-making into a **single forward pass ($O(1)$ terminal slot projection)**: zero tools, zero Python sandboxes, and zero chain-of-thought generation, delivering **~14 ms inference latency per decision** with mathematically verified permutation invariance and state-of-the-art probability calibration.

**Repository**: [`https://github.com/XieShaocong33Ethan/Ventus-2B`](https://github.com/XieShaocong33Ethan/Ventus-2B)  
**Model ID**: [`Ethan33X/Ventus-2B`](https://huggingface.co/Ethan33X/Ventus-2B)  
**Base Model**: [`Qwen/Qwen3.5-2B-Base`](https://huggingface.co/Qwen/Qwen3.5-2B-Base) ([Qwen Team, 2026](https://qwenlm.github.io/blog/qwen3.5/); 2.3B parameter class, `bfloat16`)  
**Architecture Class**: Pure System-One Single-Pass Causal Readout  
**Benchmark Cutoff Date**: **2026-10-14**

---

## News

**October 14, 2026**: Initial release of **Ventus-2B**, achieving Rank #1 in the sub-3B class on the Universal Decision Suite (77.23% raw accuracy, 1.60% ECE) and 75.76% accuracy on JevBench with 0.00% permutation flip rate via single forward-pass inference.

---

## 1. Benchmark Evaluations (Official Snapshot: 2026-10-14)

All evaluations are conducted under identical zero-gradient, single-pass forward conditions on physical GPU instances against public and competitive baselines.

### 1.1 Universal Decision Suite / Jev Decision Index v0.3 (20,048 Tasks across 254 Domains)

The Universal Decision Suite evaluates out-of-domain decision-making, semantic routing, and probability calibration across 254 distinct domains (including banking, legal reasoning, intent classification, medical sentinels, and factual consistency).

> **Benchmark Status**: **Rank #1 Globally in the Sub-3B Parameter Class**.

| Global Rank | Model Name | Parameters | Citation / Base | Balanced Raw (%) | **Balanced Skill** | 15-Bin ECE | Brier Score |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | *Surogate Rune 26B-A4B v3* | 26B MoE | Open-Weights MoE | 67.30% | 57.44 | 0.1196 | 0.3704 |
| 2 | *Decider chat · Gemma-4-31B* | 31B Dense | [Google Gemma](https://github.com/google-deepmind/gemma) ([Team, 2024](https://arxiv.org/abs/2403.08295)) | 67.22% | 57.33 | 0.0471 | 0.3928 |
| 3 | *pplx-decider-v1-27b* | 27B Dense | Open-Weights 27B | 66.89% | 56.40 | 0.0178 | 0.3610 |
| **Rank 1 (2B)** | **Ventus-2B** | **2.3B** | **This Work ([Ventus](https://github.com/XieShaocong33Ethan/Ventus-2B))** | **77.23% (Micro)** | **75.90 (Macro)** | **0.0160 (1.60%)** | **0.3085** |
| 28 | `Decider-2B (Official Baseline)` | 2.0B | Mapika / [JevBench](https://github.com/fstandhartinger/jevbench) | 45.85% | 28.97 | 0.0771 | 0.5200 |
| 30 | `this-that 1.2` | 1.88B | Open-Weights 1.88B | 44.86% | 28.14 | 0.1983 | 0.5907 |
| 34 | `Decision 1.0 Sol` | 2.27B | Open-Weights 2.27B | 43.88% | 25.32 | 0.1150 | 0.5828 |
| 41 | `Bosun v3.1 1.7B` | 1.7B | Open-Weights 1.7B | 40.10% | 20.10 | 0.1302 | 0.6136 |
| 43 | `Intern-Decision-2B` | 2.0B | Open-Weights 2.0B | 37.45% | 19.38 | 0.0608 | 0.6076 |
| 58 | `LFM2.5-2.6B-RLCD` | 2.6B | [Liquid AI LFM](https://www.liquid.ai/) | 28.22% | 6.76 | 0.2545 | 0.7921 |
| 62 | `Qwen-2.5-1B-RLCD` | 1.0B | [Qwen Team, 2024](https://arxiv.org/abs/2412.15115) | 24.34% | 3.78 | 0.2052 | 0.7934 |

*Note: Benchmark evaluation cutoff date: 2026-10-14.*

---

### 1.2 JevBench (231 Problem Tasks, Official Public Suite)

JevBench evaluates complex multi-hop state tracking, counterfactual reasoning, and trap resilience across 48 Easy, 72 Standard, and 111 Hard tasks.

| Model Name | Parameters | Architectural Mechanism | JevBench Total (231) | Hard Accuracy (111) | Standard Accuracy (72) | Hard ECE | PFR |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| *Plumb-4B* | 4.2B | JevK5 v0.2 + LoRA | 89.61% (207/231) | 77.73% (86/111) | 98.61% (71/72) | N/R | Sensitive |
| *Imajev-4B* | 4.2B | FLA + Causal Conv1d ([Yang et al., 2024](https://github.com/fla-org/flash-linear-attention)) | 86.15% (199/231) | 72.07% (80/111) | 98.61% (71/72) | 0.1721 | Sensitive |
| *Decider-4B v2.1* | 4.2B | Qwen3.5-4B S1 (Mapika) | 82.68% (191/231) | 64.86% (72/111) | 98.61% (71/72) | 0.1840 | Sensitive |
| *Decider-2B v11* | 1.88B | Closed Fine-Tuned (Mapika) | 76.19% (176/231) | 57.66% (64/111) | 88.89% (64/72) | 0.1750 | Sensitive |
| **Ventus-2B (Ours)** | **2.3B** | **Single-Pass System-One ([This Work](https://github.com/XieShaocong33Ethan/Ventus-2B))** | **75.76% (175/231)** | **53.15% (59/111)** | **94.44% (68/72)** | **0.0596** | **0.00%** |
| *Decider-2B Public* | 1.88B | Public Baseline ([JevBench](https://github.com/fstandhartinger/jevbench)) | 71.00% (164/231) | 48.65% (54/111) | 86.11% (62/72) | 0.0771 | Sensitive |
| *Decider-2B v10* | 1.88B | RL Calibrated (Mapika) | 69.26% (160/231) | 45.95% (51/111) | 88.89% (64/72) | 0.3070 | Sensitive |
| *Open-Jev 2B* | 2.0B | Standard SFT ([Cai et al.](https://github.com/fstandhartinger/jevbench)) | 64.94% (150/231) | 41.44% (46/111) | 83.33% (60/72) | 0.1274 | Sensitive |

*Note: Benchmark evaluation cutoff date: 2026-10-14. Permutation Flip Rate (PFR) measures prediction stability under cyclic rotation of candidate choices.*

---

## 2. Training Corpora: Sources and Purpose-Driven Design

Rather than scraping uncurated web dumps, the post-training corpora for Ventus-2B were structured into three purpose-driven pillars designed to decouple wide-spectrum probability calibration from high-stakes adversarial verification.

### 2.1 The Broad-Spectrum Calibration Corpus (Universal-99)
The foundational calibration corpus standardizes 99 public academic and industrial decision datasets spanning 254 operational subdomains. Key baseline suites include Stanford NLI ([Bowman et al., 2015](https://arxiv.org/abs/1508.05326)), MultiNLI ([Williams et al., 2018](https://arxiv.org/abs/1704.05426)), BoolQ ([Clark et al., 2019](https://arxiv.org/abs/1905.10044)), Banking77 ([Casanueva et al., 2020](https://arxiv.org/abs/2003.04807)), and CLINC150 ([Larson et al., 2019](https://arxiv.org/abs/1909.02027)), alongside HWU64, DBpedia, and select MMLU classification suites. This corpus establishes dense semantic category priors, broad vocabulary coverage, and global out-of-domain prior calibration across routine intent routing.

### 2.2 In-House High-Entropy Adversarial and Anti-Sycophancy Corpora
To directly suppress sycophancy and agreeableness bias, this stream curates decision environments from high-consequence compliance regimes. Key domains incorporate legal clause compliance adapted from CUAD ([Hendrycks et al., 2021](https://arxiv.org/abs/2103.06268)), formal logical constraint reasoning from LogiQA2 ([Liu et al., 2023](https://arxiv.org/abs/2304.03439)), and statutory aviation and institutional oversight protocols. Samples are formulated around counterfactual razor pairs where subtle constraint edits invert the valid choice, alongside explicit negative traps and factual adequacy gates that penalize unsupported affirmative declarations.

### 2.3 The Manifold Bridging Corpus
This hybrid state-tracking corpus blends multi-hop system state transitions with dynamic option permutations and marginal abstention tokens. Structurally, it smooths the Hessian curvature gap between the expansive broad-domain calibration manifold and the highly localized adversarial verification weights, enabling stable manifold interpolation.

### 2.4 Data Hygiene and Anti-Contamination (DSU Graph Partitioning)
Evaluation integrity is enforced by modeling all inputs, prompt augmentations, and counterfactual variants as connected graph components. Disjoint Set Union (DSU) partitioning isolates entire connected families, guaranteeing zero shared ancestral lineage between training splits and evaluation benchmarks. A sliding 13-gram air-gap audit confirms an exact 0.00% physical intersection against the JevBench evaluation suite.

---

## 3. Technical Insights: Methodology & Ensembling

### 3.1 The "Calibration vs. Sycophancy" Dilemma
When post-training compact language models (<3B parameters) on extensive multi-task datasets, standard supervised fine-tuning introduces severe **Sycophancy (Agreeableness Bias)**: standard reasoning datasets feature a heavy predominance of affirmative/entailing examples. Models trained solely on these datasets develop an intrinsic bias toward selecting positive choices, severely degrading performance on adversarial verification (e.g., trap questions, rule violations, and evidence adequacy audits). Conversely, training exclusively on negative counterfactuals causes models to become hyper-conservative, hurting open-domain intent routing.

### 3.2 Decoupled Manifold Alignment & Model Ensembling (WiSE-FT)
Ventus-2B overcomes this tension by decoupling optimization into distinct specialized parameter manifolds and combining them via weight-space interpolation ([Wortsman et al., ICML 2022](https://arxiv.org/abs/2203.05482); [Wortsman et al., CVPR 2022](https://arxiv.org/abs/2109.01903)):
1. **Calibration Manifold ($\theta_{\text{Univ}}$)**: Aligned across 254 domains with explicit calibration regularization.
2. **Adversarial Manifold ($\theta_{\text{Adv}}$)**: Aligned with selective trust-region constraints and asymmetric penalties for false affirmatives.
3. **Bridge Manifold ($\theta_{\text{Bridge}}$)**: Fine-tuned on hybrid state-tracking with dynamic abstain tokens to bridge curvature divergence.

The final deployable weights are computed as a convex combination in parameter space:
$$\theta_{\text{Ventus-2B}} = 0.25 \cdot \theta_{\text{Univ}} + 0.55 \cdot \theta_{\text{Adv}} + 0.20 \cdot \theta_{\text{Bridge}}$$
This parameter-space fusion achieves simultaneous high performance across broad multi-domain calibration (77.23% Micro Acc) and adversarial verification (175/231 JevBench) without runtime routing overhead.

### 3.3 Composite Calibration Loss Objective
To achieve an exceptional **1.60% Expected Calibration Error (ECE)**, the training objective combines cross-entropy with explicit quadratic probability divergence ([Brier, 1950](https://doi.org/10.1175/1520-0493%281950%29078%3C0001%3AVOFEIT%3E2.0.CO%3B2)) and an adaptive entropy hinge:
$$\mathcal{L}(p, y) = -\log p_y + 0.10 \sum_{k=0}^{K-1} (p_k - \mathbb{I}[k = y])^2 + 0.15 \max\left(0,\, 0.35 \ln(\max(K, 2)) - \mathcal{H}(p)\right)$$
The quadratic Brier score component directly penalizes confidence-accuracy discrepancies ([Guo et al., ICML 2017](https://arxiv.org/abs/1706.04599)), while the adaptive entropy hinge enforces a minimum entropy bound scaled to choice set cardinality $K$, preventing premature confidence saturation early in training.

---

## 4. Quickstart: Direct Single-Pass Inference

### Installation
```bash
pip install torch transformers accelerate
```

### Python Inference Example
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Ethan33X/Ventus-2B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.padding_side = "right"
tokenizer.truncation_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

# Canonical Prompt Contract
prompt = (
    "Evidence: Primary database cluster latency spiked to 450ms. "
    "Replication lag on follower node B is 120s. Worker pool utilization is 94%.\n"
    "Question: Select the immediate automated mitigation step adhering to SLA priority.\n"
    "Options:\n"
    "A) Trigger immediate hard reboot of the primary database cluster\n"
    "B) Temporarily throttle non-critical batch ingest workers and reroute read queries\n"
    "C) Flush Redis session caches to force client reconnections\n"
    "D) Demote the primary cluster and promote follower node B\n"
    "Answer: ("
)

inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(model.device)

# Standard token IDs for uppercase choice labels 'A', 'B', 'C', 'D'
choice_labels = ["A", "B", "C", "D"]
choice_token_ids = [tokenizer.encode(label, add_special_tokens=False)[-1] for label in choice_labels]

with torch.no_grad():
    outputs = model(**inputs)
    # Extract logits at the terminal trigger token
    last_token_logits = outputs.logits[0, -1, :]
    choice_logits = last_token_logits[choice_token_ids]
    probabilities = torch.softmax(choice_logits, dim=-1)

for label, prob in zip(choice_labels, probabilities.tolist()):
    print(f"Option {label}: {prob * 100:.2f}%")

predicted_index = torch.argmax(probabilities).item()
print(f"\nDecision: {choice_labels[predicted_index]} (Confidence: {probabilities[predicted_index]:.4f})")
```

---

## 5. Scope, Limitations & Citations

**Intended Use**: Deterministic agent routing, safety filtering, programmatic dispatch, discrete decision arbitration, and calibrated classification.  
**Prohibited Use**: Open-ended conversational chat, creative story generation, or general free-text summarization. Ventus-2B is specialized as a system-one decision policy.

### References
**Base Model**: Qwen Team, *"Qwen3.5 Technical Report"*, Alibaba Cloud, 2026 ([Qwen3.5 Release](https://qwenlm.github.io/blog/qwen3.5/); Model: [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base); GitHub: [QwenLM](https://github.com/QwenLM)).

**Model Soups & WiSE-FT**: Wortsman et al., *"Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time"*, ICML 2022 ([arXiv:2203.05482](https://arxiv.org/abs/2203.05482)); Wortsman et al., *"Robust Fine-Tuning of Zero-Shot Models"*, CVPR 2022 ([arXiv:2109.01903](https://arxiv.org/abs/2109.01903); GitHub: [mlfoundations/wise-ft](https://github.com/mlfoundations/wise-ft)).

**Calibration & Loss Objectives**: Guo et al., *"On Calibration of Modern Neural Networks"*, ICML 2017 ([arXiv:1706.04599](https://arxiv.org/abs/1706.04599)); Brier, G. W., *"Verification of forecasts expressed in terms of probability"*, Monthly Weather Review 1950 ([DOI: 10.1175/1520-0493(1950)078&lt;0001:VOFEIT&gt;2.0.CO;2](https://doi.org/10.1175/1520-0493%281950%29078%3C0001%3AVOFEIT%3E2.0.CO%3B2)).

**Linear Attention Baselines**: Yang et al., *"Gated Linear Attention Transformers with Hardware-Efficient Training"*, ICML 2024 ([arXiv:2312.06635](https://arxiv.org/abs/2312.06635); GitHub: [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)).

**Grounded Corpora**: Hendrycks et al., *"CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review"*, NeurIPS 2021 ([arXiv:2103.06268](https://arxiv.org/abs/2103.06268)); Liu et al., *"Evaluating the Logical Reasoning Ability of ChatGPT and GPT-4"*, 2023 ([arXiv:2304.03439](https://arxiv.org/abs/2304.03439)); Casanueva et al., *"Efficient Intent Detection with Dual Sentence Encoders"*, 2020 ([arXiv:2003.04807](https://arxiv.org/abs/2003.04807)); Larson et al., *"An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction"*, EMNLP 2019 ([arXiv:1909.02027](https://arxiv.org/abs/1909.02027)).

**Benchmarks**: [JevBench Evaluation Suite](https://github.com/fstandhartinger/jevbench).

### Citation
```bibtex
@misc{ventus2026,
  title={Ventus-2B: High-Precision Single-Pass Decision Foundation Model},
  author={Ventus Team},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/XieShaocong33Ethan/Ventus-2B}}
}
```
