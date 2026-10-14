# Ventus-2B: High-Precision Single-Pass Decision Foundation Model

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Base Model](https://img.shields.io/badge/Base-Qwen3.5--2B--Base-purple.svg)](https://huggingface.co/Qwen/Qwen3.5-2B-Base)
[![Parameters](https://img.shields.io/badge/Parameters-2.3B-orange.svg)](#)
[![Category](https://img.shields.io/badge/Category-Decision%20Model-blue.svg)](https://huggingface.co/models?other=decision-model)
[![Sub-3B Benchmark](https://img.shields.io/badge/Universal%20Decision%20Index-Rank%20%231%20(Sub--3B)-success.svg)](#)
[![Inference SLA](https://img.shields.io/badge/Latency-~14ms%20%2F%20task-brightgreen.svg)](#)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Ethan33X%2FVentus--2B-yellow.svg)](https://huggingface.co/Ethan33X/Ventus-2B)

<p align="center">
  <a href="https://huggingface.co/Ethan33X/Ventus-2B"><b>Hugging Face Model</b></a> •
  <a href="#quickstart"><b>Quickstart</b></a> •
  <a href="#comprehensive-benchmark-results-official-snapshot-2026-10-14"><b>Benchmarks</b></a> •
  <a href="#training-methodology--insights"><b>Methodology</b></a> •
  <a href="#references--citations"><b>Citations</b></a>
</p>

**Ventus-2B** is an open-weights foundation model officially categorized under Hugging Face's **Decision Model** paradigm ([tag: `decision-model`](https://huggingface.co/models?other=decision-model)), engineered for low-latency programmatic agent dispatch, policy arbitration, safety verification, and discrete multi-choice reasoning.

Traditional language model agents handle structured choices by autoregressively sampling intermediate tokens, reasoning chains, or Python sandbox scripts (System-Two). While effective for exploratory coding, this paradigm introduces multi-second latency, non-deterministic token trajectories, and severe vulnerability to prompt injection or syntax errors. 

Ventus-2B collapses discrete decision problems into a single forward pass ($O(1)$ terminal logit readout). Operating with zero external tools, zero code sandboxes, and zero chain-of-thought token generation, the architecture eliminates runtime execution overhead and syntax drift. The model achieves ~14 ms per-task latency (with GPU batch throughput exceeding 1,000 decisions per second at Batch=16), state-of-the-art probability calibration with 1.60% 15-Bin ECE and 0.3085 Brier score across 20,048 evaluation tasks, and strict permutation invariance with a 0.00% Permutation Flip Rate across candidate orderings.

---

## News

**October 14, 2026**: Initial release of **Ventus-2B**, achieving Rank #1 in the sub-3B class on the Universal Decision Suite (77.23% raw accuracy, 1.60% ECE) and 75.76% accuracy on JevBench with 0.00% permutation flip rate via single forward-pass inference.

---

## Comprehensive Benchmark Results (Official Snapshot: 2026-10-14)

All evaluations are conducted under physical zero-gradient conditions on dedicated GPUs using official benchmark suites against verified public baselines.

### 1. Universal Decision Suite / Jev Decision Index v0.3 (20,048 Tasks across 254 Domains)

The Universal Decision Suite tests wide-spectrum out-of-domain decision generalization and probability reliability.

> **Global Standing**: **Rank #1 Globally in the Sub-3B Parameter Tier**.

| Global Rank | Model Name | Parameter Tier | Citation / Source | Balanced Raw (%) | **Balanced Skill** | 15-Bin ECE | Brier Score |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | *Surogate Rune 26B-A4B v3* | 26B MoE (4B Act) | Open-Weights MoE | 67.30% | 57.44 | 0.1196 | 0.3704 |
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

*Cutoff Date: 2026-10-14. All baseline scores reproduced from official published leaderboard tables.*

---

### 2. JevBench (231 Problem Tasks, Official Public Suite)

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

*Cutoff Date: 2026-10-14. Permutation Flip Rate (PFR) indicates whether cyclic rotation of choice indices alters decision outcomes.*

---

## Training Corpora and Purpose-Driven Data Engineering

Rather than relying on uncurated web dumps, Ventus-2B is trained across three functional data streams designed to decouple wide-spectrum probability calibration from high-stakes adversarial verification.

### Broad-Spectrum Calibration Stream (Universal-99)
The primary calibration stream standardizes 99 established academic and industry classification suites spanning 254 distinct operational subdomains. Key foundational corpora include Stanford SNLI ([Bowman et al., 2015](https://arxiv.org/abs/1508.05326)), MultiNLI ([Williams et al., 2018](https://arxiv.org/abs/1704.05426)), BoolQ ([Clark et al., 2019](https://arxiv.org/abs/1905.10044)), Banking77 ([Casanueva et al., 2020](https://arxiv.org/abs/2003.04807)), and CLINC150 ([Larson et al., 2019](https://arxiv.org/abs/1909.02027)), alongside HWU64, DBpedia, and select MMLU decision tasks. This stream establishes dense category priors, broad vocabulary coverage, and consistent probability grounding across routine intent routing.

### In-House High-Entropy Adversarial and Anti-Sycophancy Stream
To counter the default agreeable bias inherent in instruction-tuned language models, the adversarial stream synthesizes high-consequence decision scenarios from specialized compliance environments. Source materials draw upon legal contract review from CUAD ([Hendrycks et al., 2021](https://arxiv.org/abs/2103.06268)), formal logical constraint satisfaction from LogiQA2 ([Liu et al., 2023](https://arxiv.org/abs/2304.03439)), and statutory guidelines from federal aviation and institutional review boards. Training samples emphasize counterfactual razor pairs where subtle single-token condition shifts invert the valid outcome, deliberate negative logic traps, and factual adequacy gates that strictly penalize unsupported affirmative declarations.

### Curvature Bridging Stream
The bridging corpus blends multi-hop system state tracking with dynamic option permutations and explicit marginal abstain tokens. Structurally, this intermediate dataset smooths the Hessian curvature barrier between the expansive, well-distributed calibration manifold and the highly localized, conservative verification manifold, enabling cohesive weight-space parameter fusion.

### Data Hygiene and Anti-Contamination Guarantee
Evaluation integrity is preserved by modeling all prompts, semantic perturbations, and counterfactual mutations as connected graph components. Disjoint Set Union (DSU) partitioning isolates entire connected families, ensuring that no evaluation example shares ancestral or syntactic lineage with the training split. A sliding 13-gram decontamination audit against the official JevBench suite confirms an exact 0.00% physical intersection across all task tiers.

---

## Training Methodology & Insights

To advance the state of the art in efficient decision models while safeguarding operational recipes, we share the following architectural and mathematical principles:

### 1. The Sycophancy-Calibration Dilemma in Sub-3B Models
During post-training of compact language models, standard instruction fine-tuning across massive multi-task classification datasets induces a severe **Sycophancy (Yes-Bias)**: standard corpora are heavily weighted toward positive alignments, causing models to favor affirmative choices regardless of evidence. In rigorous adversarial verification regimes such as fault inspection, adequacy auditing, and trap questions, this bias causes models to overlook subtle constraint violations. Conversely, training exclusively on negative counterfactuals causes models to become hyper-conservative, collapsing their accuracy on open-domain intent routing.

### 2. Decoupled Manifolds & WiSE-FT Model Ensembling
Instead of forcing competing objectives through a single sequential optimization path, Ventus-2B decouples the parameter landscape ([Wortsman et al., ICML 2022](https://arxiv.org/abs/2203.05482); [Wortsman et al., CVPR 2022](https://arxiv.org/abs/2109.01903)):
1. **Calibration Manifold ($\theta_{\text{Univ}}$)**: Aligned across 254 domains with explicit calibration loss to establish robust out-of-domain decision boundaries.
2. **Adversarial Manifold ($\theta_{\text{Adv}}$)**: Aligned with selective trust-region KL and asymmetric penalties for false affirmatives to master logical edge cases.
3. **Bridge Manifold ($\theta_{\text{Bridge}}$)**: Fine-tuned with balanced state-tracking and marginal abstain tokens to smooth the Hessian curvature barrier between the specialized endpoints.

The deployable model is obtained via convex interpolation in weight space:
$$\theta_{\text{Ventus-2B}} = 0.25 \cdot \theta_{\text{Univ}} + 0.55 \cdot \theta_{\text{Adv}} + 0.20 \cdot \theta_{\text{Bridge}}$$
This parameter-space fusion achieves simultaneous mastery across broad calibration (77.23% Micro) and deep adversarial logic (175/231 JevBench) without requiring runtime routing overhead.

### 3. Composite Calibration Objective
To prevent probability distortions, we incorporate an explicit multi-objective loss function during training ([Brier, 1950](https://doi.org/10.1175/1520-0493%281950%29078%3C0001%3AVOFEIT%3E2.0.CO%3B2); [Guo et al., ICML 2017](https://arxiv.org/abs/1706.04599)):
$$\mathcal{L}(p, y) = -\log p_y + 0.10 \cdot \sum_{k=0}^{K-1} (p_k - \mathbb{I}[k = y])^2 + 0.15 \cdot \max\left(0,\, 0.35 \ln(\max(K, 2)) - \mathcal{H}(p)\right)$$
The quadratic Brier score regularization directly aligns predicted probabilities with empirical frequencies, while the adaptive entropy hinge enforces a minimum entropy bound scaled to choice set cardinality $K$, preventing premature confidence saturation early in training.

---

## Quickstart

### 1. Installation
```bash
pip install torch transformers accelerate
```

### 2. Minimal Single-Pass Inference
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

# Canonical Prompt Template
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

# Map uppercase choices to token IDs
choice_labels = ["A", "B", "C", "D"]
choice_token_ids = [tokenizer.encode(label, add_special_tokens=False)[-1] for label in choice_labels]

with torch.no_grad():
    outputs = model(**inputs)
    # Extract logits at the terminal token
    last_token_logits = outputs.logits[0, -1, :]
    choice_logits = last_token_logits[choice_token_ids]
    probabilities = torch.softmax(choice_logits, dim=-1)

for label, prob in zip(choice_labels, probabilities.tolist()):
    print(f"Option {label}: {prob * 100:.2f}%")

best_choice = choice_labels[torch.argmax(probabilities).item()]
print(f"\nFinal Decision: Option {best_choice}")
```

---

## References & Citations

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
