#!/usr/bin/env python3
"""quickstart.py

Minimal example demonstrating zero-shot decision inference with Ventus-2B.
Scores candidate choices via direct option-letter causal logits.
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_id = "Ethan33X/Ventus-2B"
    print(f"Loading Ventus-2B from {model_id}...")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Example multi-criteria decision task
    prompt = (
        "Evidence: A real-time edge robotics system must balance path optimality against obstacle reaction latency under intermittent sensor dropouts.\n"
        "Question: Which control strategy provides the highest safety margin?\n"
        "Options:\n"
        "A) Static Dijkstra route recalculation on full topological grid\n"
        "B) Model Predictive Path Integral (MPPI) control with rollouts updated per LiDAR frame\n"
        "C) Open-loop proportional feedback without state estimator\n"
        "D) Unfiltered gradient ascent on potential field\n"
        "Answer: ("
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]

    # Evaluate candidate options ('A', 'B', 'C', 'D')
    option_letters = ["A", "B", "C", "D"]
    option_token_ids = [tokenizer.encode(letter, add_special_tokens=False)[-1] for letter in option_letters]

    raw_scores = torch.tensor([logits[tid].item() for tid in option_token_ids])
    probs = torch.softmax(raw_scores, dim=-1)

    print("\n=== Ventus-2B Decision Output ===")
    for letter, prob in zip(option_letters, probs):
        bar = "█" * int(prob.item() * 40)
        print(f"Option {letter}: {prob.item():6.2%} | {bar}")

    best_idx = torch.argmax(probs).item()
    print(f"\nWinning Decision: Option {option_letters[best_idx]}")

if __name__ == "__main__":
    main()
