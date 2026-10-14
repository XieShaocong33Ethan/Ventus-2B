#!/usr/bin/env python3
"""scripts/run_inference_ventus_2b.py

Official Single-Pass Inference Utility for Ventus-2B.
Loads the model from Hugging Face ("Ethan33X/Ventus-2B") or local checkpoint,
and executes deterministic discrete choice inference in a single forward pass.
"""

import argparse
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_inference(
    model_id_or_path: str,
    evidence: str,
    question: str,
    options: list[str],
    device: str = "auto",
):
    print(f"Loading tokenizer & model from: {model_id_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }

    target_device = device
    if device == "auto":
        try:
            import accelerate  # noqa: F401
            model_kwargs["device_map"] = "auto"
        except ImportError:
            target_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        **model_kwargs,
    )
    if "device_map" not in model_kwargs:
        model = model.to(target_device)
    model.eval()

    # Format choices
    slot_labels = [chr(ord("A") + i) for i in range(len(options))]
    options_text = "\n".join([f"{label}) {opt}" for label, opt in zip(slot_labels, options)])
    
    prompt = f"Evidence: {evidence}\nQuestion: {question}\nOptions:\n{options_text}\nAnswer: ("

    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(model.device)
    slot_token_ids = [tokenizer.encode(label, add_special_tokens=False)[-1] for label in slot_labels]

    with torch.no_grad():
        outputs = model(**inputs)
        last_token_logits = outputs.logits[0, -1, :]
        choice_logits = last_token_logits[slot_token_ids]
        probabilities = torch.softmax(choice_logits, dim=-1)

    print("\n" + "=" * 50)
    print("Ventus-2B Probability Distribution:")
    print("=" * 50)
    for label, opt, prob in zip(slot_labels, options, probabilities.tolist()):
        print(f"  [{label}] ({prob * 100:6.2f}%): {opt}")

    best_idx = torch.argmax(probabilities).item()
    print("-" * 50)
    print(f"Final Decision: Option {slot_labels[best_idx]} (Confidence: {probabilities[best_idx]:.4f})")
    print("=" * 50 + "\n")
    return slot_labels[best_idx], probabilities.tolist()


def main():
    parser = argparse.ArgumentParser(description="Ventus-2B Inference CLI")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Ethan33X/Ventus-2B",
        help="HuggingFace model ID or local directory path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target inference execution device ('auto', 'cuda', 'cpu', 'mps')",
    )
    args = parser.parse_args()

    demo_evidence = (
        "Primary database cluster latency spiked to 450ms. "
        "Replication lag on follower node B is 120s. Worker pool utilization is 94%."
    )
    demo_question = "Select the immediate automated mitigation step adhering to SLA priority."
    demo_options = [
        "Trigger immediate hard reboot of the primary database cluster",
        "Temporarily throttle non-critical batch ingest workers and reroute read queries",
        "Flush Redis session caches to force client reconnections",
        "Demote the primary cluster and promote follower node B",
    ]

    run_inference(args.model_id, demo_evidence, demo_question, demo_options, device=args.device)


if __name__ == "__main__":
    main()
