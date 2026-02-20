import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import csv
import time
import argparse


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate AAV2 sequences using a causal language model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-n", "--num-generations",
        type=int,
        default=100,
        dest="num_df",
        help="Number of sequences to generate.",
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=64,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Higher = more random.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling probability threshold (0.0–1.0). Use 1.0 to disable.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-K sampling. Pass 0 or omit to disable (i.e. top_k=None).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=500,
        help="Maximum total token length (prompt + generated tokens).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../datasets",
        help="Directory to save the output CSV.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Output file name (without .csv). Defaults to 'AAV2-final-final-{num_generations}'.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve dependent defaults
    top_k = None if (args.top_k is None or args.top_k == 0) else args.top_k
    name = args.name or f"AAV2-final-final-{args.num_df}"

    print("=" * 50)
    print("Run configuration")
    print("=" * 50)
    print(f"  Num generations  : {args.num_df:,}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Temperature      : {args.temperature}")
    print(f"  Top-p            : {args.top_p}")
    print(f"  Top-k            : {top_k}")
    print(f"  Max length       : {args.max_length}")
    print(f"  Output           : {args.output_dir}/{name}.csv")
    print("=" * 50)

    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # ── Model & tokenizer ────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        "Moreza009/AAVGen",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Moreza009/AAVGen", padding_side="left", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile enabled")

    # ── Dataset & dataloader ─────────────────────────────────────────────────
    df = pd.DataFrame({"first": ["M"] * args.num_df})

    def collate_fn(batch):
        return tokenizer(batch, return_tensors="pt", padding=False, truncation=False)

    dataloader = DataLoader(
        TextDataset(df["first"].tolist()),
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # ── Generation ───────────────────────────────────────────────────────────
    outputs = []
    torch.set_grad_enabled(False)

    for batch in tqdm(dataloader, desc="Generating", dynamic_ncols=True):
        batch = {k: v.to(device) for k, v in batch.items()}

        generated = model.generate(
            **batch,
            max_new_tokens=max(1, args.max_length - batch["input_ids"].shape[1]),
            do_sample=True,
            top_k=top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        input_len = batch["input_ids"].shape[1]
        decoded = [
            tokenizer.decode(g[input_len:], skip_special_tokens=True)
            for g in generated
        ]
        outputs.extend(decoded)

    # ── Save ─────────────────────────────────────────────────────────────────
    df["generate_seqs"] = outputs

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed / 3600:.2f} hours")

    out_path = f"{args.output_dir}/{name}.csv"
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\")

    print(f"Output saved to : {out_path}")
    print(f"Sample outputs  : {outputs[:5]}")


if __name__ == "__main__":
    main()