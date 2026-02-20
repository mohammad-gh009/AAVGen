import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

_model = None
_tokenizer = None
_device = None


def _load_model():
    global _model, _tokenizer, _device

    if _model is not None:
        return  # Already loaded

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    _tokenizer = AutoTokenizer.from_pretrained(
        "Moreza009/AAVGen",
        padding_side="left",
        trust_remote_code=True,
    )
    _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        "Moreza009/AAVGen",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    _model.eval()


def generate(
    input_sequences: list[str],
    max_length: int = 500,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    batch_size: int = 64,
) -> list[str]:
    """
    Generate AAV2 sequences from one or more prompt strings.

    Args:
        input_sequences: List of prompt strings, e.g. ["M", "MAAG"].
        max_length:       Maximum total token length (prompt + generated).
        temperature:      Sampling temperature. Higher = more random.
        top_p:            Nucleus sampling threshold (1.0 = disabled).
        top_k:            Top-K sampling (None = disabled).
        batch_size:       Number of sequences to process per batch.

    Returns:
        List of generated sequences (one per input, prompt excluded).
    """
    _load_model()

    results = []

    with torch.inference_mode():
        for i in range(0, len(input_sequences), batch_size):
            batch_prompts = input_sequences[i : i + batch_size]

            inputs = _tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            inputs = {k: v.to(_device) for k, v in inputs.items()}

            input_len = inputs["input_ids"].shape[1]
            max_new_tokens = max(1, max_length - input_len)

            generated = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                use_cache=True,
                pad_token_id=_tokenizer.eos_token_id,
            )

            # Decode the full sequence (prompt + generated tokens)
            decoded = [
                _tokenizer.decode(g, skip_special_tokens=True)
                for g in generated
            ]
            results.extend(decoded)

    return results