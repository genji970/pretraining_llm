from __future__ import annotations

import torch
from torch import nn

from training.pre_training.tokenization.regex_tokenizer import RegexTokenizer


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer: RegexTokenizer,
    prompt: str,
    device: torch.device,
    max_seq_len: int,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    model.eval()
    token_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    for _ in range(max_new_tokens):
        context = token_ids[-max_seq_len:]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        logits = model(input_ids, attention_mask)[:, -1, :] / temperature

        if top_k > 0:
            k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, k=k, dim=-1).values[:, -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        probabilities = torch.softmax(logits, dim=-1)
        next_token_id = int(torch.multinomial(probabilities, num_samples=1).item())
        token_ids.append(next_token_id)
        if next_token_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(token_ids, skip_special_tokens=False)


if __name__ == "__main__":
    from training.pre_training.model.language_model import DecoderLanguageModel

    tokenizer = RegexTokenizer()
    tokenizer.train(["hello world", "hello model"])
    model = DecoderLanguageModel(
        tokenizer.vocab_size,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ffn_multiplier=2,
        max_seq_len=16,
    )
    print(
        generate(
            model,
            tokenizer,
            prompt="hello",
            device=torch.device("cpu"),
            max_seq_len=16,
            max_new_tokens=3,
            top_k=3,
        )
    )
