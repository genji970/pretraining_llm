from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.tokenizer import Tokenizer
from model.model import DecoderLanguageModel


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device(name)


def load_model(
    checkpoint_path: Path,
    tokenizer_path: Path,
    device: torch.device,
) -> tuple[DecoderLanguageModel, Tokenizer, int]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    tokenizer = Tokenizer.load(tokenizer_path)

    model = DecoderLanguageModel(
        vocab_size=len(tokenizer),
        block_num=int(config["block_num"]),
        embed_dim=int(config["embed_dim"]),
        context_length=int(config["context_length"]),
        num_head=int(config["num_heads"]),
        dropout=float(config.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer, int(checkpoint.get("global_step", 0))


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, k=k, dim=-1)
        probabilities = torch.softmax(values, dim=-1)
        sampled_index = torch.multinomial(probabilities, num_samples=1)
        return indices.gather(-1, sampled_index)

    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


@torch.no_grad()
def generate(
    model: DecoderLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> tuple[str, str]:
    prompt_ids = tokenizer.encode(
        prompt,
        add_bos=True,
        add_eos=False,
    )
    generated = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )
    prompt_length = generated.size(1)
    eos_id = tokenizer.dictionary.get("<eos>")

    for _ in range(max_new_tokens):
        # Context window보다 길어지면 최근 토큰만 모델에 넣는다.
        model_input = generated[:, -model.context_length :]
        attention_mask = torch.ones_like(model_input)

        logits = model(
            model_input,
            attention_mask=attention_mask,
        )
        next_token = sample_next_token(
            logits=logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
        )
        generated = torch.cat([generated, next_token], dim=1)

        if eos_id is not None and int(next_token.item()) == eos_id:
            break

    all_ids = generated[0].tolist()
    continuation_ids = all_ids[prompt_length:]

    return (
        tokenizer.decode(all_ids),
        tokenizer.decode(continuation_ids),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/wiki_pretrain/final_checkpoint.pt"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="기본값: checkpoint와 같은 폴더의 tokenizer.json",
    )
    parser.add_argument("--prompt", type=str, default="Artificial intelligence is")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0이면 greedy decoding, 0보다 크면 sampling",
    )
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    tokenizer_path = args.tokenizer or args.checkpoint.parent / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    model, tokenizer, global_step = load_model(
        checkpoint_path=args.checkpoint,
        tokenizer_path=tokenizer_path,
        device=device,
    )

    full_text, continuation = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print(f"checkpoint step: {global_step}")
    print(f"device: {device}")
    print("\n=== prompt ===")
    print(args.prompt)
    print("\n=== prediction ===")
    print(continuation)
    print("\n=== full text ===")
    print(full_text)


if __name__ == "__main__":
    main()