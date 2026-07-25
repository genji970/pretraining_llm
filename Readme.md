# Modular decoder-only pretraining

The original notebook modules were separated by function and connected with a
single immutable `argparse` configuration. The default public corpus is the
English `wikimedia/wikipedia` subset `20231101.en`, loaded in streaming mode and
bounded by `--max_samples`.

## Structure

```text
llm_pretraining_modular/
├── config.py               # the only argparse definition
├── main.py                 # wiring only
├── data/
│   ├── load_data.py        # Hugging Face / toy text loading
│   ├── tokenizer.py        # original regex tokenizer + vocab save/load
│   └── dataset.py          # packed next-token dataset and collator
├── model/
│   └── model.py            # RoPE, attention, blocks, decoder LM
└── train/
    └── trainer.py          # optimizer, CE loss, checkpoints, loop
```

## Install

```bash
pip install -r requirements.txt
```

## Small network-free end-to-end test

```bash
python main.py \
  --dataset_name toy \
  --dataset_config "" \
  --max_samples 4 \
  --output_dir outputs/toy \
  --context_length 16 \
  --batch_size 2 \
  --block_num 2 \
  --embed_dim 64 \
  --num_heads 4 \
  --epochs 1 \
  --max_steps 3 \
  --log_every 1 \
  --device cpu
```

## Wikipedia sample pretraining

```bash
python main.py \
  --dataset_name wikimedia/wikipedia \
  --dataset_config 20231101.en \
  --dataset_split train \
  --text_column text \
  --streaming \
  --max_samples 10000 \
  --max_vocab_size 30000 \
  --min_token_frequency 2 \
  --context_length 256 \
  --batch_size 8 \
  --block_num 8 \
  --embed_dim 512 \
  --num_heads 8 \
  --epochs 1 \
  --learning_rate 2e-4 \
  --output_dir outputs/wiki_pretrain
```

All runtime values come from `config.py`. There is no second dictionary or
`TrainingArguments` object that can silently replace a command-line value.
`vocab_size` is intentionally derived from the tokenizer dictionary instead of
being separately configurable.

## Test each file independently

Run from the project root:

```bash
python config.py
python -m data.load_data
python -m data.tokenizer
python -m data.dataset
python -m model.model
python -m train.trainer
```

Each test uses a tiny local sample. Only the real Wikipedia command requires a
network connection.

## Important scaling limit

This keeps the original simple design: selected documents are materialized in
memory, the vocabulary is built in one pass, and token IDs are packed into one
in-memory tensor. It is appropriate for debugging and sampled-corpus training,
but not yet for full multi-billion-token pretraining. A later scale-up should
replace only `data/dataset.py` with a sharded/streaming token dataset; the model,
trainer, CLI, and main wiring can remain.
