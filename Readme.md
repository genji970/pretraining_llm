Repository for recording progress on implementing LLM pretraining from scratch.

## In Progress

### 1. Scaling

- **Paper 1-1: Training Compute-Optimal Large Language Models**
    - The number of training tokens and model size should be scaled together.
    - Note: The paper used Huber loss for fitting the scaling-law model.

### 2. Data

- **Paper 2-1: FineWeb** / (below is from Fineweb post.)
    - **2-1-1. Related resources**
        - [FineWeb blog post](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
    - **2-1-2. Large-scale data processing library**
        - [Hugging Face DataTrove](https://github.com/huggingface/datatrove)
    - **2-1-3. Collecting high-quality data**
        - **Perplexity-based evaluation**
            - Low perplexity does not always lead to better downstream performance.
            - Medical, mathematical, and other specialized data may have distributions different from Wikipedia.
            - Even when such data has high perplexity under a Wikipedia-trained model, it may still be useful for domain-specific tasks.
        - **Small-model evaluation**
            - Train small models on representative subsets of candidate datasets.
            - Evaluate the models on multiple downstream benchmarks.
            - Using diverse benchmarks helps prevent overfitting the data-selection process to a single benchmark.
            - lighteval.(https://github.com/huggingface/lighteval/) / (Below is from lighteval repo.)
                  - General knowledge : MMLU , MMMU , BIG-Bench
                  - Question Answering : TriviaQA , Natural Questions, SimpleQA , Humanity's last question
                  - Math and Code Benchmark :
                  - Chat Model Evaluation :
                      - Instruction Following : IFEval , IFEval-fr
                      - Reasoning : MUSR, DROP(discrete reasoning)

        - **How to crwal**
            - preprocess CommonCrawl data(already in data format)
            - commoncrawl has two format : 1) WARC(raw data) 2) WET(text only version)
                  - can extract text data from WARC using **trafilatura library**
                  - WET was poor compared to text extracted from WARC 
                  - for tradeoff(budget and quality), it is good to use WET
                    
        - **Piepline**
            - text extraction
            - base filtering
                - removing lower quality data in pipeline.
                      - filtering basis
                          - blocklist(URL filtering)
                          - fastText language classifier
                          - MassiveText filters(quality and repetition)
                - Deduplicating
                      - speed up method
                          - hashing
                          - efficient data structures in indexing
                      - if searching is only for complete word, it's better to use hash table, inverted index, trie, etc in general rather than suffix array.
                      - RefinedWeb, using MinHash & LSH bounding with (for 8 hash values(MinHash values) in one bounding and 14 bounding. So,8*14=112)
                      - For common crawling, it periodically crawl data.
                          - e.g) 2024-04-10 , 2024-04-11..(snapshot/ each is called dump).
                              - If we deduplicate inside each snapshot, deplication amoong snapshots remain.
                                  - From recent dump, deduplicate 1) inside dump 2) among dumps
                      -  
                      




              

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
