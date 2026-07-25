# From-scratch LLM pretraining

Colab notebook code를 VS Code에서 실행 가능한 패키지형 프로젝트로 분리한 버전입니다. 현재는 pretraining만 구현하며, 이후 `src/training/sft/`, `src/training/ppo/`를 같은 수준에 추가할 수 있습니다.

## 구조

```text
.
├── artifacts/                         # tokenizer와 checkpoint 출력
├── data/
│   ├── raw/                           # 원본 데이터
│   └── process/                       # 정규화된 학습 데이터
├── scripts/
│   └── pretraining.py                 # 유일한 전체 실행 진입점
└── src/training/pre_training/
    ├── config.py                      # Python dataclass 설정
    ├── data/
    │   ├── source/                    # local/Hugging Face 데이터 읽기
    │   ├── format/                    # plain text/MCQA 정규화
    │   ├── storage/                   # 처리 데이터 저장
    │   └── process.py                 # source → format → storage
    ├── tokenization/                  # tokenizer 인터페이스와 regex tokenizer
    ├── dataset/                       # causal-LM dataset과 dynamic padding
    ├── model/                         # embedding, RoPE, attention, block, LM
    ├── loss.py
    ├── scheduler.py
    ├── checkpoint.py
    ├── trainer.py
    ├── generation.py
    ├── pipeline.py
    └── utils.py
```

별도의 `toy.toml`, `create_toy_data.py`, `train_tokenizer.py`, `prepare_data.py` 같은 실행 스크립트는 없습니다. 설정은 `config.py`의 dataclass와 `scripts/pretraining.py`의 CLI 인자로 관리합니다.

## 설치

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -e .
```

Hugging Face 데이터셋도 사용할 경우:

```powershell
pip install -e ".[huggingface]"
```

## 각 기능 파일 자체 테스트

각 핵심 `.py` 파일 아래에는 작은 샘플을 실행하는 `if __name__ == "__main__":` 블록이 있습니다. 프로젝트 루트에서 모듈로 실행합니다.

```bash
python -m training.pre_training.data.source.local_jsonl
python -m training.pre_training.data.format.mcqa
python -m training.pre_training.data.process
python -m training.pre_training.tokenization.regex_tokenizer
python -m training.pre_training.dataset.collator
python -m training.pre_training.model.rope
python -m training.pre_training.model.attention
python -m training.pre_training.model.language_model
python -m training.pre_training.trainer
python -m training.pre_training.pipeline
```

## 전체 pretraining 동작 확인

별도 toy 파일을 만들지 않고 임시 디렉터리에서 2-step 학습을 수행합니다.

```bash
python scripts/pretraining.py --smoke-test
```

## 로컬 JSONL 데이터로 학습

`data/raw/train.jsonl` 예시:

```json
{"text": "first pretraining document"}
{"text": "second pretraining document"}
```

실행:

```bash
python scripts/pretraining.py \
  --train-path data/raw/train.jsonl \
  --eval-path data/raw/eval.jsonl \
  --formatter plain_text \
  --d-model 512 \
  --n-heads 8 \
  --n-layers 8 \
  --max-seq-len 1024 \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --epochs 1 \
  --amp
```

Windows PowerShell에서는 줄 연결 문자를 백틱으로 바꿉니다.

## MCQA 데이터로 학습

행에 `question`, `options`, `answer` 또는 `answer_letter`가 있으면:

```bash
python scripts/pretraining.py \
  --train-path data/raw/train.jsonl \
  --formatter mcqa
```

## Hugging Face 데이터로 확장

```bash
python scripts/pretraining.py \
  --source huggingface \
  --dataset-name m-a-p/SuperGPQA \
  --train-split train \
  --formatter mcqa
```

새 데이터 저장소는 `DataSource`, 새 변환 형식은 `RecordFormatter`, 새 처리 데이터 저장 방식은 `DataStorage`, 새 tokenizer는 `TokenizerBase`, 새 embedding 공간은 `TokenEmbeddingSpace`를 구현해 추가합니다.
