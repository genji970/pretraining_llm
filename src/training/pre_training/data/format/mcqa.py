from __future__ import annotations

from typing import Any

from training.pre_training.data.format.base import RecordFormatter


class MCQAFormatter(RecordFormatter):
    """Formats common multiple-choice rows as causal-LM text.

    The option field may be a list of strings or dictionaries containing
    ``key`` and ``answer``. The answer may be stored as a letter or text.
    """

    def __init__(
        self,
        question_field: str = "question",
        options_field: str = "options",
        answer_field: str = "answer",
        answer_letter_field: str = "answer_letter",
        target_field: str = "text",
    ):
        self.question_field = question_field
        self.options_field = options_field
        self.answer_field = answer_field
        self.answer_letter_field = answer_letter_field
        self.target_field = target_field

    def _options(self, raw_options: Any) -> list[tuple[str, str]]:
        if not isinstance(raw_options, list) or not raw_options:
            raise ValueError("options must be a non-empty list")

        normalized: list[tuple[str, str]] = []
        for index, option in enumerate(raw_options):
            default_key = chr(65 + index)
            if isinstance(option, dict):
                key = str(option.get("key", default_key)).strip()
                value = option.get("answer", option.get("text", option.get("value", "")))
                text = str(value).strip()
            else:
                key = default_key
                text = str(option).strip()
            if not text:
                raise ValueError(f"Option {index} is empty")
            normalized.append((key, text))
        return normalized

    def format(self, record: dict[str, Any]) -> dict[str, Any]:
        question = str(record[self.question_field]).strip()
        if not question:
            raise ValueError("question cannot be empty")

        options = self._options(record[self.options_field])
        options_text = "\n".join(f"{key}. {text}" for key, text in options)

        answer_text = str(record.get(self.answer_field, "")).strip()
        answer_letter = str(record.get(self.answer_letter_field, "")).strip()
        if not answer_text and answer_letter:
            answer_text = next(
                (text for key, text in options if key == answer_letter),
                answer_letter,
            )
        if not answer_text:
            raise ValueError("An answer or answer_letter is required for pretraining")

        answer = f"{answer_letter}. {answer_text}" if answer_letter else answer_text
        text = (
            f"<question>\n{question}\n\n"
            f"<choices>\n{options_text}\n\n"
            f"<answer>\n{answer}"
        )
        return {self.target_field: text}


if __name__ == "__main__":
    formatter = MCQAFormatter()
    sample = {
        "question": "What is 2 + 2?",
        "options": ["3", "4", "5"],
        "answer_letter": "B",
        "answer": "4",
    }
    print(formatter.format(sample)["text"])
