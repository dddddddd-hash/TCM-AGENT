import re
from typing import Any, Dict, List

from loguru import logger


class TXTParser:
    def __init__(self, min_text_length: int = 20):
        self.min_text_length = min_text_length
        self.encoding_candidates = [
            "utf-8",
            "utf-8-sig",
            "gb18030",
            "gbk",
            "utf-16",
        ]

    def _read_text(self, file_path: str) -> str:
        last_error = None
        for encoding in self.encoding_candidates:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError as exc:
                last_error = exc
                continue

        raise UnicodeDecodeError(
            "txt",
            b"",
            0,
            1,
            f"Unable to decode {file_path} with supported encodings: "
            f"{self.encoding_candidates}. Last error: {last_error}",
        )

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        content = self._read_text(file_path)
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        content = re.sub(r"[ \t]+", " ", content)
        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n+", content)
            if p.strip()
        ]

        results: List[Dict[str, Any]] = []
        for i, paragraph in enumerate(paragraphs):
            paragraph = re.sub(r"\s*\n\s*", " ", paragraph)
            if len(paragraph) < self.min_text_length:
                continue

            results.append(
                {
                    "content": paragraph,
                    "metadata": {
                        "source": file_path,
                        "paragraph_index": i,
                        "type": "txt",
                    },
                }
            )

        logger.info(f"TXT parse complete: {file_path}, paragraphs={len(results)}")
        return results
