import re
import fitz  # PyMuPDF
from typing import List, Dict, Any
from loguru import logger


class PDFParser:
    """PDF 文本解析器。"""

    def __init__(self, min_line_length: int = 10):
        self.min_line_length = min_line_length

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        pages_content = []

        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                blocks = page.get_text("blocks", sort=True)
                page_text_lines = []

                for block in blocks:
                    if block[6] != 0:
                        continue
                    text = block[4].strip()

                    text = re.sub(r"\s*\n\s*", " ", text)
                    text = re.sub(r"[ \t]+", " ", text)

                    if len(text) >= self.min_line_length:
                        page_text_lines.append(text)

                if page_text_lines:
                    raw = "\n".join(page_text_lines)
                    raw = re.sub(r"(?<![。！？；])\n", " ", raw)
                    raw = re.sub(r"\n{2,}", "\n", raw)

                    pages_content.append(
                        {
                            "content": raw,
                            "metadata": {
                                "source": file_path,
                                "page": page_num + 1,
                                "type": "pdf",
                            },
                        }
                    )

        logger.info(f"PDF 解析完成: {file_path}，共 {len(pages_content)} 页有效内容")
        return pages_content
