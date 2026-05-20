import re
from typing import List, Dict, Any
from loguru import logger


class MarkdownParser:
    """Markdown 章节解析器。"""

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        sections = re.split(r"(?=^#{1,3}\s)", content, flags=re.MULTILINE)
        result = []

        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            title_match = re.match(r"^(#{1,3})\s+(.+)", section)
            title = title_match.group(2) if title_match else "无标题"
            level = len(title_match.group(1)) if title_match else 0

            result.append(
                {
                    "content": section,
                    "metadata": {
                        "source": file_path,
                        "section_index": i,
                        "title": title,
                        "heading_level": level,
                        "type": "markdown",
                    },
                }
            )

        logger.info(f"Markdown 解析完成: {file_path}，共 {len(result)} 个章节块")
        return result
