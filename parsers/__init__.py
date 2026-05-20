from parsers.pdf_parser import PDFParser
from parsers.md_parser import MarkdownParser
from parsers.json_parser import JSONParser
from parsers.txt_parser import TXTParser
from typing import List, Dict, Any, Optional
from loguru import logger


class ParserFactory:
    """根据文件扩展名选择解析器。"""

    _PARSER_MAP = {
        ".pdf": PDFParser,
        ".md": MarkdownParser,
        ".json": JSONParser,
        ".txt": TXTParser,
    }

    def __init__(self):
        self._parser_instances: Dict[str, Any] = {}

    def _get_parser(self, ext: str) -> Optional[Any]:
        """获取或创建解析器实例。"""
        if ext not in self._PARSER_MAP:
            return None

        if ext not in self._parser_instances:
            parser_class = self._PARSER_MAP[ext]
            self._parser_instances[ext] = parser_class()
            logger.debug(f"[ParserFactory] 初始化解析器: {parser_class.__name__}")

        return self._parser_instances[ext]

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """自动识别文件类型并解析。"""
        import os

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        parser = self._get_parser(ext)
        if parser is None:
            supported = list(self._PARSER_MAP.keys())
            raise ValueError(
                f"不支持的文件格式: '{ext}'，当前支持: {supported}"
            )

        logger.info(f"[ParserFactory] 使用 {type(parser).__name__} 解析: {file_path}")
        return parser.parse(file_path)

    def parse_directory(
        self,
        dir_path: str,
        recursive: bool = False,
    ) -> List[Dict[str, Any]]:
        """批量解析目录下所有支持格式的文件。"""
        import os

        all_results = []
        supported_exts = set(self._PARSER_MAP.keys())

        if recursive:
            file_paths = []
            for root, _, files in os.walk(dir_path):
                for filename in files:
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in supported_exts:
                        file_paths.append(os.path.join(root, filename))
        else:
            file_paths = [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if os.path.isfile(os.path.join(dir_path, f))
                and os.path.splitext(f)[1].lower() in supported_exts
            ]

        if not file_paths:
            logger.warning(f"[ParserFactory] 目录中未发现支持格式的文件: {dir_path}")
            return []

        logger.info(f"[ParserFactory] 共发现 {len(file_paths)} 个待解析文件")

        for filepath in file_paths:
            try:
                results = self.parse(filepath)
                all_results.extend(results)
                logger.info(f"  ✓ {os.path.basename(filepath)} -> {len(results)} 个块")
            except Exception as e:
                logger.error(f"  ✗ {os.path.basename(filepath)} 解析失败: {e}")
                continue

        logger.info(
            f"[ParserFactory] 批量解析完成：共 {len(file_paths)} 个文件 -> "
            f"{len(all_results)} 个文档块"
        )
        return all_results


__all__ = [
    "PDFParser",
    "MarkdownParser",
    "JSONParser",
    "TXTParser",
    "ParserFactory",
]
