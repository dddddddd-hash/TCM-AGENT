import json
import os
from typing import List, Dict, Any
from loguru import logger


class JSONParser:
    """
    JSON 文档解析器

    支持两种 JSON 结构：
    1. 数组型：每个元素转换为一个独立文档块
    2. 对象型：单对象直接转换，嵌套对象按 key 展开
    """

    PRIORITY_FIELDS = [
        "名称",
        "name",
        "方名",
        "药材名",
        "功效",
        "effect",
        "功能主治",
        "主治",
        "indication",
        "组成",
        "composition",
        "药物组成",
        "性味",
        "性味归经",
        "归经",
        "用法用量",
        "usage",
        "禁忌",
        "contraindication",
        "注意事项",
        "来源",
        "出处",
        "source",
        "描述",
        "description",
        "content",
    ]

    def __init__(self, max_text_length: int = 2000):
        self.max_text_length = max_text_length

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        解析 JSON 文件，返回统一格式的文档块列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON 文件不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败 [{file_path}]: {e}")

        logger.info(f"开始解析 JSON 文件: {file_path}")

        if isinstance(raw_data, list):
            result = self._parse_array(raw_data, file_path)
        elif isinstance(raw_data, dict):
            result = self._parse_object(raw_data, file_path)
        else:
            logger.warning(f"不支持的 JSON 根类型: {type(raw_data)}，跳过")
            return []

        logger.info(f"JSON 解析完成: {file_path}，共生成 {len(result)} 个文档块")
        return result

    def _parse_array(self, data: list, file_path: str) -> List[Dict[str, Any]]:
        result = []
        for idx, item in enumerate(data):
            if not isinstance(item, (dict, str)):
                logger.debug(f"跳过非文本数组元素 [index={idx}]: {type(item)}")
                continue

            if isinstance(item, str):
                content = item.strip()
            else:
                content = self._dict_to_text(item)

            if not content:
                continue

            if len(content) > self.max_text_length:
                content = content[: self.max_text_length] + "...[内容截断]"

            item_name = self._extract_item_name(item) if isinstance(item, dict) else f"条目{idx}"

            result.append(
                {
                    "content": content,
                    "metadata": {
                        "source": file_path,
                        "file_name": os.path.basename(file_path),
                        "record_index": idx,
                        "item_name": item_name,
                        "type": "json",
                        "json_structure": "array",
                        "chunk_id": f"{file_path}_record_{idx}",
                    },
                }
            )

        return result

    def _parse_object(self, data: dict, file_path: str) -> List[Dict[str, Any]]:
        all_values_complex = all(isinstance(v, (dict, list)) for v in data.values())

        if all_values_complex and len(data) > 1:
            result = []
            for key, value in data.items():
                if isinstance(value, dict):
                    enriched = {"名称": key, **value}
                    content = self._dict_to_text(enriched)
                elif isinstance(value, list):
                    content = f"{key}：\n" + self._list_to_text(value)
                else:
                    content = f"{key}：{value}"

                if not content.strip():
                    continue

                result.append(
                    {
                        "content": content[: self.max_text_length],
                        "metadata": {
                            "source": file_path,
                            "file_name": os.path.basename(file_path),
                            "item_name": str(key),
                            "type": "json",
                            "json_structure": "nested_object",
                            "chunk_id": f"{file_path}_key_{key}",
                        },
                    }
                )
            return result

        content = self._dict_to_text(data)
        if not content.strip():
            return []

        return [
            {
                "content": content[: self.max_text_length],
                "metadata": {
                    "source": file_path,
                    "file_name": os.path.basename(file_path),
                    "item_name": self._extract_item_name(data),
                    "type": "json",
                    "json_structure": "single_object",
                    "chunk_id": f"{file_path}_single",
                },
            }
        ]

    def _dict_to_text(self, data: dict, indent_level: int = 0) -> str:
        if not data:
            return ""

        lines = []
        indent = "  " * indent_level
        sorted_keys = self._sort_keys_by_priority(list(data.keys()))

        for key in sorted_keys:
            value = data[key]

            if value in (None, "", [], {}):
                continue

            if isinstance(value, dict):
                nested_text = self._dict_to_text(value, indent_level + 1)
                if nested_text:
                    lines.append(f"{indent}{key}：\n{nested_text}")
            elif isinstance(value, list):
                list_text = self._list_to_text(value)
                if list_text:
                    lines.append(f"{indent}{key}：{list_text}")
            elif isinstance(value, bool):
                lines.append(f"{indent}{key}：{'是' if value else '否'}")
            elif isinstance(value, (int, float)):
                lines.append(f"{indent}{key}：{value}")
            else:
                str_value = str(value).strip()
                if str_value:
                    lines.append(f"{indent}{key}：{str_value}")

        return "\n".join(lines)

    def _list_to_text(self, data: list) -> str:
        if not data:
            return ""

        if all(isinstance(item, (str, int, float)) for item in data):
            return "、".join(str(item) for item in data if str(item).strip())

        if all(isinstance(item, dict) for item in data):
            lines = []
            for item in data:
                item_text = " ".join(
                    f"{k}：{v}"
                    for k, v in item.items()
                    if not isinstance(v, (dict, list)) and v
                )
                if item_text:
                    lines.append(item_text)
            return "\n".join(lines)

        parts = []
        for item in data:
            if isinstance(item, dict):
                parts.append(self._dict_to_text(item))
            else:
                parts.append(str(item))
        return "；".join(p for p in parts if p.strip())

    def _sort_keys_by_priority(self, keys: List[str]) -> List[str]:
        priority_set = {f.lower(): i for i, f in enumerate(self.PRIORITY_FIELDS)}

        def sort_key(key: str) -> int:
            return priority_set.get(key.lower(), len(self.PRIORITY_FIELDS))

        return sorted(keys, key=sort_key)

    def _extract_item_name(self, data: dict) -> str:
        name_fields = ["名称", "name", "方名", "药材名", "title", "标题", "id"]
        for field in name_fields:
            for key in data.keys():
                if key.lower() == field.lower():
                    value = data[key]
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        return "未命名条目"

    def parse_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        all_results = []
        json_files = [
            f
            for f in os.listdir(dir_path)
            if f.endswith(".json") and os.path.isfile(os.path.join(dir_path, f))
        ]

        if not json_files:
            logger.info(f"目录 {dir_path} 中未发现 JSON 文件")
            return []

        for filename in json_files:
            filepath = os.path.join(dir_path, filename)
            try:
                results = self.parse(filepath)
                all_results.extend(results)
            except (ValueError, FileNotFoundError) as e:
                logger.error(f"JSON 文件解析失败，跳过: {filename} -> {e}")
                continue

        logger.info(f"目录批量解析完成: {dir_path}，共 {len(all_results)} 个文档块")
        return all_results
