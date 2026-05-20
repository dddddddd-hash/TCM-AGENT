from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple


@dataclass
class SafetyDecision:
    action: str = "proceed"
    risk_level: str = "low"
    reasons: List[str] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)
    matched_terms: List[str] = field(default_factory=list)


class TCMMedicalSafetyGuardrail:
    """Lightweight medical safety checks for the TCM QA agent."""

    red_flag_terms = {
        "possible_emergency": [
            "胸痛",
            "胸闷",
            "呼吸困难",
            "喘不上气",
            "意识不清",
            "昏迷",
            "抽搐",
            "剧烈腹痛",
            "吐血",
            "呕血",
            "便血",
            "黑便",
            "持续高热",
            "高热不退",
            "意识模糊",
            "脱水",
            "休克",
            "自杀",
            "服药过量",
            "压榨样疼痛",
        ],
        "special_population": [
            "孕妇",
            "怀孕",
            "妊娠",
            "婴儿",
            "新生儿",
            "老人",
            "高龄",
        ],
    }

    symptom_terms = [
        "痛",
        "胀",
        "拉肚子",
        "腹泻",
        "便秘",
        "咳嗽",
        "发热",
        "怕冷",
        "乏力",
        "失眠",
        "头晕",
        "恶心",
        "呕吐",
        "食欲",
        "纳差",
        "口苦",
        "口干",
        "出汗",
        "月经",
        "胃",
        "腹",
        "胸",
        "心悸",
        "出汗",
    ]

    knowledge_intent_terms = [
        "组成",
        "主治",
        "功效",
        "方义",
        "出处",
        "条文",
        "是什么",
        "主要讲",
        "讲了什么",
        "解释",
        "了解",
        "比较",
        "鉴别要点",
        "常见",
        "为什么",
        "理论",
    ]

    symptom_consult_intent_terms = [
        "证型",
        "辨证",
        "调理",
        "帮我看看",
        "是什么问题",
        "怎么治",
        "吃什么",
        "开什么",
        "开个",
    ]

    missing_info_checks = [
        ("病程和发作频率", ["多久", "几天", "几周", "几月", "一年", "多年", "频率", "每天", "反复"]),
        ("寒热、汗出和口渴情况", ["怕冷", "怕热", "发热", "出汗", "盗汗", "口渴", "口干"]),
        ("饮食、二便和睡眠情况", ["食欲", "纳差", "大便", "小便", "睡眠", "失眠", "便秘", "腹泻"]),
        ("舌象和脉象", ["舌", "苔", "脉"]),
        ("既往诊断、用药和特殊人群情况", ["诊断", "用药", "服药", "怀孕", "孕", "儿童", "老人"]),
    ]

    strong_diagnosis_terms = [
        "高度指向",
        "基本可以确定",
        "无疑就是",
        "一定是",
        "必然是",
        "确诊为",
    ]

    dosage_pattern = re.compile(r"\d+(\.\d+)?\s*(g|克|毫升|ml|片|丸|袋|次/日|每日)", re.IGNORECASE)

    def assess_query(self, query: str) -> SafetyDecision:
        red_flags = self._matched_terms(query, self._flatten(self.red_flag_terms.values()))
        if red_flags:
            return SafetyDecision(
                action="urgent_referral",
                risk_level="high",
                reasons=["命中红旗症状或特殊人群，需要优先线下评估"],
                matched_terms=red_flags,
            )

        if not self._is_symptom_question(query) or self._is_knowledge_question(query):
            return SafetyDecision(action="proceed", risk_level="low")

        missing_info = self._missing_info(query)
        if len(missing_info) >= 3:
            return SafetyDecision(
                action="clarify",
                risk_level="medium",
                reasons=["症状类问题缺少辨证关键信息"],
                missing_info=missing_info,
                matched_terms=self._matched_terms(query, self.symptom_terms),
            )

        return SafetyDecision(
            action="proceed",
            risk_level="low",
            missing_info=missing_info,
            matched_terms=self._matched_terms(query, self.symptom_terms),
        )

    def should_clarify_after_react(
        self,
        decision: SafetyDecision,
        confidence: float,
        evidence_count: int,
        next_action: str,
    ) -> bool:
        if next_action == "clarify":
            return True
        if decision.action == "clarify" and confidence < 0.8:
            return True
        if confidence < 0.5 or evidence_count <= 1:
            return True
        return False

    def build_urgent_response(self, decision: SafetyDecision) -> str:
        terms = "、".join(decision.matched_terms) if decision.matched_terms else "相关高风险表现"
        return (
            "我先不建议只按中医辨证自行处理。你描述中出现了需要优先排查风险的表现："
            f"{terms}。\n\n"
            "建议尽快到正规医疗机构就诊，必要时选择急诊。线上问答无法完成体格检查、生命体征评估"
            "和必要检查，因此不适合在这里给出诊断或用药方案。\n\n"
            "就医前可以记录：症状开始时间、加重或缓解因素、体温、疼痛部位和程度、大小便情况、"
            "既往病史及正在使用的药物。"
        )

    def build_clarification_response(
        self,
        query: str,
        decision: SafetyDecision,
        confidence: float,
        evidence_count: int,
    ) -> str:
        missing_info = decision.missing_info or self._missing_info(query)
        if not missing_info:
            missing_info = ["病程和发作频率", "舌象和脉象", "饮食、二便和睡眠情况"]

        questions = [f"{idx}. 请补充{item}。" for idx, item in enumerate(missing_info[:4], 1)]
        return (
            "目前信息还不足以稳妥辨证，我更倾向于先追问，而不是直接下结论或给方药。\n\n"
            f"当前系统置信度约为 {confidence:.0%}，可用证据数为 {evidence_count} 条。\n\n"
            "请你补充以下信息：\n"
            + "\n".join(questions)
            + "\n\n补充后我可以再结合检索证据，给出更谨慎的辨证倾向、鉴别方向和证据边界。"
        )

    def sanitize_answer(self, answer: str) -> Tuple[str, List[str]]:
        flags: List[str] = []
        if self.dosage_pattern.search(answer):
            flags.append("contains_exact_dosage")
        if any(term in answer for term in self.strong_diagnosis_terms):
            flags.append("contains_overcertain_language")

        notes = []
        if "contains_exact_dosage" in flags:
            notes.append("涉及具体方药剂量的内容不宜仅凭线上问答确定，需由中医师面诊辨证后决定。")
        if "contains_overcertain_language" in flags:
            notes.append("以上辨证仅代表基于当前信息和证据的倾向性判断，不应视为确诊。")

        if notes:
            answer = answer.rstrip() + "\n\n【安全提示】" + " ".join(notes)
        return answer, flags

    def _is_symptom_question(self, query: str) -> bool:
        return any(term in query for term in self.symptom_terms)

    def _is_knowledge_question(self, query: str) -> bool:
        if any(marker in query for marker in ["不是让我诊断", "不涉及个人用药", "只想了解", "解释为什么"]):
            return True
        if any(term in query for term in self.symptom_consult_intent_terms):
            return False
        return any(term in query for term in self.knowledge_intent_terms)

    def _missing_info(self, query: str) -> List[str]:
        missing = []
        for label, terms in self.missing_info_checks:
            if not any(term in query for term in terms):
                missing.append(label)
        return missing

    @staticmethod
    def _matched_terms(text: str, terms: Iterable[str]) -> List[str]:
        matched = []
        for term in terms:
            if term not in text:
                continue
            term_index = text.find(term)
            negation_window = text[max(0, term_index - 8):term_index]
            if (
                f"没有{term}" in text
                or f"无{term}" in text
                or f"未见{term}" in text
                or "没有" in negation_window
                or "无" in negation_window
                or "未见" in negation_window
            ):
                continue
            matched.append(term)
        return matched

    @staticmethod
    def _flatten(groups: Iterable[Iterable[str]]) -> List[str]:
        return [term for group in groups for term in group]
