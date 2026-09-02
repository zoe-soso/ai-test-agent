"""
测试用例生成模块（Day 9）

------------------------------------------------------------------
Day 9 和 Day 7 的区别
------------------------------------------------------------------
Day 7 搭好了技术链路：structured.generate()
    LLM -> JSON -> 解析 -> 校验 -> 自修正
它"能用"，但它不关心业务——给你几条用例就交差，
至于有没有覆盖全三类场景，它不管。

Day 9 在它之上加一层**业务封装**，补上三件事：

    1. 场景覆盖自检：正常 / 异常 / 边界，缺哪类报出来
    2. 定向补充：缺了就再调一次 LLM，**专门补这一类**
    3. 统一出入口：generate / generate_from_file / save

------------------------------------------------------------------
关键设计：为什么"补充"比"重新生成"好
------------------------------------------------------------------
发现缺"边界"场景时，你有两个选择：

    A. 重新生成  —— 把已有的 8 条全扔掉，让模型再来一遍
       缺点：贵（8 条的 token 重新付一遍），
             而且可能这次又缺"异常"，来回折腾。

    B. 定向补充  —— 告诉模型"已有这些，缺边界，只补边界"
       优点：便宜（只生成 1~3 条）、精准、保住已有成果。

B 明显更好，但**前提是你能观察到结果、并据此决定下一步**——
这正是 Day 8 讲的 Agent 味道。

    观察（覆盖自检发现缺边界）
      ↓
    决策（只补边界，不全量重来）
      ↓
    行动（调补充模板）
      ↓
    再观察（重新自检）

所以 Day 9 这一层不是简单的"封装"，而是**第二个闭环**。
第一个闭环是 Day 7 的自修正（格式不对 -> 修格式），
第二个闭环是今天的覆盖补充（场景不全 -> 补场景）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent import json_utils, llm_client, requirement_reader, structured, validator
from agent.models import TestCase, TestSuite
from agent.validator import VALID_CASE_TYPES
from config import settings
from prompts import loader
from tools import file_io
from tools.exceptions import LLMResponseError
from tools.logger import get_logger

logger = get_logger(__name__)

# 复用 Day 7 校验器里的契约常量，避免两处各写一份导致不一致
REQUIRED_TYPES = VALID_CASE_TYPES
SUPPLEMENT_TEMPLATE = "supplement"


@dataclass
class CoverageReport:
    """场景覆盖自检报告。"""

    counts: dict[str, int] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    total: int = 0
    supplemented: int = 0      # 本次自动补了几条
    attempts: int = 0          # LLM 调用次数（含补充）

    @property
    def ok(self) -> bool:
        """三类场景都至少有一条，才算覆盖完整。"""
        return not self.missing

    def describe(self) -> str:
        parts = [f"{t} {self.counts.get(t, 0)} 条" for t in REQUIRED_TYPES]
        text = f"共 {self.total} 条（" + "，".join(parts) + "）"
        if self.missing:
            text += f" 缺：{'、'.join(self.missing)}"
        if self.supplemented:
            text += f"（自动补了 {self.supplemented} 条）"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "总数": self.total,
            **{t: self.counts.get(t, 0) for t in REQUIRED_TYPES},
            "缺失": "、".join(self.missing) if self.missing else "无",
            "自动补充": self.supplemented,
        }


class TestCaseGenerator:
    """测试用例生成器（业务层）。

    用法：
        gen = TestCaseGenerator()
        suite, report = gen.generate("用户登录功能", "邮箱+密码登录...")
        gen.save(suite)
    """

    def __init__(
        self,
        client: llm_client.LLMClient | None = None,
        template: str | None = None,
        *,
        auto_supplement: bool = True,
        max_repairs: int = 1,
    ) -> None:
        self.client = client or llm_client.get_client()
        self.template = template or structured.DEFAULT_TEMPLATE
        self.auto_supplement = auto_supplement
        self.max_repairs = max_repairs

    # ------------------------------------------------------------------
    # 核心入口
    # ------------------------------------------------------------------
    def generate(
        self, feature: str, description: str = "（无额外描述）"
    ) -> tuple[TestSuite, CoverageReport]:
        """生成测试用例，并保证三类场景都覆盖到。

        返回 (用例集, 覆盖报告)。
        注意：即使补充之后仍然缺某类，也**不抛异常**，
        而是如实写在 report.missing 里——宁可给出不完美的结果加警告，
        也不要让整条流水线崩掉。
        """
        result = structured.generate(
            feature,
            description,
            template=self.template,
            max_repairs=self.max_repairs,
            client=self.client,
        )

        cases: list[TestCase] = list(result.cases)
        report = self.check_coverage(cases)
        report.attempts = result.attempts

        logger.info("首次生成结果：%s", report.describe())

        # ---- 第二个闭环：发现缺场景 -> 定向补充 ----
        if report.missing and self.auto_supplement:
            logger.warning(
                "缺少 %s 场景，触发定向补充（不是全量重新生成）",
                "、".join(report.missing),
            )
            added = self._supplement(feature, cases, report.missing)

            if added:
                cases.extend(added)
                report = self.check_coverage(cases)
                report.supplemented = len(added)
                report.attempts = result.attempts + 1
                logger.info("补充后：%s", report.describe())
            else:
                logger.warning("补充生成没有拿到可用用例，保留原结果")

        suite: TestSuite = {"feature": feature, "cases": cases}
        return suite, report

    def generate_from_file(self, path: str | Path) -> tuple[TestSuite, CoverageReport]:
        """从需求文件生成（复用 Day 3 的需求解析）。"""
        requirement = requirement_reader.load(path)
        return self.generate(requirement.feature, requirement.description)

    def save(self, suite: TestSuite, path: str | Path | None = None) -> Path:
        """落盘 YAML。"""
        target = Path(path) if path else settings.OUTPUT_DIR / "testcases.yaml"
        file_io.write_yaml(target, suite)
        logger.info("已写入 YAML：%s", target)
        return target

    # ------------------------------------------------------------------
    # 覆盖自检
    # ------------------------------------------------------------------
    @staticmethod
    def check_coverage(cases: list[TestCase]) -> CoverageReport:
        """统计三类场景各有多少条，缺哪类。

        单独做成静态方法，是因为它**不依赖 LLM**，
        可以单独测试，也可以拿去检查任何来源的用例
        （手写的、AI 生成的、从 YAML 读回来的）。
        """
        counts: dict[str, int] = {t: 0 for t in REQUIRED_TYPES}
        for case in cases:
            case_type = case.get("case_type")
            if case_type in counts:
                counts[case_type] += 1

        missing = [t for t in REQUIRED_TYPES if counts[t] == 0]
        return CoverageReport(counts=counts, missing=missing, total=len(cases))

    # ------------------------------------------------------------------
    # 定向补充
    # ------------------------------------------------------------------
    def _supplement(
        self, feature: str, cases: list[TestCase], missing: list[str]
    ) -> list[TestCase]:
        """针对缺失的场景类型，再调一次 LLM 专门补。

        返回补充到的合格用例（已过滤，只保留目标类型）。
        失败返回空列表，不抛异常——补充是"尽力而为"的优化，
        不是必须成功的步骤。
        """
        existing = "\n".join(
            f"- {case['id']} [{case['case_type']}] {case['name']}" for case in cases
        )
        present = [t for t in REQUIRED_TYPES if t not in missing]

        prompt = loader.load(
            SUPPLEMENT_TEMPLATE,
            feature=feature,
            existing=existing or "（暂无）",
            present="、".join(present) or "无",
            missing="、".join(missing),
        )

        raw = self.client.chat_messages(prompt.to_messages(), mock_hint="json")

        try:
            data = json_utils.extract_json(raw)
            raws = json_utils.extract_cases(data)
        except LLMResponseError as exc:
            logger.warning("补充生成的结果无法解析：%s", exc)
            return []

        # 注意：validate_cases 返回 3 个值（合格, 全部详情, 未通过详情）。
        # structured 里那个返回 4 个的是它自己包装的私有函数，别搞混。
        passed, _, failed = validator.validate_cases(raws, "AUTO")

        # 只保留目标场景类型，防止模型"顺手"又生成一堆已有类型
        kept = [case for case in passed if case["case_type"] in missing]

        if failed:
            logger.warning("补充生成中有 %d 条校验未通过，已丢弃", len(failed))
        logger.info("补充生成：拿到 %d 条，其中目标类型 %d 条", len(passed), len(kept))
        return kept
