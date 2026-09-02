"""
Prompt A/B 对比实验（Day 6）

Day 6 要回答的问题只有一个：
    **同样是"设计登录测试用例"这句话，
      换个说法，模型给出的东西能差多少？**

答案是：差非常多。这个项目里最便宜、回报最高的一次优化，
就是把 prompt 从"请生成登录测试用例"改写成带角色 + 约束 + 示例的版本。
改 prompt 不花钱、不写代码，但效果经常比换模型还明显。

------------------------------------------------------------------
四个必须掌握的概念
------------------------------------------------------------------

1. System Prompt（系统提示词）
    给模型"定人设、定规则"，在整个对话里长期生效。
    类比：你给新同事发的《岗位说明书》，说一次，后面一直按这个来。

2. User Prompt（用户提示词）
    这一次具体要它干什么。类比：你递给同事的一张具体任务单。

3. Role（角色）
    "你是一名有 8 年经验的资深测试工程师"——这句话不是废话。
    模型的输出分布会朝这个角色的语料偏移，
    同样的问题，挂上角色后专业度和术语准确度明显不同。

4. Constraint（约束）
    新手最容易漏的东西。你不说"必须覆盖边界场景"，
    模型就默认给你 3 条最显而易见的正向用例。
    约束要写死、要可检查：
        差：尽量全面一些
        好：必须覆盖正常/异常/边界三类，每类至少 1 条

5. Few-shot（示例）
    光用文字描述"我要什么格式"说不清，直接给它一个例子。
    本项目模板里的【示例】就是 one-shot（给一个例子）。
    给 2~3 个叫 few-shot。对输出格式的约束力比任何描述都强。

------------------------------------------------------------------
这个文件做什么
------------------------------------------------------------------
把两个版本的 prompt 喂给模型，拿到两份输出，
做一次粗粒度打分，然后把**原文 + 分数**一起存下来。

注意 quick_score 的定位：
    它是"纯文本统计"的粗略参考，不调 LLM，成本为零。
    真正严谨的评测在 Day 7 的 eval/ 里（结构化解析之后再算分）。
    这里存在的意义是让你**当场就能看到差距**，而不是靠感觉。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent import llm_client
from prompts import loader
from tools.logger import get_logger

logger = get_logger(__name__)

# 模板名 -> 喂给 Mock 的提示（告诉假模型该演哪种风格）
# 真实模型不需要这个，Mock 模式下靠它模拟"不同 prompt 得到不同质量的输出"
MOCK_HINTS: dict[str, str] = {
    "testcase_v0_naive": "naive",
    "testcase_v1_engineered": "engineered",
    "testcase_v2_json": "json",
    "testcase_v3_method": "json",
}

# 默认对比的两个版本
DEFAULT_VARIANTS: tuple[str, ...] = ("testcase_v0_naive", "testcase_v1_engineered")


@dataclass
class CompareItem:
    """一个 prompt 版本的实验结果。"""

    name: str
    system: str
    user: str
    output: str
    score: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_chars(self) -> int:
        """Prompt 本身有多长——直接决定输入 token 成本。"""
        return len(self.system) + len(self.user)


def quick_score(text: str) -> dict[str, Any]:
    """粗粒度打分：不调 LLM，纯文本统计，成本为零。

    只看几件事，因为它对应"这份输出能不能直接给自动化用"：
        - 有几条用例（太少说明覆盖不足）
        - 有没有边界、异常场景（朴素 prompt 最容易漏这两类）
        - 预期结果写没写（没写就没法转成断言）
        - 有没有优先级（没有就没法排执行顺序）
    """
    return {
        "用例条数": count_cases(text),
        "字符数": len(text),
        "覆盖边界场景": "边界" in text,
        "覆盖异常场景": "异常" in text,
        "写了预期结果": ("expected" in text) or ("预期" in text),
        "标了优先级": bool(re.search(r"\bP[0-3]\b", text)),
        "有可执行步骤": ("steps" in text) or ("步骤" in text),
    }


def count_cases(text: str) -> int:
    """数文本里有几条用例。

    两种写法都认：
        结构化：id: TC_LOGIN_001
        自然语：1. 正常登录

    先数 TC_ 编号（更准），数不到再退化成数行首编号。
    """
    ids = re.findall(r"TC_[A-Z]+_\d+", text)
    if ids:
        return len(set(ids))

    numbered = re.findall(r"(?m)^\s*(\d+)\s*[.、)]", text)
    return len(numbered)


def compare(
    feature: str,
    description: str = "（无额外描述）",
    variants: tuple[str, ...] | list[str] = DEFAULT_VARIANTS,
    client: llm_client.LLMClient | None = None,
) -> list[CompareItem]:
    """跑一次 A/B 对比。

    为什么 prompt 从文件加载而不是写在代码里？
        因为调 prompt 是最高频的操作，一天改十几次。
        抽成文件后：改文案不用动代码、能做版本对比、非程序员也能改。
        详见 prompts/loader.py 的模块说明。

    client 可以从外面传进来（写测试时方便注入），不传就用全局默认。
    """
    client = client or llm_client.get_client()
    items: list[CompareItem] = []

    for name in variants:
        prompt = loader.load(name, feature=feature, description=description)

        logger.info("对比实验：%s（system %d 字，user %d 字）",
                    name, len(prompt.system), len(prompt.user))

        output = client.chat_messages(
            prompt.to_messages(),
            mock_hint=MOCK_HINTS.get(name),
        )

        items.append(
            CompareItem(
                name=name,
                system=prompt.system,
                user=prompt.user,
                output=output,
                score=quick_score(output),
            )
        )

    return items


def render_report(items: list[CompareItem], feature: str) -> str:
    """把对比结果渲染成 Markdown，方便直接存进 outputs/ 慢慢看。"""
    lines: list[str] = [
        f"# Prompt A/B 对比报告",
        "",
        f"- 被测功能：{feature}",
        f"- 对比版本：{len(items)} 个",
        "",
        "## 一、分数对比",
        "",
    ]

    # 表头
    if items:
        metrics = list(items[0].score.keys())
        lines.append("| 指标 | " + " | ".join(i.name for i in items) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in items) + " |")
        for metric in metrics:
            cells = []
            for item in items:
                value = item.score.get(metric)
                cells.append("✅" if value is True else ("❌" if value is False else str(value)))
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")

        lines.append("")
        lines.append("| 版本 | Prompt 长度（字符） | 输出长度（字符） |")
        lines.append("| --- | --- | --- |")
        for item in items:
            lines.append(f"| {item.name} | {item.prompt_chars} | {len(item.output)} |")

    lines.append("")
    lines.append("## 二、原始输出")
    for item in items:
        lines.extend(
            [
                "",
                f"### {item.name}",
                "",
                "````text",
                item.output,
                "````",
            ]
        )

    return "\n".join(lines)
