"""
从大模型的"脏输出"里抠出 JSON（Day 7 第一块）

------------------------------------------------------------------
为什么需要这个文件
------------------------------------------------------------------
你要求模型"请只输出 JSON"，它依然会给你这些：

    ```json
    {"cases": [...]}
    ```
    ↑ 最常见的：包一层 Markdown 代码围栏

    好的，我帮你设计了以下用例：
    ```json
    {...}
    ```
    如果还需要性能测试，随时告诉我。
    ↑ 前后都加了客套话（模型被训练成"有礼貌的助手"，改不掉）

    {"cases": [...],}      ← 尾部多了个逗号，JSON 标准不允许
    {'cases': [...]}       ← 用了单引号，JSON 标准不允许

所以**不要相信模型会乖乖输出纯 JSON**，
要写一个"不管你怎么包，我都能抠出来"的解析器。
这是 LLM 应用工程里最常见、也最容易被忽略的一块。

------------------------------------------------------------------
设计原则：逐级降级，不要一步到位
------------------------------------------------------------------
直接用正则去抠，短平快，但遇到嵌套结构、字符串里含大括号就会错。
所以这里采用**从保守到激进**的策略，前一级成功就不再往后试：

    1. 整个文本直接 json.loads          ← 最理想，模型很乖
    2. 剥掉 ``` 代码围栏后再试           ← 覆盖 80% 的真实情况
    3. 只取代码块里的内容再试
    4. 截取第一个 { 到最后一个 }         ← 对付前后有废话的
    5. 截取第一个 [ 到最后一个 ]
    6. 上面都失败，做一次"修复"再试      ← 尾逗号 / 注释 / 单引号
    7. 还是不行，抛异常，交给上层的自修正逻辑

好处：每一级都简单、可单独测试、能打日志说清"在第几步救回来的"。
这些日志以后就是你的"模型有多不听话"的统计数据来源。
"""

from __future__ import annotations

import json
import re
from typing import Any

from tools.exceptions import LLMResponseError
from tools.logger import get_logger

logger = get_logger(__name__)

# 匹配 ```json ... ``` 或 ``` ... ```，非贪婪，跨行
FENCE_RE = re.compile(r"```[ \t]*(\w+)?[ \t]*\r?\n(.*?)```", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """剥掉 Markdown 代码围栏，返回中间的内容。

    有多个代码块时，取**最长的那个**——
    因为模型偶尔会先给一段示例代码、再给真正的 JSON，
    长的那个通常才是我们要的。
    """
    blocks = FENCE_RE.findall(text)
    if not blocks:
        return text.strip()

    contents = [body.strip() for _lang, body in blocks if body.strip()]
    if not contents:
        return text.strip()

    return max(contents, key=len)


def _repair(text: str) -> str:
    """修几个 JSON 标准不允许、但模型经常犯的小毛病。

    只做安全、无歧义的修复，不碰可能改变语义的东西：
        - 删掉 // 和 /* */ 注释
        - 删掉对象和数组末尾多余的逗号
    单引号转双引号不做——字符串内容里可能有英文撇号（don't），
    盲改会炸，宁可让上层重试。
    """
    # 删行注释和块注释
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"(?m)//.*$", "", text)
    # 删尾逗号： , 后面（允许空白）紧跟 } 或 ]
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text.strip()


def _candidates(text: str) -> list[str]:
    """按"保守 → 激进"的顺序列出所有可尝试的候选文本。"""
    raw = text.strip()
    out: list[str] = []

    def add(value: str) -> None:
        value = value.strip()
        if value and value not in out:
            out.append(value)

    add(raw)                        # 1. 原文直接解析
    add(strip_code_fence(raw))      # 2~3. 剥围栏

    # 4. 第一个 { 到最后一个 }
    first_brace, last_brace = raw.find("{"), raw.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        add(raw[first_brace : last_brace + 1])

    # 5. 第一个 [ 到最后一个 ]
    first_bracket, last_bracket = raw.find("["), raw.rfind("]")
    if first_bracket != -1 and last_bracket > first_bracket:
        add(raw[first_bracket : last_bracket + 1])

    return out


def extract_json(text: str) -> Any:
    """从任意文本里提取 JSON，返回 Python 对象（dict / list / ...）。

    失败时抛 LLMResponseError，并把原文前 500 字符记进日志——
    排查线上问题时，这段日志是唯一线索，一定要留。

    用法：
        data = extract_json(model_output)
        cases = data["cases"]     # 已知约定好的结构
    """
    if not text or not text.strip():
        raise LLMResponseError("模型返回内容为空，无法提取 JSON")

    # 第一轮：原样尝试
    for index, candidate in enumerate(_candidates(text), start=1):
        try:
            result = json.loads(candidate)
            if index > 1:
                logger.info("JSON 在第 %d 次尝试时解析成功（模型输出不干净）", index)
            return result
        except json.JSONDecodeError:
            continue

    # 第二轮：先修复再尝试
    for index, candidate in enumerate(_candidates(text), start=1):
        try:
            result = json.loads(_repair(candidate))
            logger.info("JSON 经修复后解析成功（第 %d 个候选）", index)
            return result
        except json.JSONDecodeError:
            continue

    preview = text.strip()[:500]
    logger.error("JSON 提取失败，原始输出预览：%s", preview)
    raise LLMResponseError(
        "无法从模型输出中解析出 JSON"
        f"（已尝试去代码围栏、截取花括号、修复尾逗号等 7 种方式）。"
        f"输出预览：{preview}"
    )


def extract_cases(data: Any) -> list[dict[str, Any]]:
    """从解析后的 JSON 里取出用例列表。

    模型对"外层包什么"的理解很随意，常见四种：
        {"cases": [...]}
        {"test_cases": [...]}
        {"feature": "...", "cases": [...]}
        [ {...}, {...} ]         ← 直接给数组

    统一在这里收敛，后面的代码就不用到处判断类型了。
    """
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        # 按优先级找第一个是列表的字段
        for key in ("cases", "test_cases", "testcases", "data", "result"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

        # 都没找到，但只有一个列表字段，那就是它
        lists = [v for v in data.values() if isinstance(v, list)]
        if len(lists) == 1:
            return [item for item in lists[0] if isinstance(item, dict)]

    raise LLMResponseError(
        f"JSON 结构里找不到用例列表，实际类型：{type(data).__name__}"
    )
