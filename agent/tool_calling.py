"""
Tool Calling（工具调用）机制（Day 13）

这是本项目第二个非常关键的 AI 知识点（第一个是 Day 7 的 Structured Output）。

------------------------------------------------------------------
Tool Calling 到底解决什么问题
------------------------------------------------------------------
大模型有两个天然缺陷：

    1. **它不知道新信息**
       训练数据截止之后发生的事，它一概不知。
       你问"我项目里 tests/ 目录下有几个测试文件"，它只能瞎编。

    2. **它不能真的做事**
       它只能"说"，不能"做"。
       它能写出一段完美的 pytest 命令，但没法真的去执行它。

Tool Calling 就是补上这两块短板：
**让模型能够"点名"调用你写的 Python 函数，拿到真实结果，再继续说话。**

    ┌──────────────────────────────────────────────────────┐
    │  你（Python）          模型               你（Python）  │
    │                                                        │
    │  ① 我能用这些工具 ────>                                │
    │     [run_pytest, generate_testcase, read_file]         │
    │                                                        │
    │                     ② 思考后决定：                     │
    │                        "我要调 run_pytest"             │
    │  <──── 请执行 run_pytest(target="tests/") ─────────────│
    │                                                        │
    │  ③ 真的去执行（模型做不到这一步）                       │
    │     拿到：7 passed, 2 failed                           │
    │                                                        │
    │  ④ 把结果塞回对话 ────>                                │
    │                     ⑤ 看到结果，用自然语言总结           │
    │  <──── "测试有 2 条失败，分别是……" ─────────────────────│
    └──────────────────────────────────────────────────────┘

**关键点：模型从头到尾没有执行任何代码。**
它只是"说"了要调什么，真正执行的是你的 Python。
这个区别非常重要 —— 它意味着**控制权始终在你手里**：
你可以拒绝执行、可以加人工确认、可以记录审计日志。
Day 19 的 Human-in-the-loop 就是建立在这个基础上的。

------------------------------------------------------------------
本文件的设计取舍
------------------------------------------------------------------
工具定义用**装饰器 + 类型注解 + docstring** 自动生成 JSON Schema，
而不是手写一大坨 schema 字典。

为什么？
    手写 schema 最大的问题是"三处不同步"：
        函数签名改了 → schema 忘了改 → 模型传错参数 → 运行时才炸
    从签名自动生成，三者永远一致，改函数就等于改文档。

    代价是要写一段解析注解和 docstring 的代码（下面 100 行）。
    这个代价值得，而且这段代码本身也是很好的 Python 练习
    （inspect / typing / get_type_hints 的实际用法）。
"""

from __future__ import annotations

import inspect
import json
import re
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

from tools.logger import get_logger

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------
@dataclass
class ToolCall:
    """模型发起的一次工具调用请求。"""

    id: str                 # 调用 ID，回传结果时要用它对上号
    name: str               # 工具名
    arguments: dict[str, Any]


@dataclass
class AssistantMessage:
    """模型返回的一条消息（可能带工具调用）。

    为什么要自己定义这个类型，而不直接用 openai 的返回对象？
        因为 Mock 模式没有 openai 对象。
        定义一个中间结构，真实调用和 Mock 走同一条代码路径 ——
        这样 Mock 测过的逻辑，在真实调用时才能真的成立。
    """

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class ToolSpec:
    """一个已注册的工具。"""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]


@dataclass
class StepRecord:
    """循环里的一步，用于事后复盘和命令行展示。"""

    step: int
    tool: str
    arguments: dict[str, Any]
    result: str
    error: str = ""

    def __str__(self) -> str:
        args = json.dumps(self.arguments, ensure_ascii=False)
        head = f"  [{self.step}] {self.tool}({args})"
        if self.error:
            return f"{head}\n      ✗ {self.error}"
        preview = self.result if len(self.result) <= 200 else self.result[:197] + "..."
        return f"{head}\n      -> {preview}"


@dataclass
class ToolLoopResult:
    """一次工具调用循环的完整结果。"""

    answer: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    iterations: int = 0
    stopped_reason: str = ""

    @property
    def tool_names(self) -> list[str]:
        return [s.tool for s in self.steps]

    def summary(self) -> dict[str, Any]:
        return {
            "循环轮数": self.iterations,
            "调用工具": " → ".join(self.tool_names) if self.steps else "无",
            "结束原因": self.stopped_reason,
        }


# ----------------------------------------------------------------------
# 从函数签名 + docstring 生成 JSON Schema
# ----------------------------------------------------------------------
ARGS_HEADS = ("Args:", "参数:", "Arguments:", "Parameters:")
STOP_HEADS = ("Returns:", "返回:", "Raises:", "Yields:",
              "Example:", "示例:", "Example", "Note:", "注意:")
_PARAM_LINE = re.compile(r"^(\w+)\s*[：:]\s*(.*)$")


def parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """解析 Google 风格 docstring，返回 (整体描述, {参数名: 参数说明})。

    支持这种写法：

        def run_pytest(target: str) -> str:
            \"\"\"执行 pytest 并返回结果摘要。

            Args:
                target: 要执行的测试路径
            \"\"\"

    参数说明不是可有可无的装饰 —— **它会原样发给模型**，
    模型全靠这段文字判断该传什么值。写得含糊，模型就会传错。
    """
    description_lines: list[str] = []
    param_docs: dict[str, str] = {}
    section = "desc"
    current: str | None = None

    for raw in (doc or "").splitlines():
        stripped = raw.strip()

        if stripped in ARGS_HEADS:
            section, current = "args", None
            continue

        if any(stripped.startswith(head) for head in STOP_HEADS):
            section, current = "other", None
            continue

        if section == "desc":
            if stripped:
                description_lines.append(stripped)

        elif section == "args":
            matched = _PARAM_LINE.match(stripped)
            if matched:
                current = matched.group(1)
                param_docs[current] = matched.group(2).strip()
            elif current and stripped:
                # 上一参数的说明换行续写了，接上去
                param_docs[current] += " " + stripped

    return " ".join(description_lines).strip(), param_docs


def _type_to_schema(annotation: Any) -> dict[str, Any]:
    """把 Python 类型注解翻译成 JSON Schema 的类型描述。"""
    origin = get_origin(annotation)

    # str | None 这类可选类型：剥掉 None，用剩下的那个类型
    if origin in (Union, types.UnionType):
        candidates = [a for a in get_args(annotation) if a is not type(None)]
        if len(candidates) == 1:
            return _type_to_schema(candidates[0])
        # 多个候选类型说不清，退化成字符串，让模型自己组织
        return {"type": "string"}

    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if origin is list:
        item = get_args(annotation)
        return {"type": "array", "items": _type_to_schema(item[0]) if item else {"type": "string"}}

    # 认不出来的类型一律当字符串。
    # 宁可让模型传字符串进来我们再转，也不要生成非法 schema 把 API 调用搞挂。
    return {"type": "string"}


_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")


def _clean_description(text: str) -> str:
    """把 docstring 里的 Markdown 记号去掉，只留纯文本。

    为什么要清洗？
        写 docstring 时习惯用 **加粗** 强调重点，那是为了让人读代码时好读。
        但这段文字会原样发给模型 —— 模型不认 Markdown，
        `**` 对它来说只是两个无意义的星号，白白占 token，偶尔还会被它学去。
        代码里该好看的照样好看，发给模型时干净就行。
    """
    return _MD_BOLD.sub(r"\1", text).strip()


def build_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """从一个函数生成 OpenAI tools 需要的 parameters JSON Schema。"""
    signature = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:  # noqa: BLE001 - 注解写得很花哨时 get_type_hints 会失败
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if name in ("self", "cls"):
            continue

        annotation = hints.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = str

        schema = _type_to_schema(annotation)
        schema["description"] = ""   # 稍后由 docstring 的参数说明填充
        properties[name] = schema

        if param.default is inspect.Parameter.empty:
            required.append(name)

    description, param_docs = parse_docstring(inspect.getdoc(func) or "")
    for name, text in param_docs.items():
        if name in properties and text:
            properties[name]["description"] = _clean_description(text)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


# ----------------------------------------------------------------------
# 工具注册表
# ----------------------------------------------------------------------
class ToolRegistry:
    """工具的登记处。

    用法：

        registry = ToolRegistry()

        @registry.tool
        def run_pytest(target: str) -> str:
            \"\"\"执行 pytest。

            Args:
                target: 测试路径
            \"\"\"
            ...
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """装饰器：把一个函数登记成工具。"""
        description, _ = parse_docstring(inspect.getdoc(func) or "")
        name = func.__name__
        self._tools[name] = ToolSpec(
            name=name,
            description=_clean_description(description) or f"调用 {name}",
            parameters=build_schema(func),
            func=func,
        )
        logger.debug("已注册工具：%s", name)
        return func

    def register(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """`register(...)` 形式（等价装饰器，读起来更明确时用这个）。"""
        return self.tool(func)

    # ---- 查询 ----
    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """生成发给模型的 tools 数组。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self._tools.values()
        ]

    def describe(self) -> str:
        """给人看的工具清单（命令行 --list 用）。"""
        if not self._tools:
            return "（尚未注册任何工具）"
        lines = []
        for spec in self._tools.values():
            params = spec.parameters.get("properties", {})
            required = set(spec.parameters.get("required", []))
            rendered = ", ".join(
                f"{n}{'*' if n in required else ''}" for n in params
            ) or "无参数"
            lines.append(f"  {spec.name}({rendered})")
            lines.append(f"      {spec.description}")
            for name, meta in params.items():
                if meta.get("description"):
                    lines.append(f"      - {name}: {meta['description']}")
        lines.append("  （带 * 的是必填参数）")
        return "\n".join(lines)

    # ---- 执行 ----
    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """执行一个工具，返回**字符串**结果。

        为什么返回值必须是字符串？
            tools 结果要以文本的形式回传给模型，
            模型读的是文字，不是 Python 对象。
            所以工具实现里要自己把结果转成好读的文本。

        这里不抛异常：工具出错时把错误信息作为结果返回给模型，
        让它自己决定怎么办（换个参数重试 / 换别的工具 / 向用户报告）。
        这比直接崩溃好得多 —— 而且这正是 Agent 该有的样子。
        """
        spec = self._tools.get(name)
        if spec is None:
            return (
                f"[错误] 没有名为 {name} 的工具。"
                f"可用的工具有：{', '.join(self.names)}"
            )

        try:
            result = spec.func(**arguments)
        except TypeError as exc:
            # 参数对不上，十有八九是模型传了不存在的参数名
            return (
                f"[错误] 调用 {name} 时参数不匹配：{exc}。"
                f"该工具接受的参数：{', '.join(spec.parameters.get('properties', {}))}"
            )
        except Exception as exc:  # noqa: BLE001 - 工具内部异常要变成模型能读的反馈
            logger.error("工具 %s 执行异常：%s", name, exc, exc_info=True)
            return f"[错误] 工具 {name} 执行失败：{type(exc).__name__}: {exc}"

        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, indent=2)
        return result


# ----------------------------------------------------------------------
# 工具调用循环
# ----------------------------------------------------------------------
def run_tool_loop(
    client: Any,
    messages: list[dict[str, Any]],
    registry: ToolRegistry,
    *,
    max_iterations: int = 5,
    on_step: Callable[[StepRecord], None] | None = None,
) -> ToolLoopResult:
    """跑"模型要工具 → 我们执行 → 结果回传"的循环，直到模型不再要工具。

    参数：
        client          带 chat_with_tools 方法的 LLM 客户端
        messages        对话历史（会被**就地修改**，把工具结果追加进去）
        registry        工具注册表
        max_iterations  最多循环几轮
        on_step         每执行完一个工具的回调，用于实时打印

    ------------------------------------------------------------------
    为什么要限制 max_iterations
    ------------------------------------------------------------------
    这是**必须的**安全护栏，不是可选优化。
    模型可能陷入"调工具 → 结果不满意 → 换个参数再调"的死循环，
    每一轮都是真金白银。没有上限的话，一个跑飞的 Agent
    能在你意识到之前烧掉几十块钱。

    5 轮的取值依据：本项目的任务（生成用例 → 保存 → 执行 → 分析）
    正常 2~3 轮就收敛。给到 5 轮留足余量，又能挡住跑飞的情况。

    ------------------------------------------------------------------
    为什么工具出错不中断循环
    ------------------------------------------------------------------
    工具报错时，我们把错误信息作为结果回传给模型，
    而不是抛异常终止。因为**模型有能力自我纠正**：
    它可能传错了参数，看到错误提示后往往能改对。
    直接中断等于放弃了这个机会 —— 也就不叫 Agent 了。
    """
    result = ToolLoopResult()

    for iteration in range(1, max_iterations + 1):
        result.iterations = iteration

        message = client.chat_with_tools(messages, registry.schemas())

        # 模型没要工具 —— 它给出了最终答复，循环结束
        if not message.wants_tools:
            result.answer = message.content or ""
            result.stopped_reason = "模型给出最终答复"
            return result

        # 把"模型要调工具"这个事实记进对话历史。
        # 缺了这一步，模型下一轮就不知道自己刚才说过要调什么，
        # 会出现反复调同一个工具的现象。
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ],
        })

        for call in message.tool_calls:
            logger.info("第 %d 轮：模型请求调用 %s(%s)",
                        iteration, call.name, call.arguments)

            outcome = registry.call(call.name, call.arguments)

            record = StepRecord(
                step=iteration,
                tool=call.name,
                arguments=call.arguments,
                result=outcome,
                error="" if not outcome.startswith("[错误]") else outcome,
            )
            result.steps.append(record)
            if on_step:
                on_step(record)

            # 把执行结果回传给模型，用 tool_call_id 对上号
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": outcome,
            })

    # 循环耗尽
    result.stopped_reason = f"已达最大轮数 {max_iterations}，强制结束"
    result.answer = (
        f"（达到最大工具调用轮数 {max_iterations}，未获得最终答复）"
    )
    logger.warning(result.stopped_reason)
    return result
