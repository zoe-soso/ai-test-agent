"""
大模型客户端（Day 5 —— 项目的第一个关键节点）

Day 5 只要搞懂一件事：Python 到底是怎么调用大模型的？

答案其实很朴素：**就是发一个 HTTP 请求**。

    Python 代码
        │
        │  ① 组装 messages（对话历史）
        │  ② 加上 API Key 做鉴权
        ▼
    POST https://api.deepseek.com/chat/completions
    {
      "model": "deepseek-chat",
      "messages": [
        {"role": "system",    "content": "你是一名资深测试工程师"},
        {"role": "user",      "content": "请帮我设计登录功能的测试用例"}
      ],
      "temperature": 0.2
    }
        │
        │  ③ 拿到 JSON 响应
        ▼
    {"choices": [{"message": {"role": "assistant", "content": "……"}}],
     "usage": {"prompt_tokens": 42, "completion_tokens": 317, "total_tokens": 359}}

`openai` 这个库只是帮你把这层 HTTP 封装成了 client.chat.completions.create()。
**不要把它想成什么黑魔法。**

几个必须理解的概念：
    API Key   你的身份凭证 + 计费账号。绝对不能提交到 Git。
    Token    模型计费和长度的单位。中文大约 1 个字 ≈ 0.6~1 个 token，
             不是"字符数"。max_tokens 限制的是**回复**的长度。
    Prompt   你发给模型的全部文字（system + user）。
    messages 对话历史。模型本身没有记忆，
             你每次都要把上下文重新发一遍 —— 这点以后做 Agent 时极其重要。
    temperature
             0 = 每次输出几乎一样（适合生成代码、结构化数据）
             1+ = 更有创造性，但也更不稳定
             生成测试用例建议 0.1~0.3。

为什么用 OpenAI 兼容协议？
    因为 DeepSeek、智谱、通义千问、Kimi、硅基流动全都兼容它。
    换供应商 = 改 .env 里两行，代码一行不动。这是本项目最重要的解耦之一。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from agent.mock_llm import MockLLM
from config import settings
from tools.exceptions import (
    LLMNotConfiguredError,
    LLMRequestError,
    LLMResponseError,
)
from tools.logger import get_logger

logger = get_logger(__name__)

# 默认的系统提示词。
# 现在真正的提示词都放在 prompts/*.txt 里（Day 6 抽出去的），
# 这个常量只用于 `client.chat("一句话")` 这种临时问一句的场景。
DEFAULT_SYSTEM_PROMPT = "你是一名资深软件测试工程师，回答简洁、专业、可执行。"

# Mock 模式下没有真实 usage，用字符数粗估一个量级，让成本指标也能演示。
# 中文约 1 字 0.6~1 token，英文约 4 字符 1 token，中英混排取折中。
ESTIMATE_CHARS_PER_TOKEN = 2.5


@dataclass
class TokenUsage:
    """累计的 token 消耗与成本（Day 7 引入）。

    为什么 Day 7 就要管成本？
        LLM 项目在真实工程里受两个硬约束：**钱** 和 **延迟**。
        一个跑得很爽的 Agent，如果每处理一条需求要花 5 毛钱，
        上线第一天就会被叫停。

        面试时能说出"我把单条需求生成 8 条用例的成本控制在 X 分钱"，
        比"我用了大模型"专业一个量级。

    这个对象会挂在 LLMClient 上自动累加，
    也能跨调用累计（评测脚本一次跑 5 条需求，要算总成本）。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    errors: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost(self) -> float:
        """按 settings 里的单价估算成本（元）。

        注意是**估算**：真实账单以供应商后台为准。
        这里的作用是比较不同 prompt / 不同模型的相对贵贱，
        以及给评测脚本一个"单次成本"指标。
        """
        return (
            self.prompt_tokens / 1_000_000 * settings.PRICE_INPUT_PER_1M
            + self.completion_tokens / 1_000_000 * settings.PRICE_OUTPUT_PER_1M
        )

    def add(self, usage: Any) -> None:
        """把一次真实调用的 usage 累加进来。"""
        self.calls += 1
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

    def add_estimate(self, prompt_text: str, answer_text: str) -> None:
        """Mock 模式下按字符数粗估累加。

        估算出来的数字不是真实账单，但量级是对的，
        足以支撑"哪个 prompt 更费钱"这类比较。
        """
        self.calls += 1
        self.prompt_tokens += int(len(prompt_text) / ESTIMATE_CHARS_PER_TOKEN)
        self.completion_tokens += int(len(answer_text) / ESTIMATE_CHARS_PER_TOKEN)

    def count_error(self) -> None:
        """记一次失败调用。失败率本身也是要盯的指标。"""
        self.errors += 1

    def reset(self) -> None:
        """清零，评测脚本每次开跑前会调一次。"""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.errors = 0

    def describe(self) -> dict[str, Any]:
        return {
            "调用次数": self.calls,
            "输入 token": self.prompt_tokens,
            "输出 token": self.completion_tokens,
            "合计 token": self.total_tokens,
            "估算成本": f"{settings.CURRENCY}{self.cost:.4f}",
        }

    def __str__(self) -> str:
        return (
            f"{self.calls} 次调用 / {self.total_tokens} tokens"
            f"（输入 {self.prompt_tokens}，输出 {self.completion_tokens}）"
            f"，约 {settings.CURRENCY}{self.cost:.4f}"
        )


class LLMClient:
    """大模型客户端：真实调用 + Mock 两种模式。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        mock_mode: str | None = None,
        mock_style: str | None = None,
    ) -> None:
        self.api_key = api_key or settings.LLM_API_KEY
        self.base_url = base_url or settings.LLM_BASE_URL
        self.model = model or settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE if temperature is None else temperature
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        self.timeout = timeout or settings.LLM_TIMEOUT
        self.system_prompt = system_prompt
        self.mock_mode = (mock_mode or settings.LLM_MOCK_MODE).lower()

        # 延迟创建：不在 __init__ 里建 OpenAI 对象。
        # 这样"没配 Key"不会导致 import 就崩，而是在真正要调用时才报错。
        self._client: Any = None

        # 离线假模型（会模拟真实模型的各种坏毛病，见 agent/mock_llm.py）
        # style 可显式指定，写测试时能强制复现某一种"坏毛病"
        self._mock_engine = MockLLM(style=mock_style or settings.LLM_MOCK_STYLE)

        # 累计的 token 消耗与成本，每次调用自动累加（Day 7）
        self.usage = TokenUsage()

    # ------------------------------------------------------------------
    # 模式判断
    # ------------------------------------------------------------------
    @property
    def is_mock(self) -> bool:
        """是否走离线 Mock。"""
        if self.mock_mode == "true":
            return True
        if self.mock_mode == "false":
            return False
        return not self.api_key  # auto：没 Key 就 Mock

    def _ensure_client(self) -> Any:
        """懒加载 OpenAI 客户端。"""
        if self._client is not None:
            return self._client

        if self.is_mock:
            return None

        if not self.api_key:
            raise LLMNotConfiguredError(
                "未配置 LLM_API_KEY",
                base_url=self.base_url,
                model=self.model,
            )

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMRequestError("openai 库未安装，请执行 pip install -r requirements.txt") from exc

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        logger.info("已创建 LLM 客户端：%s / %s", self.base_url, self.model)
        return self._client

    # ------------------------------------------------------------------
    # 核心：对话
    # ------------------------------------------------------------------
    def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 2,
        mock_hint: str | None = None,
    ) -> str:
        """发一句话给模型，拿回模型的回复文本。"""
        system = system_prompt or self.system_prompt
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.chat_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            mock_hint=mock_hint,
        )

    def chat_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 2,
        mock_hint: str | None = None,
    ) -> str:
        """直接发一组 messages（Day 6 起，Prompt 模板用的就是这个方法）。

        为什么要单独开一个方法？
            `chat()` 只能发"一句 user 话"，但 Prompt 模板里 system 和 user
            是分开设计的，有时还要塞 few-shot 的 assistant 示例。
            让 `chat()` 去拼 messages 只是个便捷封装，真正的入口是这里。

        retries：失败自动重试次数。
            网络抖动、限流在调用外部 API 时是家常便饭，
            简单重试能挡掉大部分偶发故障 —— 这是生产代码和玩具代码的分水岭。
        """
        if self.is_mock:
            logger.warning("当前为 Mock 模式，不会真实调用大模型")
            system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
            user = messages[-1]["content"] if messages else ""
            answer = self._mock_engine.respond(user, system=system, hint=mock_hint)
            # Mock 没有真实 usage，按字符数估个量级，让成本指标也能演示
            self.usage.add_estimate(system + user, answer)
            return answer

        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return self._request(messages, temperature, max_tokens)
            except LLMRequestError as exc:
                last_error = exc
                if attempt == retries:
                    break
                wait = attempt * 2
                logger.warning(
                    "第 %d/%d 次调用失败，%d 秒后重试：%s", attempt, retries, wait, exc
                )
                time.sleep(wait)

        raise LLMRequestError(
            f"调用大模型失败（已重试 {retries} 次）：{last_error}",
            model=self.model,
            base_url=self.base_url,
        )

    def _request(
        self,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        """真正发请求的地方。所有 openai 库的异常都在这里翻译成业务异常。"""
        client = self._ensure_client()

        total_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info(
            "调用 LLM：model=%s，messages=%d 条，共 %d 字符",
            self.model,
            len(messages),
            total_chars,
        )
        started = time.perf_counter()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
        except Exception as exc:  # openai 的异常类型很多，统一收敛
            # 常见：APIConnectionError(网络) / APITimeoutError(超时)
            #       AuthenticationError(Key 错) / RateLimitError(限流)
            self.usage.count_error()
            logger.error("LLM 请求异常：%s: %s", type(exc).__name__, exc)
            raise LLMRequestError(
                f"{type(exc).__name__}: {exc}",
                model=self.model,
                error_type=type(exc).__name__,
            ) from exc

        cost = time.perf_counter() - started

        # usage 里是本次调用的 token 消耗，直接决定花多少钱。
        # 记日志 + 累加到 self.usage（Day 7 的成本指标就靠这个）
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage.add(usage)
            logger.info(
                "LLM 返回：耗时 %.2fs，prompt=%s，completion=%s，total=%s | 累计 %s",
                cost,
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
                getattr(usage, "total_tokens", "?"),
                self.usage,
            )
        else:
            logger.info("LLM 返回：耗时 %.2fs（响应中没有 usage 字段）", cost)

        # ---- 截断检测（Day 12 补上）----
        #
        # 这是 LLM 应用里最阴险的一类故障，特点是没有明显症状：
        #   - HTTP 状态码 200，不报错
        #   - 返回的内容看起来就是正常的 JSON，只是**没写完**
        #   - 下游解析失败时，你会以为是"模型格式又乱了"，
        #     回头改 prompt、加自修正，折腾半天 ——
        #     其实真正的原因是自己的 max_tokens 设小了
        #
        # finish_reason == "length" 是唯一的线索，必须在这里抓住并明确报出来。
        try:
            finish_reason = response.choices[0].finish_reason
        except (AttributeError, IndexError):
            finish_reason = None

        if finish_reason == "length":
            self.usage.count_error()
            limit = max_tokens or self.max_tokens
            logger.error(
                "模型输出被截断：finish_reason=length，当前 max_tokens=%s。"
                "内容不完整，JSON 大概率解析失败。", limit,
            )
            raise LLMResponseError(
                f"模型输出被 max_tokens={limit} 截断，内容不完整。"
                f"请调大 LLM_MAX_TOKENS，或减少单次生成的内容量"
                f"（例如减少数据组数、缩短超长数据的长度）。",
                model=self.model,
            )

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise LLMResponseError("响应结构异常，取不到 choices[0].message.content") from exc

        if not content:
            raise LLMResponseError("模型返回内容为空", model=self.model)

        return content

    def chat_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """让模型返回 JSON，并自动解析成 Python 对象。

        现在只是"请求 + 尽力解析"，Day 7（Structured Output）会重点解决
        "模型不听话、在 JSON 前后加废话"的问题。
        """
        raw = self.chat(prompt, system_prompt=system_prompt, **kwargs)
        cleaned = _strip_code_fence(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("JSON 解析失败，原始返回：%s", raw)
            raise LLMResponseError(
                f"返回内容不是合法 JSON：{exc}", model=self.model
            ) from exc


    # ------------------------------------------------------------------
    # 工具调用（Day 13）
    # ------------------------------------------------------------------
    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 2,
    ) -> Any:
        """让模型在给定工具里挑一个调用，返回带工具调用请求的消息。

        为什么单独一个方法？
            Tool Calling（Day 13）和"纯聊天"是两种不同的 API 用法：
            普通 chat 只问一句话；工具调用要在请求里带上 tools 列表，
            模型可能返回"我要调某个工具"而不是文字。返回结构也不同，
            所以这里直接返回 `AssistantMessage`（含 tool_calls），
            而不是一段字符串。

        tool_choice="auto"：让模型自己决定调不调、调哪个；
        tool_choice="none"：强制模型只说文字、不许调工具
        （用于"最后该收尾了，别再调工具"的场景）。

        失败同样自动重试，和 chat_messages 一致。
        """
        from agent.tool_calling import AssistantMessage, ToolCall

        if self.is_mock:
            # 离线也要能跑通"模型要工具 → 我们执行 → 回传结果"的循环，
            # 否则 Day 13/26 的 Tool Calling 逻辑在没 Key 时永远验证不到。
            # MockLLM 内置了"按关键词决定调哪个工具"的剧本
            # （generate_testcase / rerun_test / analyze_failure ...），
            # 这里直接交给它，和真实路径走同一条 run_tool_loop。
            return self._mock_engine.respond_with_tools(messages, tools)

        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return self._request_with_tools(
                    messages, tools, tool_choice, temperature, max_tokens
                )
            except LLMRequestError as exc:
                last_error = exc
                if attempt == retries:
                    break
                wait = attempt * 2
                logger.warning(
                    "第 %d/%d 次工具调用失败，%d 秒后重试：%s", attempt, retries, wait, exc
                )
                time.sleep(wait)

        raise LLMRequestError(
            f"工具调用失败（已重试 {retries} 次）：{last_error}",
            model=self.model,
            base_url=self.base_url,
        )

    def _request_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str,
        temperature: float | None,
        max_tokens: int | None,
    ) -> Any:
        """真正发一次带 tools 的请求，并把返回解析成 AssistantMessage。

        把解析逻辑单独抽出来，和 _request 对称，也方便以后加测试桩。
        """
        from agent.tool_calling import AssistantMessage, ToolCall

        client = self._ensure_client()

        total_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info(
            "工具调用 LLM：model=%s，messages=%d 条，tools=%d 个，共 %d 字符",
            self.model, len(messages), len(tools), total_chars,
        )
        started = time.perf_counter()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
        except Exception as exc:  # openai 异常类型很多，统一收敛
            self.usage.count_error()
            logger.error("工具调用请求异常：%s: %s", type(exc).__name__, exc)
            raise LLMRequestError(
                f"{type(exc).__name__}: {exc}",
                model=self.model,
                error_type=type(exc).__name__,
            ) from exc

        cost = time.perf_counter() - started

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage.add(usage)
            logger.info(
                "工具调用返回：耗时 %.2fs，prompt=%s，completion=%s，total=%s | 累计 %s",
                cost,
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
                getattr(usage, "total_tokens", "?"),
                self.usage,
            )
        else:
            logger.info("工具调用返回：耗时 %.2fs（无 usage 字段）", cost)

        message = response.choices[0].message
        content = message.content

        tool_calls: list[ToolCall] = []
        for raw_call in (message.tool_calls or []):
            try:
                arguments = json.loads(raw_call.function.arguments or "{}")
            except json.JSONDecodeError:
                # 模型偶尔会吐出非法 JSON，宁可当成空参数，也不让整轮崩溃
                logger.warning("工具 %s 的参数不是合法 JSON，按空参数处理", raw_call.function.name)
                arguments = {}
            tool_calls.append(ToolCall(
                id=raw_call.id,
                name=raw_call.function.name,
                arguments=arguments,
            ))

        logger.info(
            "模型返回：%s工具调用 x%d",
            "有" if tool_calls else "无",
            len(tool_calls),
        )
        return AssistantMessage(content=content, tool_calls=tool_calls)


# ----------------------------------------------------------------------
# 模块级便捷函数
# ----------------------------------------------------------------------
_default_client: LLMClient | None = None


def get_client(**kwargs: Any) -> LLMClient:
    """获取全局默认客户端（单例）。"""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient(**kwargs)
    return _default_client


def chat(prompt: str, **kwargs: Any) -> str:
    """一句话调用：`chat("请帮我设计登录功能的测试用例")`。"""
    return get_client().chat(prompt, **kwargs)


# ----------------------------------------------------------------------
# 内部辅助
# ----------------------------------------------------------------------
def _strip_code_fence(text: str) -> str:
    """去掉模型爱加的 ```json ... ``` 代码围栏。

    这是和 LLM 打交道时的标准操作：你要求它输出 JSON，
    它有相当概率给你包一层 Markdown 代码块。别跟它较真，剥掉就行。
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # 去掉首行 ```json 和末行 ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# 注：Day 5 这里原本有一个返回固定字符串的 _mock_answer，
# Day 6 已被 agent/mock_llm.py 取代 —— 那个假模型会轮换模拟
# clean / fenced / chatty / broken 四种真实模型的坏毛病。
# 留着旧实现只会让人困惑"到底哪个在生效"，所以删掉了。
