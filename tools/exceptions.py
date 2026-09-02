"""
自定义异常（Day 4）

知识点：为什么要自定义异常，而不是到处 raise Exception / ValueError？

1. 调用方能精确捕获
       try:
           call_llm()
       except LLMNotConfiguredError:      # 没配 Key，去提示用户配置
           ...
       except LLMResponseError:           # 模型返回了脏数据，重试
           ...
    如果全都 raise Exception，你就只能一把 catch 住，然后靠字符串判断错误类型，很脆。

2. 可以带上下文
    自定义异常能挂额外字段（比如 model、status_code），
    打印日志时信息量比一行字符串大得多。

3. 形成异常层级
    这里所有异常都继承 AgentError，所以调用方既能精确捕获子类，
    也能用 `except AgentError` 一把兜住"本项目的所有业务错误"。
    这在写 Agent 时非常关键——LLM 调用可能因为几十种原因失败，
    你得能区分"配置问题 / 网络问题 / 模型返回格式问题"。

记住一个原则：
    异常是给"程序员"看的，要精准、带上下文；
    错误信息是给"用户"看的，要人话、要告诉人家下一步怎么办。
"""

from __future__ import annotations


class AgentError(Exception):
    """本项目所有业务异常的基类。"""

    # 给用户看的默认提示。子类可以覆盖。
    user_message = "程序出错了，请查看日志了解详情。"

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        # context 用来挂额外信息：model=..., status_code=..., path=...
        self.context = context

    def __str__(self) -> str:
        if not self.context:
            return self.message
        extra = ", ".join(f"{k}={v}" for k, v in self.context.items())
        return f"{self.message} [{extra}]"


# ----------------------------------------------------------------------
# 配置类
# ----------------------------------------------------------------------
class ConfigError(AgentError):
    """配置缺失或非法。"""

    user_message = "配置有问题，请检查 .env 文件。"


# ----------------------------------------------------------------------
# 需求 / 用例类
# ----------------------------------------------------------------------
class RequirementError(AgentError):
    """需求读取或解析失败。"""

    user_message = "需求文件读取失败，请检查文件路径和内容。"


class TestCaseError(AgentError):
    """测试用例结构不合法。"""

    user_message = "生成的测试用例不符合预期结构，已丢弃。"


# ----------------------------------------------------------------------
# 大模型类
# ----------------------------------------------------------------------
class LLMError(AgentError):
    """大模型调用相关的基类异常。"""

    user_message = "调用大模型失败，请稍后重试。"


class LLMNotConfiguredError(LLMError):
    """没配 API Key。"""

    user_message = "还没配置 LLM_API_KEY，请复制 .env.example 为 .env 并填入 Key。"


class LLMRequestError(LLMError):
    """网络层失败：超时、连接不上、鉴权失败等。"""

    user_message = "请求大模型服务失败，请检查网络与 API Key。"


class LLMResponseError(LLMError):
    """模型返回了，但内容不符合预期（比如不是合法 JSON）。"""

    user_message = "大模型返回的内容无法解析，已记录日志。"
