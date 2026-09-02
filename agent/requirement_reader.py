"""
需求读取器（Day 3）

知识点：@dataclass
    当你需要一个"只装数据、没什么复杂行为"的类时，手写 __init__ 很啰嗦：

        class Requirement:
            def __init__(self, feature, description, raw):
                self.feature = feature
                self.description = description
                self.raw = raw

    @dataclass 帮你自动生成这些，还顺带生成了 __repr__（打印出来好看）
    和 __eq__（可以直接 == 比较）。

    什么时候用 dataclass、什么时候用 dict？
        - 数据结构固定、字段明确、要被多处传递 -> dataclass，编辑器能补全
        - 结构不固定、来自外部（比如 LLM 返回的 JSON）-> dict

设计约定：
    需求文件的第一行是"功能名"，后面是详细描述。
    现在只是简单按行解析，Day 9 之后会换成让 LLM 来解析需求 --
    但输入输出的契约不变，所以到时候只换实现，不用改调用方。
"""

from __future__ import annotations

from dataclasses import dataclass

from tools import file_io
from tools.exceptions import RequirementError
from tools.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Requirement:
    """一份测试需求。"""

    feature: str        # 功能名，如 "用户登录功能"
    description: str    # 详细描述
    source: str         # 来源文件路径，方便回溯


def parse_text(text: str, source: str = "<string>") -> Requirement:
    """把需求文本解析成 Requirement。

    规则很简单：
        第一个非空行 = 功能名
        其余内容     = 详细描述

    为什么要 strip()？
        文件里每行末尾可能有 \\r\\n（Windows 换行）和不可见空格，
        不 strip 就会出现"用户登录功能\\r"这种看起来一样但实际不同的字符串，
        后面做功能名匹配时会莫名其妙匹配不上。
    """
    lines = [line.strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]

    if not non_empty:
        # 抛自己定义的异常，带上 source 上下文，日志里一眼能看出是哪个文件
        raise RequirementError("需求内容为空", source=source)

    feature = non_empty[0]
    description = "\n".join(non_empty[1:])

    logger.info("解析需求成功：feature=%s（%d 行描述）", feature, len(description.splitlines()))

    return Requirement(feature=feature, description=description, source=str(source))


def load(path: str) -> Requirement:
    """从文件读取并解析需求。"""
    try:
        text = file_io.read_text(path)
    except OSError as exc:
        # 把底层异常"翻译"成业务异常，同时用 from exc 保留原始调用栈。
        # 没有 `from exc` 的话，日志里看不到最初的 FileNotFoundError，
        # 排查问题时就断了一截。
        raise RequirementError(f"需求文件读取失败：{path}", source=str(path)) from exc

    return parse_text(text, source=str(path))
