"""
日志工具（Day 4）

知识点：为什么 AI Agent 项目必须有日志，而 print 不行？

1. print 没有时间戳。
   你根本不知道"卡住的这一步"是跑了 3 秒还是 3 分钟。

2. print 没有级别。
   调试信息和致命错误混在一起，出事的时候找不到重点。

3. print 只往屏幕输出。
   程序挂了、终端关了，线索全丢。日志文件是唯一的"黑匣子"。

4. LLM 的输出天然不稳定。
   同一句 prompt，今天返回 JSON，明天可能前面多一句"好的，这是结果："。
   没有日志，你根本复现不了昨天那次失败。

logging 的五个级别（从轻到重）：
    DEBUG    调试细节，开发时开
    INFO     关键流程节点，默认开
    WARNING  有问题但不影响继续
    ERROR    某个操作失败，但程序还能跑
    CRITICAL 程序活不下去了

本项目约定：
    - 每个模块开头 logger = get_logger(__name__)，用 __name__ 当名字，
      日志里就能直接看出是哪一行代码打的。
    - 同时输出到控制台和文件，文件按大小自动轮转，避免把磁盘写满。
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from config import settings

# 模块级标记：保证"配置日志"这件事只做一次。
# 不做这个判断的话，每次 get_logger 都会加一个 handler，
# 结果同一条日志被打印 N 遍 —— 这是个非常经典的坑。
_CONFIGURED = False

# 单行格式：时间 | 级别 | 哪个模块 | 消息
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | None = None) -> None:
    """配置根 logger：控制台 + 滚动文件。

    为什么用 root logger 而不是给每个 logger 单独配？
        因为第三方库（openai、httpx）也用 logging，
        配 root 能顺带把它们的警告也收进日志文件，
        排查网络问题时这些信息非常有用。
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings.ensure_dirs()
    log_level = getattr(logging, (level or settings.LOG_LEVEL).upper(), logging.INFO)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(log_level)

    # --- 控制台 handler ---
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    # Windows 控制台是 GBK，logging 往里写中文可能炸，这里做一个安全兜底。
    try:
        console.stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        pass
    root.addHandler(console)

    # --- 文件 handler（按大小轮转）---
    file_handler = RotatingFileHandler(
        settings.LOG_DIR / "agent.log",
        maxBytes=2 * 1024 * 1024,  # 单文件 2MB
        backupCount=5,             # 最多留 5 个历史文件
        encoding="utf-8",          # 必须显式指定，否则 Windows 上是 GBK
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str = "ai-test-agent") -> logging.Logger:
    """获取一个 logger。用法：logger = get_logger(__name__)。"""
    setup_logging()
    return logging.getLogger(name)


if __name__ == "__main__":
    # 直接运行本文件，看一眼五个级别长什么样
    demo = get_logger("demo")
    demo.debug("这是 DEBUG，开发调试用")
    demo.info("这是 INFO，记录关键流程")
    demo.warning("这是 WARNING，有问题但还能跑")
    demo.error("这是 ERROR，这一步失败了")
    demo.critical("这是 CRITICAL，程序要挂了")
    print(f"\n日志已写入：{settings.LOG_DIR / 'agent.log'}")
