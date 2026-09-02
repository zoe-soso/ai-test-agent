"""
文件读写工具（Day 3）

知识点：json / yaml / open / with

三个必须记住的点：

1. 永远用 with
       with open(path, "r", encoding="utf-8") as f:
           data = f.read()
   with 会在代码块结束后自动关闭文件，哪怕中间抛异常也关。
   手写 f = open(...); ...; f.close() 一旦中间报错，文件句柄就泄漏了。

2. Windows 上必须显式写 encoding="utf-8"
   Python 的 open() 默认编码跟操作系统走，Windows 是 GBK。
   读 UTF-8 文件会 UnicodeDecodeError，写中文会变成乱码。
   这是 Windows 用户最常踩的坑之一，没有之一。

3. json 和 yaml 的职责不同
   json：机器之间交换数据用，几乎所有语言都支持，但不支持注释。
   yaml：给人看/给人写给机器读，支持注释和缩进，配置文件和测试用例首选。
   本项目的测试用例用 YAML，因为测试工程师要人工 Review 和改。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from tools.logger import get_logger

logger = get_logger(__name__)


def read_text(path: str | Path) -> str:
    """读取文本文件，返回字符串。"""
    logger.debug("读取文件：%s", path)
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> Path:
    """写入文本文件（覆盖写）。返回路径方便链式调用。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    logger.debug("写入文件：%s（%d 字符）", target, len(content))
    return target


def read_yaml(path: str | Path) -> Any:
    """读取 YAML 文件，转成 Python 的 dict / list。"""
    logger.debug("读取 YAML：%s", path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: str | Path, data: Any) -> Path:
    """把 Python 对象写成 YAML。

    safe_dump 的三个关键参数：
        allow_unicode=True   —— 不加的话中文会被转义成 \\u767b\\u5f55，没法看
        sort_keys=False      —— 保持你写字典时的顺序，不要按字母重排
        indent=2             —— 缩进 2 空格，这是 YAML 的通行写法
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            indent=2,
            default_flow_style=False,
        )
    logger.info("已写入 YAML：%s", target)
    return target


def read_json(path: str | Path) -> Any:
    """读取 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> Path:
    """把 Python 对象写成 JSON（ensure_ascii=False 保住中文）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return target
