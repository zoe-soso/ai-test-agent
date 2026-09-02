"""
Prompt 模板加载器（Day 6）

知识点：为什么要把 Prompt 抽成 .txt 文件，而不是写在代码里？

1. **改文案不用改代码**
   调 Prompt 是这个项目最高频的操作 —— 你一天会改十几次。
   如果 prompt 埋在 .py 里，每次改动都是一次代码变更，
   git diff 里混着逻辑和文案，没法 review，也没法回滚对比。

2. **能做 A/B 对比**
   两个版本的 prompt 就是两个文件，可以并排跑、并排看输出、算分。
   这是 Day 7 评测脚本的基础。

3. **非程序员也能改**
   测试同事可以直接编辑 .txt 帮你调优，不需要碰 Python。

模板文件的格式约定（很简单）：

    ### SYSTEM
    （系统提示词，可有可无）

    ### USER
    （用户提示词，必须有）

占位符用 `$feature`、`$description` 这种 shell 风格，
**不用 `{}`**，因为 Prompt 里经常要写 JSON 示例，大括号会和
`str.format` 打架。所以这里用 `string.Template.safe_substitute`。
`safe_substitute` 的好处：变量没提供时不报错，原样保留。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template

from config import settings
from tools.logger import get_logger

logger = get_logger(__name__)

SYSTEM_MARK = "### SYSTEM"
USER_MARK = "### USER"


@dataclass
class Prompt:
    """一份完整的可发送 Prompt。"""

    name: str
    system: str
    user: str

    def to_messages(self) -> list[dict[str, str]]:
        """转成 LLM API 需要的 messages 格式。"""
        messages: list[dict[str, str]] = []
        if self.system.strip():
            messages.append({"role": "system", "content": self.system.strip()})
        messages.append({"role": "user", "content": self.user.strip()})
        return messages

    def __str__(self) -> str:
        return f"<Prompt {self.name}: system={len(self.system)}字, user={len(self.user)}字>"


def parse_template(text: str, name: str = "<string>") -> tuple[str, str]:
    """把模板文本拆成 (system, user) 两段。

    规则：
        `### SYSTEM` 之后、`### USER` 之前  → system
        `### USER` 之后到文件末尾          → user
        没有 `### SYSTEM` 段时，system 为空字符串
    """
    system, user = "", ""

    if USER_MARK in text:
        head, _, user = text.partition(USER_MARK)
        if SYSTEM_MARK in head:
            _, _, system = head.partition(SYSTEM_MARK)
    else:
        # 没有分节标记，整段都当 user（兼容最简单的模板）
        user = text

    return system.strip(), user.strip()


def load(name: str, **variables: object) -> Prompt:
    """按名字加载模板并填充变量。

    用法：
        load("testcase_v1_engineered", feature="用户登录功能", description="...")
    """
    path = settings.PROMPT_DIR / name
    if not path.suffix:
        path = path.with_suffix(".txt")

    if not path.exists():
        raise FileNotFoundError(f"Prompt 模板不存在：{path}")

    raw = path.read_text(encoding="utf-8")
    system, user = parse_template(raw, name=path.stem)

    # 占位符替换：把 $feature 换成实际值
    system = Template(system).safe_substitute(**variables)
    user = Template(user).safe_substitute(**variables)

    logger.debug("加载 Prompt：%s（%d 变量）", path.stem, len(variables))
    return Prompt(name=path.stem, system=system, user=user)


def list_prompts() -> list[str]:
    """列出 prompts/ 目录下所有模板名（不含扩展名）。"""
    if not settings.PROMPT_DIR.exists():
        return []
    return sorted(p.stem for p in settings.PROMPT_DIR.glob("*.txt"))


if __name__ == "__main__":
    for prompt_name in list_prompts():
        prompt = load(prompt_name, feature="用户登录功能", description="（示例）")
        print(f"{prompt_name:32} -> {prompt}")
