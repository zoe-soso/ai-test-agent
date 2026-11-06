# Web 测试用例生成与缺陷分析 Agent

![CI](https://github.com/zoe-soso/ai-test-agent/actions/workflows/test.yml/badge.svg)

输入测试需求，调用 DeepSeek 生成用例和 Playwright 脚本，再借用 [电商自动化项目](https://github.com/zoe-soso/ecommerce-test-automation) 的页面对象与测试环境执行。失败后收集日志和截图路径，交给模型分析原因。

```text
需求 → 用例生成和评审 → 代码生成和检查 → 人工确认 → pytest → 结果与失败分析
```

代码检查包含 Python 语法、POM 使用、方法存在性和可静态确定的调用参数。不合格代码不会进入执行，但静态检查不能保证业务断言正确。默认执行前询问确认，`--auto` 会跳过确认。

## 运行

两个项目各自建立虚拟环境。先按电商项目的 README 安装依赖和 Chromium，再在本项目根目录执行：

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中填写 `LLM_API_KEY`，确认 `LLM_BASE_URL`、`LLM_MODEL`；真实调用使用 `LLM_MOCK=false`。密钥文件不要提交。

```powershell
# 检查环境
.\venv\Scripts\python.exe main.py check

# 生成、检查并执行一份登录脚本，执行前人工确认
.\venv\Scripts\python.exe main.py run --profile ecommerce --feature "用户登录功能" --limit 1

# 已确认生成代码的使用范围时，可加 --auto 跳过询问
# --allure 可额外输出 Allure 原始数据

# 离线单元测试，不调用模型
$env:LLM_MOCK = "true"
.\venv\Scripts\python.exe -m pytest -q
# 恢复由 .env 决定运行模式
Remove-Item Env:LLM_MOCK
```

目标项目由 `config/profiles/ecommerce.yaml` 指定。默认路径适用于当前目录布局；目录不同时设置 `TEST_PROJECT_DIR`，解释器可用 `TEST_PROJECT_PYTHON` 指定。接其他项目还需核对页面类、方法、fixture 和配置读取方式。

## 文件与结果

`agent/` 放生成、评审和分析逻辑，`tools/` 放执行器与工具，`prompts/` 放提示模板。生成脚本在 `generated_tests/`，用例、汇总报告和运行产物在 `outputs/`。运行会保存 `outputs/pipeline_report.yaml`；命令退出码为 0 表示通过，1 表示执行失败，2 表示取消，3 表示生成未通过检查。

多文件执行只包含本次选中的脚本，不会顺带运行旧文件。执行器复用目标项目源码，日志和报告写入 Agent 的输出目录。

## 测试范围

CI 分别运行离线单元测试，以及已保存登录脚本的 Chromium 集成冒烟。集成冒烟验证与电商项目的衔接，不调用 DeepSeek；真实生成效果需要配置密钥单独运行。依赖关联项目或特定平台的单元测试可能跳过。

`eval/` 下有场景覆盖、代码检查通过率和缺陷分类评测。其中代码通过率是**静态检查通过率**，不是浏览器执行通过率；关键词覆盖也不等于完整业务覆盖。
