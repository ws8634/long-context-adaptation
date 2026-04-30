# Long Context Adaptation Library

一个用于处理大语言模型(LLM)长上下文适配的Python库，提供多种策略将长文本适配到有限的上下文窗口。

## 策略概览

| 策略名称 | 一句话优势 | 一句话劣势 |
|---------|-----------|-----------|
| **sliding_window_truncation** (滑动/尾部优先截断) | 实现简单，保留最新信息，语义不被破坏 | 丢失开头的全部内容，可能遗漏重要前置信息 |
| **segment_summary_concatenation** (分段摘要后拼接) | 保留各段落的核心信息，覆盖更广的文本范围 | "摘要"只是段落开头，可能丢失段落内的关键细节 |
| **deterministic_retrieval_chunk_assembly** (确定性检索式块装配) | 可复现的随机选择，可能捕获不同位置的多样信息块 | 无语义理解的随机选择可能遗漏真正重要的内容 |

## 安装

### 方式一：使用requirements.txt

```bash
# 克隆或下载项目后，在项目根目录运行
pip install -r requirements.txt
```

## 使用方法

### CLI 调用

#### 基础用法

```bash
# 显示帮助信息
python -m context_adapter --help

# 显示tokenizer规则
python -m context_adapter --show-rules

# 滑动/尾部优先截断策略
python -m context_adapter input.txt --strategy sliding_window

# 分段摘要后拼接策略
python -m context_adapter input.txt --strategy segment_summary

# 确定性检索式块装配策略（使用环境变量设置seed）
CONTEXT_ADAPTER_SEED=7 python -m context_adapter input.txt --strategy deterministic_retrieval

# 确定性检索式块装配策略（使用--seed参数）
python -m context_adapter input.txt --strategy deterministic_retrieval --seed 7
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `input_file` | 输入的UTF-8文本文件路径 | 必填 |
| `--strategy, -s` | 选择适配策略，可选值：`sliding_window`, `segment_summary`, `deterministic_retrieval` | 必填 |
| `--context-limit, -l` | 上下文token上限 | 4096 |
| `--dry-run, -d` | 试运行模式，只输出估算结果，`output_text`为`null` | 关闭 |
| `--show-rules` | 显示tokenizer规则后退出 | 关闭 |
| `--seed` | 确定性检索策略的随机种子（覆盖环境变量） | 由环境变量决定 |

#### 环境变量

| 变量名 | 说明 | 默认值 |
|-------|------|-------|
| `CONTEXT_ADAPTER_SEED` | 确定性检索式块装配策略的随机种子 | 0 |

#### 输出格式

CLI通过标准输出输出JSON格式结果：

```json
{
  "strategy": "sliding_window_truncation",
  "truncated": true,
  "tokens_in": 10000,
  "tokens_out": 4096,
  "discarded_blocks": 12,
  "output_text": "处理后的文本内容...",
  "explanation": "Truncated from 10000 to 4096 tokens..."
}
```

| 字段 | 说明 |
|-----|------|
| `strategy` | 使用的策略名称 |
| `truncated` | 是否发生截断（布尔值） |
| `tokens_in` | 输入文本的token估算数 |
| `tokens_out` | 输出文本的token估算数 |
| `discarded_blocks` | 丢弃的块数 |
| `output_text` | 处理后的文本（`--dry-run`时为`null`） |
| `explanation` | 简短说明 |

#### 错误处理

当遇到不支持的参数或错误时，CLI会输出包含`error`字段的JSON并以非零退出码终止：

```json
{
  "error": "File not found: nonexistent.txt",
  "strategy": "sliding_window",
  "truncated": null,
  "tokens_in": null,
  "tokens_out": null,
  "discarded_blocks": null,
  "output_text": null,
  "explanation": "File not found: nonexistent.txt"
}
```

## 运行测试

本项目使用pytest进行测试。

### 运行所有测试

```bash
# 在项目根目录运行
pytest -v
```

### 运行特定测试

```bash
# 只运行滑动窗口策略的测试
pytest tests/test_strategies.py::TestSlidingWindowTruncation -v

# 只运行验收测试
pytest tests/test_strategies.py::TestAcceptanceCriteria -v
```

### 测试覆盖范围

每种策略包含以下测试用例：

1. **超长必截断**: 测试文本显著超过上下文限制时的截断行为
2. **贴上限边界**: 测试文本刚好等于或略低于上下文限制时的行为
3. **Dry-run模式**: 测试`--dry-run`参数，验证`output_text`为`null`但其他字段齐全

额外测试：
- CLI功能测试（帮助信息、规则显示、错误处理）
- 确定性检索策略的可复现性测试
- 验收测试（>=5000字符文本，seed=7时三种策略各运行一次）

## 伪Tokenizer规则

本项目使用一个固定规则的伪tokenizer进行token估算，规则如下：

| 类型 | 规则 |
|-----|------|
| 中文字符 | 每个字符 = 1 token |
| 英文单词 | 每个单词(a-z/A-Z序列) = 1 token |
| 数字 | 每个数字序列 = 1 token |
| 空白字符 | 不计入 |
| 其他标点/符号 | 每个 = 1 token |

可以通过以下命令查看：

```bash
python -m context_adapter --show-rules
```

## 项目结构

```
11-llm-long-context-adaptation/
├── context_adapter/          # 主包
│   ├── __init__.py          # 包初始化
│   ├── __main__.py          # 模块入口
│   ├── cli.py               # CLI实现
│   ├── tokenizer.py         # 伪Tokenizer实现
│   └── strategies.py        # 三种策略实现
├── tests/                   # 测试目录
│   ├── __init__.py
│   ├── conftest.py          # pytest fixtures
│   └── test_strategies.py   # 测试用例
├── requirements.txt          # 依赖文件
└── README.md                # 本文档
```

## 验收标准验证

为验证项目符合验收标准：

1. **测试全部通过**:
   ```bash
   pytest -v
   ```

2. **针对5000+字符fixture，seed=7时三种策略各运行一次**:
   
   验收测试已包含在`TestAcceptanceCriteria`类中，会自动验证：
   - 输出token均不超过上限
   - 实际运行结果与`--dry-run`估算结果一致

   手动验证示例：
   ```bash
   # 创建一个5000+字符的测试文件
   python -c "print('中' * 6000)" > test_long.txt
   
   # 滑动窗口策略
   python -m context_adapter test_long.txt --strategy sliding_window --context-limit 1000 --dry-run
   python -m context_adapter test_long.txt --strategy sliding_window --context-limit 1000
   
   # 分段摘要策略
   python -m context_adapter test_long.txt --strategy segment_summary --context-limit 1000 --dry-run
   python -m context_adapter test_long.txt --strategy segment_summary --context-limit 1000
   
   # 确定性检索策略（seed=7）
   CONTEXT_ADAPTER_SEED=7 python -m context_adapter test_long.txt --strategy deterministic_retrieval --context-limit 1000 --dry-run
   CONTEXT_ADAPTER_SEED=7 python -m context_adapter test_long.txt --strategy deterministic_retrieval --context-limit 1000
   
   # 清理
   rm test_long.txt
   ```

## 许可证

MIT License
