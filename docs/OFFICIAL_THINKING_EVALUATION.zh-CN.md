# Gemma 4 原生 thinking BBEH 评测

<p align="center">
  <a href="./OFFICIAL_THINKING_EVALUATION.md"><img src="https://img.shields.io/badge/Language-English-0969DA?style=for-the-badge" alt="English"></a>
  <a href="./OFFICIAL_THINKING_EVALUATION.zh-CN.md"><img src="https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-DE2910?style=for-the-badge" alt="简体中文"></a>
</p>

## 为什么必须单独评测

Gemma 4 技术报告表 5 给出 E2B 在 BBEH 上 21.9% 的微平均准确率，并说明表中
模型除非特别标注都启用了 thinking。仓库原始冻结 RL 评测则有意采用无 system
消息、贪心解码以及 64/256 token 上限。原结果对其注册的部署协议仍然有效，但若
直接与 21.9% 比较，就会把模型能力、提示和推理模式混为一谈。

因此原生 thinking 配置是独立的事后协议修正，不会改写注册研究。它有两个目标：

1. 让未修改的基础 E2B 在完整 4,520 条 BBEH 上运行，作为唯一能与论文直接描述性
   比较的 cell；
2. 在同一个公开 thinking 配置下，只用 index ≥ 50 的 3,370 条冻结题比较 Base、
   GRPO 和 VOLT。

## 固定的 v3 协议

| 组件 | 固定值 |
|---|---|
| 基础模型 | `google/gemma-4-E2B-it` revision `70af34e20bd4b7a91f0de6b22675850c43922a03` |
| 精度 | BF16，不量化 |
| BBEH | revision `80d12ca916b7158f22293fcf3144f4d3d854d4be` |
| 提示 | `task.json` 输入 + BBEH 附录 C 公布的精确评测后缀 |
| 消息 | 一条 `user`；不手工提供 system 消息 |
| Thinking | `apply_chat_template(..., enable_thinking=True)` |
| 采样 | 一次；temperature 1.0、top-p 0.95、top-k 64 |
| 输出上限 | 8,192 个新 token |
| Seed | `20260709`，每批有效 seed 都写入记录 |
| 解析 | 固定 tokenizer 的 `parse_response`；只评分 `content` |
| 评分 | 固定的上游 `bbeh/bbeh/evaluate.py` |
| 失败处理 | 超长提示直接中止；解析失败按空答案；单独统计截断 |

8,192 token 上限必须明确展示。Google 没有公开表 5 的 BBEH 输出上限、seed、采样
次数或完整内部 harness。因此即使分数相同，也只能称为“基于公开信息的最佳复现”，
不能证明实现完全一致。

## 精确输入与输出契约

评测器在每条 BBEH 输入后追加论文公布的后缀：

```text
Think step by step, and when you provide the final answer, please use the prefix
"The answer is:" without any modification, and provide the answer directly, with
no formatting, no bolding, and no markup. ...
```

随后把整段文字作为一条 user 消息，并设置 `enable_thinking=True`。固定的 Gemma
模板会自动生成且只生成一个 system thinking 控制 turn：

```text
<bos><|turn>system
<|think|>
<turn|>
<|turn>user
... BBEH 题目与后缀 ...<turn|>
<|turn>model
```

符合协议的模型输出形态为：

```text
<|channel>thought
... 私有推理 ...<channel|>The answer is: disproved<turn|>
```

不能用 `decode(..., skip_special_tokens=True)` 评分，因为它会把推理与最终答案拼接。
`parse_response` 会得到：

```json
{
  "thinking": "... 私有推理 ...",
  "content": "The answer is: disproved"
}
```

只有 `content` 会交给官方 BBEH 评分器。`predictions.jsonl` 同时保留 thinking、
最终内容、带控制 token 的原始响应、token 数、停止 token、截断标记、解析错误和
精确匹配结果，便于审计。

最初有一次只用校准行的 smoke test 没有附加 BBEH 公布后缀。它使 Gemma 在 final
channel 内又输出一篇长解释，官方评分器无法稳定提取答案，因此该草案被主动判为
无效；当时没有看到完整集或冻结集结果。之后一次已带正确后缀、按长度排序的 40 行
工程运行也在没有生成 summary 的情况下停止，以便 v3 增加权重/配置/代码哈希和
独占输出锁。这两次局部运行都不可报告；v3 在全新完整运行前固定科学协议。

## Cell 与统计比较

| Cell | 样本数 | 允许的结论 |
|---|---:|---|
| Base，完整 BBEH | 4,520 | 与论文 21.9% 作描述性比较 |
| 从完整 Base 预测中筛出的冻结行 | 3,370 | 冻结基础基线 |
| GRPO checkpoint 45 | 3,370 | 冻结、计算量匹配的 RL 基线 |
| VOLT checkpoint 45 | 3,370 | 冻结新方法结果 |

所有 cell 完成后，比较脚本会先验证键完全一致和分母，再报告微平均准确率、平均
completion/thinking token、截断和解析失败、配对胜负、精确双侧 McNemar 检验、
按任务分层 bootstrap 区间与 token 变化。由于批次按输入长度排序，不解释任何
中途准确率。

## 运行与恢复

```bash
# 在逻辑 GPU 0 依次运行 Base/完整、GRPO/冻结、VOLT/冻结，再做配对分析。
CUDA_VISIBLE_DEVICES=0 ./scripts/run_official_thinking_evals.sh
```

每个 cell 按完整批次追加并 fsync。若进程中断，重新运行 shell 脚本会添加
`--resume`，验证全部不可变配置，只生成缺失 prompt ID。它会拒绝 revision 混用、
重复 ID、协议变化、静默跳过提示，以及未显式声明 resume 的已有输出目录。

远程产物写入：

```text
/data/benwulab/gemma4-rl/runs/evals-official-thinking/v3/
  base-all/
  grpo-frozen-test/
  volt-frozen-test/
  comparison.json
  comparison.md
```

版本控制中的规范为
[`experiments/rl/official_thinking_e2b_bbeh.json`](../experiments/rl/official_thinking_e2b_bbeh.json)，
实现为 [`rl/eval_official_thinking.py`](../rl/eval_official_thinking.py)。
