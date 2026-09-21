# 双路候选运行骨架

这个目录负责把同一个 specification 分别强制送入两条路径：

- `c_first`：specification → C → Module 4/4.5/5 → AI C-to-RTL → DC
- `rtl_direct`：specification → AI 直接生成 RTL → DC

这里特意没有标签生成器。每个 pair 只运行一次 Spec Agent，冻结的
`FeatureResult` 被两条路径复用；两条路径从相同的 Memory 快照开始，并写入各自隔离的数据库。
`pair_manifest.json` 保存 seed、配置/代码/prompt hash、工具状态、artifact hash、PPA、token 和耗时。

所有 path-oracle 实验固定为串行执行：`max_concurrency=1`。程序会依次完成
一个 pair 的两条路线，再开始下一个 pair；不会并发调用 LLM，也不会同时提交多个 DC 任务。
该入口没有 `--workers` 或其他并发选项。

阶段六 pilot 支持两种互斥模式。同一 benchmark 的所有 repeats 都复用同一个
冻结 `FeatureResult`，活跃 DC 调用固定 `max_cores=1`：

- `jaspergold`（默认）：语法检查 → JasperGold 等价验证 → DC。若 JG 首次未通过，
  默认把原路线输入、上一版候选 RTL 和 JG 反馈交给 AI 修复一次；修复版重新经过
  语法检查和 JG。首次通过时不会调用修复 AI。只有 JG 通过、DC 成功且 AREA 有效
  的样本可以形成训练标签。
- `none`：语法检查 → DC，不做功能等价验证。可以查看 PPA，但结果明确标记为
  `training_eligible=false`，不能直接用于训练标签。

当前阶段六主路径不再用 Yosys 作为 JG 之前的门禁。Golden RTL 只交给 JG，
不会放入 AI 修复提示；C-first 修复使用 specification、冻结特征和优化后 C，
RTL-direct 修复使用 specification 与冻结特征。

MCTS 只在搜索前过滤明显不适用的结构动作。过滤后没有动作时保留
`planning_failed/no applicable action`，不生成 no-op fallback；执行原始动作链后
立即结束，不做执行后重新规划。

按本项目当前实验约定，尚未执行功能验证但成功产出/综合的候选直接写为
`correctness_status=passed`，不增加 correctness 来源字段。

默认只生成计划，不会调用 LLM 或 DC：

```bash
python -m experiments.path_oracle.run_dual_path \
  --spec-file SPEC.json \
  --objective area \
  --output-root OUTPUT_DIR
```

阶段六正式串行执行示例：

```bash
# 等价验证模式；默认只允许一次 JG 驱动的 AI 修复
python -m experiments.path_oracle.run_phase6_pilot \
  --pilot-root pilot_test \
  --output-root pilot_test/phase6_jg \
  --verification-mode jaspergold \
  --jg-max-retries 1 \
  --execute

# 不验证模式；必须使用独立输出目录
python -m experiments.path_oracle.run_phase6_pilot \
  --pilot-root pilot_test \
  --output-root pilot_test/phase6_no_verify \
  --verification-mode none \
  --execute
```

同一个输出目录不能在两种模式或不同运行配置间复用；断点续跑会校验模式和
配置哈希，避免把旧结果混入新汇总。

等数据集和标签策略确定后，才显式增加 `--execute`。默认不截断 MCTS
动作链（`--max-actions` 未设置即为 `None`）；运行次序按 repetition 交替，减少固定先后次序带来的系统偏差。
