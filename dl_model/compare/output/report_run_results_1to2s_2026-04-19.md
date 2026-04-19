# compare / compare_stander 运行结果总结（1~2s 随机训练）

## 数据来源（本次实际读取）

- `dl_model/compare/output/summary_manifest_aggregate_1to2s.csv`（2026-04-19 16:13:10）
- `dl_model/compare/output/summary_manifest_test_duration_aggregate_1to2s.csv`（2026-04-19 16:13:10）
- `dl_model/compare_stander/output_official/results_aggregate_1to2s.csv`（2026-04-19 14:48:00）
- `dl_model/compare_stander/output_official/results_test_duration_aggregate_1to2s.csv`（2026-04-19 14:48:00）

说明：当前每个模型 `runs=1`，因此标准差列均为 `0.0`。

## 1. compare 总体结果（主测试口径=1.5s）

| 模型 | Test ACC | Test F1 | Test ERR | Test Loss | 训练耗时(s) |
|---|---:|---:|---:|---:|---:|
| sincnet | 0.7207 | 0.6915 | 0.2793 | 0.6004 | 1193.24 |
| redimnet | 0.5991 | 0.6691 | 0.4009 | 0.7472 | 713.40 |
| final_model | 0.6914 | 0.6514 | 0.3086 | 0.6015 | 1754.47 |
| ecapatdnn | 0.5743 | 0.5953 | 0.4257 | 0.7516 | 267.38 |
| tdnn | 0.6329 | 0.4759 | 0.3671 | 0.6302 | 364.01 |

结论：
- `compare` 里按 F1 最优是 `sincnet`（`F1=0.6915`）。
- 若看 ACC，`sincnet` 也最高（`ACC=0.7207`）。

## 2. compare_stander 总体结果（主测试口径=1.5s）

| 模型 | Test ACC | Test F1 | Test ERR | Test Loss | 训练耗时(s) |
|---|---:|---:|---:|---:|---:|
| ecapatdnn | 0.6486 | 0.6355 | 0.3514 | 0.8735 | 475.61 |
| tdnn | 0.5946 | 0.5238 | 0.4054 | 1.4593 | 217.05 |
| resnet | 0.6104 | 0.5181 | 0.3896 | 1.1948 | 2173.93 |

结论：
- `compare_stander` 里按 F1 与 ACC 都是 `ecapatdnn` 最优。

## 3. 分时长测试结果（1.0s / 1.5s / 2.0s）

### compare：每个时长的最佳模型（按 F1）

| 时长 | 最佳模型 | ACC | F1 |
|---|---|---:|---:|
| 1.0s | sincnet | 0.7140 | 0.6752 |
| 1.5s | sincnet | 0.7207 | 0.6915 |
| 2.0s | sincnet | 0.7410 | 0.7268 |

趋势：
- `sincnet` 随测试时长增加，`ACC/F1` 均上升（1.0s < 1.5s < 2.0s）。
- `redimnet` 也表现出明显随时长上升的趋势（F1: `0.6362 -> 0.6691 -> 0.7027`）。

### compare_stander：每个时长的最佳模型（按 F1）

| 时长 | 最佳模型 | ACC | F1 |
|---|---|---:|---:|
| 1.0s | ecapatdnn | 0.6374 | 0.6552 |
| 1.5s | ecapatdnn | 0.6486 | 0.6355 |
| 2.0s | ecapatdnn | 0.6779 | 0.6587 |

趋势：
- `ecapatdnn` 在三个时长下都稳定领先。
- `resnet` 也随着时长增长而改善（F1: `0.4921 -> 0.5181 -> 0.5658`）。

## 4. 两组公共模型对比（compare - compare_stander，主口径1.5s）

| 模型 | ΔACC | ΔF1 | 训练耗时比（stander/compare） |
|---|---:|---:|---:|
| ecapatdnn | -0.0743 | -0.0402 | 1.78x |
| tdnn | +0.0383 | -0.0479 | 0.60x |

解读：
- `ecapatdnn`：compare_stander 的精度更高，但 compare 训练更快。
- `tdnn`：compare 的 ACC 更高，但 compare_stander 的 F1 更高，且 compare 更慢。

## 5. 可直接写论文的简要结论

- 在本次 `1~2s` 随机训练设置下，`compare` 体系中 `sincnet` 综合最优，且在 `1.0/1.5/2.0s` 三个测试时长都保持第一。
- `compare_stander` 体系中 `ecapatdnn` 综合最优，并在三个测试时长下均表现稳定。
- 多数模型在更长测试窗口（2.0s）下性能提升，说明该任务对上下文时长仍有明显依赖。
