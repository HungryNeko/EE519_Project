# Compare vs Compare_Standard 简要实验报告（2026-04-19）

## 1. 数据来源（本次读取文件）

- `dl_model/compare/output/summary_manifest_aggregate.csv`（更新时间：2026-04-19 05:40:06）
- `dl_model/compare/output/summary_manifest_runs.csv`（更新时间：2026-04-19 05:40:06）
- `dl_model/compare_standard/output_official/results_aggregate.csv`（更新时间：2026-04-19 04:11:09）
- `dl_model/compare_standard/output_official/results_runs.csv`（更新时间：2026-04-19 04:11:09）

## 2. 统一实验口径

- 数据划分：`train=8946, val=696, test=444`
- 选择准则：`select_metric=f1`（基于验证集选择最佳权重）
- 本次每个模型 `runs=1`，因此 `std=0`
- 关键指标：`test_acc`, `test_f1`, `test_err`, `test_loss`, `train_time_seconds`, `test_time_seconds`, `total_time_seconds`

## 3. compare（当前实现）结果汇总

| model | test_acc | test_f1 | test_err | test_loss | train_time(s) |
|---|---:|---:|---:|---:|---:|
| final_model | 0.7680 | 0.7482 | 0.2320 | 0.5530 | 1551.68 |
| sincnet | 0.7230 | 0.6870 | 0.2770 | 0.5707 | 1557.18 |
| ecapatdnn | 0.6689 | 0.5905 | 0.3311 | 0.6019 | 403.60 |
| redimnet | 0.5946 | 0.5833 | 0.4054 | 0.6834 | 336.93 |
| tdnn | 0.6374 | 0.4984 | 0.3626 | 0.6316 | 419.57 |

## 4. compare_standard（官方标准实现）结果汇总

| model | test_acc | test_f1 | test_err | test_loss | train_time(s) |
|---|---:|---:|---:|---:|---:|
| ecapatdnn | 0.6689 | 0.6016 | 0.3311 | 1.5871 | 706.03 |
| tdnn | 0.6014 | 0.5735 | 0.3986 | 1.5308 | 208.84 |
| resnet | 0.6779 | 0.5489 | 0.3221 | 1.7252 | 1967.20 |

## 5. 重叠模型对比（compare - official）

| model | Δtest_acc | Δtest_f1 | 训练耗时比（official/compare） |
|---|---:|---:|---:|
| ecapatdnn | +0.0000 | -0.0111 | 1.75x |
| tdnn | +0.0360 | -0.0751 | 0.50x |

说明：
- `ecapatdnn`：两者 `test_acc` 基本一致；官方 `test_f1` 略高；compare 训练更快。
- `tdnn`：compare 的 `test_acc` 更高，但官方的 `test_f1` 更高；官方版训练更快。

## 6. 单次运行细节（best epoch / early stop 后训练轮数）

### compare

- tdnn：best_epoch=22, epochs_trained=32, val_f1=0.9089, test_f1=0.4984
- final_model：best_epoch=14, epochs_trained=24, val_f1=0.8698, test_f1=0.7482
- ecapatdnn：best_epoch=12, epochs_trained=22, val_f1=0.8838, test_f1=0.5905
- redimnet：best_epoch=7, epochs_trained=17, val_f1=0.7388, test_f1=0.5833
- sincnet：best_epoch=14, epochs_trained=24, val_f1=0.8742, test_f1=0.6870

### compare_standard

- tdnn：best_epoch=12, epochs_trained=22, val_f1=0.8794, test_f1=0.5735
- ecapatdnn：best_epoch=20, epochs_trained=30, val_f1=0.9137, test_f1=0.6016
- resnet：best_epoch=15, epochs_trained=25, val_f1=0.9059, test_f1=0.5489

## 7. 可直接写入论文的简要结论

- 在当前 `compare` 实验中，`final_model` 在测试集上达到最优（`ACC=0.7680, F1=0.7482`）。
- 在 `official` 对照组中，`ecapatdnn` 的测试 `F1` 最好（`F1=0.6016`），`resnet` 的测试 `ACC` 略高（`ACC=0.6779`）但 `F1` 较低。
- 两组脚本在同名模型（`tdnn`, `ecapatdnn`）上的结果存在差异，说明实现细节、损失定义或预处理流程会明显影响最终指标；论文中应将两组实验分开报告并注明实现来源。
