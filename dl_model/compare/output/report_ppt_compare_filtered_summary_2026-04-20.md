# CSV Pair Merged Tables

Rule: each experiment merges 2 CSV files (`aggregate` + `runs`) into 1 table.

## C1 compare baseline (merged from 2 CSV)

| Model | Best Epoch | Epochs | Samples T/V/Te | Test ACC | Test F1 | Test ERR | Test Loss | Val F1(mean) | Val F1@Best | Train Time(s) | Test Time(s) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sincnet | 14 | 24 | 8946/696/444 | 0.7230 | 0.6870 | 0.2770 | 0.5707 | 0.8742 | 0.8742 | 1557.18 | 0.8101 |
| ecapatdnn | 12 | 22 | 8946/696/444 | 0.6689 | 0.5905 | 0.3311 | 0.6019 | 0.8838 | 0.8838 | 403.60 | 0.2791 |
| redimnet | 7 | 17 | 8946/696/444 | 0.5946 | 0.5833 | 0.4054 | 0.6834 | 0.7388 | 0.7388 | 336.93 | 0.3915 |
| tdnn | 22 | 32 | 8946/696/444 | 0.6374 | 0.4984 | 0.3626 | 0.6316 | 0.9089 | 0.9089 | 419.57 | 0.2056 |

## S1 stander baseline (merged from 2 CSV)

| Model | Best Epoch | Epochs | Samples T/V/Te | Test ACC | Test F1 | Test ERR | Test Loss | Val F1@Best | Train Time(s) | Test Time(s) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ecapatdnn | 20 | 30 | 8946/696/444 | 0.6689 | 0.6016 | 0.3311 | 1.5871 | 0.9137 | 706.03 | 0.3643 |
| tdnn | 12 | 22 | 8946/696/444 | 0.6014 | 0.5735 | 0.3986 | 1.5308 | 0.8794 | 208.84 | 0.3997 |
| resnet | 15 | 25 | 8946/696/444 | 0.6779 | 0.5489 | 0.3221 | 1.7252 | 0.9059 | 1967.20 | 1.4921 |

## C2 compare 1to2s (merged from 2 CSV)

| Model | Best Epoch | Epochs | Samples T/V/Te | Test ACC(1.5s) | Test F1(1.5s) | F1@1.0s | F1@2.0s | Val F1@Best | Train Time(s) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| sincnet | 7 | 17 | 8946/696/444 | 0.7207 | 0.6915 | 0.6752 | 0.7268 | 0.8603 | 1193.24 |
| redimnet | 23 | 33 | 8946/696/444 | 0.5991 | 0.6691 | 0.6362 | 0.7027 | 0.6831 | 713.40 |
| ecapatdnn | 4 | 14 | 8946/696/444 | 0.5743 | 0.5953 | 0.6196 | 0.5939 | 0.8160 | 267.38 |
| tdnn | 18 | 28 | 8946/696/444 | 0.6329 | 0.4759 | 0.4809 | 0.4904 | 0.8815 | 364.01 |

## S2 stander 1to2s (merged from 2 CSV)

| Model | Best Epoch | Epochs | Samples T/V/Te | Test ACC(1.5s) | Test F1(1.5s) | F1@1.0s | F1@2.0s | Val F1@Best | Train Time(s) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ecapatdnn | 12 | 22 | 8946/696/444 | 0.6486 | 0.6355 | 0.6552 | 0.6587 | 0.8691 | 475.61 |
| tdnn | 14 | 24 | 8946/696/444 | 0.5946 | 0.5238 | 0.4749 | 0.5302 | 0.8272 | 217.05 |
| resnet | 19 | 29 | 8946/696/444 | 0.6104 | 0.5181 | 0.4921 | 0.5658 | 0.8895 | 2173.93 |

## C3 compare change_val_test_train (merged from 2 CSV)

| Model | Best Epoch | Epochs | Samples T/V/Te | Test ACC | Test F1 | Test ERR | Test Loss | Val F1(mean) | Val F1@Best | Train Time(s) | Test Time(s) | Student ms |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sincnet | 20 | 30 | 8946/444/696 | 0.8276 | 0.8519 | 0.1724 | 0.4800 | 0.7563 | 0.7563 | 2020.55 | 1.3002 | 2.2812 |
| ecapatdnn | 21 | 31 | 8946/444/696 | 0.7816 | 0.8203 | 0.2184 | 0.5883 | 0.7178 | 0.7178 | 499.47 | 0.3857 | 6.7775 |
| tdnn | 2 | 12 | 8946/444/696 | 0.6466 | 0.7377 | 0.3534 | 0.6678 | 0.6828 | 0.6828 | 142.62 | 0.2704 | 2.0634 |
| redimnet | 25 | 35 | 8946/444/696 | 0.5647 | 0.6936 | 0.4353 | 0.9207 | 0.6945 | 0.6945 | 653.14 | 0.6067 | 2.9362 |

## B1 baseline output_official (merged from 2 CSV, inference-only)

Source files:
- `baseline/output_official/summary_manifest_runs.csv`
- `baseline/output_official/summary_manifest_aggregate.csv`

| Model | Best Epoch | Epochs | Samples T/V/Te | Train ACC | Train F1 | Val ACC | Val F1 | Test ACC | Test F1 | Test ERR | Train Time(s) | Test Time(s) | Total Time(s) | Student ms | Official Source |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| speechbrain_ecapa | 0 | 0 | 8946/696/444 | 0.7474 | 0.7655 | 0.8822 | 0.8941 | 0.8176 | 0.8121 | 0.1824 | 466.50 | 21.8527 | 488.35 | 49.2177 | speechbrain/spkrec-ecapa-voxceleb |
| speechbrain_xvector | 0 | 0 | 8946/696/444 | 0.5000 | 0.6667 | 0.5000 | 0.6667 | 0.5000 | 0.6667 | 0.5000 | 76.66 | 3.3792 | 80.04 | 7.6108 | speechbrain/spkrec-xvect-voxceleb |
| resemblyzer_ge2e | 0 | 0 | 8946/696/444 | 0.5869 | 0.6996 | 0.5991 | 0.7138 | 0.6644 | 0.7315 | 0.3356 | 120.86 | 5.6897 | 126.55 | 12.8146 | resemblyzer/GE2E |
| wespeaker_english | 0 | 0 | 8946/696/444 | 0.8154 | 0.7928 | 0.9052 | 0.8988 | 0.7410 | 0.6588 | 0.2590 | 1710.17 | 79.9722 | 1790.15 | 180.1176 | wespeaker_nuaazs:english |
