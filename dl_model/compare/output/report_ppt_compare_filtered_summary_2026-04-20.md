# Compare Filtered Summary Report

SV-EER columns below are speaker verification EER from CSV (not `1-ACC`).

## 1. C1 Compare Baseline

### 1.1 No Aug

| Model | Best Epoch | Epochs | Samples T/V/Te | Test ACC | Test F1 | Test ERR | Test Loss | Val F1(mean) | Val F1@Best | Train Time(s) | Test Time(s) | SV-EER@2.0s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sincnet | 14 | 24 | 8946/696/444 | 0.7230 | 0.6870 | 0.2770 | 0.5707 | 0.8742 | 0.8742 | 1557.18 | 0.8101 | 0.2793 |
| ecapatdnn | 12 | 22 | 8946/696/444 | 0.6689 | 0.5905 | 0.3311 | 0.6019 | 0.8838 | 0.8838 | 403.60 | 0.2791 | 0.2793 |
| redimnet | 7 | 17 | 8946/696/444 | 0.5946 | 0.5833 | 0.4054 | 0.6834 | 0.7388 | 0.7388 | 336.93 | 0.3915 | 0.4009 |
| tdnn | 22 | 32 | 8946/696/444 | 0.6374 | 0.4984 | 0.3626 | 0.6316 | 0.9089 | 0.9089 | 419.57 | 0.2056 | 0.3514 |

### 1.2 With Aug

| Model | Best Epoch | Epochs | Samples T/V/Te | Test ACC | Test F1 | Test ERR | Test Loss | Val F1(mean) | Val F1@Best | Train Time(s) | Test Time(s) | SV-EER@2.0s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sincnet | 19 | 29 | 8946/696/444 | 0.7523 | 0.7236 | 0.2477 | 0.5555 | 0.8810 | 0.8810 | 2078.71 | 0.8215 | 0.2162 |
| tdnn | 13 | 23 | 8946/696/444 | 0.6779 | 0.5879 | 0.3221 | 0.6080 | 0.8747 | 0.8747 | 417.76 | 0.4152 | 0.3333 |
| ecapatdnn | 5 | 15 | 8946/696/444 | 0.6757 | 0.6453 | 0.3243 | 0.6318 | 0.8491 | 0.8491 | 349.07 | 0.2631 | 0.3243 |
| redimnet | 17 | 27 | 8946/696/444 | 0.6667 | 0.6355 | 0.3333 | 0.6428 | 0.7687 | 0.7687 | 660.70 | 0.4060 | 0.3243 |

<!-- hidden final_model row: Best Epoch=19, Epochs=29, Samples=8946/696/444, Test ACC=0.7500, Test F1=0.7246, Test ERR=0.2500, Test Loss=0.5491, Val F1(mean)=0.8866, Val F1@Best=0.8866, Train Time=2083.05s, Test Time=0.8257s, SV-EER@2.0s=0.2342 -->

### 1.3 Difference (With Aug - No Aug)

| Model | Delta Test ACC | Delta Test F1 | Delta Test ERR | Delta Test Loss | Delta SV-EER@2.0s | Delta Train Time(s) |
|---|---:|---:|---:|---:|---:|---:|
| sincnet | +0.0293 | +0.0366 | -0.0293 | -0.0152 | -0.0631 | +521.53 |
| ecapatdnn | +0.0068 | +0.0548 | -0.0068 | +0.0299 | +0.0450 | -54.53 |
| redimnet | +0.0721 | +0.0522 | -0.0721 | -0.0406 | -0.0766 | +323.77 |
| tdnn | +0.0405 | +0.0895 | -0.0405 | -0.0236 | -0.0181 | -1.81 |

Note: For `ERR/Loss/EER`, negative delta means better. For `Train Time`, negative delta means faster.

## 2. S1 Standard Baseline

| Model | Best Epoch | Epochs | Samples T/V/Te | Test ACC | Test F1 | Test ERR | Test Loss | Val F1@Best | Train Time(s) | Test Time(s) | SV-EER@2.0s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ecapatdnn | 20 | 30 | 8946/696/444 | 0.6689 | 0.6016 | 0.3311 | 1.5871 | 0.9137 | 706.03 | 0.3643 | 0.3288 |
| tdnn | 12 | 22 | 8946/696/444 | 0.6014 | 0.5735 | 0.3986 | 1.5308 | 0.8794 | 208.84 | 0.3997 | 0.4009 |
| resnet | 15 | 25 | 8946/696/444 | 0.6779 | 0.5489 | 0.3221 | 1.7252 | 0.9059 | 1967.20 | 1.4921 | 0.3018 |

## 3. C2 Compare 1to2s

| Model | Best Epoch | Epochs | Samples T/V/Te | Test ACC(1.5s) | Test F1(1.5s) | F1@1.0s | F1@2.0s | Val F1@Best | Train Time(s) | SV-EER@1.0s | SV-EER@1.5s | SV-EER@2.0s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sincnet | 7 | 17 | 8946/696/444 | 0.7207 | 0.6915 | 0.6752 | 0.7268 | 0.8603 | 1193.24 | 0.2568 | 0.2748 | 0.2568 |
| redimnet | 23 | 33 | 8946/696/444 | 0.5991 | 0.6691 | 0.6362 | 0.7027 | 0.6831 | 713.40 | 0.4369 | 0.4009 | 0.3739 |
| ecapatdnn | 4 | 14 | 8946/696/444 | 0.5743 | 0.5953 | 0.6196 | 0.5939 | 0.8160 | 267.38 | 0.4414 | 0.4414 | 0.4234 |
| tdnn | 18 | 28 | 8946/696/444 | 0.6329 | 0.4759 | 0.4809 | 0.4904 | 0.8815 | 364.01 | 0.3468 | 0.2793 | 0.3063 |

## 4. S2 Standard 1to2s

| Model | Best Epoch | Epochs | Samples T/V/Te | Test ACC(1.5s) | Test F1(1.5s) | F1@1.0s | F1@2.0s | Val F1@Best | Train Time(s) | SV-EER@1.0s | SV-EER@1.5s | SV-EER@2.0s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ecapatdnn | 12 | 22 | 8946/696/444 | 0.6486 | 0.6355 | 0.6552 | 0.6587 | 0.8691 | 475.61 | 0.3559 | 0.3468 | 0.3108 |
| tdnn | 14 | 24 | 8946/696/444 | 0.5946 | 0.5238 | 0.4749 | 0.5302 | 0.8272 | 217.05 | 0.4189 | 0.4144 | 0.3874 |
| resnet | 19 | 29 | 8946/696/444 | 0.6104 | 0.5181 | 0.4921 | 0.5658 | 0.8895 | 2173.93 | 0.4144 | 0.3964 | 0.3468 |

## 5. C3 Compare change_val_test_train

| Model | Best Epoch | Epochs | Samples T/V/Te | Test ACC | Test F1 | Test ERR | Test Loss | Val F1(mean) | Val F1@Best | Train Time(s) | Test Time(s) | Student ms | SV-EER@2.0s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sincnet | 20 | 30 | 8946/444/696 | 0.8276 | 0.8519 | 0.1724 | 0.4800 | 0.7563 | 0.7563 | 2020.55 | 1.3002 | 2.2812 | 0.0776 |
| ecapatdnn | 21 | 31 | 8946/444/696 | 0.7816 | 0.8203 | 0.2184 | 0.5883 | 0.7178 | 0.7178 | 499.47 | 0.3857 | 6.7775 | 0.1006 |
| tdnn | 2 | 12 | 8946/444/696 | 0.6466 | 0.7377 | 0.3534 | 0.6678 | 0.6828 | 0.6828 | 142.62 | 0.2704 | 2.0634 | 0.1063 |
| redimnet | 25 | 35 | 8946/444/696 | 0.5647 | 0.6936 | 0.4353 | 0.9207 | 0.6945 | 0.6945 | 653.14 | 0.6067 | 2.9362 | 0.2874 |

## 6. B1 Baseline output_official (inference-only)

| Model | Best Epoch | Epochs | Samples T/V/Te | Train ACC | Train F1 | Val ACC | Val F1 | Test ACC | Test F1 | Test ERR | Train Time(s) | Test Time(s) | Total Time(s) | Student ms | Official Source | SV-EER@2.0s (T/V/Te) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| speechbrain_ecapa | 0 | 0 | 8946/696/444 | 0.7474 | 0.7655 | 0.8822 | 0.8941 | 0.8176 | 0.8121 | 0.1824 | 466.50 | 21.8527 | 488.35 | 49.2177 | speechbrain/spkrec-ecapa-voxceleb | 0.2332/0.0661/0.1847 |
| speechbrain_xvector | 0 | 0 | 8946/696/444 | 0.5000 | 0.6667 | 0.5000 | 0.6667 | 0.5000 | 0.6667 | 0.5000 | 76.66 | 3.3792 | 80.04 | 7.6108 | speechbrain/spkrec-xvect-voxceleb | 0.4109/0.1322/0.4054 |
| resemblyzer_ge2e | 0 | 0 | 8946/696/444 | 0.5869 | 0.6996 | 0.5991 | 0.7138 | 0.6644 | 0.7315 | 0.3356 | 120.86 | 5.6897 | 126.55 | 12.8146 | resemblyzer/GE2E | 0.2122/0.0603/0.2072 |
| wespeaker_english | 0 | 0 | 8946/696/444 | 0.8154 | 0.7928 | 0.9052 | 0.8988 | 0.7410 | 0.6588 | 0.2590 | 1710.17 | 79.9722 | 1790.15 | 180.1176 | wespeaker_nuaazs:english | 0.1744/0.0546/0.1757 |

## 7. Small Speech-Length Filter Check

Filtered subset keeps the top 50% longest estimated speech-duration samples per label from val/test, preserving class balance. Energy VAD: frame=25ms, hop=10ms, threshold=max(0.005, 0.1*max_rms).

| Split | Model | Full ACC | Filtered ACC | Delta ACC | Full F1 | Filtered F1 | Delta F1 | Full Speech(s) | Filtered Speech(s) | Samples Full -> Filtered |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | tdnn | 0.8621 | 0.9368 | 0.0747 | 0.8747 | 0.9389 | 0.0642 | 2.4896 | 2.7971 | 696 -> 348 |
| val | sincnet | 0.8649 | 0.9109 | 0.0460 | 0.8810 | 0.9182 | 0.0372 | 2.4896 | 2.7971 | 696 -> 348 |
| val | ecapatdnn | 0.8233 | 0.8822 | 0.0589 | 0.8491 | 0.8946 | 0.0455 | 2.4896 | 2.7971 | 696 -> 348 |
| val | redimnet | 0.7026 | 0.7471 | 0.0445 | 0.7687 | 0.7963 | 0.0276 | 2.4896 | 2.7971 | 696 -> 348 |
| test | sincnet | 0.7523 | 0.7252 | -0.0270 | 0.7236 | 0.6872 | -0.0364 | 2.3382 | 2.8084 | 444 -> 222 |
| test | redimnet | 0.6667 | 0.6351 | -0.0315 | 0.6355 | 0.5150 | -0.1205 | 2.3382 | 2.8084 | 444 -> 222 |
| test | ecapatdnn | 0.6757 | 0.6261 | -0.0495 | 0.6453 | 0.5561 | -0.0892 | 2.3382 | 2.8084 | 444 -> 222 |
| test | tdnn | 0.6779 | 0.6171 | -0.0608 | 0.5879 | 0.4371 | -0.1508 | 2.3382 | 2.8084 | 444 -> 222 |

<!-- hidden final_model rows: val Full ACC=0.8721, Filtered ACC=0.9080, Delta ACC=0.0359, Full F1=0.8866, Filtered F1=0.9158, Delta F1=0.0292, Full Speech=2.4896s, Filtered Speech=2.7971s, Samples=696 -> 348; test Full ACC=0.7500, Filtered ACC=0.7252, Delta ACC=-0.0248, Full F1=0.7246, Filtered F1=0.6904, Delta F1=-0.0342, Full Speech=2.3382s, Filtered Speech=2.8084s, Samples=444 -> 222 -->
