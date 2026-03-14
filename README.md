# EE519 Project

这个仓库是一个面向多语种、跨语言插入式混音与语音分析的研究型工作区，不是单一可执行产品。

当前主线围绕三套数据集展开：

- `Corpus`：英印语音数据
- `ascend`：英中语音数据
- `hinglish`：英印语音数据

项目中已经包含了：

- 数据集语言切分与标注脚本
- 基于 crossfade 的跨数据集语音插入式混音生成
- Whisper / WhisperX 级别的分段与语言切换分析
- 真假插入边界切换过滤
- 两套说话人分离实验脚本
- 一个基于模拟特征的说话人切换分类原型

## 仓库定位

这个仓库更像“实验平台”而不是“打包好的应用”：

- `datasets/` 保存原始数据、语言切分结果和部分生成结果
- `preprocess/` 做混音样本构造和 TTS 实验
- `translate/` 做 Whisper / WhisperX 标注、翻译和语言切换分析
- `split_models/` 放说话人分离实验
- `dl_model/` 放一个基于模拟数据的二分类原型

根目录的 `main.py` 目前只是占位文件，不是项目入口。

## 目录结构

```text
.
├─ datasets/
│  ├─ Corpus/                      # 英印数据集与语言切分结果
│  ├─ ascend/                      # 英中数据集与语言切分结果
│  ├─ hinglish/                    # Hinglish 数据集与语言切分结果
│  ├─ crossfade_insertions*/       # 生成的插入式混音结果
│  └─ download_ASCEND.py           # 从 Hugging Face 导出 ASCEND 音频
├─ preprocess/
│  ├─ insert_with_crossfade.py     # 主混音生成脚本
│  ├─ audio_create.py              # Edge TTS 生成中英语音与混音示例
│  ├─ mix_language_create.py       # 其他 TTS/拼接实验
│  └─ generated_batch/             # 样例 TTS 音频
├─ translate/
│  ├─ Whisper/                     # Whisper 早期实验脚本
│  └─ whisperX/                    # WhisperX/Whisper 标注与切换分析
├─ split_models/
│  ├─ mossformer2/                 # MossFormer2 分离实验
│  └─ speechbrain/                 # SpeechBrain SepFormer 分离实验
├─ dl_model/                       # 模拟数据 + MLP 原型
├─ run_whisper.slurm               # WhisperX 标注作业脚本
└─ run_whisper2.slurm              # WhisperX Hinglish 标注作业脚本
```

## 当前数据资产

### 1. 语言切分后的数据清单

这些数字来自当前仓库里的 JSON 文件，表示过滤后的可用条目数。

| Dataset | Monolingual A | Monolingual B | Mixed | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Corpus` | `en=1737` | `hi=2272` | `346` | `820` | `5175` |
| `ascend` | `en=4312` | `zh=5209` | `2333` | `216` | `12070` |
| `hinglish` | `en=2654` | `hi=19674` | `23995` | `9599` | `55922` |

说明：

- `Corpus` 由 `datasets/Corpus/data_filt_en_hi.py` 生成语言拆分结果
- `ascend` 由 `datasets/ascend/data_filt_en_zh.py` 生成语言拆分结果
- `hinglish` 由 `datasets/hinglish/data_filt_hinglish.py` 生成语言拆分结果
- `Other` 表示不只包含目标双语标签，或语言标签落在目标集合之外

### 2. 标注类 JSON

每个数据集现在都维护两类核心 JSON：

- `whisper_segment_<dataset>.json`
  - 逐音频保存 Whisper/WhisperX 风格的 `segments`
  - 每个 segment 内带 `language_spans`
- `whisper_language_switch_<dataset>.json`
  - 从 `language_spans` 汇总得到的语言切换边界

这两类文件是后续生成混音、分析切换、过滤插入边界的核心输入。

## 推荐工作流

### 路线 A：先做真实数据标注与语言切分

1. 准备原始音频
2. 运行 Whisper/WhisperX 标注脚本，生成 `whisper_segment_*.json`
3. 从 segment JSON 生成语言切换 JSON
4. 运行数据切分脚本，得到各语言子集

推荐脚本：

```bash
python datasets/Corpus/data_filt_en_hi.py
python datasets/ascend/data_filt_en_zh.py
python datasets/hinglish/data_filt_hinglish.py
```

Whisper/WhisperX 标注主脚本：

```bash
python translate/whisperX/whisper_wordlevel_json_corss_new.py --dataset ascend
python translate/whisperX/whisper_wordlevel_json_corss_new.py --dataset hinglish
python translate/whisperX/whisper_wordlevel_json_corss_new.py --dataset Corpus
```

说明：

- 这个脚本会扫描 `datasets/<dataset>/` 下的音频
- 逐条生成 `whisper_segment_<dataset>.json`
- 同时同步生成 `whisper_language_switch_<dataset>.json`
- 失败条目会记录到 `whisper_failed_<dataset>.json`

### 路线 B：生成插入式 code-switching 混音数据

主脚本：

```bash
python preprocess/insert_with_crossfade.py
```

这个脚本会：

- 从三套语言切分 JSON 中读取源语音和目标语音
- 根据目标语音时长决定插入片段时长
- 从源语音中选取一段带语音活动的片段
- 用 crossfade 方式插入到目标语音中
- 为每个任务输出：
  - `audio/*.wav`
  - `mixed_manifest.json`
  - `summary.json`

当前默认任务包含 12 个跨数据集方向，例如：

- `corpus_hi -> ascend_en`
- `ascend_zh -> corpus_en`
- `hinglish_hi -> ascend_en`
- `ascend_en -> corpus_hi`
- `corpus_en -> hinglish_hi`
- `hinglish_en -> ascend_zh`

当前脚本已经支持：

- `Corpus/corpus` 目录大小写修复
- 音频文件名大小写不一致的自动查找
- 旧绝对路径映射到当前项目内 `datasets/...`
- 在同一数据集根目录下按 basename 兜底找文件

### 路线 C：过滤“真实插入边界”与“自发语言切换”

如果已经有：

- `datasets/crossfade_insertions/**/mixed_manifest.json`
- `whisper_language_switch_crossfade_insertions.json`

可以运行：

```bash
python translate/whisperX/filter_true_insert_switches.py
```

这个脚本会把 Whisper 检测到的切换划分成两类：

- `true_insert_switches`
  - 由人工构造的插入边界造成
- `self_switches`
  - 原语音自身就带的语言切换

这一步对后续分析“人工插入是否真的被检测到”很重要。

### 路线 D：TTS + 分离 + 翻译的小规模演示

1. 用 TTS 生成中英音频和混音：

```bash
python preprocess/audio_create.py
```

2. 用 MossFormer2 做分离：

```bash
python split_models/mossformer2/split.py
```

3. 用 Whisper 实验脚本做转写和语言分析：

```bash
python translate/Whisper/trans_class.py
```

这是一个可快速跑通的 toy pipeline，不依赖真实数据集的整套标注流程。

## 关键脚本说明

### `preprocess/insert_with_crossfade.py`

这是当前仓库最重要的生成脚本之一。

核心逻辑：

- 目标语音 `< 3s` 直接跳过
- 目标语音 `3s ~ 5s` 时插入 `1.3s`
- 目标语音 `> 5s` 时插入 `2.0s`
- 在目标波形中随机选插入点
- 对插入片段做 RMS 匹配
- 可选混入目标局部噪声
- 用 crossfade 平滑插入边界

输出目录默认是：

```text
datasets/crossfade_insertions/
```

### `translate/whisperX/whisper_wordlevel_json_corss_new.py`

作用：

- 扫描数据集目录里的音频
- 用 Whisper 做 segment 级别转写
- 再对每段做词级时间戳分析
- 依据字符范围把词粗略映射到 `en / zh / hi / other`
- 合并为 `language_spans`
- 汇总语言切换位置

这个脚本是生成“语言 span + switch”两类结构化 JSON 的核心入口。

### `split_models/mossformer2/split.py`

作用：

- 将输入混音 wav 转成 mono 并重采样到 8 kHz
- 调用 ModelScope 的 MossFormer2 分离模型
- 将输出保存为 `spk0.wav`、`spk1.wav`

默认会处理 `preprocess/generated_batch/*_mixed.wav`。

### `split_models/speechbrain/`

这里存放的是另一套说话人分离实验，基于 SpeechBrain SepFormer。

- `split_2mix.py`
- `split_3mix.py`

它们更偏实验性质，没有像 MossFormer2 那样完整的批处理包装。

### `dl_model/`

这里不是完整训练流水线，而是一个模拟数据原型：

- `dataset_sim.py`：生成模拟的声学相似度特征
- `model.py`：双分支 MLP
- `train.py`：在模拟数据上训练 `same speaker / different speaker` 二分类
- `functions.py`：从真实音频片段里提取和比较声学特征

这部分更像方法验证，不直接依赖上面的 WhisperX 结果。

## 环境与依赖

这个仓库没有单一统一环境，按模块分开更稳。

### 1. Whisper / WhisperX 环境

常见依赖包括：

- `torch`
- `openai-whisper`
- `whisperx`
- `tqdm`
- `soundfile`
- `ffmpeg`

批处理脚本可参考：

- `run_whisper.slurm`
- `run_whisper2.slurm`

### 2. 数据导出与预处理环境

常见依赖包括：

- `datasets`
- `soundfile`
- `numpy`
- `scipy`
- `resampy`
- `torchaudio`

### 3. TTS 环境

常见依赖包括：

- `edge-tts`
- `torchaudio`
- `asyncio`

### 4. 说话人分离环境

按模型不同分别需要：

- `modelscope`
- `speechbrain`
- `torchaudio`

### 5. 原型分类器环境

常见依赖包括：

- `torch`
- `librosa`
- `numpy`
- `scikit-learn`

仓库里的 `translate/Whisper/env.txt`、`translate/whisperX/requ.txt` 可以作为参考，但不是完整、统一、可复现的一键环境文件。

## 常用命令

### 导出 ASCEND 音频

```bash
python datasets/download_ASCEND.py
```

### 重新生成语言切分 JSON

```bash
python datasets/Corpus/data_filt_en_hi.py
python datasets/ascend/data_filt_en_zh.py
python datasets/hinglish/data_filt_hinglish.py
```

### 生成 crossfade 插入混音

```bash
python preprocess/insert_with_crossfade.py
```

### 小规模 smoke test

```bash
python preprocess/insert_with_crossfade.py \
  --output-root datasets/crossfade_insertions_smoke \
  --max-items-per-pair 5
```

### WhisperX 标注

```bash
python translate/whisperX/whisper_wordlevel_json_corss_new.py --dataset ascend
python translate/whisperX/whisper_wordlevel_json_corss_new.py --dataset hinglish
python translate/whisperX/whisper_wordlevel_json_corss_new.py --dataset Corpus
```

### 过滤真实插入切换

```bash
python translate/whisperX/filter_true_insert_switches.py
```

### MossFormer2 分离

```bash
python split_models/mossformer2/split.py
```

### 模拟分类器训练

```bash
cd dl_model
python train.py
```

## 输出文件说明

### 语言切分输出

- `corpus_en_language.json`
- `corpus_hi_language.json`
- `corpus_mixed_language.json`
- `corpus_non_en_hi_language.json`
- `ascend_en_language.json`
- `ascend_zh_language.json`
- `hinglish_en_language.json`
- `hinglish_hi_language.json`

### 混音输出

每个任务目录下通常包含：

- `audio/*.wav`
- `mixed_manifest.json`

根目录下会包含：

- `summary.json`

### WhisperX 输出

- `whisper_segment_<dataset>.json`
- `whisper_language_switch_<dataset>.json`
- `whisper_failed_<dataset>.json`

## 已知情况

- 这是一个研究仓库，很多脚本是“能跑的实验入口”，不是统一封装后的生产代码
- 部分脚本或注释存在编码历史问题，显示会有 mojibake，但不影响理解主流程
- 不同模块依赖不同环境，直接在一个 conda 环境里安装全部依赖通常不稳定
- 根目录 `README.md` 现在描述的是当前仓库结构与主流程，优先以脚本实际行为为准

## 建议阅读顺序

如果你第一次接触这个仓库，建议按下面顺序看：

1. `preprocess/insert_with_crossfade.py`
2. `translate/whisperX/whisper_wordlevel_json_corss_new.py`
3. `translate/whisperX/filter_true_insert_switches.py`
4. `datasets/Corpus/data_filt_en_hi.py`
5. `datasets/ascend/data_filt_en_zh.py`
6. `datasets/hinglish/data_filt_hinglish.py`
7. `split_models/mossformer2/split.py`
8. `dl_model/`

这样最快能建立对“数据从哪里来、怎么变成混音、如何再被检测和评估”的整体理解。
