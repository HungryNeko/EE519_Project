from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import librosa
import whisper

from dl_model.functions import SpeakerFeatureExtractor


def load_audio(path: Path, sr=16000):
    wav, s = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if s != sr:
        wav = librosa.resample(wav, orig_sr=s, target_sr=sr)
    return wav.astype(np.float32), sr


def extract_window(wav: np.ndarray, sr: int, start_time: float, end_time: float):
    start_i = int(round(start_time * sr))
    end_i = int(round(end_time * sr))
    length = max(1, end_i - start_i)
    out = np.zeros(length, dtype=np.float32)

    src_start = max(0, start_i)
    src_end = min(len(wav), end_i)
    if src_end <= src_start:
        return out

    dst_start = src_start - start_i
    dst_end = dst_start + (src_end - src_start)
    out[dst_start:dst_end] = wav[src_start:src_end]
    return out


def detect_lang_by_char(ch: str):
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF:
        return "zh"
    if 0x0900 <= o <= 0x097F:
        return "hi"
    if "a" <= ch.lower() <= "z":
        return "en"
    return "other"


def predict(audio_path: Path, model_path: Path, model_class, device=None, window_sec: float = 1.0):
    # 固定随机种子
    torch.manual_seed(0)
    np.random.seed(0)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = model_class().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    wav, sr = load_audio(audio_path, sr=16000)

    whisper_model = whisper.load_model("large-v3", device=device)

    # ===== load full audio（与 whisper 训练数据生成一致）=====
    audio_full = whisper.load_audio(str(audio_path))
    sr_whisper = whisper.audio.SAMPLE_RATE

    # ===== 第一段 transcribe（获取 segment 边界）=====
    base_result = whisper_model.transcribe(
        str(audio_path),
        task="transcribe",
        language=None,
        verbose=False,
        fp16=False,
        word_timestamps=False,  # 第一次只需要 segment 边界
    )

    print("\n=== ASR TEXT ===")
    print(base_result.get("text", "").strip())

    segments = base_result.get("segments", [])

    language_spans = []

    # ===== segment → 再跑 whisper(word) 获取精确时间戳（与参考脚本完全一致）=====
    for seg in segments:
        seg_start = seg.get("start")
        seg_end = seg.get("end")

        seg_audio = audio_full[int(seg_start * sr_whisper): int(seg_end * sr_whisper)]

        seg_result = whisper_model.transcribe(
            seg_audio,
            task="transcribe",
            language=None,
            verbose=False,
            fp16=False,
            temperature=0.2,
            beam_size=1,
            word_timestamps=True,
        )

        seg_language = seg_result.get("language", "unknown")

        words = []
        for wseg in seg_result.get("segments", []):
            for w in wseg.get("words", []):
                if w.get("start") is None or w.get("end") is None:
                    continue
                words.append({
                    "word": w.get("word", "").strip(),
                    "start": seg_start + w["start"],
                    "end": seg_start + w["end"],
                    "score": w.get("probability", 0.0),
                })

        # ===== 构建 language_spans =====
        cur = None

        def flush():
            nonlocal cur
            if cur:
                cur["text"] = cur["text"].strip()
                cur["score"] = cur["score_sum"] / max(cur["count"], 1)
                cur.pop("score_sum")
                cur.pop("count")
                language_spans.append(cur)
                cur = None

        for w in words:
            if not w["word"]:
                continue
            lang = detect_lang_by_char(w["word"][0])
            if lang == "other":
                flush()
                continue
            if cur is None or cur["language"] != lang:
                flush()
                cur = {
                    "language": lang,
                    "start": w["start"],
                    "end": w["end"],
                    "text": w["word"],
                    "score_sum": w["score"] or 0.0,
                    "count": 1,
                }
            else:
                cur["end"] = w["end"]
                cur["text"] += " " + w["word"]
                cur["score_sum"] += w["score"] or 0.0
                cur["count"] += 1

        flush()

        if not language_spans:
            language_spans.append({
                "language": seg_language,
                "start": seg_start,
                "end": seg_end,
                "text": seg_result.get("text", "").strip(),
                "score": seg.get("avg_logprob"),
            })

    # ===== 找 switch（支持多次切换，优先选择时间长度更长的部分）=====
    if len(language_spans) < 2:
        print("\nNo switch detected")
        print("Result: mix")
        return None

    # 找出所有切换点
    switches = []
    for i in range(1, len(language_spans)):
        if language_spans[i]["language"] != language_spans[i - 1]["language"]:
            # 计算切换点前后的语言段时长
            before_dur = language_spans[i - 1]["end"] - language_spans[i - 1]["start"]
            after_dur = language_spans[i]["end"] - language_spans[i]["start"]
            switches.append({
                "switch_time": language_spans[i - 1]["end"],  # 使用前一个语言段的结束时间（与训练数据一致）
                "from_lang": language_spans[i - 1]["language"],
                "to_lang": language_spans[i]["language"],
                "from_start": language_spans[i - 1]["start"],
                "to_end": language_spans[i]["end"],
                "before_dur": before_dur,
                "after_dur": after_dur,
            })

    if not switches:
        print("\nNo switch detected")
        print("Result: mix")
        return None

    # 计算每个语言段的总时长
    lang_duration = {}
    for span in language_spans:
        lang = span["language"]
        duration = span["end"] - span["start"]
        lang_duration[lang] = lang_duration.get(lang, 0) + duration

    print(f"\nLanguage durations: {lang_duration}")
    print(f"\nSwitches found: {len(switches)}")
    for i, sw in enumerate(switches):
        print(f"  [{i}] {sw['from_lang']}→{sw['to_lang']} @ {sw['switch_time']:.3f}s "
              f"(before={sw['before_dur']:.3f}s, after={sw['after_dur']:.3f}s)")

    # 策略：选择前后时长都足够长的切换点（优先选择更可靠的切换）
    # 过滤掉前后时长都小于 window_sec 的切换点
    valid_switches = [
        sw for sw in switches
        if sw["before_dur"] >= window_sec * 0.5 and sw["after_dur"] >= window_sec * 0.5
    ]

    if valid_switches:
        # 选择前后时长之和最大的切换点
        switch = max(valid_switches, key=lambda s: s["before_dur"] + s["after_dur"])
    else:
        # 如果没有有效切换点，选择第一个
        switch = switches[0]

    switch_time = switch["switch_time"]

    print(f"\nSwitch detected at: {switch_time:.3f} sec (window={window_sec}s)")

    seg1 = extract_window(wav, sr, switch_time - 1.0, switch_time)
    seg2 = extract_window(wav, sr, switch_time, switch_time + 1.0)

    extractor = SpeakerFeatureExtractor(sr=16000)
    emb1 = extractor.extract_embedding(seg1)
    emb2 = extractor.extract_embedding(seg2)

    pair = np.stack([emb1, emb2], axis=0)
    x = torch.from_numpy(pair).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))

    label = "code_switch" if pred == 1 else "mix"

    print(f"Prediction: {label}")
    print(f"Probabilities: mix={probs[0]:.4f}, code_switch={probs[1]:.4f}")

    return {
        "switch_time": switch_time,
        "prediction": label,
        "probabilities": probs.tolist()
    }


if __name__ == "__main__":
    from dl_model.mlp.model1 import MLPModel1

    audio_path = Path(r"samples\output_spk0.wav")
    model_path = Path(r"dl_model/checkpoints/MLPModel1_best.pth")
    model_class = MLPModel1

    predict(audio_path, model_path, model_class)
