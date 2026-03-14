import soundfile as sf
import numpy as np
import librosa
import math

# ==================================================
# Audio utilities
# ==================================================

def load_audio(path, target_sr=16000):
    waveform, sr = sf.read(path)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return waveform.astype(np.float32), sr


def slice_segment(waveform, sr, t_start, t_end):
    i_start = max(0, int(t_start * sr))
    i_end = min(len(waveform), int(t_end * sr))
    return waveform[i_start:i_end]


# ==================================================
# Per-segment acoustic features
# ==================================================

def rms_energy(seg):
    if len(seg) == 0:
        return 0.0
    return float(np.sqrt(np.mean(seg ** 2)))


def energy_variance(seg, sr, frame_len=0.025, hop_len=0.01):
    if len(seg) < int(frame_len * sr):
        return 0.0

    fl = int(frame_len * sr)
    hl = int(hop_len * sr)

    energies = []
    for i in range(0, len(seg) - fl, hl):
        frame = seg[i:i + fl]
        energies.append(np.log(np.mean(frame ** 2) + 1e-10))

    return float(np.var(energies)) if energies else 0.0


def pitch_mean(seg, sr, fmin=50, fmax=400):
    if len(seg) < int(0.05 * sr):
        return 0.0

    f0 = librosa.yin(seg, fmin=fmin, fmax=fmax, sr=sr)
    f0 = f0[np.isfinite(f0)]

    if len(f0) < 5:
        return 0.0

    return float(np.mean(f0))


def zero_crossing_rate(seg):
    if len(seg) == 0:
        return 0.0
    return float(np.mean(librosa.feature.zero_crossing_rate(seg)))


def mfcc_features(seg, sr, n_mfcc=13):
    if len(seg) < int(0.05 * sr):
        return np.zeros(n_mfcc), np.zeros(n_mfcc)

    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc)

    mean = np.mean(mfcc, axis=1).astype(np.float32)
    std = np.std(mfcc, axis=1).astype(np.float32)

    return mean, std


def silence_ratio(seg, sr, silence_db=-40.0, frame_len=0.025, hop_len=0.01):
    if len(seg) < int(frame_len * sr):
        return 1.0

    fl = int(frame_len * sr)
    hl = int(hop_len * sr)

    silent = 0
    total = 0

    for i in range(0, len(seg) - fl, hl):
        frame = seg[i:i + fl]

        rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
        db = 20 * np.log10(rms + 1e-12)

        total += 1
        if db < silence_db:
            silent += 1

    return float(silent / total) if total > 0 else 1.0


def spectral_centroid_mean(seg, sr):
    if len(seg) < int(0.05 * sr):
        return 0.0

    sc = librosa.feature.spectral_centroid(y=seg, sr=sr)
    return float(np.mean(sc))


def spectral_bandwidth_mean(seg, sr):
    if len(seg) < int(0.05 * sr):
        return 0.0

    sb = librosa.feature.spectral_bandwidth(y=seg, sr=sr)
    return float(np.mean(sb))


# ==================================================
# Similarity functions
# ==================================================

def scalar_similarity(a, b, scale=1.0, eps=1e-8):
    # 处理 NaN/Inf
    if not np.isfinite(a) or not np.isfinite(b):
        return 0.0
    return float(np.exp(-abs(a - b) / (scale + eps)))


def vector_cosine_similarity(x, y, eps=1e-8):
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)

    if nx < eps or ny < eps:
        return 0.0

    return float(np.dot(x, y) / (nx * ny + eps))


def vector_dimwise_similarity(x, y, eps=1e-8):
    sims = []
    for xi, yi in zip(x, y):
        sims.append(scalar_similarity(xi, yi, 1.0, eps))
    return np.asarray(sims, dtype=np.float32)


def time_gap_similarity(t1_end, t2_start, tau=1.0):
    gap = max(0.0, t2_start - t1_end)
    return float(np.exp(-gap / tau))


# ==================================================
# Diagnostics helpers
# ==================================================

def warn_if_problem(name, a, b, sim, scale):
    msgs = []
    # NaN/Inf
    if not np.isfinite(a):
        msgs.append(f"{name}: a is not finite ({a})")
    if not np.isfinite(b):
        msgs.append(f"{name}: b is not finite ({b})")
    # identical (sim very close to 1) but values differ a lot
    if sim >= 0.9999 and abs(a - b) > max(1e-6, scale * 0.01):
        msgs.append(f"{name}: sim≈1 but |a-b|={abs(a-b):.4g} >> expected (scale={scale})")
    # sim is zero
    if sim <= 1e-8 and abs(a - b) < 1e-12:
        # identical zeros -> suspicious
        msgs.append(f"{name}: sim≈0 but values nearly identical ({a}, {b})")
    # large difference causing sim≈0
    if sim <= 1e-6 and abs(a - b) > scale * 5:
        msgs.append(f"{name}: sim≈0 because |a-b|={abs(a-b):.4g} >> scale={scale}")
    return msgs


# ==================================================
# Compare TWO segments (with debug)
# ==================================================

def compare_segments(path1, t1_start, t1_end,
                     path2, t2_start, t2_end,
                     target_sr=16000, n_mfcc=13, debug=True):

    wav1, sr1 = load_audio(path1, target_sr)
    wav2, sr2 = load_audio(path2, target_sr)

    seg1 = slice_segment(wav1, sr1, t1_start, t1_end)
    seg2 = slice_segment(wav2, sr2, t2_start, t2_end)

    mfcc_mean1, mfcc_std1 = mfcc_features(seg1, sr1, n_mfcc)
    mfcc_mean2, mfcc_std2 = mfcc_features(seg2, sr2, n_mfcc)

    f1 = {
        "rms": rms_energy(seg1),
        "energy_var": energy_variance(seg1, sr1),
        "pitch": pitch_mean(seg1, sr1),
        "zcr": zero_crossing_rate(seg1),
        "mfcc_mean": mfcc_mean1,
        "mfcc_std": mfcc_std1,
        "centroid_mean": spectral_centroid_mean(seg1, sr1),
        "bandwidth_mean": spectral_bandwidth_mean(seg1, sr1),
        "silence_ratio": silence_ratio(seg1, sr1),
        "duration": t1_end - t1_start
    }

    f2 = {
        "rms": rms_energy(seg2),
        "energy_var": energy_variance(seg2, sr2),
        "pitch": pitch_mean(seg2, sr2),
        "zcr": zero_crossing_rate(seg2),
        "mfcc_mean": mfcc_mean2,
        "mfcc_std": mfcc_std2,
        "centroid_mean": spectral_centroid_mean(seg2, sr2),
        "bandwidth_mean": spectral_bandwidth_mean(seg2, sr2),
        "silence_ratio": silence_ratio(seg2, sr2),
        "duration": t2_end - t2_start
    }

    # scales for scalar_similarity (tuned heuristically)
    scales = {
        "rms": 0.1,
        "energy_var": 0.5,
        "pitch": 50.0,
        "zcr": 0.1,
        "silence_ratio": 0.3,
        "duration": 1.0,
        "centroid_mean": 1000.0,
        "bandwidth_mean": 1500.0
    }

    sims = {
        "sim_rms": scalar_similarity(f1["rms"], f2["rms"], scale=scales["rms"]),
        "sim_energy_var": scalar_similarity(f1["energy_var"], f2["energy_var"], scale=scales["energy_var"]),
        "sim_pitch": scalar_similarity(f1["pitch"], f2["pitch"], scale=scales["pitch"]),
        "sim_zcr": scalar_similarity(f1["zcr"], f2["zcr"], scale=scales["zcr"]),
        "sim_silence_ratio": scalar_similarity(f1["silence_ratio"], f2["silence_ratio"], scale=scales["silence_ratio"]),
        "sim_duration": scalar_similarity(f1["duration"], f2["duration"], scale=scales["duration"]),
        "sim_mfcc_mean_cos": vector_cosine_similarity(f1["mfcc_mean"], f2["mfcc_mean"]),
        "sim_mfcc_mean_dim": vector_dimwise_similarity(f1["mfcc_mean"], f2["mfcc_mean"]),
        "sim_mfcc_std_cos": vector_cosine_similarity(f1["mfcc_std"], f2["mfcc_std"]),
        "sim_mfcc_std_dim": vector_dimwise_similarity(f1["mfcc_std"], f2["mfcc_std"]),
        "sim_centroid_mean": scalar_similarity(f1["centroid_mean"], f2["centroid_mean"], scale=scales["centroid_mean"]),
        "sim_bandwidth_mean": scalar_similarity(f1["bandwidth_mean"], f2["bandwidth_mean"], scale=scales["bandwidth_mean"]),
        "sim_time_gap": time_gap_similarity(t1_end, t2_start)
    }

    if debug:
        print(f"--- Debug: features path1={path1} [{t1_start},{t1_end}] path2={path2} [{t2_start},{t2_end}] ---")
        # 打印标量特征
        scalar_keys = ["rms", "energy_var", "pitch", "zcr", "silence_ratio", "duration", "centroid_mean", "bandwidth_mean"]
        for k in scalar_keys:
            a = f1[k]
            b = f2[k]
            scale = scales.get(k, 1.0)
            sim_key = ("sim_" + k) if k not in ("centroid_mean", "bandwidth_mean") else ("sim_" + k if k.startswith("sim_") else f"sim_{k}")
            # compute sim used
            if k in ("centroid_mean", "bandwidth_mean"):
                sim_used = sims[f"sim_{k}"] if f"sim_{k}" in sims else sims.get("sim_centroid_mean", sims.get("sim_bandwidth_mean", None))
            else:
                sim_used = sims.get(f"sim_{k}", None)
            print(f"{k}: a={a}  b={b}  scale={scale}  sim={sim_used}")
            problems = warn_if_problem(k, a, b, sim_used, scale)
            for p in problems:
                print("  WARNING:", p)

        # 打印 MFCC 向量相似信息（cos + dim）
        print("mfcc_mean_cos:", sims["sim_mfcc_mean_cos"])
        print("mfcc_mean_dim shape:", sims["sim_mfcc_mean_dim"].shape)
        # 打印前 5 维 dimwise 相似值方便观察
        print("mfcc_mean_dim first5:", sims["sim_mfcc_mean_dim"][:5].tolist())
        print("mfcc_std_cos:", sims["sim_mfcc_std_cos"])
        print("mfcc_std_dim first5:", sims["sim_mfcc_std_dim"][:5].tolist())

        print("--- End Debug ---\n")

    return sims


# ==================================================
# Example (will print diagnostics)
# ==================================================

if __name__ == "__main__":

    exps = [

        'Same speaker, same time',
        compare_segments(
            "./preprocess/speaker_en.wav", 1.5, 2.0,
            "./preprocess/speaker_en.wav", 1.5, 2.0),

        'Same speaker, diff time',
        compare_segments(
            "./preprocess/speaker_en.wav", 2.0, 2.5,
            "./preprocess/speaker_en.wav", 1.5, 2.0),

        'diff speaker, same lang',
        compare_segments(
            "./preprocess/same_language_diff_voice.wav", 0.0, 1.0,
            "./preprocess/same_language_diff_voice.wav", 3.5, 4.5),

        'Same speaker, diff lang',
        compare_segments(
            "./preprocess/same_voice_two_languages.wav", 0.0, 1.0,
            "./preprocess/same_voice_two_languages.wav", 3.0, 4.0),

        'diff speaker, diff lang',
        compare_segments(
            "./preprocess/two_languages.wav", 0.0, 1.0,
            "./preprocess/two_languages.wav", 3.0, 4.0),
    ]

    for sims in exps:
        if isinstance(sims, str):
            print(sims)
            continue
        # 主输出：若要更详细的逐项数值查看，请看上面的 Debug 段
        for k, v in sims.items():
            if isinstance(v, np.ndarray):
                print(k, v.shape)
            else:
                print(k, f"{v:.4f}")
        print()