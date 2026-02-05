import soundfile as sf
import numpy as np
import librosa

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
        energies.append(np.mean(frame ** 2))
    return float(np.var(energies)) if energies else 0.0


def pitch_mean(seg, sr, fmin=50, fmax=400):
    if len(seg) < int(0.05 * sr):
        return 0.0
    f0 = librosa.yin(seg, fmin=fmin, fmax=fmax, sr=sr)
    f0 = f0[np.isfinite(f0)]
    return float(np.mean(f0)) if len(f0) else 0.0


def zero_crossing_rate(seg):
    if len(seg) == 0:
        return 0.0
    return float(np.mean(librosa.feature.zero_crossing_rate(seg)))


def mfcc_mean(seg, sr, n_mfcc=13):
    if len(seg) < int(0.05 * sr):
        return np.zeros(n_mfcc, dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1).astype(np.float32)


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


# ==================================================
# Similarity functions (核心)
# ==================================================

def scalar_similarity(a, b, eps=1e-8):
    if max(abs(a), abs(b)) < eps:
        return 1.0
    return float(1.0 - abs(a - b) / max(abs(a), abs(b), eps))


def vector_cosine_similarity(x, y, eps=1e-8):
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx < eps or ny < eps:
        return 0.0
    return float(np.dot(x, y) / (nx * ny + eps))


def vector_dimwise_similarity(x, y, eps=1e-8):
    sims = []
    for xi, yi in zip(x, y):
        sims.append(scalar_similarity(xi, yi, eps))
    return np.asarray(sims, dtype=np.float32)


def time_gap_similarity(t1_end, t2_start, tau=1.0):
    gap = max(0.0, t2_start - t1_end)
    return float(np.exp(-gap / tau))


# ==================================================
# Compare TWO segments (NO LANGUAGE)
# ==================================================

def compare_segments(path1, t1_start, t1_end,
                     path2, t2_start, t2_end,
                     target_sr=16000, n_mfcc=13):
    wav1, sr1 = load_audio(path1, target_sr)
    wav2, sr2 = load_audio(path2, target_sr)

    seg1 = slice_segment(wav1, sr1, t1_start, t1_end)
    seg2 = slice_segment(wav2, sr2, t2_start, t2_end)

    # ---- extract features ----
    f1 = {
        "rms": rms_energy(seg1),
        "energy_var": energy_variance(seg1, sr1),
        "pitch": pitch_mean(seg1, sr1),
        "zcr": zero_crossing_rate(seg1),
        "mfcc": mfcc_mean(seg1, sr1, n_mfcc),
        "silence_ratio": silence_ratio(seg1, sr1),
        "duration": t1_end - t1_start
    }

    f2 = {
        "rms": rms_energy(seg2),
        "energy_var": energy_variance(seg2, sr2),
        "pitch": pitch_mean(seg2, sr2),
        "zcr": zero_crossing_rate(seg2),
        "mfcc": mfcc_mean(seg2, sr2, n_mfcc),
        "silence_ratio": silence_ratio(seg2, sr2),
        "duration": t2_end - t2_start
    }

    # ---- per-feature similarities ----
    sims = {
        "sim_rms": scalar_similarity(f1["rms"], f2["rms"]),
        "sim_energy_var": scalar_similarity(f1["energy_var"], f2["energy_var"]),
        "sim_pitch": scalar_similarity(f1["pitch"], f2["pitch"]),
        "sim_zcr": scalar_similarity(f1["zcr"], f2["zcr"]),
        "sim_silence_ratio": scalar_similarity(f1["silence_ratio"], f2["silence_ratio"]),
        "sim_duration": scalar_similarity(f1["duration"], f2["duration"]),
        "sim_mfcc_cos": vector_cosine_similarity(f1["mfcc"], f2["mfcc"]),
        "sim_mfcc_dim": vector_dimwise_similarity(f1["mfcc"], f2["mfcc"]),
        "sim_time_gap": time_gap_similarity(t1_end, t2_start)
    }

    return sims


# ==================================================
# Example
# ==================================================

if __name__ == "__main__":
    exps=[
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
        for k, v in sims.items():
            if isinstance(v, np.ndarray):
                print(k, v.shape)
            else:
                print(k, f"{v:.4f}")
        print()
