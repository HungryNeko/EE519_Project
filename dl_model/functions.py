import soundfile as sf
import numpy as np
import librosa

def slice_segment(waveform, sr, t_start, t_end):
    i_start = int(t_start * sr)
    i_end = int(t_end * sr)
    return waveform[i_start:i_end]

def get_rms_energy(waveform, sr, t_start, t_end):
    seg = slice_segment(waveform, sr, t_start, t_end)
    if len(seg) == 0:
        return 0.0
    return float(np.sqrt(np.mean(seg ** 2)))

def get_db_energy(waveform, sr, t_start, t_end, eps=1e-8):
    rms = get_rms_energy(waveform, sr, t_start, t_end)
    return float(20 * np.log10(rms + eps))



def get_pitch_stats(waveform, sr, t_start, t_end):
    seg = slice_segment(waveform, sr, t_start, t_end)
    if len(seg) < sr * 0.05:
        return 0.0, 0.0

    f0 = librosa.yin(
        seg,
        fmin=50,
        fmax=400,
        sr=sr
    )
    f0 = f0[np.isfinite(f0)]
    if len(f0) == 0:
        return 0.0, 0.0

    return float(np.mean(f0)), float(np.std(f0))
