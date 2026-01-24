import numpy as np
import soundfile as sf
import resampy

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def prepare_audio(input_path, target_sr=8000):
    """
    读取音频 -> 单声道 -> 下采样到 8kHz
    返回处理后的音频路径
    """
    audio, sr = sf.read(input_path)

    # 转单声道
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # 下采样
    if sr != target_sr:
        audio = resampy.resample(audio, sr, target_sr)

    out_path = "../../preprocess/mixed_8k.wav"
    sf.write(out_path, audio, target_sr)
    return out_path


def main():
    input_wav = "../../preprocess/mixed.wav"   # 原始混合语音
    processed_wav = prepare_audio(input_wav)
    # processed_wav= "../../preprocess/mixed_8k.wav"
    # 官方 pipeline
    separation = pipeline(
        Tasks.speech_separation,
        model="iic/speech_mossformer2_separation_temporal_8k"
    )

    result = separation(processed_wav)

    # 保存分离结果
    for i, signal in enumerate(result["output_pcm_list"]):
        sf.write(
            f"output_spk{i}.wav",
            np.frombuffer(signal, dtype=np.int16),
            8000
        )


if __name__ == "__main__":
    main()
