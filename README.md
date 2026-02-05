# Environment

```bash
conda create -n EE519_Project python=3.10 -y
conda activate EE519_Project
```
* SpeechBrain requires Windows Developer Mode to be enabled.  
* whisper needs ffmpeg installed in your computer.

## Demo

1. Run `preprocess/audio_create` to generate audio files.
2. Run `split_models/mossformer2/split.py` to downsample audio to **8 kHz** and perform speaker separation.
3. Run `translate/Whisper/trans_class.py` to detect languages, merge segments with the same language, and translate them into English.

**Note:** Each step requires a different Python environment.
