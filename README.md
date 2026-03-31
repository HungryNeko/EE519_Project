# Environment

```bash
conda create -n EE519_Project python=3.10 -y
conda activate EE519_Project
```
* SpeechBrain requires Windows Developer Mode to be enabled.  
* whisper needs ffmpeg installed in your computer.

## Demo
1. Run `pip install -r demo/requirements.txt` to set up environment
2. Run `python demo/demo_pipeline.py samples/mixed.wav` (replace `samples/mixed.wav` with your audio path)
* add `--device cuda` to run with GPU, `--device cpu` to run with CPU
* add `--whisper-model #model type#` to specify whisper model 
* add `--tdnn-weight dl_model/final_model/xxx.pth` to specify TDNN path
* add `--use-llm` to activate llm model for TDNN support


## Models
1. Run `preprocess/audio_create` to generate audio files.
2. Run `split_models/mossformer2/split.py` to downsample audio to **8 kHz** and perform speaker separation.
3. Run `translate/Whisper/trans_class.py` to detect languages, merge segments with the same language, and translate them into English.

**Note:** Each step requires a different Python environment.

## Data
1. Corpus:
* HI only    : 2727
* EN only    : 1674
* Mixed lang : 328
2. ascend
* ZH only    : 5391
* EN only    : 4289
* Mixed lang : 2180
