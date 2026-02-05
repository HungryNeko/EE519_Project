HiACC Corpus - Directory Structure and Description
===================================================
Shruti Singh, Muskaan Singh, Virender Kadyan

The HiACC corpus is a richly annotated Hinglish code-switched speech dataset featuring both adult and child speakers, designed for researchers working on code-switching, speech recognition, speaker analysis, and related NLP/ASR tasks. Each category is structured identically to maintain uniformity and support streamlined data loading, training, and analysis workflows.
  
The HiACC (Hinglish Adult & Children Code-switched Corpus) dataset is organized into two major speaker categories:
  - Adult
  - Children

Main Folder Structure:
----------------------


Corpus/
├── Adult/                          # Contains data from adult speakers
│   ├── Metadata/                   # Speaker and sentence-level metadata
│   │   ├── speaker_info.csv            # Includes speaker ID, age, gender and language spoken
│   │   └── sentence_stats.csv          # Per-sentence statistics: (total number of words, total number of englis words, total 
|	|									# number of hindi words, code mixing index, duration, word per minute, speech_rate, code 
|	|									# switching instances count, hindi to english code switching instances count, english to
| 	|									# hindi code switching instances count)									
│   ├── Audio/                      # Raw audio files organized by data split
│   │   ├── train/                      # Training audio (.wav) files
│   │   ├── test/                       # Test audio (.wav) files
│   │   └── val/                        # Validation audio (.wav) files
│   ├── Transcripts/               # Text transcriptions aligned with audio
│   │   ├── train/                      # Transcripts for training audio
│   │   ├── test/                       # Transcripts for test audio
│   │   └── val/                        # Transcripts for validation audio
│   └── Annotations/              # Code-switching and linguistic annotations
│       └── code_switch_labels.json     # JSON-formatted annotations indicating type of switching within utterances
│
├── Children/                      # Contains data from adult speakers
│   ├── Metadata/                   # Speaker and sentence-level metadata
│   │   ├── speaker_info.csv            # Includes speaker ID, age, gender and language spoken
│   │   └── sentence_stats.csv          # Per-sentence statistics: (total number of words, total number of englis words, total 
|	|									# number of hindi words, code mixing index, duration, word per minute, speech_rate, code 
|	|									# switching instances count, hindi to english code switching instances count, english to
| 	|									# hindi code switching instances count)									
│   ├── Audio/                      # Raw audio files organized by data split
│   │   ├── train/                      # Training audio (.wav) files
│   │   ├── test/                       # Test audio (.wav) files
│   │   └── val/                        # Validation audio (.wav) files
│   ├── Transcripts/               # Text transcriptions aligned with audio
│   │   ├── train/                      # Transcripts for training audio
│   │   ├── test/                       # Transcripts for test audio
│   │   └── val/                        # Transcripts for validation audio
│   └── Annotations/              # Code-switching and linguistic annotations
│       └── code_switch_labels.json     # JSON-formatted annotations indicating type of switching within utterances


Additional Notes:
-----------------
- Audio Format: All audio files are mono-channel WAV format, recorded at 16 kHz sampling rate with 16-bit PCM encoding.
- Utterance Types: The recordings include both read speech (e.g., story reading) and spontaneous speech (e.g., responses to daily-life questions and image prompts).
- Annotations: The `code_switch_labels.json` files contain utterance-wise and word-level annotations for language switching (e.g., Intre-sentential, Inter-sentential).
- Train/Test/Val Splits: All subsets are speaker-disjoint to ensure unbiased evaluation.
- Transcriptions: Text files are manually curated and aligned with corresponding audio segments.

