import csv
import json
import os
import datasets

# _ANNOT_URL = {
#     "train": "https://huggingface.co/datasets/ujs/hinglish/resolve/main/data/metadata.csv",
#     "test": "https://huggingface.co/datasets/ujs/hinglish/resolve/main/data/metadata-test.csv",
# }

_ANNOT_URL = {
    "train": "./data/metadata.csv",
    "test": "./data/metadata-test.csv"
}

# _DATA_URL = [
#     "https://huggingface.co/datasets/ujs/hinglish/resolve/main/data/train.tar.gz",
#     "https://huggingface.co/datasets/ujs/hinglish/resolve/main/data/test.tar.gz"
# ]

_DATA_URL = [
    "./data/train.tar.gz",
    "./data/test.tar.gz"
]

_DESCRIPTION = """\
A Hugginface version of the Hindi-English code-switched dataset from OpenSLR-104.
"""

class HinglishDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "path": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16_000),
                "sentence": datasets.Value("string"),
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        prompts_paths = dl_manager.download(_ANNOT_URL)
        archive = dl_manager.download(_DATA_URL)
        train_dir = 'train'
        test_dir = 'test'
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "prompts_path": prompts_paths["train"],
                    "path_to_clips": train_dir,
                    "audio_files": dl_manager.iter_archive(archive[0]),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "prompts_path": prompts_paths["test"],
                    "path_to_clips": test_dir,
                    "audio_files": dl_manager.iter_archive(archive[1]),
                },
            ),
        ]
    
    def _generate_examples(self, prompts_path, path_to_clips, audio_files):
        examples = {}
        with open(prompts_path, encoding="utf-8") as f:
            for row in f:
                data = row.strip().split(",")
                audio_path = "/".join([data[0]])
                examples[audio_path] = {
                    "path": audio_path,
                    "sentence": data[1]
                }
        inside_clips_dir = False
        id_ = 0
        for path, f in audio_files:
            if path.startswith(path_to_clips):
                inside_clips_dir = True
                if path in examples:
                    audio = {"path": path, "bytes": f.read()}
                    yield id_, {**examples[path], "audio": audio}
                    id_ += 1
            elif inside_clips_dir:
                break
