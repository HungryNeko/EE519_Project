from dl_model.compare_standard.official_common import SpeechBrainECAPAPair

MODEL_NAME = "ecapatdnn"


def build_model(args):
    return SpeechBrainECAPAPair(
        sample_rate=args.sr,
        n_mels=args.n_mels,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
    )

