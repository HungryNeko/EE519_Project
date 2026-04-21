from dl_model.compare_standard.official_common import SpeechBrainResNetPair

MODEL_NAME = "resnet"


def build_model(args):
    return SpeechBrainResNetPair(
        sample_rate=args.sr,
        n_mels=args.n_mels,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
    )

