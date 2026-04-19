from dl_model.compare_stander.official_common import SpeechBrainTDNNPair

MODEL_NAME = "tdnn"


def build_model(args):
    return SpeechBrainTDNNPair(
        sample_rate=args.sr,
        n_mels=args.n_mels,
        channels=tuple(args.student_channels),
        emb_dim=args.emb_dim,
        dropout=args.dropout,
    )

