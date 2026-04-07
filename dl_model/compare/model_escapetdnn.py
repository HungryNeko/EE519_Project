from dl_model.compare.shared import ECAPAPairStudent

MODEL_NAME = "escapetdnn"


def build_model(args):
    return ECAPAPairStudent(
        sample_rate=args.sr,
        n_mels=args.n_mels,
        channels=args.ecapa_channels,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        time_mask_max=args.time_mask_max,
        freq_mask_max=args.freq_mask_max,
        use_specaugment=True,
    )
