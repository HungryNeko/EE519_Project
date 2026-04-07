from dl_model.compare.shared import ReDimNetPairStudent

MODEL_NAME = "redimnet"


def build_model(args):
    return ReDimNetPairStudent(
        sample_rate=args.sr,
        n_mels=args.n_mels,
        base_channels=args.redimnet_channels,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        time_mask_max=args.time_mask_max,
        freq_mask_max=args.freq_mask_max,
        use_specaugment=True,
    )
