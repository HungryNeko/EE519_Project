from dl_model.compare.shared import SincTDNNPairStudent

MODEL_NAME = "sincnet_tdnn"


def build_model(args):
    return SincTDNNPairStudent(
        sample_rate=args.sr,
        channels=tuple(args.student_channels),
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        time_mask_max=args.time_mask_max,
        freq_mask_max=args.freq_mask_max,
        use_dilation=True,
        use_stats_pooling=True,
        use_pairwise_product=True,
        use_specaugment=True,
        sinc_channels=args.sinc_channels,
    )
