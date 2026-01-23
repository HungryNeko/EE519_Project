from huggingface_hub import snapshot_download

local_dir = "D:/Github/EE519_Project/pretrained_models/sepformer-wsj02mix"

snapshot_download(
    repo_id="speechbrain/sepformer-wsj02mix",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # ⭐ 关键：不使用 symlink
)
