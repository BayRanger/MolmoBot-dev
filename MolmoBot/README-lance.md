# Train MolmoBot on LeRobot Lance data

The `lerobot_lance` backend trains MolmoBot on filesystem-backed LeRobot v3
Lance products. A shared filesystem or an object-store filesystem mount can
supply the path. Use the `conversion/` directory as the dataset root.

## Environment

Run from the `MolmoBot/` Python package directory with a CUDA-enabled training
environment, Python 3.12, and FFmpeg shared libraries. Install the training and
Lance dependencies together. Use a separate environment for the MolmoSpaces
evaluation extras, which require an older PyTorch version.

```bash
uv sync --locked --python 3.12 --extra train --extra lance
. .venv/bin/activate
python -c 'from lerobot_lancedb import LanceDBDataset, lance_mp_context; from torchcodec.decoders import VideoDecoder'
```

The Lance extra uses LeRobot 0.6.1, lerobot-lancedb 0.3.1, and LanceDB 0.41.0.
It also pairs PyTorch 2.11.0 with TorchCodec 0.11.1 for RGB decoding.
LeRobot 0.6.1 requires Python 3.12 or later. [LeRobot release metadata](https://pypi.org/project/lerobot/0.6.1/) · [Reader release](https://pypi.org/project/lerobot-lancedb/0.3.1/) · [TorchCodec compatibility](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions)

## Dataset and checkpoint

Start with an existing LeRobot v3.0 dataset. To convert one from disk, run
`lerobot-lance-convert --root "$LEROBOT_DATA_ROOT" --out "$LANCE_ROOT"`. Check
the converter's options with `lerobot-lance-convert --help`.

```text
conversion/
├── frames.lance/
├── videos.lance/
├── meta.lance/
└── meta/
    └── info.json
```

- This adapter uses Franka joint actions with 8 values, a one-dimensional
  observation state with at least 8 values, and the two RGB streams listed in
  the training command. Episodes must have at least three frames. If metadata
  declares splits, it must include a contiguous `train` episode range.
- External MP4 references must resolve from every training worker. Mount paths
  must agree with those stored in the Lance product.
- Provide a MolmoBot initializer checkpoint and a precomputed normalization
  YAML file with `observation.state` and `action` under the `synthmanip` stats
  entry. For a loader smoke test only, `--no_stats` bypasses that file check.
- The Lance adapter currently supports training with one observation step,
  action horizon 16, and `--no_val`. Weighted sampling and point prompts are
  unsupported. Pass a filesystem path rather than an `s3://` URI.

## Launch training

Set `CHECKPOINT`, `LANCE_ROOT`, `STATS_PATH`, and `OUTPUT_DIR` for your run.
This two-GPU example uses batch 8 per GPU and global batch 16, and stops at
10 optimizer steps. Change `--stop_at` for longer training and set
`--global_batch_size` against the total number of ranks.

```bash
torchrun --standalone --nproc-per-node=2 launch_scripts/train_molmobot.py "$CHECKPOINT" \
  --data_backend=lerobot_lance --data_paths "$LANCE_ROOT" \
  --action_preset=franka_joint \
  --camera_names droid_shoulder_light_randomization wrist_camera_zed_mini \
  --stats_path "$STATS_PATH" --no_val \
  --n_obs_steps=1 --action_horizon=16 --n_action_steps=8 --seq_len=528 \
  --device_batch_size=8 --global_batch_size=16 \
  --num_workers=4 --prefetch_factor=4 \
  --model.mm_preprocessor.image.crop_mode=resize \
  --model.mm_preprocessor.image.max_images=2 --model.mm_preprocessor.max_frames=1 \
  --compile=null --wandb=null --log_interval=1 \
  --stop_at=10 --save_folder="$OUTPUT_DIR" --save_overwrite=False
```

For multi-node runs, use the cluster's `torchrun` rendezvous settings and make
the checkpoint, stats, dataset, and output paths visible on all nodes. A
successful smoke run logs `Direct lerobot-lancedb dataset` and finishes the
requested optimizer steps.
