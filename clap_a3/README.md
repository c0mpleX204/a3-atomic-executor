# CLAP action predictor on A3

This experiment feeds the official Generalizable CLAP fine-grained action predictor a saved A3 Planner proposal and a post-grasp observation. The Planner's front-image target hint is back-projected through depth from the initial frame. The fine stage consumes a crop of the three post-grasp RGB-D views, step text and gripper state. The existing A3 simulation replays the predicted move-and-hold action with joint and held-object collision checks.

The bundled example is `prefilled`, seed `13000`, with five cookies already held. The model predicted a position of approximately `[0.0879, 0.1102, 0.7730]` m and the closed-gripper class. The arm carried the cookies above the box in 158 control ticks. A predicted contact between a held cookie and an existing cookie blocked the descent. See [the recorded prediction](examples/prefilled-13000/prediction.json) and [execution report](examples/prefilled-13000/observed-execution.json).

The simulator adapter assumes that CLAP's Panda tool `+Y` is the jaw axis, `+Z` is the approach axis and the tip origin is the pad midpoint. This is a test mapping. The replay covers one closed-gripper prediction and stops before descent when the collision check rejects the target.

## Get the checkpoint and pinned dependencies

The [official checkpoint](https://github.com/Jianshu-Hu/Generalizable-CLAP/blob/e536857194288d4707bbdce9075ce855e3eb4b49/scripts/EVALUATION.md#pre-trained-ckpts) is stored in Git LFS parts. Restoration verifies every part and the SHA-256 of the original `model_5.pth`. Fetching and assembling uses about 8.6 GB of local disk for the parts and restored file.

```sh
git lfs install
git lfs pull --include="clap_a3/weights/model_5.pth.part-*"
python -m clap_a3.restore_weights --output artifacts/model_5.pth

git clone https://github.com/Jianshu-Hu/Generalizable-CLAP.git ../Generalizable-CLAP
git -C ../Generalizable-CLAP checkout --detach e536857194288d4707bbdce9075ce855e3eb4b49
git -C ../Generalizable-CLAP submodule update --init pvr_ckpts/dinov2

git clone https://github.com/Eter0109/a3_dual_arm_sim.git ../a3_dual_arm_sim
git -C ../a3_dual_arm_sim checkout --detach a61b13fe7253a262781b42253a0d382bf67a98e1
```

Use the repository's `requirements.txt` in a Python 3.12 simulation environment. In a separate Python 3.10 inference environment, install PyTorch with CUDA and the packages below. The tested inference path uses PyTorch point rendering on Windows when the upstream CUDA extension is unavailable.

```sh
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r clap_a3/requirements-inference.txt
hf download google/siglip-so400m-patch14-224 --revision d04cf29fca7b6374f74d8bea1969314492266b5e --include config.json special_tokens_map.json tokenizer_config.json spiece.model --local-dir artifacts/siglip-config
```

## Reproduce the replay

From the repository root, run capture and execution with the simulation environment and inference with the inference environment.

```sh
python -m clap_a3.capture --a3-root ../a3_dual_arm_sim --fixture clap_a3/examples/prefilled-13000 --output artifacts/observation.npz

python -m clap_a3.infer --upstream ../Generalizable-CLAP --checkpoint artifacts/model_5.pth --config clap_a3/weights/config.yaml --siglip-config artifacts/siglip-config --observation artifacts/observation.npz --output artifacts/prediction.json

python -m clap_a3.execute --a3-root ../a3_dual_arm_sim --fixture clap_a3/examples/prefilled-13000 --prediction artifacts/prediction.json --output artifacts/clap-replay
```

For a replay without running the model, pass `--prediction clap_a3/examples/prefilled-13000/prediction.json` to the last command. The replay writes `execution.json`, `execution.mp4` and a final MuJoCo state.
