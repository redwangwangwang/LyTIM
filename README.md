# LyTIM: Lyapunov-Guided Monotonic Mutual Refinement

LyTIM is a research extension of [TIM](https://github.com/yihengd/TIM), **Temporal Decoupling with Iterative Mutual-Refinement Model for Longitudinal Radiology Report Generation**. It preserves TIM's dataset and Stage-I architecture, but replaces unconditional fixed-step refinement with a clinically constrained proposal–verification–acceptance process.

> Every accepted refinement must reduce a non-negative longitudinal clinical inconsistency energy. A candidate that fails this test is rolled back, and refinement stops for that sample.

## What is new

LyTIM adds four trainable components on top of the frozen TIM representations:

1. **Multi-view clinical energy** combining current image–report consistency, temporal transition consistency, and backward historical consistency.
2. **Fact-level trust gate** that focuses the refinement prompt on inconsistent diseases while preserving already-correct findings.
3. **Monotonic training objective** that penalizes a refined report whenever its surrogate clinical energy does not decrease by a margin.
4. **Adaptive stopping with rollback** that accepts a candidate only if it lowers energy by at least `lytim_accept_epsilon`; otherwise the previous report is retained.

No new patients, report annotations, temporal pairs, or dataset fields are required. Current/previous CheXbert states and four temporal transition labels (`new`, `resolved`, `persistent`, `stable-negative`) are derived online from the original reports.

## Repository structure

```text
models/model_stage1.py                 # original TIM Stage I
models/model_stage2.py                 # original TIM Stage II baseline
models/model_lytim.py                  # LyTIM integration and monotonic inference
models/utils/lytim_controller.py       # energy, transition, gate, and stop modules
tests/test_lytim_controller.py         # CPU unit tests for the controller
scripts/lytim_train.sh                 # LyTIM training entry point
scripts/lytim_test.sh                  # LyTIM evaluation entry point
docs/METHOD.md                         # equations and implementation mapping
```

The original Stage-II model is deliberately retained so TIM and LyTIM can be compared from the same repository.

## Installation

```bash
conda create -n lytim python=3.9
conda activate lytim
pip install -r requirements.txt
```

Prepare the same local pretrained assets expected by TIM:

```text
pretrain_weights/
├── Llama-2-7b-chat-hf/
├── swin-base-patch4-window7-224/
├── bert-base-uncased/
└── chexbert.pth
```

Prepare MIMIC-CXR exactly as for TIM. The default paths are:

```text
dataset/mimic-cxr/annotation.json
dataset/mimic-cxr/files/...
```

## Training

### 1. Train TIM Stage I

```bash
bash scripts/stage1_train.sh
```

The default expected checkpoint is:

```text
save/longitudinal-mimic/stage1/checkpoints/best.pth
```

A different checkpoint can be supplied through an environment variable.

```bash
STAGE1_CKPT=/path/to/stage1_best.pth bash scripts/lytim_train.sh
```

### 2. Train LyTIM

```bash
bash scripts/lytim_train.sh
```

`lytim_seed_source=auto` first uses `curr_stage1_text` / `prev_stage1_text` when a custom dataloader already provides them. With the unmodified TIM dataset loader, those fields are absent, so LyTIM automatically generates the frozen Stage-I current report and backward prior report online. This makes the checked-in data pipeline directly usable without adding cached predictions to the annotation file.

For a fast debugging run that avoids online generation, use teacher reports as seeds:

```bash
python train.py \
  --stage lytim \
  --delta_file /path/to/stage1_best.pth \
  --lytim_seed_source teacher \
  --devices 1 \
  --strategy auto \
  --limit_train_batches 2 \
  --limit_val_batches 1
```

`teacher` is intended for integration debugging, not the main experiment.

### 3. Evaluate

```bash
LYTIM_CKPT=/path/to/lytim_best.pth \
STAGE1_CKPT=/path/to/stage1_best.pth \
bash scripts/lytim_test.sh
```

In addition to the original language and CheXbert metrics, LyTIM logs:

- `lytim_*_accepted_steps`
- `lytim_*_initial_energy`
- `lytim_*_final_energy`
- `lytim_*_energy_drop`
- `lytim_*_rolled_back`
- `lytim_*_stopped_by_head`

## Important hyperparameters

| Argument | Default | Meaning |
|---|---:|---|
| `--max_iteration` | `3` | Maximum number of refinement proposals |
| `--lytim_accept_epsilon` | `0.01` | Required energy decrease for acceptance |
| `--lytim_stop_energy` | `0.15` | Skip/stop refinement below this energy |
| `--lytim_stop_threshold` | `0.5` | Learned stop-head threshold |
| `--lytim_margin` | `0.02` | Training-time monotonic decrease margin |
| `--lytim_supervision_weight` | `0.2` | Online state/change supervision weight |
| `--lytim_monotonic_weight` | `0.1` | Monotonic energy penalty weight |
| `--lytim_keep_weight` | `0.05` | Already-correct fact preservation weight |

The absolute energy scale is learned, so `lytim_accept_epsilon` and `lytim_stop_energy` should be tuned on the validation split. Report these values with the final experiment.

## Tests and static checks

The controller tests do not load LLaMA, Swin, XCLIP, or CheXbert weights.

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
python -m compileall configs models tests train.py
```

A full model smoke test additionally requires the pretrained assets and at least one valid longitudinal batch.

## Compatibility fixes included

The original repository is preserved as the baseline, while the executable paths used by LyTIM also address several integration issues:

- robust loading of checkpoints saved under either `state_dict` or legacy `model` keys;
- correction of the prior-prompt attribute mismatch through a compatibility alias;
- removal of the Stage-II dependency on missing `prev_stage1_text` / `curr_stage1_text` dataset fields;
- corrected training scripts (`stage2` now selects `model_stage2`, and unsupported CLI arguments were removed);
- command-line booleans now parse `True` and `False` correctly;
- Lightning strategy selection now honors `--strategy` and supports single-device `auto` execution.

## Method scope

The monotonic guarantee applies to LyTIM's learned **surrogate clinical energy**, not directly to patient-level clinical correctness. The repository therefore logs energy trajectories and rollback behavior, and final claims should also be supported with CheXbert/RadGraph-style clinical metrics and expert review where available.

## Acknowledgement and citation

This codebase is derived from the official TIM repository and retains its architecture and evaluation code. Please cite TIM when using this work:

```bibtex
@inproceedings{dong2026tim,
  title={TIM: Temporal Decoupling with Iterative Mutual-Refinement Model for Longitudinal Radiology Report Generation},
  author={Dong, Yiheng and Lin, Yi and Huang, Shilong and Yang, Xiyan and Yang, Xin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6951--6961},
  year={2026}
}
```
