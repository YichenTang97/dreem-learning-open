# CNN-RNN Architecture Rationale (SOL Experiments)

This note documents the literature-grounded configuration used in `net.json` for the `CNNMaxPoolEpochEncoder + LSTMSequenceEncoder` model.

## Final Recommended Configuration

### CNN epoch encoder (`CNNMaxPoolEpochEncoder`)

Per-channel shared 1D CNN with channel-wise max pooling downstream:

1. Conv1: `1 -> 16`, `kernel=64`, `stride=4`, `padding=32`, `bias=false`, `activation=ELU`, `batch_norm=true`, `dropout=0.25`
2. Conv2: `16 -> 32`, `kernel=16`, `stride=4`, `padding=8`, `bias=false`, `activation=ELU`, `batch_norm=true`, `dropout=0.25`
3. Conv3: `32 -> 64`, `kernel=8`, `stride=2`, `padding=4`, `bias=false`, `activation=ELU`, `batch_norm=true`, `dropout=0.0`
4. `AdaptiveAvgPool1d(1)` to produce one embedding vector per channel per epoch.

### Sequence encoder (`LSTMSequenceEncoder`)

- `layers=2`
- `cells=64`
- `bidir=false`
- `dropout=0.5` (in this repo, dropout is applied inside stacked LSTM and again after recurrent output)

## Why These Choices

### 1) CNN depth and kernel schedule

- **Three temporal CNN layers** is a compact middle ground between very small EEG models and heavier sleep-staging models.
- The kernel schedule `64 -> 16 -> 8` at 100 Hz maps to progressively finer temporal scales:
  - `64` samples = `640 ms`
  - `16` samples = `160 ms`
  - `8` samples = `80 ms`
- This follows the common EEG principle of using a longer first temporal kernel to capture low-frequency structure, then smaller kernels for increasingly local patterns.

**Literature link:** EEGNet sets temporal filtering as the first operation and motivates kernel length around half sampling rate to capture low-frequency content efficiently.

### 2) Strides and downsampling

- `stride=4,4,2` performs aggressive but controlled temporal compression, similar in spirit to pooling-heavy sleep EEG architectures (e.g., DeepSleepNet branch designs).
- This keeps runtime and memory suitable for wearable/edge-oriented experiments while preserving enough temporal resolution before global pooling.

### 3) Activations and normalization

- **ELU + BatchNorm** is chosen to align with EEGNet and other EEG CNN conventions where ELU is often preferred over plain ReLU for stable optimization on low-SNR biosignals.
- `bias=false` in Conv layers is standard when BatchNorm follows immediately.

### 4) Dropout strategy

- **CNN dropout = 0.25** on early/mid blocks: regularizes features without over-weakening local temporal detectors.
- **RNN dropout = 0.5** at sequence output: stronger regularization for temporal modeling, consistent with sleep-staging CNN-RNN practice (DeepSleepNet uses pervasive `0.5` dropout).

### 5) RNN depth/width for this dataset regime

- **Unidirectional 2-layer LSTM (64 cells)** favors a smaller parameter budget:
  - Unidirectional supports causal/online deployment constraints.
  - Two layers increase transition-modeling capacity over one layer without the full parameter cost of bidirectional stacks.
  - 64 cells matches the slimmer CNN embedding and further reduces LOOCV train cost; widen if validation clearly underfits.
- DeepSleepNet used a heavier 2-layer **BiLSTM** (512/512 directions), which is stronger but less aligned with causal deployment and small-data regularization constraints.

### 6) Output embedding size

- Final CNN channel embedding set to **64** with a gradual width schedule `16 -> 32 -> 64` to cut parameters versus wider stacks while keeping the same kernel/stride temporal design.
- The LSTM input width matches this embedding; overall capacity is intentionally modest for small-data / LOOCV regimes.

## References

1. Lawhern et al., **EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces** (J Neural Eng, 2018).  
   - arXiv: <https://arxiv.org/abs/1611.08024>  
   - Key points used here: temporal-first filtering, ELU + BatchNorm, dropout regularization, compact design for limited EEG data.

2. Supratak et al., **DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG** (IEEE TNSRE, 2017).  
   - arXiv: <https://arxiv.org/abs/1703.04046>  
   - Key points used here: CNN + recurrent sequence modeling, sampling-rate-scaled first kernels (`Fs/2`, `4*Fs`), recurrent dropout-heavy regularization, sequence context importance.

3. Phan et al., **SeqSleepNet: End-to-End Hierarchical Recurrent Neural Network for Sequence-to-Sequence Automatic Sleep Staging** (IEEE TNSRE, 2019).  
   - PubMed: <https://pubmed.ncbi.nlm.nih.gov/30716040/>  
   - Key point used here: explicit sequence-level modeling improves staging vs epoch-only classification.

## Practical Tuning Order (if you iterate)

1. Keep the current backbone fixed and tune only `LSTM cells` in `{48, 64, 96}`.
2. If underfitting, increase final CNN out channels `64 -> 96` or `64 -> 128`, and/or raise LSTM cells toward `96`.
3. If overfitting, reduce Conv2/Conv3 widths first (`32/64 -> 24/48`) or add dropout on the last conv block before widening the LSTM.
4. Keep `bidir=false` unless you intentionally move to offline-only inference.
