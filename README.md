# Liquid-Time-Constant-Networks
# Can Liquid Time-Constant Networks Learn to See?
### A Systematic Ablation of LTCs, CNN Features, and CBAM Attention on Static Image Classification

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Result-Negative%2FInstructive-yellow)

📄 **[Read the full paper](./paper.pdf)**

## TL;DR

Liquid Time-Constant Networks (LTCs) are built to model **irregular, continuous-time dynamics** — their core mechanism is a neuron time-constant that adapts to the input. Static images have no temporal structure, so this project asks a direct question: *does that adaptive-timescale advantage transfer to image classification if you tokenize an image into patches and feed it through an LTC as a "sequence"?*

Across five architectures on CIFAR-10, the answer is **no, not on its own, and not efficiently**. The best LTC-based hybrid (CNN backbone + CBAM attention + LTC classification head) reaches 47.6% test accuracy — a 26-point improvement over a bare LTC-on-patches (21.8%), but still ~22 points behind a plain CNN baseline (69.8%), at roughly **240× the training time**. Every accuracy gain in this study came from the CNN and CBAM components; the LTC core never outperformed a standard classifier head. The value of the project is in showing *why*, not in a strong result.

## Table of Contents

- [Motivation](#motivation)
- [Background](#background)
- [Method](#method)
- [Results](#results)
- [Why LTC Fails for Images](#why-ltc-fails-for-images)
- [LTC vs. ViT vs. CNN](#ltc-vs-vit-vs-cnn)
- [Could LTC Ever Work for Visual Data?](#could-ltc-ever-work-for-visual-data)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Repository Structure](#repository-structure)
- [Pretrained Models](#pretrained-models)
- [Reproducing This](#reproducing-this)
- [References](#references)

## Motivation

Vision Transformers showed that patch tokenization lets sequence-style architectures work on images, trading convolutional inductive bias for scale and data. LTCs are a different kind of sequence model — instead of attention, they use continuous-time dynamics with an input-dependent time-constant, originally validated on genuinely irregular time-series (sensor data, gesture recognition, physical control). This project treats that as a testable hypothesis rather than an assumption: if patch tokenization is enough to make transformers work on images, is it enough to make LTCs work too, and does adding spatial priors (CNN features, CBAM attention) close the gap?

## Background

- **Liquid Time-Constant Networks** (Hasani, Lechner, Amini, Rus, Grosu — MIT CSAIL / IST Austria / TU Wien, AAAI 2021): replaces a fixed-dynamics continuous-time RNN with one where a nonlinear gate modulates both the injected signal and the effective time-constant, `dx/dt = -[1/τ + gate]·x + gate·A`. This gives an adaptive effective timescale `τ_eff = τ/(1 + τ·gate)`, richer expressivity, and bounded, stable states, validated on irregular time-series tasks (gesture segmentation, occupancy detection, human activity recognition, sequential MNIST, traffic/power/ozone forecasting) and continuous control (MuJoCo half-cheetah).
- **CBAM** (Woo et al., 2018): a lightweight, sequential channel-then-spatial attention module that refines convolutional feature maps — "what matters" (channel attention via pooled descriptors through a shared MLP) followed by "where it matters" (spatial attention via a large-kernel convolution over pooled channel maps).
- **Vision Transformers** (Dosovitskiy et al., 2020): the reference point for "patch tokenization works for vision," given enough data and self-attention. This project borrows the patch-tokenization idea but swaps self-attention for an LTC core to test a different sequence mechanism under the same premise.
- **AutoNCP wiring** (Hasani et al., 2020): a sparse connectivity pattern inspired by *C. elegans* neural circuits, used as the LTC cell's hidden-layer wiring (128 hidden neurons → 10 output classes) in all LTC variants here.

The original paper itself notes LTCs are "not well-suited for static datasets" — this project empirically tests that claim rather than taking it on faith.

## Method

Five architectures, all evaluated on CIFAR-10, isolating one variable at a time:

| # | Model | Core idea |
|---|-------|-----------|
| 1 | **Baseline CNN** | Standard Conv–BN–ReLU–Pool ×2 + FC classifier. No LTC, no attention. The control. |
| 2 | **Pure LTC** | Image → non-overlapping 4×4 patches (64 patches) → patches treated as a temporal sequence → LTC cell (AutoNCP wiring) → classify from the final timestep. No CNN, no attention. |
| 3 | **LTC + CBAM** | Same patch pipeline as (2), with CBAM channel+spatial attention applied to the patch embeddings before the LTC. |
| 4 | **Hybrid CNN-LTC** | A real CNN backbone (2 conv blocks) extracts spatial features first; the resulting feature map's spatial positions are reshaped into a sequence and classified by an LTC head instead of an FC layer. |
| 5 | **Hybrid CNN-LTC-CBAM** | Same as (4), with a CBAM attention "neck" refining the CNN features before they're handed to the LTC head. |

All LTC variants use `ncps`' `AutoNCP(128, num_classes)` wiring with the `LTC` cell (`batch_first=True`); training uses standard backpropagation through time with gradient clipping. Full model definitions are in [`models.py`](./models.py); training/eval logic in [`train.py`](./train.py) and [`evaluate.py`](./evaluate.py).
## Results

| Model | Parameters | Train Acc | Test Acc | Best Epoch | Time / Epoch |
|---|---|---|---|---|---|
| Baseline CNN | 591K | 57.7% | **69.83%** | 10 | 20 s |
| Pure LTC | 102K | 19.8% | 21.78% | 10 | 4,500 s |
| LTC + CBAM | 103K | 26.0% | 26.62% | 7 | 5,700 s |
| Hybrid CNN-LTC | 118K | 33.0% | 34.12% | 10 | 4,600 s |
| **Hybrid CNN-LTC-CBAM** | 119K | 47.6% | **47.55%** | 10 | 4,900 s |

**Computational cost:**

| Model | Total time (10 epochs) | Test Acc | Efficiency (Acc/hr) |
|---|---|---|---|
| Baseline CNN | 3.3 min | 69.83% | 1,267 %/hr |
| Pure LTC | 12.5 hrs | 21.78% | 1.7 %/hr |
| LTC + CBAM | 15.9 hrs | 26.62% | 1.7 %/hr |
| Hybrid CNN-LTC | 12.8 hrs | 34.12% | 2.7 %/hr |
| Hybrid CNN-LTC-CBAM | 13.6 hrs | 47.55% | 3.5 %/hr |

**What actually moved the needle:**
- CNN spatial features: **+12.34%** (Pure LTC → Hybrid CNN-LTC)
- CBAM attention on top of that: **+13.43%** further (Hybrid CNN-LTC → Hybrid CNN-LTC-CBAM)
- Combined, going from Pure LTC to the full hybrid: **+25.77% absolute**, at roughly the same wall-clock cost throughout (~240× the baseline CNN)

**The most telling result — per-class accuracy.** Pure LTC scores 61.0% on horses, 60.4% on ships, 44.4% on trucks, but only 2.4% on birds, 2.6% on cats, and 1.4% on dogs. Confusion matrices show it predicting "horse" for 17.7% of airplane images and 44.4% of bird images. CIFAR-10 horses, ships, and trucks tend to share simple, uniform backgrounds (fields/fences, water/sky, roads); birds, cats, and dogs appear in cluttered, varied backgrounds that require recognizing the actual object. This is strong evidence that **Pure LTC is classifying background texture, not objects** — confirmed by t-SNE on the penultimate-layer features, where the Baseline CNN shows clean, separated class clusters and Pure LTC shows near-total class overlap.

![Confusion matrices for Baseline CNN, Pure LTC, and the best hybrid](figures/confusion_matrices.png)
*Pure LTC (middle) scatters predictions and repeatedly falls back to "horse"; Baseline CNN (left) has a clean diagonal.*

![Per-class accuracy across all five models](figures/per_class_accuracy.png)
*Pure LTC swings from 61% (horses) to 1.4% (dogs) — a CNN's accuracy is far more even across classes.*

## Why LTC Fails for Images

Four issues compound:

1. **Spatial vs. temporal mismatch.** LTC's entire mechanism models how a signal changes *over time*. A static image has no time axis — treating 64 patches as timesteps 1→64 imposes an arbitrary ordering unrelated to the problem. ViT sidesteps this because self-attention is permutation-invariant; LTC's recurrence is order-dependent by construction and structurally can't do the same.
2. **No long-range spatial binding.** Recognizing an airplane needs to relate patches containing the nose, wings, and tail — potentially dozens of steps apart in the sequence. LTC's recurrence only carries local temporal context forward; by patch 50, signal from patch 10 has degraded through 40 recurrent steps.
3. **ODE overhead with no corresponding benefit.** Every patch-timestep requires solving the LTC's ODE — for genuinely dynamic data that buys adaptive, stable temporal modeling; for a static image it buys nothing, just 64 solver calls per forward pass, which is where the 240× overhead comes from.
4. **Texture-correlation shortcut.** With no spatial inductive bias to fall back on, Pure LTC appears to latch onto whatever's easiest to fit — background texture — rather than the object itself, matching the per-class pattern above exactly.

Critically, scale doesn't rescue this the way it rescued ViT: even Hybrid CNN-LTC-CBAM, where a real CNN already does the spatial feature extraction and LTC only classifies pre-extracted features, still underperforms a plain FC classifier on the same features by 22 points. That points to an architectural mismatch, not a data-scale one.

## LTC vs. ViT vs. CNN

| Property | CNN | ViT | LTC |
|---|:---:|:---:|:---:|
| Spatial inductive bias | ✓ | ✗ | ✗ |
| Permutation invariant | ✗ | ✓ | ✗ |
| Global patch interactions | ✗ | ✓ | ✗ |
| Parallel processing | ✓ | ✓ | ✗ |
| Native temporal modeling | ✗ | ✗ | ✓ |

ViT and LTC both tokenize images into patches and both process them as a "sequence" — that's where the similarity ends. ViT's self-attention gives it global, order-agnostic patch interactions, which is what actually lets it learn spatial relationships from data. LTC's sequential recurrence has neither globality nor order-independence, so the one thing ViT needed to succeed without convolutions is exactly what LTC lacks.

## Could LTC Ever Work for Visual Data?

Plausibly, but not on static images: video or other genuinely temporal visual data (though 3D CNNs / video transformers likely still win on spatial modeling within each frame), time-series of images (medical imaging over time, satellite imagery over time — LTC's actual home turf), event-camera streams (naturally asynchronous and continuous-time), and visual control tasks (LTC's original robotics domain, though CNN feature extraction would still do the spatial work). The common thread: LTC's niche is real temporal structure — static image classification isn't one.

## Limitations

- Single dataset (CIFAR-10, 50K images) vs. ViT's JFT-300M-scale study — though the hybrid results suggest this is architectural, not data-scale.
- 10-epoch budget, single-seed runs — no variance estimate across seeds.
- One patch size/ordering scheme tested (4×4, raster order) — alternatives (spiral, hierarchical) weren't explored, though the spatial-binding limitation likely persists regardless.
- AutoNCP wiring was designed for control tasks, not tuned for this setting.

## Future Work

- Test the same architectures on genuinely irregular temporal data (video, sensor streams) — the domain LTCs are actually built for, rather than static images.
- Swap CBAM for full self-attention and compare against a ViT-style baseline directly.
- Repeat with larger datasets and multiple seeds to get real variance estimates.

## Repository Structure

```
.
├── main.py              # Trains all five models sequentially, saves results.json + checkpoints
├── models.py             # All 5 architectures + parameter counting utility
├── train.py               # Shared training loop
├── evaluate.py           # Test-set evaluation + results table generation
├── models/                # Pretrained .pth checkpoints (see Pretrained Models below)
├── figures/               # Result plots (training curves, confusion matrices, t-SNE)
├── paper.pdf              # Full write-up (background, derivations, all figures, references)
├── requirements.txt
└── README.md
```

## Pretrained Models

The `models/` folder contains lightweight `.pth` checkpoint files included as example pretrained weights for reproducibility and quick testing.

These files are provided for:
- Reproducing the reported results.
- Loading a pretrained checkpoint for inference.
- Avoiding retraining from scratch.

If you only want to train from scratch, you can ignore this folder.

## Reproducing This

```bash
git clone https://github.com/raddy666/Liquid-Time-Constant-Networks.git
cd Liquid-Time-Constant-Networks
pip install -r requirements.txt

# Trains all 5 models sequentially, saves results.json + a .pth per model.
# CIFAR-10 downloads automatically on first run via torchvision — no manual setup needed.
python main.py
```
**Heads up on runtime**: on the RTX 3060 used for these results, the LTC-based models took 12.5–15.9 hours each — the full run across all five models is ~58 hours combined. To sanity-check the setup first, comment out the LTC entries in `models_config` inside `main.py` and run just the Baseline CNN (~3 minutes).

## References

1. Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021). *Liquid Time-constant Networks*. AAAI 2021. [arXiv:2006.04439](https://arxiv.org/pdf/2006.04439)
2. Woo, S., Park, J., Lee, J.-Y., & Kweon, I.S. (2018). *CBAM: Convolutional Block Attention Module*. [arXiv:1807.06521](https://arxiv.org/pdf/1807.06521)
3. Dosovitskiy, A. et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. [arXiv:2010.11929](https://arxiv.org/pdf/2010.11929)
4. Stanford CS231n. *Convolutional Neural Networks (CNNs / ConvNets)*. [cs231n.github.io/convolutional-networks](https://cs231n.github.io/convolutional-networks/)

## Author

**Md Tahmid Hamim** — Sichuan University, College of Software Engineering
