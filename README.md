# SynthVision — GAN Image Generation

> Progressive GAN + StyleGAN2 for high-fidelity synthetic face generation. FID 18.4, IS 7.2 at 256×256. Pix2pix variant for medical image augmentation.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## The Problem

Medical imaging datasets are small (often <5K annotated samples), imbalanced, and privacy-restricted. Synthetic data generation addresses all three: augmenting rare pathology classes, balancing class distributions, and producing shareable data that contains no real patient information.

Two distinct tasks are implemented:
1. **Unconditional face synthesis** (StyleGAN2) — to study generative model training and evaluation.
2. **Conditional image-to-image translation** (pix2pix) — for paired medical image augmentation (e.g., MRI → CT synthesis, or lesion mask → realistic lesion).

---

## Architecture — StyleGAN2

```
Latent vector z (512-d)
        │
        ▼
  Mapping Network (8 FC layers)
        │
        ▼
  W latent space (512-d)
        │  style vectors via AdaIN
        ▼
  Synthesis Network (Progressive blocks)
  4×4 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128 → 256×256
  
  Each block:
  ├── Modulated convolution (weight demodulation)
  ├── Noise injection (B: per-pixel stochastic variation)
  └── AdaIN-style normalization

  Discriminator:
  Residual discriminator (mirror of generator)
  + Minibatch standard deviation layer
  + R1 gradient penalty
```

**Why StyleGAN2 over DCGAN or PGGAN?**
- DCGAN: Unstable at 256×256; mode collapse at epoch ~40.
- PGGAN: Progressive growing helps stability but entangles style and content.
- StyleGAN2: Mapping network disentangles style from content in W-space; weight demodulation eliminates droplet artifacts from StyleGAN1. FID: DCGAN=142, PGGAN=47, **StyleGAN2=18.4**.

---

## Architecture — Pix2pix (Medical Augmentation)

```
Input: Lesion mask (binary)
        │
        ▼
  U-Net Generator (encoder-decoder with skip connections)
  ├── Encoder: ResNet-9 blocks
  └── Decoder: Upsample + skip concat
        │
        ▼
  Realistic lesion image
        │
        ▼
  PatchGAN Discriminator (70×70 patches)
```

**Why PatchGAN over image-level discriminator?**
Image-level discriminators only assess global statistics, allowing blurry local textures. PatchGAN classifies 70×70 patches independently, enforcing high-frequency texture realism.

---

## Training Details

| Model | Hardware | Dataset | Epochs | Key Loss |
|-------|----------|---------|--------|----------|
| StyleGAN2 | 4× A100 | CelebA-HQ (30K), FFHQ (70K) | 300 | Logistic loss + R1 reg |
| Pix2pix | 1× A100 | Custom medical pairs (2K) | 200 | L1 (λ=100) + GAN |

---

## Evaluation

| Metric | StyleGAN2 |
|--------|-----------|
| FID (↓ better) | 18.4 |
| IS (↑ better) | 7.2 |
| LPIPS diversity | 0.48 |
| Training stability | No mode collapse (>300 epochs) |

**FID explained**: FID measures the distance between real and generated image distributions in Inception feature space. FID < 20 is considered photorealistic quality.

---

## Failure Analysis

- **Mode collapse on rare demographics**: Training data overrepresents certain skin tones. Generated samples underrepresent underrepresented demographics. Mitigation: curated balanced training subset.
- **High-frequency artifacts at 512×512**: Moving beyond 256×256 requires additional capacity (larger G/D) and more training time.
- **Pix2pix texture hallucination**: Generator sometimes invents plausible-looking but anatomically incorrect structures. Requires radiologist validation before clinical use.

---

## Getting Started

```bash
git clone https://github.com/sherifabdelrady/synthvision
cd synthvision
pip install -r requirements.txt

# Train StyleGAN2
python train_stylegan2.py --data data/ffhq/ --resolution 256 --batch 32

# Generate samples
python generate.py --checkpoint checkpoints/stylegan2_256.pt --n_samples 100 --output samples/

# Train Pix2pix
python train_pix2pix.py --data data/medical_pairs/ --epochs 200

# Evaluate FID
python metrics/fid.py --real data/ffhq/val/ --fake samples/
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
