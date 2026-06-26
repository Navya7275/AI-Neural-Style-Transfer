---
title: Neural Style Transfer Studio
emoji: "\U0001F3A8"
colorFrom: amber
colorTo: orange
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
---

# Neural Style Transfer Studio

A production-ready implementation of **Gatys et al. (2015)** neural style transfer, built with PyTorch and served through a gallery-aesthetic Gradio interface. Upload any photograph and a style reference to reconstruct the content image as if it were painted in the style of the reference.

## Architecture

```
Content Image ──┐                           ┌── Content Loss (relu4_2)
                ├─▶ VGG-19 Feature ─────────┤
Style Image  ───┘   Extractor              └── Style Loss (relu1_1 → relu5_1)
                         │                          │
                         ▼                          ▼
                   L-BFGS Optimizer ◄──── Weighted Loss Sum
                         │                    + TV Regularisation
                         ▼
                   Generated Image ──▶ Post-processing ──▶ Output
```

### How It Works

1. **Feature Extraction** — A pretrained VGG-19 network extracts multi-scale feature representations. Content structure is captured at `relu4_2` (deep, semantic features), while style texture is captured across five layers (`relu1_1` through `relu5_1`) via Gram matrices.

2. **Gram-Matrix Style Representation** — For each style layer, the Gram matrix of the feature maps encodes texture statistics (correlations between feature channels) independent of spatial arrangement.

3. **Iterative Optimisation** — Starting from the content image, L-BFGS jointly minimises content loss (preserve structure), style loss (match texture statistics), and total variation loss (suppress noise).

4. **Post-processing** — Edge cropping removes pooling-boundary artifacts; optional sharpening and contrast enhancement produce a polished final image.

## Project Structure

```
├── app.py                          # Application entry point
├── nst/                            # Core algorithm package
│   ├── config.py                   # Hyperparameters and device utilities
│   ├── engine.py                   # L-BFGS optimisation loop
│   ├── pipeline.py                 # High-level stylise() orchestrator
│   ├── losses/
│   │   ├── content.py              # Content loss (feature-space MSE)
│   │   ├── style.py                # Style loss (Gram-matrix MSE)
│   │   └── regularization.py       # Total variation regulariser
│   ├── model/
│   │   ├── normalization.py        # ImageNet normalisation layer
│   │   └── vgg.py                  # VGG-19 builder with loss injection
│   ├── preprocessing/
│   │   ├── loader.py               # Aspect-preserving image loader
│   │   └── histogram.py            # Colour histogram transfer
│   └── postprocessing/
│       └── enhance.py              # Sharpening and contrast adjustment
├── ui/                             # Gradio frontend
│   ├── css.py                      # Gallery-aesthetic stylesheet
│   ├── theme.py                    # Custom Gradio theme
│   ├── components.py               # Reusable HTML fragments
│   └── layout.py                   # Interface assembly
├── examples/                       # Sample images
├── requirements.txt
└── README.md
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backbone | VGG-19 (ImageNet pretrained) |
| Optimiser | L-BFGS (second-order, fast convergence) |
| Style Metric | Gram matrices across 5 layers |
| Regulariser | Total variation loss |
| Framework | PyTorch |
| Frontend | Gradio 4.x |
| Colour Transfer | Per-channel histogram matching |
| Post-processing | PIL (sharpening, contrast) |

## Quick Start

### Local

```bash
git clone https://github.com/Navya7275/AI-Neural-Style-Transfer.git
cd AI-Neural-Style-Transfer
pip install -r requirements.txt
python app.py
```

The interface opens at `http://localhost:7860`.

### Hugging Face Spaces

This repository is configured to deploy directly to HF Spaces — push and it auto-builds.

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| Style Intensity | 5 | Controls style weight (exponential scale) |
| Content Fidelity | 6 | Controls content weight — higher preserves structure |
| Smoothing | 2 | TV regularisation strength |
| Iterations | 400 | L-BFGS steps (more = finer, slower) |
| Resolution | 512 | Longest edge in pixels |

## References

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). *A Neural Algorithm of Artistic Style*. arXiv:1508.06576
- Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. arXiv:1409.1556
