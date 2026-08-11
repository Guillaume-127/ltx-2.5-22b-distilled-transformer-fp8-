---
license: openrail++
license_name: ltx-2-community-license-agreement
license_link: https://huggingface.co/Lightricks/LTX-2.5
language:
- en
tags:
- comfyui
- fp8
- float8_e4m3fn
- ltx-2.5
- ltx-video
- lightricks
- text-to-video
- image-to-video
- video-to-video
- audio-to-video
- text-to-audio-video
- image-to-audio-video
- diffusion-transformer
- dit
base_model: Lightricks/LTX-2.5
pipeline_tag: text-to-video
---

# 🎬 LTX 2.5 (22B Distilled Transformer) - FP8 (e4m3fn) for ComfyUI

<div align="center">

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Download%20Model%20Weights-yellow.svg?style=for-the-badge)](https://huggingface.co/guillaume127/LTX-2.5-FP8)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Supported-blue.svg?style=for-the-badge)](https://comfy.org)
[![Hardware](https://img.shields.io/badge/Tested%20On-RTX%204090%2024GB-76B900.svg?style=for-the-badge&logo=nvidia)](https://huggingface.co/guillaume127/LTX-2.5-FP8)
[![License](https://img.shields.io/badge/License-OpenRAIL%2B%2B-lightgrey.svg?style=for-the-badge)](https://huggingface.co/Lightricks/LTX-2.5)

</div>

Official repository for **FP8 (`float8_e4m3fn`) selective quantization** of Lightricks' **LTX 2.5 (22B Distilled Transformer)**.

This project provides an open-source Python converter and pre-quantized weights engineered specifically for **ComfyUI** on consumer GPUs (e.g. NVIDIA RTX 4090 24GB), achieving full video generation in **~99 seconds** without quality loss or PCIe swapping.

> 📥 **Looking for the converted model weights?**  
> Download the `.safetensors` model file directly from the [Hugging Face Model Repository](https://huggingface.co/guillaume127/LTX-2.5-FP8).

---

## 🎯 Why Choose This FP8 Build?

Unlike experimental NVFP4 / Blackwell-only builds that require enterprise GPUs (SM120) or unstable `torchao` dependencies:

- ⚡ **Native RTX 40-Series & Consumer GPU Support**: Engineered specifically for **NVIDIA RTX 4090, RTX 4080, RTX 3090** and standard Ada Lovelace / Ampere GPUs using native PyTorch `float8_e4m3fn`.
- 🛠️ **Zero Experimental Dependencies**: Works 100% out-of-the-box in vanilla ComfyUI without needing custom C++ compilation or experimental nightly packages.
- 🛡️ **Selective DiT Precision**: Preserves LayerNorm, AdaLN modulation, and dual-stream audio/video conditioning in BF16 to eliminate visual noise and audio-video desync.

---

## ⚠️ Disclaimer & Experimental Status (v1)

> [!WARNING]
> **Experimental Community Build (v1)**: This FP8 quantization is an early community build (v1). While thoroughly verified and producing high-quality video outputs in standard testing, it requires further community testing across edge cases (e.g., extreme prompt lengths, specific audio-video sync patterns, unusual frame rates, etc.). 
> Use at your own risk, test it, report any edge-case issues in the **Community** tab on Hugging Face, and share your feedback!

---

## 💻 Tested Hardware & Performance Setup

This build was converted, tuned, and verified on the following hardware setup:
- **GPU**: NVIDIA GeForce RTX 4090 (24 GB VRAM)
- **System RAM**: 64 GB DDR5 / DDR4
- **Software**: ComfyUI (Native PyTorch 2.x + CUDA 12)
- **Generation Performance**: **~99 seconds** total for full video generation (down from 10+ minutes with PCIe RAM offloading!).

---

## 💡 Crucial Pro-Tip for ComfyUI Users

To achieve maximum generation speed (~99s) and prevent VRAM overflow on 24GB GPUs:

1. **Use a VRAM Cleanup Node**: Place a **VRAM Cleanup / Free GPU VRAM** node right before your KSampler or LTX 2.5 Transformer loader.
2. **Why?** The Gemma 4 12B Text Encoder (`LTXAVTEModel_`) consumes ~14.6 GB VRAM. If it stays in VRAM alongside LTX 2.5 FP8 (20 GB), ComfyUI will trigger slow *dynamic VRAM loading* over PCIe (16s/step). Purging Gemma 4 from VRAM before sampling allows LTX 2.5 FP8 to fit 100% inside your 24GB VRAM!
3. **Startup Flag**: Alternatively, launch ComfyUI with `--highvram`.

---

## 🌟 Key Features & Selective Precision

Naive FP8 casting on DiT (Diffusion Transformer) architectures breaks normalization layers and bias vectors, causing black frames or visual noise. This conversion uses **Selective Precision Preservation**:

- ⚡ **FP8 (`float8_e4m3fn`)**: Applied to large 2D weight matrices (MLP projections, Attention linear layers).
- 🛡️ **BF16 Preservation**: Applied to LayerNorm/RMSNorm, AdaLN modulation (scale, shift, gate), dual-stream audio/video input blocks (`audio_in`, `video_in`, `time_in`, `guidance_in`), biases, and position/time embeddings.

### Benefits
- **Size Reduction**: Reduced from 42 GB (BF16) down to **~21.5 GB (FP8)** — fits inside 24GB VRAM!
- **Zero Visual Degradation**: Preserves dynamic range, lighting, contrast, and dual-stream audio/video sync.
- **No PCIe Swapping**: Prevents GPU-RAM thrashing when properly configured in ComfyUI.

---

## 🚀 How to Install in ComfyUI

1. Download `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn.safetensors` from [Hugging Face](https://huggingface.co/guillaume127/LTX-2.5-FP8).
2. Move the `.safetensors` file into your ComfyUI models directory:
   ```text
   ComfyUI/models/diffusion_models/
   ```
   *(or `ComfyUI/models/unet/`)*
3. Select `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn.safetensors` in your `Load Diffusion Model` / `UNETLoader` node.

---

## 🛠️ Run the Conversion Locally

If you want to convert your own weights from BF16 to FP8 locally:

```bash
# Clone the repository
git clone https://github.com/Guillaume-127/LTX-2.5-FP8.git
cd LTX-2.5-FP8

# Run conversion using ComfyUI's virtual environment
python convert_ltx25_fp8.py --input "path/to/ltx-2.5-22b-distilled-transformer-bf16.safetensors" --output "ltx-2.5-22b-distilled-transformer-fp8_e4m3fn.safetensors"
```

---

## 🙏 Credits & References

- **Original Architecture & Weights**: [Lightricks LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5)
- **Hugging Face Model Repository**: [guillaume127/LTX-2.5-FP8](https://huggingface.co/guillaume127/LTX-2.5-FP8)
- **FP8 Selective Quantization & Release**: [Guillaume-127](https://github.com/Guillaume-127)
