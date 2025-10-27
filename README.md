
## Overview

VibeVoice is a novel framework designed for generating expressive, long-form, multi-speaker conversational audio, such as podcasts, from text. It addresses significant challenges in traditional Text-to-Speech (TTS) systems, particularly in scalability, speaker consistency, and natural turn-taking.

- **GPU Required**: This notebook requires a GPU to run. Free tier T4 GPU is sufficient.
- **First Run**: Initial model download may take 3-5 minutes
- **Generation Time**: Depends on text length and diffusion steps (typically 100-300 seconds)
- **Voice Quality**: Better voice samples = better cloning results
- **Recommended Voice Length**: 15-30 seconds of clear speech per speaker

## Quick Start

### Prerequisites

**GPU Runtime Required**: Before running, enable GPU in Colab:
1. Go to `Runtime` → `Change runtime type`
2. Select `Hardware accelerator`: **GPU**
3. Select `GPU type`: **T4** (recommended)
4. Click `Save`

### Choose Your Interface

#### Option 1: Gradio Interface (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nurimator/VibeVoice-Large-Q8-Colab/blob/main/vibevoice_gradio_inference.ipynb)

#### Option 2: ipywidgets Interface

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nurimator/VibeVoice-Large-Q8-Colab/blob/main/vibevoice_standalone_inference.ipynb)

## Features

- **Voice Cloning**: Upload custom voice samples for personalized speech synthesis
- **Single & Multi-Speaker Support**: Generate speech with one or multiple voices
- **Pause Control**: Add natural pauses using `[pause]` or `[pause:milliseconds]` tags
- **8-bit Quantization**: Efficient model loading for Google Colab's free tier
- **Automatic Text Chunking**: Handle long texts seamlessly
- **Progress Tracking**: Real-time generation status updates

## Model Information

**Model**: [FabioSarracino/VibeVoice-Large-Q8](https://huggingface.co/FabioSarracino/VibeVoice-Large-Q8)

**Source Repository**: [Enemyx-net/VibeVoice-ComfyUI](https://github.com/Enemyx-net/VibeVoice-ComfyUI)

**Quantization**: 
- Uses `bitsandbytes` for 8-bit quantization
- Optimized for T4 GPU (Google Colab free tier)
- Reduced memory footprint while maintaining quality

**Architecture**:
- Diffusion-based speech generation
- Multi-speaker voice cloning capability
- Custom voice embedding system

##  Usage Examples

### Single Speaker

Upload a voice sample for Speaker 1, then use:

```
Hello, this is a test. [pause:500] Let me continue speaking. [pause] And here's the final part.
```

### Multi-Speaker

Upload different voice samples for multiple speakers:

```
[1]: Hello, how are you today?
[2]: I'm doing great, thanks for asking!
[1]: That's wonderful to hear. [pause:1000] What are your plans for the weekend?
[2]: I'm planning to relax and maybe watch some movies.
```

### Pause Tags

- `[pause]` - Default pause
- `[pause:500]` - 500 millisecond pause


## ⚠️ Important Notes

This project uses **VibeVoice**, a TTS model developed by **Microsoft**. We extend our deepest gratitude to the Microsoft team for their groundbreaking work in expressive speech synthesis.

### Responsible AI Guidelines

In respect to Microsoft's research and ethical AI principles, we strongly emphasize:

**Prohibited Uses** - DO NOT use this model for:
- Creating deepfakes or impersonating individuals without explicit consent
- Generating misleading, deceptive, or fraudulent content
- Harassment, bullying, or malicious activities
- Creating content that violates laws or regulations
- Spreading misinformation or disinformation
- Any activity that could harm individuals or communities

**Ethical Guidelines**:
- Always obtain proper consent before cloning someone's voice
- Clearly disclose when audio is AI-generated
- Use the technology for legitimate, beneficial purposes
- Respect privacy and intellectual property rights
- Follow all applicable laws and regulations

By using this repository and the VibeVoice model, you agree to abide by these guidelines and accept full responsibility for your usage.
