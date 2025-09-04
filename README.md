# VibeVoice ComfyUI Nodes

A comprehensive ComfyUI integration for Microsoft's VibeVoice text-to-speech model, enabling high-quality single and multi-speaker voice synthesis directly within your ComfyUI workflows.

## Features

### Core Functionality
- 🎤 **Single Speaker TTS**: Generate natural speech with optional voice cloning
- 👥 **Multi-Speaker Conversations**: Support for up to 4 distinct speakers
- 🎯 **Voice Cloning**: Clone voices from audio samples
- 📝 **Text File Loading**: Load scripts from text files

### Model Options
- 🚀 **Two Model Sizes**: 1.5B (faster) and 7B (higher quality)
- 🔧 **Flexible Configuration**: Control temperature, sampling, and guidance scale

### Performance & Optimization
- ⚡ **Attention Mechanisms**: Choose between auto, eager, sdpa, or flash_attention_2
- 🎛️ **Diffusion Steps**: Adjustable quality vs speed trade-off (default: 20)
- 💾 **Memory Management**: Toggle automatic VRAM cleanup after generation
- 🧹 **Free Memory Node**: Manual memory control for complex workflows

## Video Demo
<p align="center">
  <a href="https://www.youtube.com/watch?v=fIBMepIBKhI">
    <img src="https://img.youtube.com/vi/fIBMepIBKhI/maxresdefault.jpg" alt="VibeVoice ComfyUI Wrapper Demo" />
  </a>
  <br>
  <strong>Click to watch the demo video</strong>
</p>

## Installation

### Automatic Installation (Recommended)
1. Clone this repository into your ComfyUI custom nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
```

2. Restart ComfyUI - the nodes will automatically install requirements on first use

## Available Nodes

### 1. VibeVoice Load Text From File
Loads text content from files in ComfyUI's input/output/temp directories.
- **Supported formats**: .txt
- **Output**: Text string for TTS nodes

### 2. VibeVoice Single Speaker
Generates speech from text using a single voice.
- **Text Input**: Direct text or connection from Load Text node
- **Models**: VibeVoice-1.5B or VibeVoice-7B-Preview
- **Voice Cloning**: Optional audio input for voice cloning
- **Parameters** (in order):
  - `text`: Input text to convert to speech
  - `model`: VibeVoice-1.5B, VibeVoice-Large-Preview (deprecated), or VibeVoice-Large
  - `attention_type`: auto, eager, sdpa, or flash_attention_2 (default: auto)
  - `free_memory_after_generate`: Free VRAM after generation (default: True)
  - `diffusion_steps`: Number of denoising steps (5-100, default: 20)
  - `seed`: Random seed for reproducibility (default: 42)
  - `cfg_scale`: Classifier-free guidance (1.0-2.0, default: 1.3)
  - `use_sampling`: Enable/disable deterministic generation (default: False)
- **Optional Parameters**:
  - `voice_to_clone`: Audio input for voice cloning
  - `temperature`: Sampling temperature (0.1-2.0, default: 0.95)
  - `top_p`: Nucleus sampling parameter (0.1-1.0, default: 0.95)

### 3. VibeVoice Multiple Speakers
Generates multi-speaker conversations with distinct voices.
- **Speaker Format**: Use `[N]:` notation where N is 1-4
- **Voice Assignment**: Optional voice samples for each speaker
- **Recommended Model**: VibeVoice-7B-Preview for better multi-speaker quality
- **Parameters** (in order):
  - `text`: Input text with speaker labels
  - `model`: VibeVoice-1.5B, VibeVoice-Large-Preview (deprecated), or VibeVoice-Large
  - `attention_type`: auto, eager, sdpa, or flash_attention_2 (default: auto)
  - `free_memory_after_generate`: Free VRAM after generation (default: True)
  - `diffusion_steps`: Number of denoising steps (5-100, default: 20)
  - `seed`: Random seed for reproducibility (default: 42)
  - `cfg_scale`: Classifier-free guidance (1.0-2.0, default: 1.3)
  - `use_sampling`: Enable/disable deterministic generation (default: False)
- **Optional Parameters**:
  - `speaker1_voice` to `speaker4_voice`: Audio inputs for voice cloning
  - `temperature`: Sampling temperature (0.1-2.0, default: 0.95)
  - `top_p`: Nucleus sampling parameter (0.1-1.0, default: 0.95)

### 4. VibeVoice Free Memory
Manually frees all loaded VibeVoice models from memory.
- **Input**: `audio` - Connect audio output to trigger memory cleanup
- **Output**: `audio` - Passes through the input audio unchanged
- **Use Case**: Insert between nodes to free VRAM/RAM at specific workflow points
- **Example**: `[VibeVoice Node] → [Free Memory] → [Save Audio]`

## Multi-Speaker Text Format

For multi-speaker generation, format your text using the `[N]:` notation:

```
[1]: Hello, how are you today?
[2]: I'm doing great, thanks for asking!
[1]: That's wonderful to hear.
[3]: Hey everyone, mind if I join the conversation?
[2]: Not at all, welcome!
```

**Important Notes:**
- Use `[1]:`, `[2]:`, `[3]:`, `[4]:` for speaker labels
- Maximum 4 speakers supported
- The system automatically detects the number of speakers from your text
- Each speaker can have an optional voice sample for cloning

## Model Information

### VibeVoice-1.5B
- **Size**: ~5GB download
- **Speed**: Faster inference
- **Quality**: Good for single speaker
- **Use Case**: Quick prototyping, single voices

### VibeVoice-Large-Preview (Deprecated)
- **Size**: ~17GB download
- **Speed**: Slower inference
- **Quality**: Superior, especially for multi-speaker
- **Use Case**: Legacy support only - use VibeVoice-Large instead
- **Note**: This is the preview version, now superseded by VibeVoice-Large

### VibeVoice-Large
- **Size**: ~17GB download
- **Speed**: Slower inference but optimized
- **Quality**: Best available quality
- **Use Case**: Highest quality production, multi-speaker conversations
- **Note**: Latest official release from Microsoft

Models are automatically downloaded on first use and cached in `ComfyUI/models/vibevoice/`.

## Generation Modes

### Deterministic Mode (Default)
- `use_sampling = False`
- Produces consistent, stable output
- Recommended for production use

### Sampling Mode
- `use_sampling = True`
- More variation in output
- Uses temperature and top_p parameters
- Good for creative exploration

## Voice Cloning

To clone a voice:
1. Connect an audio node to the `voice_to_clone` input (single speaker)
2. Or connect to `speaker1_voice`, `speaker2_voice`, etc. (multi-speaker)
3. The model will attempt to match the voice characteristics

**Requirements for voice samples:**
- Clear audio with minimal background noise
- Minimum 3–10 seconds. Recommended at least 30 seconds for better quality
- Automatically resampled to 24kHz

## Tips for Best Results

1. **Text Preparation**:
   - Use proper punctuation for natural pauses
   - Break long texts into paragraphs
   - For multi-speaker, ensure clear speaker transitions

2. **Model Selection**:
   - Use 1.5B for quick single-speaker tasks (fastest, ~8GB VRAM)
   - Use Large-Preview (deprecated) for legacy workflows (~16GB VRAM)
   - Use Large for best quality and multi-speaker (~16GB VRAM)

3. **Seed Management**:
   - Default seed (42) works well for most cases
   - Save good seeds for consistent character voices
   - Try random seeds if default doesn't work well

4. **Performance**:
   - First run downloads models (5-17GB)
   - Subsequent runs use cached models
   - GPU recommended for faster inference

## System Requirements

### Hardware
- **Minimum**: 8GB VRAM for VibeVoice-1.5B
- **Recommended**: 16GB+ VRAM for VibeVoice-7B
- **RAM**: 16GB+ system memory

### Software
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- ComfyUI (latest version)

## Troubleshooting

### Installation Issues
- Ensure you're using ComfyUI's Python environment
- Try manual installation if automatic fails
- Restart ComfyUI after installation

### Generation Issues
- If voices sound unstable, try deterministic mode
- For multi-speaker, ensure text has proper `[N]:` format
- Check that speaker numbers are sequential (1,2,3 not 1,3,5)

### Memory Issues
- 7B model requires ~16GB VRAM
- Use 1.5B model for lower VRAM systems
- Models use bfloat16 precision for efficiency

## Examples

### Single Speaker
```
Text: "Welcome to our presentation. Today we'll explore the fascinating world of artificial intelligence."
Model: VibeVoice-1.5B
cfg_scale: 1.3
use_sampling: False
```

### Two Speakers
```
[1]: Have you seen the new AI developments?
[2]: Yes, they're quite impressive!
[1]: I think voice synthesis has come a long way.
[2]: Absolutely, it sounds so natural now.
```

### Four Speaker Conversation
```
[1]: Welcome everyone to our meeting.
[2]: Thanks for having us!
[3]: Glad to be here.
[4]: Looking forward to the discussion.
[1]: Let's begin with the agenda.
```

## Performance Benchmarks

| Model                  | VRAM Usage | Context Length | Max Audio Duration |
|------------------------|------------|----------------|-------------------|
| VibeVoice-1.5B         | ~8GB | 64K tokens | ~90 minutes |
| VibeVoice-Large-Preview | ~16GB | 32K tokens | ~45 minutes |
| VibeVoice-Large | ~16GB | 32K tokens | ~45 minutes |

## Known Limitations

- Maximum 4 speakers in multi-speaker mode
- Works best with English and Chinese text
- Some seeds may produce unstable output
- Background music generation cannot be directly controlled

## License

This ComfyUI wrapper is released under the MIT License. See LICENSE file for details.

**Note**: The VibeVoice model itself is subject to Microsoft's licensing terms:
- VibeVoice is for research purposes only
- Check Microsoft's VibeVoice repository for full model license details

## Links

- [Original VibeVoice Repository](https://github.com/microsoft/VibeVoice) - Official Microsoft VibeVoice repository

## Credits

- **VibeVoice Model**: Microsoft Research
- **ComfyUI Integration**: Fabio Sarracino
- **Base Model**: Built on Qwen2.5 architecture

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review ComfyUI logs for error messages
3. Ensure VibeVoice is properly installed
4. Open an issue with detailed error information

## Contributing

Contributions welcome! Please:
1. Test changes thoroughly
2. Follow existing code style
3. Update documentation as needed
4. Submit pull requests with clear descriptions

## Changelog

### Version 1.0.9
- Embedded VibeVoice code directly into the wrapper
  - Added vvembed folder containing the complete VibeVoice code (MIT licensed)
  - No longer requires external VibeVoice installation
  - Ensures continued functionality for all users

### Version 1.0.8
- BFloat16 Compatibility Fix
  - Fixed tensor type compatibility issues with audio processing nodes
  - Input audio tensors are now converted from BFloat16 to Float32 for numpy compatibility
  - Output audio tensors are explicitly converted to Float32 to ensure compatibility with downstream nodes
  - Resolves "Got unsupported ScalarType BFloat16" errors when using voice cloning or saving audio

### Version 1.0.7
- Added interruption handler to detect user's cancel request
- Bug fixing

### Version 1.0.6
- Fixed a bug that prevented VibeVoice nodes from receiving audio directly from another VibeVoice node

### Version 1.0.5
- Added support for Microsoft's official VibeVoice-Large model (stable release)

### Version 1.0.4
- Improved tokenizer dependency handling

### Version 1.0.3
- Added `attention_type` parameter to both Single Speaker and Multi Speaker nodes for performance optimization
  - auto (default): Automatic selection of best implementation
  - eager: Standard implementation without optimizations
  - sdpa: PyTorch's optimized Scaled Dot Product Attention
  - flash_attention_2: Flash Attention 2 for maximum performance (requires compatible GPU)
- Added `diffusion_steps` parameter to control generation quality vs speed trade-off
  - Default: 20 (VibeVoice default)
  - Higher values: Better quality, longer generation time
  - Lower values: Faster generation, potentially lower quality

### Version 1.0.2
- Added `free_memory_after_generate` toggle to both Single Speaker and Multi Speaker nodes
- New dedicated "Free Memory Node" for manual memory management in workflows
- Improved VRAM/RAM usage optimization
- Enhanced stability for long generation sessions
- Users can now choose between automatic or manual memory management

### Version 1.0.1
- Fixed issue with line breaks in speaker text (both single and multi-speaker nodes)
- Line breaks within individual speaker text are now automatically removed before generation
- Improved text formatting handling for all generation modes

### Version 1.0.0
- Initial release
- Single speaker node with voice cloning
- Multi-speaker node with automatic speaker detection
- Text file loading from ComfyUI directories
- Deterministic and sampling generation modes
- Support for VibeVoice 1.5B and 7B models