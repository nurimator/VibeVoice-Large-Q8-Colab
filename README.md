# VibeVoice ComfyUI Nodes

A comprehensive ComfyUI integration for Microsoft's VibeVoice text-to-speech model, enabling high-quality single and multi-speaker voice synthesis directly within your ComfyUI workflows.

## Features

### Core Functionality
- 🎤 **Single Speaker TTS**: Generate natural speech with optional voice cloning
- 👥 **Multi-Speaker Conversations**: Support for up to 4 distinct speakers
- 🎯 **Voice Cloning**: Clone voices from audio samples
- 🎨 **LoRA Support**: Fine-tune voices with custom LoRA adapters (v1.4.0+)
- 📝 **Text File Loading**: Load scripts from text files
- 📚 **Automatic Text Chunking**: Handles long texts seamlessly with configurable chunk size
- ⏸️ **Custom Pause Tags**: Insert silences with `[pause]` and `[pause:ms]` tags (wrapper feature)
- 🔄 **Node Chaining**: Connect multiple VibeVoice nodes for complex workflows
- ⏹️ **Interruption Support**: Cancel operations before or between generations

### Model Options
- 🚀 **Three Model Variants**: 
  - VibeVoice 1.5B (faster, lower memory)
  - VibeVoice-Large (best quality, ~17GB VRAM)
  - VibeVoice-Large-Quant-4Bit (balanced, ~7GB VRAM)
- 🔧 **Flexible Configuration**: Control temperature, sampling, and guidance scale

### Performance & Optimization
- ⚡ **Attention Mechanisms**: Choose between auto, eager, sdpa, flash_attention_2 or sage
- 🎛️ **Diffusion Steps**: Adjustable quality vs speed trade-off (default: 20)
- 💾 **Memory Management**: Toggle automatic VRAM cleanup after generation
- 🧹 **Free Memory Node**: Manual memory control for complex workflows
- 🍎 **Apple Silicon Support**: Native GPU acceleration on M1/M2/M3 Macs via MPS
- 🔢 **4-Bit Quantization**: Reduced memory usage with minimal quality loss

### Compatibility & Installation
- 📦 **Self-Contained**: Embedded VibeVoice code, no external dependencies
- 🔄 **Universal Compatibility**: Adaptive support for transformers v4.51.3+
- 🖥️ **Cross-Platform**: Works on Windows, Linux, and macOS
- 🎮 **Multi-Backend**: Supports CUDA, CPU, and MPS (Apple Silicon)

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
- **Models**: VibeVoice-1.5B or VibeVoice-Large
- **Voice Cloning**: Optional audio input for voice cloning
- **Parameters** (in order):
  - `text`: Input text to convert to speech
  - `model`: VibeVoice-1.5B, VibeVoice-Large or VibeVoice-Large-Quant-4Bit
  - `attention_type`: auto, eager, sdpa, flash_attention_2 or sage (default: auto)
  - `free_memory_after_generate`: Free VRAM after generation (default: True)
  - `diffusion_steps`: Number of denoising steps (5-100, default: 20)
  - `seed`: Random seed for reproducibility (default: 42)
  - `cfg_scale`: Classifier-free guidance (1.0-2.0, default: 1.3)
  - `use_sampling`: Enable/disable deterministic generation (default: False)
- **Optional Parameters**:
  - `voice_to_clone`: Audio input for voice cloning
  - `lora`: LoRA configuration from VibeVoice LoRA node
  - `temperature`: Sampling temperature (0.1-2.0, default: 0.95)
  - `top_p`: Nucleus sampling parameter (0.1-1.0, default: 0.95)
  - `max_words_per_chunk`: Maximum words per chunk for long texts (100-500, default: 250)

### 3. VibeVoice Multiple Speakers
Generates multi-speaker conversations with distinct voices.
- **Speaker Format**: Use `[N]:` notation where N is 1-4
- **Voice Assignment**: Optional voice samples for each speaker
- **Recommended Model**: VibeVoice-Large for better multi-speaker quality
- **Parameters** (in order):
  - `text`: Input text with speaker labels
  - `model`: VibeVoice-1.5B, VibeVoice-Large or VibeVoice-Large-Quant-4Bit
  - `attention_type`: auto, eager, sdpa, flash_attention_2 or sage (default: auto)
  - `free_memory_after_generate`: Free VRAM after generation (default: True)
  - `diffusion_steps`: Number of denoising steps (5-100, default: 20)
  - `seed`: Random seed for reproducibility (default: 42)
  - `cfg_scale`: Classifier-free guidance (1.0-2.0, default: 1.3)
  - `use_sampling`: Enable/disable deterministic generation (default: False)
- **Optional Parameters**:
  - `speaker1_voice` to `speaker4_voice`: Audio inputs for voice cloning
  - `lora`: LoRA configuration from VibeVoice LoRA node
  - `temperature`: Sampling temperature (0.1-2.0, default: 0.95)
  - `top_p`: Nucleus sampling parameter (0.1-1.0, default: 0.95)

### 4. VibeVoice Free Memory
Manually frees all loaded VibeVoice models from memory.
- **Input**: `audio` - Connect audio output to trigger memory cleanup
- **Output**: `audio` - Passes through the input audio unchanged
- **Use Case**: Insert between nodes to free VRAM/RAM at specific workflow points
- **Example**: `[VibeVoice Node] → [Free Memory] → [Save Audio]`

### 5. VibeVoice LoRA
Configure and load custom LoRA adapters for fine-tuned VibeVoice models.
- **LoRA Selection**: Dropdown menu with available LoRA adapters
- **LoRA Location**: Place your LoRA folders in `ComfyUI/models/vibevoice/loras/`
- **Parameters**:
  - `lora_name`: Select from available LoRA adapters or "None" to disable
  - `llm_strength`: Strength of the language model LoRA (0.0-2.0, default: 1.0)
  - `use_llm`: Apply language model LoRA component (default: True)
  - `use_diffusion_head`: Apply diffusion head replacement (default: True)
  - `use_acoustic_connector`: Apply acoustic connector LoRA (default: True)
  - `use_semantic_connector`: Apply semantic connector LoRA (default: True)
- **Output**: `lora` - LoRA configuration to connect to speaker nodes
- **Usage**: `[VibeVoice LoRA] → [Single/Multiple Speaker Node]`

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

### VibeVoice-Large
- **Size**: ~17GB download
- **Speed**: Slower inference but optimized
- **Quality**: Best available quality
- **Use Case**: Highest quality production, multi-speaker conversations
- **Note**: Latest official release from Microsoft

### VibeVoice-Large-Quant-4Bit
- **Size**: ~7GB download
- **Speed**: Balanced inference
- **Quality**: Good quality
- **Use Case**: Good quality production with less VRAM, multi-speaker conversations
- **Note**: Quantized by DevParker

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

## LoRA Support

### Overview
Starting from version 1.4.0, VibeVoice ComfyUI supports custom LoRA (Low-Rank Adaptation) adapters for fine-tuning voice characteristics. This allows you to train and use specialized voice models while maintaining the base VibeVoice capabilities.

### Setting Up LoRA Adapters

1. **LoRA Directory Structure**:
   Place your LoRA adapter folders in: `ComfyUI/models/vibevoice/loras/`
   ```
   ComfyUI/
   └── models/
       └── vibevoice/
           └── loras/
               ├── my_custom_voice/
               │   ├── adapter_config.json
               │   ├── adapter_model.safetensors
               │   └── diffusion_head/  (optional)
               ├── character_voice/
               └── style_adaptation/
   ```

2. **Required Files**:
   - `adapter_config.json`: LoRA configuration
   - `adapter_model.safetensors` or `adapter_model.bin`: Model weights
   - Optional components:
     - `diffusion_head/`: Custom diffusion head weights
     - `acoustic_connector/`: Acoustic connector adaptation
     - `semantic_connector/`: Semantic connector adaptation

### Using LoRA in ComfyUI

1. **Add VibeVoice LoRA Node**:
   - Create a "VibeVoice LoRA" node in your workflow
   - Select your LoRA from the dropdown menu
   - Configure component settings and strength

2. **Connect to Speaker Nodes**:
   - Connect the LoRA node's output to the `lora` input of speaker nodes
   - Both Single Speaker and Multiple Speakers nodes support LoRA

3. **LoRA Parameters**:
   - **llm_strength**: Controls the influence of the language model LoRA (0.0-2.0)
   - **Component toggles**: Enable/disable specific LoRA components
   - Select "None" to disable LoRA and use the base model

### Training Your Own LoRA

To create custom LoRA adapters for VibeVoice, use the official fine-tuning repository:
- **Repository**: [VibeVoice Fine-tuning](https://github.com/voicepowered-ai/VibeVoice-finetuning)
- **Features**:
  - Parameter-efficient fine-tuning
  - Support for custom datasets
  - Adjustable LoRA rank and scaling
  - Optional diffusion head adaptation

### Best Practices

- **Voice Consistency**: Use the same LoRA across all chunks for long texts
- **Memory Management**: LoRA adds minimal memory overhead (~100-500MB)
- **Compatibility**: LoRA adapters are compatible with all VibeVoice model variants
- **Strength Tuning**: Start with default strength (1.0) and adjust based on results

### Compatibility Note

⚠️ **Transformers Version**: The LoRA implementation was developed and tested with `transformers==4.51.3`. While our wrapper supports `transformers>=4.51.3`, LoRA functionality with newer versions of transformers is not guaranteed. If you experience issues with LoRA loading, consider using `transformers==4.51.3` specifically:
```bash
pip install transformers==4.51.3
```

### Credits

LoRA implementation by [@jpgallegoar](https://github.com/jpgallegoar) (PR #127)

## Pause Tags Support

### Overview
The VibeVoice wrapper includes a custom pause tag feature that allows you to insert silences between text segments. **This is NOT a standard Microsoft VibeVoice feature** - it's an original implementation of our wrapper to provide more control over speech pacing.

**Available from version 1.3.0**

### Usage
You can use two types of pause tags in your text:
- `[pause]` - Inserts a 1-second silence (default)
- `[pause:ms]` - Inserts a custom duration silence in milliseconds (e.g., `[pause:2000]` for 2 seconds)

### Examples

#### Single Speaker
```
Welcome to our presentation. [pause] Today we'll explore artificial intelligence. [pause:500] Let's begin!
```

#### Multi-Speaker  
```
[1]: Hello everyone [pause] how are you doing today?
[2]: I'm doing great! [pause:500] Thanks for asking.
[1]: Wonderful to hear!
```

### Important Notes

⚠️ **Context Limitation Warning**:
> **Note: The pause forces the text to be split into chunks. This may worsen the model's ability to understand the context. The model's context is represented ONLY by its own chunk.**

This means:
- Text before a pause and text after a pause are processed separately
- The model cannot see across pause boundaries when generating speech
- This may affect prosody and intonation consistency
- This may affect prosody and intonation consistency

### How It Works
1. The wrapper parses your text to find pause tags
2. Text segments between pauses are processed independently 
3. Silence audio is generated for each pause duration
4. All audio segments (speech and silence) are concatenated

### Best Practices
- Use pauses at natural breaking points (end of sentences, paragraphs)
- Avoid pauses in the middle of phrases where context is important
- Test different pause durations to find what sounds most natural

## Tips for Best Results

1. **Text Preparation**:
   - Use proper punctuation for natural pauses
   - Break long texts into paragraphs
   - For multi-speaker, ensure clear speaker transitions
   - Use pause tags sparingly to maintain context continuity

2. **Model Selection**:
   - Use 1.5B for quick single-speaker tasks (fastest, ~8GB VRAM)
   - Use Large for best quality and multi-speaker (~16GB VRAM)
   - Use Large-Quant-4Bit for good quality and low VRAM usage (~7GB VRAM)

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
- **Recommended**: 17GB+ VRAM for VibeVoice-Large
- **RAM**: 16GB+ system memory

### Software
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- Transformers 4.51.3+
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
- Large model requires ~16GB VRAM
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
| VibeVoice-Large | ~17GB | 32K tokens | ~45 minutes |
| VibeVoice-Large-Quant-4Bit | ~7GB | 32K tokens | ~45 minutes |

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

- [Original VibeVoice Repository](https://github.com/microsoft/VibeVoice) - Official Microsoft VibeVoice repository (currently unavailable)

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

### Version 1.4.1
- Fixed HuggingFace authentication error when loading locally cached models
  - Resolved 401 authorization errors for already downloaded models
  - Node now correctly uses local model snapshots without requiring HuggingFace API authentication
  - Prevents unnecessary API calls when models exist in `ComfyUI/models/vibevoice/`

### Version 1.4.0
- Added LoRA (Low-Rank Adaptation) support for fine-tuned models
  - New "VibeVoice LoRA" node for configuring custom voice adaptations
  - Support for language model, diffusion head, and connector adaptations
  - Dropdown menu for easy LoRA selection from `ComfyUI/models/vibevoice/loras/`
  - Adjustable LoRA strength and component toggles
  - Compatible with both Single and Multiple Speaker nodes
  - Minimal memory overhead (~100-500MB per LoRA)
  - Credits: Implementation by [@jpgallegoar](https://github.com/jpgallegoar)

### Version 1.3.0
- Added custom pause tag support for speech pacing control
  - New `[pause]` tag for 1-second silence (default)
  - New `[pause:ms]` tag for custom duration in milliseconds (e.g., `[pause:2000]` for 2 seconds)
  - Works with both Single Speaker and Multiple Speakers nodes
  - Automatically splits text at pause points while maintaining voice consistency
  - Note: This is a wrapper feature, not part of Microsoft's VibeVoice

### Version 1.2.5
- Bug Fixing

### Version 1.2.4
- Added automatic text chunking for long texts in Single Speaker node
  - Single Speaker node now automatically splits texts longer than 250 words to prevent audio acceleration issues
  - New optional parameter `max_words_per_chunk` (range: 100-500 words, default: 250)
  - Maintains consistent voice characteristics across all chunks using the same seed
  - Seamlessly concatenates audio chunks for smooth, natural output

### Version 1.2.3
- Added SageAttention support for inference speedup
  - New attention option "sage" using quantized attention (INT8/FP8) for faster generation
  - Requirements: NVIDIA GPU with CUDA and sageattention library installation

### Version 1.2.2
- Added 4-bit quantized model support
  - New model in menu: `VibeVoice-Large-Quant-4Bit` using ~7GB VRAM instead of ~17GB
  - Requirements: NVIDIA GPU with CUDA and bitsandbytes library installed

### Version 1.2.1
- Bug Fixing

### Version 1.2.0
- MPS Support for Apple Silicon:
  - Added GPU acceleration support for Mac with Apple Silicon (M1/M2/M3)
  - Automatically detects and uses MPS backend when available, providing significant performance improvements over CPU

### Version 1.1.1
- Universal Transformers Compatibility:
  - Implemented adaptive system that automatically adjusts to different transformers versions
  - Guaranteed compatibility from v4.51.3 onwards
  - Auto-detects and adapts to API changes between versions

### Version 1.1.0
- Updated the URL for downloading the VibeVoice-Large model
- Removed VibeVoice-Large-Preview deprecated model

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
- Support for VibeVoice 1.5B and Large models