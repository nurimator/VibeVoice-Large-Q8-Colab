# VibeVoice ComfyUI Nodes

A comprehensive ComfyUI integration for Microsoft's VibeVoice text-to-speech model, enabling high-quality single and multi-speaker voice synthesis directly within your ComfyUI workflows.

## Features

- 🎤 **Single Speaker TTS**: Generate natural speech with optional voice cloning
- 👥 **Multi-Speaker Conversations**: Support for up to 4 distinct speakers
- 🎯 **Voice Cloning**: Clone voices from audio samples
- 📝 **Text File Loading**: Load scripts from text files
- 🔧 **Flexible Configuration**: Control temperature, sampling, and guidance scale
- 🚀 **Two Model Options**: 1.5B (faster) and 7B (higher quality)

## Installation

### Automatic Installation (Recommended)
1. Clone this repository into your ComfyUI custom nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
```

2. Restart ComfyUI - the nodes will automatically install VibeVoice on first use

### Manual Installation
If automatic installation fails:
```bash
cd ComfyUI
python_embeded/python.exe -m pip install git+https://github.com/microsoft/VibeVoice.git
```

## Available Nodes

### 1. Load Text From File
Loads text content from files in ComfyUI's input/output/temp directories.
- **Supported formats**: .txt
- **Output**: Text string for TTS nodes

### 2. VibeVoice Single Speaker
Generates speech from text using a single voice.
- **Text Input**: Direct text or connection from Load Text node
- **Models**: VibeVoice-1.5B or VibeVoice-7B-Preview
- **Voice Cloning**: Optional audio input for voice cloning
- **Parameters**:
  - `cfg_scale`: Classifier-free guidance (1.0-3.0, default: 1.3)
  - `seed`: Random seed for reproducibility (default: 42)
  - `use_sampling`: Enable/disable deterministic generation
  - `temperature`: Sampling temperature (0.1-2.0)
  - `top_p`: Nucleus sampling parameter (0.1-1.0)

### 3. VibeVoice Multiple Speakers
Generates multi-speaker conversations with distinct voices.
- **Speaker Format**: Use `[N]:` notation where N is 1-4
- **Voice Assignment**: Optional voice samples for each speaker
- **Recommended Model**: VibeVoice-7B-Preview for better multi-speaker quality

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

### VibeVoice-7B-Preview
- **Size**: ~17GB download
- **Speed**: Slower inference
- **Quality**: Superior, especially for multi-speaker
- **Use Case**: Production quality, multi-speaker conversations

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
- Preferably 3-10 seconds of speech
- Automatically resampled to 24kHz

## Tips for Best Results

1. **Text Preparation**:
   - Use proper punctuation for natural pauses
   - Break long texts into paragraphs
   - For multi-speaker, ensure clear speaker transitions

2. **Model Selection**:
   - Use 1.5B for quick single-speaker tasks
   - Use 7B for multi-speaker or when quality is priority

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

| Model | VRAM Usage | Context Length | Max Audio Duration |
|-------|------------|----------------|-------------------|
| VibeVoice-1.5B | ~8GB | 64K tokens | ~90 minutes |
| VibeVoice-7B | ~16GB | 32K tokens | ~45 minutes |

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

### Version 1.0.0
- Initial release
- Single speaker node with voice cloning
- Multi-speaker node with automatic speaker detection
- Text file loading from ComfyUI directories
- Deterministic and sampling generation modes
- Support for VibeVoice 1.5B and 7B models
