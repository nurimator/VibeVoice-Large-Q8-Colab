# Created by Fabio Sarracino
# Base class for VibeVoice nodes with common functionality

import logging
import os
import tempfile
import torch
import numpy as np
import re
import gc
from typing import List, Optional, Tuple, Any

# Setup logging
logger = logging.getLogger("VibeVoice")

# Import for interruption support
try:
    import execution
    INTERRUPTION_SUPPORT = True
except ImportError:
    INTERRUPTION_SUPPORT = False
    logger.warning("Interruption support not available")

# Check for SageAttention availability
try:
    from sageattention import sageattn
    SAGE_AVAILABLE = True
    logger.info("SageAttention available for acceleration")
except ImportError:
    SAGE_AVAILABLE = False
    logger.debug("SageAttention not available - install with: pip install sageattention")

def get_optimal_device():
    """Get the best available device (cuda, mps, or cpu)"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_device_map():
    """Get device map for model loading"""
    device = get_optimal_device()
    # Note: device_map "auto" might work better for MPS in some cases
    return device if device != "mps" else "mps"

class BaseVibeVoiceNode:
    """Base class for VibeVoice nodes containing common functionality"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_path = None
        self.current_attention_type = None
    
    def free_memory(self):
        """Free model and processor from memory"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            self.current_model_path = None
            
            # Force garbage collection and clear CUDA cache if available
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("Model and processor memory freed successfully")
            
        except Exception as e:
            logger.error(f"Error freeing memory: {e}")
    
    def _check_dependencies(self):
        """Check if VibeVoice is available and import it with fallback installation"""
        try:
            import sys
            import os
            
            # Add vvembed to path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            vvembed_path = os.path.join(parent_dir, 'vvembed')
            
            if vvembed_path not in sys.path:
                sys.path.insert(0, vvembed_path)
            
            # Import from embedded version
            from modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
            
            logger.info(f"Using embedded VibeVoice from {vvembed_path}")
            return None, VibeVoiceForConditionalGenerationInference
            
        except ImportError as e:
            logger.error(f"Embedded VibeVoice import failed: {e}")
            
            # Try fallback to installed version if available
            try:
                import vibevoice
                from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
                logger.warning("Falling back to system-installed VibeVoice")
                return vibevoice, VibeVoiceForConditionalGenerationInference
            except ImportError:
                pass
            
            raise Exception(
                "VibeVoice embedded module import failed. Please ensure the vvembed folder exists "
                "and transformers>=4.51.3 is installed."
            )
    
    def _apply_sage_attention(self):
        """Apply SageAttention to the loaded model by monkey-patching attention layers"""
        try:
            from sageattention import sageattn
            import torch.nn.functional as F
            
            # Counter for patched layers
            patched_count = 0
            
            def patch_attention_forward(module):
                """Recursively patch attention layers to use SageAttention"""
                nonlocal patched_count
                
                # Check if this module has scaled_dot_product_attention
                if hasattr(module, 'forward'):
                    original_forward = module.forward
                    
                    # Create wrapper that replaces F.scaled_dot_product_attention with sageattn
                    def sage_forward(*args, **kwargs):
                        # Temporarily replace F.scaled_dot_product_attention
                        original_sdpa = F.scaled_dot_product_attention
                        
                        def sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
                            """Wrapper that converts sdpa calls to sageattn"""
                            # Log any unexpected parameters for debugging
                            if kwargs:
                                unexpected_params = list(kwargs.keys())
                                logger.debug(f"SageAttention: Ignoring unsupported parameters: {unexpected_params}")
                            
                            try:
                                # SageAttention expects tensors in specific format
                                # Transformers typically use (batch, heads, seq_len, head_dim)
                                
                                # Check tensor dimensions to determine layout
                                if query.dim() == 4:
                                    # 4D tensor: (batch, heads, seq, dim)
                                    batch_size = query.shape[0]
                                    num_heads = query.shape[1]
                                    seq_len_q = query.shape[2]
                                    seq_len_k = key.shape[2]
                                    head_dim = query.shape[3]
                                    
                                    # Reshape to (batch*heads, seq, dim) for HND layout
                                    query_reshaped = query.reshape(batch_size * num_heads, seq_len_q, head_dim)
                                    key_reshaped = key.reshape(batch_size * num_heads, seq_len_k, head_dim)
                                    value_reshaped = value.reshape(batch_size * num_heads, seq_len_k, head_dim)
                                    
                                    # Call sageattn with HND layout
                                    output = sageattn(
                                        query_reshaped, key_reshaped, value_reshaped,
                                        is_causal=is_causal,
                                        tensor_layout="HND"  # Heads*batch, seqN, Dim
                                    )
                                    
                                    # Output should be (batch*heads, seq_len_q, head_dim)
                                    # Reshape back to (batch, heads, seq, dim)
                                    if output.dim() == 3:
                                        output = output.reshape(batch_size, num_heads, seq_len_q, head_dim)
                                    
                                    return output
                                else:
                                    # For 3D tensors, assume they're already in HND format
                                    output = sageattn(
                                        query, key, value,
                                        is_causal=is_causal,
                                        tensor_layout="HND"
                                    )
                                    return output
                                    
                            except Exception as e:
                                # If SageAttention fails, fall back to original implementation
                                logger.debug(f"SageAttention failed, using original: {e}")
                                # Call with proper arguments - scale is a keyword argument in PyTorch 2.0+
                                # Pass through any additional kwargs that the original sdpa might support
                                if scale is not None:
                                    return original_sdpa(query, key, value, attn_mask=attn_mask, 
                                                       dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
                                else:
                                    return original_sdpa(query, key, value, attn_mask=attn_mask, 
                                                       dropout_p=dropout_p, is_causal=is_causal, **kwargs)
                        
                        # Replace the function
                        F.scaled_dot_product_attention = sage_sdpa
                        
                        try:
                            # Call original forward with patched attention
                            result = original_forward(*args, **kwargs)
                        finally:
                            # Restore original function
                            F.scaled_dot_product_attention = original_sdpa
                        
                        return result
                    
                    # Check if this module likely uses attention
                    # Look for common attention module names
                    module_name = module.__class__.__name__.lower()
                    if any(name in module_name for name in ['attention', 'attn', 'multihead']):
                        module.forward = sage_forward
                        patched_count += 1
                
                # Recursively patch child modules
                for child in module.children():
                    patch_attention_forward(child)
            
            # Apply patching to the entire model
            patch_attention_forward(self.model)
            
            logger.info(f"Patched {patched_count} attention layers with SageAttention")
            
            if patched_count == 0:
                logger.warning("No attention layers found to patch - SageAttention may not be applied")
                
        except Exception as e:
            logger.error(f"Failed to apply SageAttention: {e}")
            logger.warning("Continuing with standard attention implementation")
    
    def load_model(self, model_name: str, model_path: str, attention_type: str = "auto"):
        """Load VibeVoice model with specified attention implementation
        
        Args:
            model_name: The display name of the model (e.g., "VibeVoice-Large-Quant-4Bit")
            model_path: The HuggingFace model path
            attention_type: The attention implementation to use
        """
        # Check if we need to reload model due to attention type change
        current_attention = getattr(self, 'current_attention_type', None)
        if (self.model is None or 
            getattr(self, 'current_model_path', None) != model_path or
            current_attention != attention_type):
            try:
                vibevoice, VibeVoiceInferenceModel = self._check_dependencies()
                
                # Set ComfyUI models directory
                import folder_paths
                models_dir = folder_paths.get_folder_paths("checkpoints")[0]
                comfyui_models_dir = os.path.join(os.path.dirname(models_dir), "vibevoice")
                os.makedirs(comfyui_models_dir, exist_ok=True)
                
                # Force HuggingFace to use ComfyUI directory
                original_hf_home = os.environ.get('HF_HOME')
                original_hf_cache = os.environ.get('HUGGINGFACE_HUB_CACHE')
                
                os.environ['HF_HOME'] = comfyui_models_dir
                os.environ['HUGGINGFACE_HUB_CACHE'] = comfyui_models_dir
                
                # Import time for timing
                import time
                start_time = time.time()
                
                # Suppress verbose logs
                import transformers
                import warnings
                transformers.logging.set_verbosity_error()
                warnings.filterwarnings("ignore", category=UserWarning)
                
                # Check if model exists locally
                model_dir = os.path.join(comfyui_models_dir, f"models--{model_path.replace('/', '--')}")
                model_exists_in_comfyui = os.path.exists(model_dir)
                
                # Check if this is a quantized model based on the model name
                is_quantized_4bit = "Quant-4Bit" in model_name
                is_quantized_8bit = "Quant-8Bit" in model_name  # Future support
                
                # Prepare attention implementation kwargs
                model_kwargs = {
                    "cache_dir": comfyui_models_dir,
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16,
                    "device_map": get_device_map(),
                }
                
                # Handle 4-bit quantized model loading
                if is_quantized_4bit:
                    # Check if CUDA is available (required for 4-bit quantization)
                    if not torch.cuda.is_available():
                        raise Exception("4-bit quantized models require a CUDA GPU. Please use standard models on CPU/MPS.")
                    
                    # Try to import bitsandbytes
                    try:
                        from transformers import BitsAndBytesConfig
                        logger.info("Loading 4-bit quantized model with bitsandbytes...")
                        
                        # Configure 4-bit quantization
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type='nf4'
                        )
                        model_kwargs["quantization_config"] = bnb_config
                        model_kwargs["device_map"] = "cuda"  # Force CUDA for 4-bit
                        model_kwargs["subfolder"] = "4bit"  # Point to 4bit subfolder
                        
                    except ImportError:
                        raise Exception(
                            "4-bit quantized models require 'bitsandbytes' library.\n"
                            "Please install it with: pip install bitsandbytes\n"
                            "Or use the standard VibeVoice models instead."
                        )
                
                # Set attention implementation based on user selection
                use_sage_attention = False
                if attention_type == "sage":
                    # SageAttention requires special handling - can't be set via attn_implementation
                    if not SAGE_AVAILABLE:
                        logger.warning("SageAttention not installed, falling back to sdpa")
                        logger.warning("Install with: pip install sageattention")
                        model_kwargs["attn_implementation"] = "sdpa"
                    elif not torch.cuda.is_available():
                        logger.warning("SageAttention requires CUDA GPU, falling back to sdpa")
                        model_kwargs["attn_implementation"] = "sdpa"
                    else:
                        # Don't set attn_implementation for sage, will apply after loading
                        use_sage_attention = True
                        logger.info("Will apply SageAttention after model loading")
                elif attention_type != "auto":
                    model_kwargs["attn_implementation"] = attention_type
                    logger.info(f"Using {attention_type} attention implementation")
                else:
                    # Auto mode - let transformers decide the best implementation
                    logger.info("Using auto attention implementation selection")
                
                # Try to load locally first
                try:
                    if model_exists_in_comfyui:
                        model_kwargs["local_files_only"] = True
                        logger.info(f"Loading model from local cache: {model_path}")
                        if is_quantized_4bit:
                            logger.info(f"Using 4-bit quantization with subfolder: {model_kwargs.get('subfolder', 'None')}")
                        self.model = VibeVoiceInferenceModel.from_pretrained(
                            model_path,
                            **model_kwargs
                        )
                    else:
                        raise FileNotFoundError("Model not found locally")
                except (FileNotFoundError, OSError) as e:
                    logger.info(f"Downloading {model_path}...")
                    if is_quantized_4bit:
                        logger.info(f"Downloading 4-bit quantized model with subfolder: {model_kwargs.get('subfolder', 'None')}")
                    
                    model_kwargs["local_files_only"] = False
                    self.model = VibeVoiceInferenceModel.from_pretrained(
                        model_path,
                        **model_kwargs
                    )
                    elapsed = time.time() - start_time
                else:
                    elapsed = time.time() - start_time
                
                # Verify model was loaded
                if self.model is None:
                    raise Exception("Model failed to load - model is None after loading")
                
                # Load processor with proper error handling
                from processor.vibevoice_processor import VibeVoiceProcessor
                
                # Prepare processor kwargs
                processor_kwargs = {
                    "trust_remote_code": True,
                    "cache_dir": comfyui_models_dir
                }
                
                # Add subfolder for quantized models
                if is_quantized_4bit:
                    processor_kwargs["subfolder"] = "4bit"
                
                try:
                    # First try with local files if model was loaded locally
                    if model_exists_in_comfyui:
                        processor_kwargs["local_files_only"] = True
                        self.processor = VibeVoiceProcessor.from_pretrained(
                            model_path, 
                            **processor_kwargs
                        )
                    else:
                        # Download from HuggingFace
                        self.processor = VibeVoiceProcessor.from_pretrained(
                            model_path,
                            **processor_kwargs
                        )
                except Exception as proc_error:
                    logger.warning(f"Failed to load processor from {model_path}: {proc_error}")
                    
                    # Check if error is about missing Qwen tokenizer
                    if "Qwen" in str(proc_error) and "tokenizer" in str(proc_error).lower():
                        logger.info("Downloading required Qwen tokenizer files...")
                        # The processor needs the Qwen tokenizer, ensure it's available
                        try:
                            from transformers import AutoTokenizer
                            # Pre-download the Qwen tokenizer that VibeVoice depends on
                            _ = AutoTokenizer.from_pretrained(
                                "Qwen/Qwen2.5-1.5B",
                                trust_remote_code=True,
                                cache_dir=comfyui_models_dir
                            )
                            logger.info("Qwen tokenizer downloaded, retrying processor load...")
                        except Exception as tokenizer_error:
                            logger.warning(f"Failed to download Qwen tokenizer: {tokenizer_error}")
                    
                    logger.info("Attempting to load processor with fallback method...")
                    
                    # Fallback: try loading without local_files_only constraint
                    try:
                        self.processor = VibeVoiceProcessor.from_pretrained(
                            model_path,
                            local_files_only=False,
                            trust_remote_code=True,
                            cache_dir=comfyui_models_dir
                        )
                    except Exception as fallback_error:
                        logger.error(f"Processor loading failed completely: {fallback_error}")
                        raise Exception(
                            f"Failed to load VibeVoice processor. Error: {fallback_error}\n"
                            f"This might be due to missing tokenizer files. Try:\n"
                            f"1. Ensure you have internet connection for first-time download\n"
                            f"2. Clear the ComfyUI/models/vibevoice folder and retry\n"
                            f"3. Install transformers: pip install transformers>=4.51.3\n"
                            f"4. Manually download Qwen tokenizer: from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')"
                        )
                
                # Restore environment variables
                if original_hf_home is not None:
                    os.environ['HF_HOME'] = original_hf_home
                elif 'HF_HOME' in os.environ:
                    del os.environ['HF_HOME']
                    
                if original_hf_cache is not None:
                    os.environ['HUGGINGFACE_HUB_CACHE'] = original_hf_cache
                elif 'HUGGINGFACE_HUB_CACHE' in os.environ:
                    del os.environ['HUGGINGFACE_HUB_CACHE']
                
                # Move to appropriate device (skip for quantized models as they use device_map)
                if not is_quantized_4bit and not is_quantized_8bit:
                    device = get_optimal_device()
                    if device == "cuda":
                        self.model = self.model.cuda()
                    elif device == "mps":
                        self.model = self.model.to("mps")
                else:
                    logger.info("Quantized model already mapped to device via device_map")
                
                # Apply SageAttention if requested and available
                if use_sage_attention and SAGE_AVAILABLE:
                    self._apply_sage_attention()
                    logger.info("SageAttention successfully applied to model")
                    
                self.current_model_path = model_path
                self.current_attention_type = attention_type
                
            except Exception as e:
                logger.error(f"Failed to load VibeVoice model: {str(e)}")
                raise Exception(f"Model loading failed: {str(e)}")
    
    def _create_synthetic_voice_sample(self, speaker_idx: int) -> np.ndarray:
        """Create synthetic voice sample for a specific speaker"""
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, samples, False)
        
        # Create realistic voice-like characteristics for each speaker
        # Use different base frequencies for different speaker types
        base_frequencies = [120, 180, 140, 200]  # Mix of male/female-like frequencies
        base_freq = base_frequencies[speaker_idx % len(base_frequencies)]
        
        # Create vowel-like formants (like "ah" sound) - unique per speaker
        formant1 = 800 + speaker_idx * 100  # First formant
        formant2 = 1200 + speaker_idx * 150  # Second formant
        
        # Generate more voice-like waveform
        voice_sample = (
            # Fundamental with harmonics (voice-like)
            0.6 * np.sin(2 * np.pi * base_freq * t) +
            0.25 * np.sin(2 * np.pi * base_freq * 2 * t) +
            0.15 * np.sin(2 * np.pi * base_freq * 3 * t) +
            
            # Formant resonances (vowel-like characteristics)
            0.1 * np.sin(2 * np.pi * formant1 * t) * np.exp(-t * 2) +
            0.05 * np.sin(2 * np.pi * formant2 * t) * np.exp(-t * 3) +
            
            # Natural breath noise (reduced)
            0.02 * np.random.normal(0, 1, len(t))
        )
        
        # Add natural envelope (like human speech pattern)
        # Quick attack, slower decay with slight vibrato (unique per speaker)
        vibrato_freq = 4 + speaker_idx * 0.3  # Slightly different vibrato per speaker
        envelope = (np.exp(-t * 0.3) * (1 + 0.1 * np.sin(2 * np.pi * vibrato_freq * t)))
        voice_sample *= envelope * 0.08  # Lower volume
        
        return voice_sample.astype(np.float32)
    
    def _prepare_audio_from_comfyui(self, voice_audio, target_sample_rate: int = 24000) -> Optional[np.ndarray]:
        """Prepare audio from ComfyUI format to numpy array"""
        if voice_audio is None:
            return None
            
        # Extract waveform from ComfyUI audio format
        if isinstance(voice_audio, dict) and "waveform" in voice_audio:
            waveform = voice_audio["waveform"]
            input_sample_rate = voice_audio.get("sample_rate", target_sample_rate)
            
            # Convert to numpy (handling BFloat16 tensors)
            if isinstance(waveform, torch.Tensor):
                # Convert to float32 first as numpy doesn't support BFloat16
                audio_np = waveform.cpu().float().numpy()
            else:
                audio_np = np.array(waveform)
            
            # Handle different audio shapes
            if audio_np.ndim == 3:  # (batch, channels, samples)
                audio_np = audio_np[0, 0, :]  # Take first batch, first channel
            elif audio_np.ndim == 2:  # (channels, samples)
                audio_np = audio_np[0, :]  # Take first channel
            # If 1D, leave as is
            
            # Resample if needed
            if input_sample_rate != target_sample_rate:
                target_length = int(len(audio_np) * target_sample_rate / input_sample_rate)
                audio_np = np.interp(np.linspace(0, len(audio_np), target_length), 
                                   np.arange(len(audio_np)), audio_np)
            
            # Ensure audio is in correct range [-1, 1]
            audio_max = np.abs(audio_np).max()
            if audio_max > 0:
                audio_np = audio_np / max(audio_max, 1.0)  # Normalize
            
            return audio_np.astype(np.float32)
        
        return None
    
    def _get_model_mapping(self) -> dict:
        """Get model name mappings"""
        return {
            "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
            "VibeVoice-Large": "aoi-ot/VibeVoice-Large",
            "VibeVoice-Large-Quant-4Bit": "DevParker/VibeVoice7b-low-vram"
        }
    
    def _split_text_into_chunks(self, text: str, max_words: int = 250) -> List[str]:
        """Split long text into manageable chunks at sentence boundaries
        
        Args:
            text: The text to split
            max_words: Maximum words per chunk (default 250 for safety)
        
        Returns:
            List of text chunks
        """
        import re
        
        # Split into sentences (handling common abbreviations)
        # This regex tries to split on sentence endings while avoiding common abbreviations
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # If regex split didn't work well, fall back to simple split
        if len(sentences) == 1 and len(text.split()) > max_words:
            # Fall back to splitting on any period followed by space
            sentences = text.replace('. ', '.|').split('|')
            sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            # If single sentence is too long, split it further
            if sentence_word_count > max_words:
                # Split long sentence at commas or semicolons
                sub_parts = re.split(r'[,;]', sentence)
                for part in sub_parts:
                    part = part.strip()
                    if not part:
                        continue
                    part_words = part.split()
                    part_word_count = len(part_words)
                    
                    if current_word_count + part_word_count > max_words and current_chunk:
                        # Save current chunk
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [part]
                        current_word_count = part_word_count
                    else:
                        current_chunk.append(part)
                        current_word_count += part_word_count
            else:
                # Check if adding this sentence would exceed the limit
                if current_word_count + sentence_word_count > max_words and current_chunk:
                    # Save current chunk and start new one
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_word_count = sentence_word_count
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sentence)
                    current_word_count += sentence_word_count
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # If no chunks were created, return the original text
        if not chunks:
            chunks = [text]
        
        logger.info(f"Split text into {len(chunks)} chunks (max {max_words} words each)")
        for i, chunk in enumerate(chunks):
            word_count = len(chunk.split())
            logger.debug(f"Chunk {i+1}: {word_count} words")
        
        return chunks
    
    def _format_text_for_vibevoice(self, text: str, speakers: list) -> str:
        """Format text with speaker information for VibeVoice using correct format"""
        # Remove any newlines from the text to prevent parsing issues
        # The processor splits by newline and expects each line to have "Speaker N:" format
        text = text.replace('\n', ' ').replace('\r', ' ')
        # Clean up multiple spaces
        text = ' '.join(text.split())
        
        # VibeVoice expects format: "Speaker 1: text" not "Name: text"
        if len(speakers) == 1:
            return f"Speaker 1: {text}"
        else:
            # Check if text already has proper Speaker N: format
            if re.match(r'^\s*Speaker\s+\d+\s*:', text, re.IGNORECASE):
                return text
            # If text has name format, convert to Speaker N format
            elif any(f"{speaker}:" in text for speaker in speakers):
                formatted_text = text
                for i, speaker in enumerate(speakers):
                    formatted_text = formatted_text.replace(f"{speaker}:", f"Speaker {i+1}:")
                return formatted_text
            else:
                # Plain text, assign to first speaker
                return f"Speaker 1: {text}"
    
    def _generate_with_vibevoice(self, formatted_text: str, voice_samples: List[np.ndarray], 
                                cfg_scale: float, seed: int, diffusion_steps: int, use_sampling: bool,
                                temperature: float = 0.95, top_p: float = 0.95) -> dict:
        """Generate audio using VibeVoice model"""
        try:
            # Ensure model and processor are loaded
            if self.model is None or self.processor is None:
                raise Exception("Model or processor not loaded")
            
            # Set seeds for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # For multi-GPU
            
            # Also set numpy seed for any numpy operations
            np.random.seed(seed)
            
            # Set diffusion steps
            self.model.set_ddpm_inference_steps(diffusion_steps)
            logger.info(f"Starting audio generation with {diffusion_steps} diffusion steps...")
            
            # Check for interruption before starting generation
            if INTERRUPTION_SUPPORT:
                try:
                    import comfy.model_management as mm
                    
                    # Check if we're being interrupted right now
                    # The interrupt flag is reset by ComfyUI before each node execution
                    # So we only check model_management's throw_exception_if_processing_interrupted
                    # which is the proper way to check for interruption
                    mm.throw_exception_if_processing_interrupted()
                    
                except ImportError:
                    # If comfy.model_management is not available, skip this check
                    pass
            
            # Prepare inputs using processor
            inputs = self.processor(
                [formatted_text],  # Wrap text in list
                voice_samples=[voice_samples], # Provide voice samples for reference
                return_tensors="pt",
                return_attention_mask=True
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Estimate tokens for user information (not used as limit)
            text_length = len(formatted_text.split())
            estimated_tokens = int(text_length * 2.5)  # More accurate estimate for display
            
            # Log generation start with explanation
            logger.info(f"Generating audio with {diffusion_steps} diffusion steps...")
            logger.info(f"Note: Progress bar shows max possible tokens, not actual needed (~{estimated_tokens} estimated)")
            logger.info("The generation will stop automatically when audio is complete")
            
            # Create stop check function for interruption support
            stop_check_fn = None
            if INTERRUPTION_SUPPORT:
                def check_comfyui_interrupt():
                    """Check if ComfyUI has requested interruption"""
                    try:
                        if hasattr(execution, 'PromptExecutor') and hasattr(execution.PromptExecutor, 'interrupted'):
                            interrupted = execution.PromptExecutor.interrupted
                            if interrupted:
                                logger.info("Generation interrupted by user via stop_check_fn")
                            return interrupted
                    except:
                        pass
                    return False
                
                stop_check_fn = check_comfyui_interrupt
            
            # Generate with official parameters
            with torch.no_grad():
                if use_sampling:
                    # Use sampling mode (less stable but more varied)
                    output = self.model.generate(
                        **inputs,
                        tokenizer=self.processor.tokenizer,
                        cfg_scale=cfg_scale,
                        max_new_tokens=None,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        stop_check_fn=stop_check_fn,
                    )
                else:
                    # Use deterministic mode like official examples
                    output = self.model.generate(
                        **inputs,
                        tokenizer=self.processor.tokenizer,
                        cfg_scale=cfg_scale,
                        max_new_tokens=None,
                        do_sample=False,  # More deterministic generation
                        stop_check_fn=stop_check_fn,
                    )
                
                # Check if we got actual audio output
                if hasattr(output, 'speech_outputs') and output.speech_outputs:
                    speech_tensors = output.speech_outputs
                    
                    if isinstance(speech_tensors, list) and len(speech_tensors) > 0:
                        audio_tensor = torch.cat(speech_tensors, dim=-1)
                    else:
                        audio_tensor = speech_tensors
                    
                    # Ensure proper format (1, 1, samples)
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
                    elif audio_tensor.dim() == 2:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    
                    # Convert to float32 for compatibility with downstream nodes (Save Audio, etc.)
                    # Many audio processing nodes don't support BFloat16
                    return {
                        "waveform": audio_tensor.cpu().float(),
                        "sample_rate": 24000
                    }
                    
                elif hasattr(output, 'sequences'):
                    logger.error("VibeVoice returned only text tokens, no audio generated")
                    raise Exception("VibeVoice failed to generate audio - only text tokens returned")
                    
                else:
                    logger.error(f"Unexpected output format from VibeVoice: {type(output)}")
                    raise Exception(f"VibeVoice returned unexpected output format: {type(output)}")
                
        except Exception as e:
            # Re-raise interruption exceptions without wrapping
            import comfy.model_management as mm
            if isinstance(e, mm.InterruptProcessingException):
                raise  # Let the interruption propagate
            
            # For real errors, log and re-raise with context
            logger.error(f"VibeVoice generation failed: {e}")
            raise Exception(f"VibeVoice generation failed: {str(e)}")