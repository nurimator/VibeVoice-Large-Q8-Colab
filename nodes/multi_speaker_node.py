# Created by Fabio Sarracino

import logging
import os
import re
import tempfile
import torch
import numpy as np
from typing import List, Optional

from .base_vibevoice import BaseVibeVoiceNode, get_available_models

# Setup logging
logger = logging.getLogger("VibeVoice")

class VibeVoiceMultipleSpeakersNode(BaseVibeVoiceNode):
    def __init__(self):
        super().__init__()
        # Register this instance for memory management
        try:
            from .free_memory_node import VibeVoiceFreeMemoryNode
            VibeVoiceFreeMemoryNode.register_multi_speaker(self)
        except:
            pass
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available models dynamically
        available_models = get_available_models()
        model_choices = [display_name for _, display_name in available_models]
        # Try to select Large model by default if available
        default_model = "VibeVoice-Large"
        if default_model not in model_choices:
            default_model = model_choices[0] if model_choices else "No models found"

        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "[1]: Hello, this is the first speaker.\n[2]: Hi there, I'm the second speaker.\n[1]: Nice to meet you!\n[2]: Nice to meet you too!",
                    "tooltip": "Text with speaker labels. Use '[N]:' format where N is 1-4. Gets disabled when connected to another node.",
                    "forceInput": False,
                    "dynamicPrompts": True
                }),
                "model": (model_choices if model_choices else ["No models found"], {
                    "default": default_model,
                    "tooltip": "Select a model from ComfyUI/models/vibevoice/ folder. Large is recommended for multi-speaker"
                }),
                "attention_type": (["auto", "eager", "sdpa", "flash_attention_2", "sage"], {
                    "default": "auto",
                    "tooltip": "Attention implementation. Auto selects the best available, eager is standard, sdpa is optimized PyTorch, flash_attention_2 requires compatible GPU, sage uses quantized attention for speedup (CUDA only)"
                }),
                "quantize_llm": (["full precision", "4bit"], {
                    "default": "full precision",
                    "tooltip": "Dynamically quantize only the LLM component for non-quantized models. 4bit: significantly faster generation with major VRAM savings and minimal quality loss. Full precision: original quality. Note: ignored for pre-quantized models. Requires CUDA GPU."
                }),
                "free_memory_after_generate": ("BOOLEAN", {"default": True, "tooltip": "Free model from memory after generation to save VRAM/RAM. Disable to keep model loaded for faster subsequent generations"}),
                "diffusion_steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1, "tooltip": "Number of denoising steps. More steps = theoretically better quality but slower. Default: 20"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32-1, "tooltip": "Random seed for generation. Default 42 is used in official examples"}),
                "cfg_scale": ("FLOAT", {"default": 1.3, "min": 0.5, "max": 3.5, "step": 0.05, "tooltip": "Classifier-free guidance scale (official default: 1.3)"}),
                "use_sampling": ("BOOLEAN", {"default": False, "tooltip": "Enable sampling mode. When False (default), uses deterministic generation like official examples"}),
            },
            "optional": {
                "speaker1_voice": ("AUDIO", {"tooltip": "Optional: Voice sample for Speaker 1. If not provided, synthetic voice will be used."}),
                "speaker2_voice": ("AUDIO", {"tooltip": "Optional: Voice sample for Speaker 2. If not provided, synthetic voice will be used."}),
                "speaker3_voice": ("AUDIO", {"tooltip": "Optional: Voice sample for Speaker 3. If not provided, synthetic voice will be used."}),
                "speaker4_voice": ("AUDIO", {"tooltip": "Optional: Voice sample for Speaker 4. If not provided, synthetic voice will be used."}),
                "lora": ("LORA_CONFIG", {"tooltip": "Optional: LoRA configuration from VibeVoice LoRA node"}),
                "temperature": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Only used when sampling is enabled"}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Only used when sampling is enabled"}),
                "voice_speed_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.8,
                    "max": 1.2,
                    "step": 0.01,
                    "tooltip": "1.0 = normal speed, <1.0 = slower speed, >1.0 = faster speed (applies to all speakers)"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "VibeVoiceWrapper"
    DESCRIPTION = "Generate multi-speaker conversations with up to 4 distinct voices using Microsoft VibeVoice"

    def _prepare_voice_sample(self, voice_audio, speaker_idx: int, voice_speed_factor: float = 1.0) -> Optional[np.ndarray]:
        """Prepare a single voice sample from input audio with speed adjustment"""
        return self._prepare_audio_from_comfyui(voice_audio, speed_factor=voice_speed_factor)
    
    def generate_speech(self, text: str = "", model: str = "VibeVoice-7B-Preview",
                       attention_type: str = "auto", quantize_llm: str = "full precision",
                       free_memory_after_generate: bool = True,
                       diffusion_steps: int = 20, seed: int = 42, cfg_scale: float = 1.3,
                       use_sampling: bool = False, lora=None,
                       speaker1_voice=None, speaker2_voice=None,
                       speaker3_voice=None, speaker4_voice=None,
                       temperature: float = 0.95, top_p: float = 0.95,
                       voice_speed_factor: float = 1.0):
        """Generate multi-speaker speech from text using VibeVoice"""
        
        try:
            # Check text input
            if not text or not text.strip():
                raise Exception("No text provided. Please enter text with speaker labels (e.g., '[1]: Hello' or '[2]: Hi')")
            
            # First detect how many speakers are in the text
            bracket_pattern = r'\[(\d+)\]\s*:'
            speakers_numbers = sorted(list(set([int(m) for m in re.findall(bracket_pattern, text)])))
            
            # Limit to 1-4 speakers
            if not speakers_numbers:
                num_speakers = 1  # Default to 1 if no speaker format found
            else:
                num_speakers = min(max(speakers_numbers), 4)  # Max speaker number, capped at 4
                if max(speakers_numbers) > 4:
                    print(f"[VibeVoice] Warning: Found {max(speakers_numbers)} speakers, limiting to 4")
            
            # Direct conversion from [N]: to Speaker (N-1): for VibeVoice processor
            # This avoids multiple conversion steps
            converted_text = text
            
            # Find all [N]: patterns in the text
            speakers_in_text = sorted(list(set([int(m) for m in re.findall(bracket_pattern, text)])))
            
            if not speakers_in_text:
                # No [N]: format found, try Speaker N: format
                speaker_pattern = r'Speaker\s+(\d+)\s*:'
                speakers_in_text = sorted(list(set([int(m) for m in re.findall(speaker_pattern, text)])))
                
                if speakers_in_text:
                    # Text already in Speaker N format, convert to 0-based
                    for speaker_num in sorted(speakers_in_text, reverse=True):
                        pattern = f'Speaker\\s+{speaker_num}\\s*:'
                        replacement = f'Speaker {speaker_num - 1}:'
                        converted_text = re.sub(pattern, replacement, converted_text)
                else:
                    # No speaker format found
                    speakers_in_text = [1]
                    
                    # Parse pause keywords even for single speaker
                    pause_segments = self._parse_pause_keywords(text)
                    
                    # Store speaker segments for pause processing
                    speaker_segments_with_pauses = []
                    segments = []
                    
                    for seg_type, seg_content in pause_segments:
                        if seg_type == 'pause':
                            speaker_segments_with_pauses.append(('pause', seg_content, None))
                        else:
                            # Clean up newlines
                            text_clean = seg_content.replace('\n', ' ').replace('\r', ' ')
                            text_clean = ' '.join(text_clean.split())
                            
                            if text_clean:
                                speaker_segments_with_pauses.append(('text', text_clean, 1))
                                segments.append(f"Speaker 0: {text_clean}")
                    
                    # Join all segments for fallback
                    converted_text = '\n'.join(segments) if segments else f"Speaker 0: {text}"
            else:
                # Convert [N]: directly to Speaker (N-1): and handle multi-line text
                # Split text to preserve speaker segments while cleaning up newlines within each segment
                segments = []
                
                # Find all speaker markers with their positions
                speaker_matches = list(re.finditer(f'\\[({"|".join(map(str, speakers_in_text))})\\]\\s*:', converted_text))
                
                # Store speaker segments for pause processing
                speaker_segments_with_pauses = []
                
                for i, match in enumerate(speaker_matches):
                    speaker_num = int(match.group(1))
                    start = match.end()
                    
                    # Find where this speaker's text ends (at next speaker or end of text)
                    if i + 1 < len(speaker_matches):
                        end = speaker_matches[i + 1].start()
                    else:
                        end = len(converted_text)
                    
                    # Extract the speaker's text (keep pause keywords for now)
                    speaker_text = converted_text[start:end].strip()
                    
                    # Parse pause keywords within this speaker's text
                    pause_segments = self._parse_pause_keywords(speaker_text)
                    
                    # Process each segment (text or pause) for this speaker
                    for seg_type, seg_content in pause_segments:
                        if seg_type == 'pause':
                            # Add pause segment
                            speaker_segments_with_pauses.append(('pause', seg_content, None))
                        else:
                            # Clean up the text segment
                            text_clean = seg_content.replace('\n', ' ').replace('\r', ' ')
                            text_clean = ' '.join(text_clean.split())
                            
                            if text_clean:  # Only add non-empty text
                                # Add text segment with speaker info
                                speaker_segments_with_pauses.append(('text', text_clean, speaker_num))
                                # Also build the traditional segments for fallback
                                segments.append(f'Speaker {speaker_num - 1}: {text_clean}')
                
                # Join all segments with newlines (required for multi-speaker format) - for fallback
                converted_text = '\n'.join(segments) if segments else ""
            
            # Build speaker names list - these are just for logging, not used by processor
            # The processor uses the speaker labels in the text itself
            speakers = [f"Speaker {i}" for i in range(len(speakers_in_text))]
            
            # Get the actual folder path for the selected model
            available_models = get_available_models()
            model_path = None
            for folder, display_name in available_models:
                if display_name == model:
                    model_path = folder
                    break

            if not model_path:
                raise Exception(f"Model '{model}' not found in models/vibevoice/")

            # Extract LoRA configuration if provided
            lora_path = None
            llm_lora_strength = 1.0
            if lora and isinstance(lora, dict):
                lora_path = lora.get("path", None)
                llm_lora_strength = lora.get("llm_strength", 1.0)

                # Set LoRA component flags based on configuration
                self.use_llm_lora = lora.get("use_llm", True)
                self.use_diffusion_head_lora = lora.get("use_diffusion_head", True)
                self.use_acoustic_connector_lora = lora.get("use_acoustic_connector", True)
                self.use_semantic_connector_lora = lora.get("use_semantic_connector", True)

                if lora_path:
                    logger.info(f"Using LoRA from: {lora_path}")

            # Load model with optional LoRA
            self.load_model(model, model_path, attention_type, quantize_llm=quantize_llm, lora_path=lora_path)
            
            voice_inputs = [speaker1_voice, speaker2_voice, speaker3_voice, speaker4_voice]
            
            # Prepare voice samples in order of appearance
            voice_samples = []
            for i, speaker_num in enumerate(speakers_in_text):
                idx = speaker_num - 1  # Convert to 0-based for voice array
                
                # Try to use provided voice sample
                if idx < len(voice_inputs) and voice_inputs[idx] is not None:
                    voice_sample = self._prepare_voice_sample(voice_inputs[idx], idx, voice_speed_factor)
                    if voice_sample is None:
                        # Use the actual speaker index for consistent synthetic voice
                        voice_sample = self._create_synthetic_voice_sample(idx)
                else:
                    # Use the actual speaker index for consistent synthetic voice
                    voice_sample = self._create_synthetic_voice_sample(idx)
                    
                voice_samples.append(voice_sample)
            
            # Ensure voice_samples count matches detected speakers
            if len(voice_samples) != len(speakers_in_text):
                logger.error(f"Mismatch: {len(speakers_in_text)} speakers but {len(voice_samples)} voice samples!")
                raise Exception(f"Voice sample count mismatch: expected {len(speakers_in_text)}, got {len(voice_samples)}")
            
            # Check if we have pause segments to process
            if 'speaker_segments_with_pauses' in locals() and speaker_segments_with_pauses:
                # Process segments with pauses
                all_audio_segments = []
                sample_rate = 24000  # VibeVoice uses 24kHz
                
                # Group consecutive text segments from same speaker for efficiency
                grouped_segments = []
                current_group = []
                current_speaker = None
                
                for seg_type, seg_content, speaker_num in speaker_segments_with_pauses:
                    if seg_type == 'pause':
                        # Save current group if any
                        if current_group:
                            grouped_segments.append(('text_group', current_group, current_speaker))
                            current_group = []
                            current_speaker = None
                        # Add pause
                        grouped_segments.append(('pause', seg_content, None))
                    else:
                        # Text segment
                        if speaker_num == current_speaker:
                            # Same speaker, add to current group
                            current_group.append(seg_content)
                        else:
                            # Different speaker, save current group and start new one
                            if current_group:
                                grouped_segments.append(('text_group', current_group, current_speaker))
                            current_group = [seg_content]
                            current_speaker = speaker_num
                
                # Save last group if any
                if current_group:
                    grouped_segments.append(('text_group', current_group, current_speaker))
                
                # Process grouped segments
                for seg_type, seg_content, speaker_num in grouped_segments:
                    if seg_type == 'pause':
                        # Generate silence
                        duration_ms = seg_content
                        logger.info(f"Adding {duration_ms}ms pause")
                        silence_audio = self._generate_silence(duration_ms, sample_rate)
                        all_audio_segments.append(silence_audio)
                    else:
                        # Process text group for a speaker
                        combined_text = ' '.join(seg_content)
                        formatted_text = f"Speaker {speaker_num - 1}: {combined_text}"
                        
                        # Get voice sample for this speaker
                        speaker_idx = speakers_in_text.index(speaker_num)
                        speaker_voice_samples = [voice_samples[speaker_idx]]
                        
                        logger.info(f"Generating audio for Speaker {speaker_num}: {len(combined_text.split())} words")
                        
                        # Generate audio for this speaker's text
                        segment_audio = self._generate_with_vibevoice(
                            formatted_text, speaker_voice_samples, cfg_scale, seed,
                            diffusion_steps, use_sampling, temperature, top_p,
                            llm_lora_strength=llm_lora_strength
                        )
                        
                        all_audio_segments.append(segment_audio)
                
                # Concatenate all audio segments
                if all_audio_segments:
                    logger.info(f"Concatenating {len(all_audio_segments)} audio segments (including pauses)...")
                    
                    # Extract waveforms
                    waveforms = []
                    for audio_segment in all_audio_segments:
                        if isinstance(audio_segment, dict) and "waveform" in audio_segment:
                            waveforms.append(audio_segment["waveform"])
                    
                    if waveforms:
                        # Filter out None values if any
                        valid_waveforms = [w for w in waveforms if w is not None]
                        
                        if valid_waveforms:
                            # Concatenate along time dimension
                            combined_waveform = torch.cat(valid_waveforms, dim=-1)
                            
                            audio_dict = {
                                "waveform": combined_waveform,
                                "sample_rate": sample_rate
                            }
                            logger.info(f"Successfully generated multi-speaker audio with pauses")
                        else:
                            raise Exception("No valid audio waveforms generated")
                    else:
                        raise Exception("Failed to extract waveforms from audio segments")
                else:
                    raise Exception("No audio segments generated")
            else:
                # Fallback to original method without pause support
                logger.info("Processing without pause support (no pause keywords found)")
                audio_dict = self._generate_with_vibevoice(
                    converted_text, voice_samples, cfg_scale, seed, diffusion_steps,
                    use_sampling, temperature, top_p, llm_lora_strength=llm_lora_strength
                )
            
            # Free memory if requested
            if free_memory_after_generate:
                self.free_memory()
            
            return (audio_dict,)
                    
        except Exception as e:
            # Check if this is an interruption by the user
            import comfy.model_management as mm
            if isinstance(e, mm.InterruptProcessingException):
                # User interrupted - just log it and re-raise to stop the workflow
                logger.info("Generation interrupted by user")
                raise  # Propagate the interruption to stop the workflow
            else:
                # Real error - show it
                logger.error(f"Multi-speaker speech generation failed: {str(e)}")
                raise Exception(f"Error generating multi-speaker speech: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, text="", model="VibeVoice-7B-Preview",
                   speaker1_voice=None, speaker2_voice=None,
                   speaker3_voice=None, speaker4_voice=None, lora=None, **kwargs):
        """Cache key for ComfyUI"""
        voices_hash = hash(str([speaker1_voice, speaker2_voice, speaker3_voice, speaker4_voice]))
        lora_hash = hash(str(lora)) if lora else 0
        return f"{hash(text)}_{model}_{voices_hash}_{lora_hash}_{kwargs.get('cfg_scale', 1.3)}_{kwargs.get('seed', 0)}"