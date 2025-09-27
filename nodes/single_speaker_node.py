# Created by Fabio Sarracino

import logging
import os
import tempfile
import torch
import numpy as np
import re
from typing import List, Optional

from .base_vibevoice import BaseVibeVoiceNode, get_available_models

# Setup logging
logger = logging.getLogger("VibeVoice")

class VibeVoiceSingleSpeakerNode(BaseVibeVoiceNode):
    def __init__(self):
        super().__init__()
        # Register this instance for memory management
        try:
            from .free_memory_node import VibeVoiceFreeMemoryNode
            VibeVoiceFreeMemoryNode.register_single_speaker(self)
        except:
            pass
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available models dynamically
        available_models = get_available_models()
        model_choices = [display_name for _, display_name in available_models]
        default_model = model_choices[0] if model_choices else "No models found"

        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is a test of the VibeVoice text-to-speech system.",
                    "tooltip": "Text to convert to speech. Gets disabled when connected to another node.",
                    "forceInput": False,
                    "dynamicPrompts": True
                }),
                "model": (model_choices if model_choices else ["No models found"], {
                    "default": default_model,
                    "tooltip": "Select a model from ComfyUI/models/vibevoice/ folder"
                }),
                "attention_type": (["auto", "eager", "sdpa", "flash_attention_2", "sage"], {
                    "default": "auto",
                    "tooltip": "Attention implementation. Auto selects the best available, eager is standard, sdpa is optimized PyTorch, flash_attention_2 requires compatible GPU, sage uses quantized attention for speedup (CUDA only)"
                }),
                "free_memory_after_generate": ("BOOLEAN", {"default": True, "tooltip": "Free model from memory after generation to save VRAM/RAM. Disable to keep model loaded for faster subsequent generations"}),
                "diffusion_steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1, "tooltip": "Number of denoising steps. More steps = theoretically better quality but slower. Default: 20"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32-1, "tooltip": "Random seed for generation. Default 42 is used in official examples"}),
                "cfg_scale": ("FLOAT", {"default": 1.3, "min": 0.5, "max": 3.5, "step": 0.05, "tooltip": "Classifier-free guidance scale (official default: 1.3)"}),
                "use_sampling": ("BOOLEAN", {"default": False, "tooltip": "Enable sampling mode. When False (default), uses deterministic generation like official examples"}),
            },
            "optional": {
                "voice_to_clone": ("AUDIO", {"tooltip": "Optional: Reference voice to clone. If not provided, synthetic voice will be used."}),
                "lora": ("LORA_CONFIG", {"tooltip": "Optional: LoRA configuration from VibeVoice LoRA node"}),
                "temperature": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Only used when sampling is enabled"}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Only used when sampling is enabled"}),
                "max_words_per_chunk": ("INT", {"default": 250, "min": 100, "max": 500, "step": 50, "tooltip": "Maximum words per chunk for long texts. Lower values prevent speed issues but create more chunks."}),
                "voice_speed_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.8,
                    "max": 1.2,
                    "step": 0.01,
                    "tooltip": "1.0 = normal speed, <1.0 = slower speed, >1.0 = faster speed"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "VibeVoiceWrapper"
    DESCRIPTION = "Generate speech from text using Microsoft VibeVoice with optional voice cloning"

    def _prepare_voice_samples(self, speakers: list, voice_to_clone, voice_speed_factor: float = 1.0) -> List[np.ndarray]:
        """Prepare voice samples from input audio or create synthetic ones"""

        if voice_to_clone is not None:
            # Use the base class method to prepare audio with speed adjustment
            audio_np = self._prepare_audio_from_comfyui(voice_to_clone, speed_factor=voice_speed_factor)
            if audio_np is not None:
                return [audio_np]
        
        # Create synthetic voice samples for speakers
        voice_samples = []
        for i, speaker in enumerate(speakers):
            voice_sample = self._create_synthetic_voice_sample(i)
            voice_samples.append(voice_sample)
            
        return voice_samples
    
    def generate_speech(self, text: str = "", model: str = "VibeVoice-1.5B",
                       attention_type: str = "auto", free_memory_after_generate: bool = True,
                       diffusion_steps: int = 20, seed: int = 42, cfg_scale: float = 1.3,
                       use_sampling: bool = False, voice_to_clone=None, lora=None,
                       temperature: float = 0.95, top_p: float = 0.95,
                       max_words_per_chunk: int = 250, voice_speed_factor: float = 1.0):
        """Generate speech from text using VibeVoice"""
        
        try:
            # Use text directly (it now serves as both manual input and connection input)
            if text and text.strip():
                final_text = text
            else:
                raise Exception("No text provided. Please enter text or connect from LoadTextFromFile node.")
            
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
            self.load_model(model, model_path, attention_type, lora_path=lora_path)
            
            # For single speaker, we just use ["Speaker 1"]
            speakers = ["Speaker 1"]
            
            # Parse pause keywords from text
            segments = self._parse_pause_keywords(final_text)
            
            # Process segments
            all_audio_segments = []
            voice_samples = None  # Will be created on first text segment
            sample_rate = 24000  # VibeVoice uses 24kHz
            
            for seg_idx, (seg_type, seg_content) in enumerate(segments):
                if seg_type == 'pause':
                    # Generate silence for pause
                    duration_ms = seg_content
                    logger.info(f"Adding {duration_ms}ms pause")
                    silence_audio = self._generate_silence(duration_ms, sample_rate)
                    all_audio_segments.append(silence_audio)
                    
                elif seg_type == 'text':
                    # Process text segment (with chunking if needed)
                    word_count = len(seg_content.split())
                    
                    if word_count > max_words_per_chunk:
                        # Split long text into chunks
                        logger.info(f"Text segment {seg_idx+1} has {word_count} words, splitting into chunks...")
                        text_chunks = self._split_text_into_chunks(seg_content, max_words_per_chunk)
                        
                        for chunk_idx, chunk in enumerate(text_chunks):
                            logger.info(f"Processing chunk {chunk_idx+1}/{len(text_chunks)} of segment {seg_idx+1}...")
                            
                            # Format chunk for VibeVoice
                            formatted_text = self._format_text_for_vibevoice(chunk, speakers)
                            
                            # Create voice samples on first text segment
                            if voice_samples is None:
                                voice_samples = self._prepare_voice_samples(speakers, voice_to_clone, voice_speed_factor)
                            
                            # Generate audio for this chunk
                            chunk_audio = self._generate_with_vibevoice(
                                formatted_text, voice_samples, cfg_scale,
                                seed,  # Use same seed for voice consistency
                                diffusion_steps, use_sampling, temperature, top_p,
                                llm_lora_strength=llm_lora_strength
                            )
                            
                            all_audio_segments.append(chunk_audio)
                    else:
                        # Process as single chunk
                        logger.info(f"Processing text segment {seg_idx+1} ({word_count} words)")
                        
                        # Format text for VibeVoice
                        formatted_text = self._format_text_for_vibevoice(seg_content, speakers)
                        
                        # Create voice samples on first text segment
                        if voice_samples is None:
                            voice_samples = self._prepare_voice_samples(speakers, voice_to_clone, voice_speed_factor)
                        
                        # Generate audio
                        segment_audio = self._generate_with_vibevoice(
                            formatted_text, voice_samples, cfg_scale, seed, diffusion_steps,
                            use_sampling, temperature, top_p, llm_lora_strength=llm_lora_strength
                        )
                        
                        all_audio_segments.append(segment_audio)
            
            # Concatenate all audio segments (including pauses)
            if all_audio_segments:
                logger.info(f"Concatenating {len(all_audio_segments)} audio segments (including pauses)...")
                
                # Extract waveforms from all segments
                waveforms = []
                for audio_segment in all_audio_segments:
                    if isinstance(audio_segment, dict) and "waveform" in audio_segment:
                        waveforms.append(audio_segment["waveform"])
                
                if waveforms:
                    # Filter out None values if any
                    valid_waveforms = [w for w in waveforms if w is not None]
                    
                    if valid_waveforms:
                        # Concatenate along the time dimension (last dimension)
                        combined_waveform = torch.cat(valid_waveforms, dim=-1)
                        
                        # Create final audio dict
                        audio_dict = {
                            "waveform": combined_waveform,
                            "sample_rate": sample_rate
                        }
                        logger.info(f"Successfully generated audio with {len(segments)} segments")
                    else:
                        raise Exception("No valid audio waveforms generated")
                else:
                    raise Exception("Failed to extract waveforms from audio segments")
            else:
                raise Exception("No audio segments generated")
            
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
                logger.error(f"Single speaker speech generation failed: {str(e)}")
                raise Exception(f"Error generating speech: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, text="", model="VibeVoice-1.5B", voice_to_clone=None, lora=None, **kwargs):
        """Cache key for ComfyUI"""
        voice_hash = hash(str(voice_to_clone)) if voice_to_clone else 0
        lora_hash = hash(str(lora)) if lora else 0
        return f"{hash(text)}_{model}_{voice_hash}_{lora_hash}_{kwargs.get('cfg_scale', 1.3)}_{kwargs.get('seed', 0)}"