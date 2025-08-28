# Created by Fabio Sarracino

import logging
import os
import re
import tempfile
import torch
import numpy as np
from typing import List, Optional

from .base_vibevoice import BaseVibeVoiceNode

# Setup logging
logger = logging.getLogger("VibeVoice")

class VibeVoiceMultipleSpeakersNode(BaseVibeVoiceNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "[1]: Hello, this is the first speaker.\n[2]: Hi there, I'm the second speaker.\n[1]: Nice to meet you!\n[2]: Nice to meet you too!", 
                    "tooltip": "Text with speaker labels. Use '[N]:' format where N is 1-4. Gets disabled when connected to another node.",
                    "forceInput": False,
                    "dynamicPrompts": True
                }),
                "model": (["VibeVoice-1.5B", "VibeVoice-7B-Preview"], {
                    "default": "VibeVoice-7B-Preview",  # 7B recommended for multi-speaker
                    "tooltip": "Model to use. VibeVoice-7B is recommended for multi-speaker generation"
                }),
                "cfg_scale": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Classifier-free guidance scale (official default: 1.3)"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32-1, "tooltip": "Random seed for generation. Default 42 is used in official examples"}),
                "use_sampling": ("BOOLEAN", {"default": False, "tooltip": "Enable sampling mode. When False (default), uses deterministic generation like official examples"}),
            },
            "optional": {
                "speaker1_voice": ("AUDIO", {"tooltip": "Optional: Voice sample for Speaker 1. If not provided, synthetic voice will be used."}),
                "speaker2_voice": ("AUDIO", {"tooltip": "Optional: Voice sample for Speaker 2. If not provided, synthetic voice will be used."}),
                "speaker3_voice": ("AUDIO", {"tooltip": "Optional: Voice sample for Speaker 3. If not provided, synthetic voice will be used."}),
                "speaker4_voice": ("AUDIO", {"tooltip": "Optional: Voice sample for Speaker 4. If not provided, synthetic voice will be used."}),
                "temperature": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Only used when sampling is enabled"}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Only used when sampling is enabled"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "VibeVoiceWrapper"
    DESCRIPTION = "Generate multi-speaker conversations with up to 4 distinct voices using Microsoft VibeVoice"

    def _prepare_voice_sample(self, voice_audio, speaker_idx: int) -> Optional[np.ndarray]:
        """Prepare a single voice sample from input audio"""
        return self._prepare_audio_from_comfyui(voice_audio)
    
    def generate_speech(self, text: str = "", model: str = "VibeVoice-7B-Preview",
                       speaker1_voice=None, speaker2_voice=None, speaker3_voice=None, speaker4_voice=None,
                       cfg_scale: float = 1.3, seed: int = 42, use_sampling: bool = False,
                       temperature: float = 0.95, top_p: float = 0.95):
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
                    converted_text = f"Speaker 0: {text}"
            else:
                # Convert [N]: directly to Speaker (N-1):
                for speaker_num in sorted(speakers_in_text, reverse=True):
                    pattern = f'\\[{speaker_num}\\]\\s*:'
                    replacement = f'Speaker {speaker_num - 1}:'
                    converted_text = re.sub(pattern, replacement, converted_text)
            
            # Build speaker names list - these are just for logging, not used by processor
            # The processor uses the speaker labels in the text itself
            speakers = [f"Speaker {i}" for i in range(len(speakers_in_text))]
            
            # Get model mapping and load model
            model_mapping = self._get_model_mapping()
            model_path = model_mapping.get(model, model)
            self.load_model(model_path)
            
            voice_inputs = [speaker1_voice, speaker2_voice, speaker3_voice, speaker4_voice]
            
            # Prepare voice samples in order of appearance
            voice_samples = []
            for i, speaker_num in enumerate(speakers_in_text):
                idx = speaker_num - 1  # Convert to 0-based for voice array
                
                # Try to use provided voice sample
                if idx < len(voice_inputs) and voice_inputs[idx] is not None:
                    voice_sample = self._prepare_voice_sample(voice_inputs[idx], idx)
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
            
            # Generate audio with converted text (0-based speaker indexing)
            audio_dict = self._generate_with_vibevoice(
                converted_text, voice_samples, cfg_scale, seed, use_sampling, temperature, top_p
            )
            
            return (audio_dict,)
                    
        except Exception as e:
            logger.error(f"Multi-speaker speech generation failed: {str(e)}")
            raise Exception(f"Error generating multi-speaker speech: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, text="", model="VibeVoice-7B-Preview",
                   speaker1_voice=None, speaker2_voice=None, 
                   speaker3_voice=None, speaker4_voice=None, **kwargs):
        """Cache key for ComfyUI"""
        voices_hash = hash(str([speaker1_voice, speaker2_voice, speaker3_voice, speaker4_voice]))
        return f"{hash(text)}_{model}_{voices_hash}_{kwargs.get('cfg_scale', 1.3)}_{kwargs.get('seed', 0)}"