# Created by Fabio Sarracino
# Nodes module for VibeVoiceWrapper
"""
This module contains all the ComfyUI nodes for VibeVoice integration.
"""

from .load_text_node import LoadTextFromFileNode
from .single_speaker_node import VibeVoiceSingleSpeakerNode
from .multi_speaker_node import VibeVoiceMultipleSpeakersNode

__all__ = ['LoadTextFromFileNode', 'VibeVoiceSingleSpeakerNode', 'VibeVoiceMultipleSpeakersNode']