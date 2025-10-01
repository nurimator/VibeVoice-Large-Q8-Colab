# Created by Fabio Sarracino
# Original LoRa code implementation by jpgallegoar-vpai user via PR #127 
# LoRA configuration node for VibeVoice

import logging
import os
from typing import Dict, Any, List

# Setup logging
logger = logging.getLogger("VibeVoice")

# Cache for LoRA scanning to avoid repeated logs
_lora_cache = {
    "first_load_logged": False
}

def get_available_loras() -> List[str]:
    """Get list of available LoRA folders in ComfyUI/models/vibevoice/loras"""
    try:
        import folder_paths

        # Get the ComfyUI models directory
        models_dir = folder_paths.get_folder_paths("checkpoints")[0]
        # Navigate to vibevoice/loras directory
        loras_dir = os.path.join(os.path.dirname(models_dir), "vibevoice", "loras")

        # Create directory if it doesn't exist
        os.makedirs(loras_dir, exist_ok=True)

        # List all directories in the loras folder
        lora_folders = []
        if os.path.exists(loras_dir):
            for item in os.listdir(loras_dir):
                item_path = os.path.join(loras_dir, item)
                if os.path.isdir(item_path):
                    # Check if it contains LoRA files
                    adapter_config = os.path.join(item_path, "adapter_config.json")
                    adapter_model_st = os.path.join(item_path, "adapter_model.safetensors")
                    adapter_model_bin = os.path.join(item_path, "adapter_model.bin")

                    # Consider it a valid LoRA if it has config or model files
                    if os.path.exists(adapter_config) or os.path.exists(adapter_model_st) or os.path.exists(adapter_model_bin):
                        lora_folders.append(item)

        # Only log on first scan to avoid spam
        if not _lora_cache["first_load_logged"]:
            if not lora_folders:
                logger.info("No LoRA adapters found in ComfyUI/models/vibevoice/loras")
            _lora_cache["first_load_logged"] = True

        # Always include "None" option to disable LoRA
        if not lora_folders:
            return ["None"]

        # Sort alphabetically and add None option at the beginning
        lora_folders.sort()
        return ["None"] + lora_folders

    except Exception as e:
        logger.error(f"Error listing LoRA folders: {e}")
        return ["None"]

class VibeVoiceLoRANode:
    """Node for configuring LoRA adapters for VibeVoice models"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # Get available LoRA folders dynamically
        available_loras = get_available_loras()

        return {
            "required": {
                "lora_name": (available_loras, {
                    "default": "None",
                    "tooltip": "Select a LoRA adapter from ComfyUI/models/vibevoice/loras folder"
                }),
                "llm_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Strength of the LLM LoRA adapter. Controls how much the LoRA affects the language model"
                }),
                "use_llm": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply LLM (language model) LoRA component when available"
                }),
                "use_diffusion_head": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply diffusion head LoRA/replacement when available"
                }),
                "use_acoustic_connector": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply acoustic connector LoRA component when available"
                }),
                "use_semantic_connector": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply semantic connector LoRA component when available"
                }),
            }
        }

    RETURN_TYPES = ("LORA_CONFIG",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "configure_lora"
    CATEGORY = "VibeVoiceWrapper"
    DESCRIPTION = "Configure LoRA adapters for fine-tuned VibeVoice models. Place LoRA folders in ComfyUI/models/vibevoice/loras/"

    def configure_lora(self, lora_name: str = "None", llm_strength: float = 1.0,
                      use_llm: bool = True, use_diffusion_head: bool = True,
                      use_acoustic_connector: bool = True, use_semantic_connector: bool = True):
        """Configure LoRA settings and validate the path"""

        # Handle "None" selection
        if lora_name == "None":
            logger.info("No LoRA selected, using base model")
            return ({
                "path": None,
                "llm_strength": llm_strength,
                "use_llm": use_llm,
                "use_diffusion_head": use_diffusion_head,
                "use_acoustic_connector": use_acoustic_connector,
                "use_semantic_connector": use_semantic_connector
            },)

        try:
            import folder_paths

            # Build full path to the LoRA folder
            models_dir = folder_paths.get_folder_paths("checkpoints")[0]
            loras_dir = os.path.join(os.path.dirname(models_dir), "vibevoice", "loras")
            lora_path = os.path.join(loras_dir, lora_name)

            # Validate the path exists
            if not os.path.exists(lora_path):
                logger.error(f"LoRA path does not exist: {lora_path}")
                raise Exception(f"LoRA folder not found: {lora_name}")

            if not os.path.isdir(lora_path):
                logger.error(f"LoRA path is not a directory: {lora_path}")
                raise Exception(f"LoRA path must be a directory: {lora_name}")

            # Check for required files
            adapter_config = os.path.join(lora_path, "adapter_config.json")
            adapter_model_st = os.path.join(lora_path, "adapter_model.safetensors")
            adapter_model_bin = os.path.join(lora_path, "adapter_model.bin")

            if not os.path.exists(adapter_config):
                logger.warning(f"adapter_config.json not found in {lora_name}")

            if not os.path.exists(adapter_model_st) and not os.path.exists(adapter_model_bin):
                logger.warning(f"No adapter model file found in {lora_name}")
                logger.warning("Expected: adapter_model.safetensors or adapter_model.bin")

            logger.info(f"LoRA configured: {lora_name} ({lora_path})")

            # Check for optional components
            components_found = []
            diffusion_head_path = os.path.join(lora_path, "diffusion_head")
            acoustic_connector_path = os.path.join(lora_path, "acoustic_connector")
            semantic_connector_path = os.path.join(lora_path, "semantic_connector")

            if os.path.exists(diffusion_head_path):
                components_found.append("diffusion_head")
            if os.path.exists(acoustic_connector_path):
                components_found.append("acoustic_connector")
            if os.path.exists(semantic_connector_path):
                components_found.append("semantic_connector")

            if components_found:
                logger.info(f"Additional LoRA components found: {', '.join(components_found)}")

            # Create configuration dictionary
            lora_config = {
                "path": lora_path,
                "llm_strength": llm_strength,
                "use_llm": use_llm,
                "use_diffusion_head": use_diffusion_head,
                "use_acoustic_connector": use_acoustic_connector,
                "use_semantic_connector": use_semantic_connector
            }

            # Log configuration
            enabled_components = []
            if use_llm:
                enabled_components.append(f"LLM (strength: {llm_strength})")
            if use_diffusion_head:
                enabled_components.append("Diffusion Head")
            if use_acoustic_connector:
                enabled_components.append("Acoustic Connector")
            if use_semantic_connector:
                enabled_components.append("Semantic Connector")

            if enabled_components:
                logger.info(f"LoRA components enabled: {', '.join(enabled_components)}")
            else:
                logger.warning("All LoRA components are disabled")

            return (lora_config,)

        except ImportError:
            logger.error("Could not import folder_paths from ComfyUI")
            raise Exception("Failed to access ComfyUI folders")
        except Exception as e:
            logger.error(f"Error configuring LoRA: {e}")
            raise

    @classmethod
    def IS_CHANGED(cls, lora_name: str = "None", **kwargs):
        """Cache key for ComfyUI - includes all parameters"""
        return f"{lora_name}_{kwargs.get('llm_strength', 1.0)}_{kwargs.get('use_llm', True)}_{kwargs.get('use_diffusion_head', True)}_{kwargs.get('use_acoustic_connector', True)}_{kwargs.get('use_semantic_connector', True)}"