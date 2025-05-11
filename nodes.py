import os
import ast
import torch
import tempfile
import torchaudio
import librosa
from transformers import UMT5EncoderModel, AutoTokenizer

from ace_step.pipeline_ace_step import ACEStepPipeline as AP
from ace_step.music_dcae.music_dcae_pipeline import MusicDCAE
from ace_step.ace_models.ace_step_transformer import ACEStepTransformer2DModel

import folder_paths

cache_dir = folder_paths.get_temp_directory()
models_dir = folder_paths.models_dir

# ===================== AudioCacher =====================
class AudioCacher:
    def __init__(self, cache_dir=None, default_format="wav"):
        if cache_dir is None:
            self.cache_dir = tempfile.gettempdir()
        else:
            self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
            except OSError as e:
                raise
        self.default_format = default_format.lstrip('.')
        self._files_to_cleanup_in_context = []

    def cache_audio_tensor(self, audio_tensor: torch.Tensor, sample_rate: int, filename_prefix: str = "cached_audio_", audio_format: str = None) -> str:
        current_format = (audio_format or self.default_format).lstrip('.')
        try:
            with tempfile.NamedTemporaryFile(
                prefix=filename_prefix,
                suffix=f".{current_format}",
                dir=self.cache_dir,
                delete=False
            ) as tmp_file:
                temp_filepath = tmp_file.name
                torchaudio.save(temp_filepath, audio_tensor, sample_rate)
                self._files_to_cleanup_in_context.append(temp_filepath)
                return temp_filepath
        except Exception as e:
            if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except OSError:
                    pass
            raise RuntimeError(f"Failed to save audio: {e}") from e

    def cleanup_file(self, filepath: str) -> bool:
        if not filepath:
            return True
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                if filepath in self._files_to_cleanup_in_context:
                    self._files_to_cleanup_in_context.remove(filepath)
                return True
            except OSError:
                return False
        else:
            if filepath in self._files_to_cleanup_in_context:
                self._files_to_cleanup_in_context.remove(filepath)
            return True

    def cleanup_all_tracked_files(self) -> None:
        for f_path in list(self._files_to_cleanup_in_context):
            self.cleanup_file(f_path)
        self._files_to_cleanup_in_context.clear()

    def __enter__(self):
        self._files_to_cleanup_in_context = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all_tracked_files()
        return False

# ===================== Model Loader Node =====================
class ACEModelLoaderZveroboy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dcae_checkpoint": ("STRING", {"default": "models/ace_step/music_dcae_f8c8"}),
                "vocoder_checkpoint": ("STRING", {"default": "models/ace_step/music_vocoder"}),
                "ace_step_checkpoint": ("STRING", {"default": "models/ace_step/ace_step_transformer"}),
                "text_encoder_checkpoint": ("STRING", {"default": "models/ace_step/umt5-base"}),
            }
        }

    RETURN_TYPES = ("ACE_MODELS",)
    RETURN_NAMES = ("models",)
    FUNCTION = "load"
    CATEGORY = "zveroboy/ACE-Step"

    def load(self, dcae_checkpoint, vocoder_checkpoint, ace_step_checkpoint, text_encoder_checkpoint):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        for path in [dcae_checkpoint, vocoder_checkpoint, ace_step_checkpoint, text_encoder_checkpoint]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint not found: {path}")

        models = (
            MusicDCAE(
                dcae_checkpoint_path=dcae_checkpoint,
                vocoder_checkpoint_path=vocoder_checkpoint
            ),
            ACEStepTransformer2DModel.from_pretrained(ace_step_checkpoint, torch_dtype=dtype),
            UMT5EncoderModel.from_pretrained(text_encoder_checkpoint, torch_dtype=dtype),
            AutoTokenizer.from_pretrained(text_encoder_checkpoint),
            device,
            dtype
        )
        return (models,)

# ===================== Generation Nodes =====================
class ACEStepGenerateZveroboy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": ("ACE_MODELS",),
                "prompt": ("STRING", {"forceInput": True}),
                "lyrics": ("STRING", {"forceInput": True}),
                "parameters": ("STRING", {"forceInput": True}),
                "unload_model": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "ref_audio": ("AUDIO",),
                "ref_audio_strength": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
        }

    CATEGORY = "zveroboy/ACE-Step"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("music",)
    FUNCTION = "acestepgen"

    def acestepgen(self, models, prompt, lyrics, parameters, ref_audio=None, ref_audio_strength=None, unload_model=True):
        music_dcae, ace_step, umt5encoder, text_tokenizer, device, dtype = models
        parameters = ast.literal_eval(parameters)
        ap = AP(music_dcae, ace_step, umt5encoder, text_tokenizer, device=device, dtype=dtype)
        ac = AudioCacher(cache_dir=cache_dir)
        audio2audio_enable = False
        ref_audio_input = None

        if ref_audio is not None:
            ref_audio_path = ac.cache_audio_tensor(ref_audio["waveform"].squeeze(0), ref_audio["sample_rate"], filename_prefix="ref_audio_")
            audio2audio_enable = True
            ref_audio_input = ref_audio_path

        audio_output = ap(
            prompt=prompt,
            lyrics=lyrics,
            task="audio2audio" if audio2audio_enable else "text2audio",
            audio2audio_enable=audio2audio_enable if audio2audio_enable else None,
            ref_audio_strength=ref_audio_strength if audio2audio_enable else None,
            ref_audio_input=ref_audio_input,
            **parameters
        )
        audio, sr = audio_output[0][0].unsqueeze(0), audio_output[0][1]

        if unload_model:
            ap.cleanup()
        if ref_audio is not None:
            ac.cleanup_file(ref_audio_input)
        return ({"waveform": audio, "sample_rate": sr},)

class ACEStepRepaintZveroboy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": ("ACE_MODELS",),
                "src_audio": ("AUDIO",),
                "prompt": ("STRING", {"forceInput": True}),
                "lyrics": ("STRING", {"forceInput": True}),
                "parameters": ("STRING", {"forceInput": True}),
                "repaint_start": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "repaint_end": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "repaint_variance": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295, "step": 1}),
                "unload_model": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "zveroboy/ACE-Step"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("music",)
    FUNCTION = "acesteprepainting"

    def acesteprepainting(self, models, src_audio, prompt, lyrics, parameters, repaint_start, repaint_end, repaint_variance, seed, unload_model=True):
        music_dcae, ace_step, umt5encoder, text_tokenizer, device, dtype = models
        retake_seeds = [str(seed)]
        ac = AudioCacher(cache_dir=cache_dir)
        src_audio_path = ac.cache_audio_tensor(src_audio["waveform"].squeeze(0), src_audio["sample_rate"], filename_prefix="src_audio_")

        audio_duration = librosa.get_duration(filename=src_audio_path)
        if repaint_end > audio_duration:
            repaint_end = audio_duration

        parameters = ast.literal_eval(parameters)
        parameters["audio_duration"] = audio_duration

        ap = AP(music_dcae, ace_step, umt5encoder, text_tokenizer, device=device, dtype=dtype)

        audio_output = ap(
            prompt=prompt,
            lyrics=lyrics,
            task="repaint",
            retake_seeds=retake_seeds,
            src_audio_path=src_audio_path,
            repaint_start=repaint_start,
            repaint_end=repaint_end,
            retake_variance=repaint_variance,
            **parameters
        )

        audio, sr = audio_output[0][0].unsqueeze(0), audio_output[0][1]
        ac.cleanup_file(src_audio_path)
        if unload_model:
            ap.cleanup()
        return ({"waveform": audio, "sample_rate": sr},)

class ACEStepEditZveroboy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": ("ACE_MODELS",),
                "src_audio": ("AUDIO",),
                "prompt": ("STRING", {"forceInput": True}),
                "lyrics": ("STRING", {"forceInput": True}),
                "parameters": ("STRING", {"forceInput": True}),
                "edit_prompt": ("STRING", {"forceInput": True}),
                "edit_lyrics": ("STRING", {"forceInput": True}),
                "edit_n_min": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edit_n_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295, "step": 1}),
                "unload_model": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "zveroboy/ACE-Step"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("music",)
    FUNCTION = "acestepedit"

    def acestepedit(self, models, src_audio, prompt, lyrics, parameters, edit_prompt, edit_lyrics, edit_n_min, edit_n_max, seed, unload_model=True):
        music_dcae, ace_step, umt5encoder, text_tokenizer, device, dtype = models
        retake_seeds = [str(seed)]
        ac = AudioCacher(cache_dir=cache_dir)
        src_audio_path = ac.cache_audio_tensor(src_audio["waveform"].squeeze(0), src_audio["sample_rate"], filename_prefix="src_audio_")

        audio_duration = librosa.get_duration(filename=src_audio_path)
        parameters = ast.literal_eval(parameters)
        parameters["audio_duration"] = audio_duration

        ap = AP(music_dcae, ace_step, umt5encoder, text_tokenizer, device=device, dtype=dtype)

        audio_output = ap(
            prompt=prompt,
            lyrics=lyrics,
            task="edit",
            retake_seeds=retake_seeds,
            src_audio_path=src_audio_path,
            edit_target_prompt=edit_prompt,
            edit_target_lyrics=edit_lyrics,
            edit_n_min=edit_n_min,
            edit_n_max=edit_n_max,
            **parameters
        )

        audio, sr = audio_output[0][0].unsqueeze(0), audio_output[0][1]
        ac.cleanup_file(src_audio_path)
        if unload_model:
            ap.cleanup()
        return ({"waveform": audio, "sample_rate": sr},)

class ACEStepExtendZveroboy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": ("ACE_MODELS",),
                "src_audio": ("AUDIO",),
                "prompt": ("STRING", {"forceInput": True}),
                "lyrics": ("STRING", {"forceInput": True}),
                "parameters": ("STRING", {"forceInput": True}),
                "left_extend_length": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "right_extend_length": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295, "step": 1}),
                "unload_model": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "zveroboy/ACE-Step"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("music",)
    FUNCTION = "acestepextend"

    def acestepextend(self, models, src_audio, prompt, lyrics, parameters, left_extend_length, right_extend_length, seed, unload_model=True):
        music_dcae, ace_step, umt5encoder, text_tokenizer, device, dtype = models
        retake_seeds = [str(seed)]
        ac = AudioCacher(cache_dir=cache_dir)
        src_audio_path = ac.cache_audio_tensor(src_audio["waveform"].squeeze(0), src_audio["sample_rate"], filename_prefix="src_audio_")

        audio_duration = librosa.get_duration(filename=src_audio_path)
        repaint_start = -left_extend_length
        repaint_end = audio_duration + right_extend_length

        parameters = ast.literal_eval(parameters)
        parameters["audio_duration"] = audio_duration

        ap = AP(music_dcae, ace_step, umt5encoder, text_tokenizer, device=device, dtype=dtype)

        audio_output = ap(
            prompt=prompt,
            lyrics=lyrics,
            task="extend",
            retake_seeds=retake_seeds,
            src_audio_path=src_audio_path,
            repaint_start=repaint_start,
            repaint_end=repaint_end,
            retake_variance=1.0,
            **parameters
        )

        audio, sr = audio_output[0][0].unsqueeze(0), audio_output[0][1]
        ac.cleanup_file(src_audio_path)
        if unload_model:
            ap.cleanup()
        return ({"waveform": audio, "sample_rate": sr},)

# ===================== Node Mappings =====================
NODE_CLASS_MAPPINGS = {
    "ACEModelLoaderZveroboy": ACEModelLoaderZveroboy,
    "ACEStepGenerateZveroboy": ACEStepGenerateZveroboy,
    "ACEStepRepaintZveroboy": ACEStepRepaintZveroboy,
    "ACEStepEditZveroboy": ACEStepEditZveroboy,
    "ACEStepExtendZveroboy": ACEStepExtendZveroboy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACEModelLoaderZveroboy": "ACE Model Loader (Zveroboy)",
    "ACEStepGenerateZveroboy": "ACE Generate (Zveroboy)",
    "ACEStepRepaintZveroboy": "ACE Repaint (Zveroboy)",
    "ACEStepEditZveroboy": "ACE Edit (Zveroboy)",
    "ACEStepExtendZveroboy": "ACE Extend (Zveroboy)",
}
