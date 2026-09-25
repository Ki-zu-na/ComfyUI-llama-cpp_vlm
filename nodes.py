import os
import io
import gc
import ctypes
import inspect
import json
import re
import base64
import random
import wave
import torch

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from .support.cqdm import cqdm
from .support.gguf_layers import get_layer_count
from .support.prompt_enhancer_preset import *

import folder_paths
import comfy.model_management as mm
import comfy.utils

import llama_cpp
from llama_cpp import Llama

MIN_LLAMA_CPP_VERSION = "0.3.40"
MIN_QWEN38_LLAMA_CPP_VERSION = "0.3.46"
MIN_TEXT_ENCODER_LLAMA_CPP_VERSION = "0.3.46"

def parse_version_tuple(version):
    parts = re.split(r"[^\d]+", str(version).split("+", 1)[0])
    return tuple(int(part) for part in parts if part != "")

def llama_cpp_version_at_least(version):
    current = parse_version_tuple(getattr(llama_cpp, "__version__", "0"))
    required = parse_version_tuple(version)
    return current >= required

try:
    from llama_cpp.llama_chat_format import (
        Llava15ChatHandler, Llava16ChatHandler, MoondreamChatHandler,
        NanoLlavaChatHandler, Llama3VisionAlphaChatHandler,
        MiniCPMv26ChatHandler,
    )
except ImportError as e:
    raise RuntimeError(
        f"[llama-cpp_vlm] llama-cpp-python (JamePeng fork) >= {MIN_LLAMA_CPP_VERSION} is required, "
        f"but version {getattr(llama_cpp, '__version__', 'unknown')} is installed.\n"
        "Please update it from 'https://github.com/JamePeng/llama-cpp-python/releases'\n"
        f"Missing symbol: {e}"
    ) from e

def _optional_chat_handler(name):
    try:
        module = __import__("llama_cpp.llama_chat_format", fromlist=[name])
        return getattr(module, name)
    except (ImportError, AttributeError):
        return None

def _optional_chat_handler_any(*names):
    for name in names:
        handler = _optional_chat_handler(name)
        if handler is not None:
            return handler
    return None

MTMDChatHandler = _optional_chat_handler("MTMDChatHandler")
MiniCPMv45ChatHandler = _optional_chat_handler("MiniCPMv45ChatHandler")
MiniCPMv46ChatHandler = _optional_chat_handler_any("MiniCPMv46ChatHandler", "MiniCPMV46ChatHandler")
Gemma3ChatHandler = _optional_chat_handler("Gemma3ChatHandler")
Gemma4ChatHandler = _optional_chat_handler("Gemma4ChatHandler")
Qwen25VLChatHandler = _optional_chat_handler("Qwen25VLChatHandler")
Qwen3VLChatHandler = _optional_chat_handler("Qwen3VLChatHandler")
Qwen35ChatHandler = _optional_chat_handler("Qwen35ChatHandler")
Qwen3ASRChatHandler = _optional_chat_handler("Qwen3ASRChatHandler")
GLM46VChatHandler = _optional_chat_handler("GLM46VChatHandler")
GLM41VChatHandler = _optional_chat_handler("GLM41VChatHandler")
Step3VLChatHandler = _optional_chat_handler("Step3VLChatHandler")
LFM2VLChatHandler = _optional_chat_handler("LFM2VLChatHandler")
LFM25VLChatHandler = _optional_chat_handler("LFM25VLChatHandler")
PaddleOCRChatHandler = _optional_chat_handler("PaddleOCRChatHandler")
GraniteDoclingChatHandler = _optional_chat_handler("GraniteDoclingChatHandler")

_MTMD = MTMDChatHandler is not None

chat_handlers = [
    "None",
    "LLaVA-1.5", "LLaVA-1.6", "Moondream2", "nanoLLaVA", "llama3-Vision-Alpha",
    "MiniCPM-v2.6",
]
if MiniCPMv45ChatHandler is not None:
    chat_handlers.append("MiniCPM-v4.5")
if MiniCPMv46ChatHandler is not None:
    chat_handlers.append("MiniCPM-v4.6")
if Gemma3ChatHandler is not None:
    chat_handlers.append("Gemma3")
if Gemma4ChatHandler is not None:
    chat_handlers.append("Gemma4")
if Qwen25VLChatHandler is not None:
    chat_handlers += ["Qwen2.5-VL", "MinerU2.5-Pro"]
if Qwen3VLChatHandler is not None:
    chat_handlers.append("Qwen3-VL")
if Qwen35ChatHandler is not None:
    chat_handlers += ["Qwen3.5", "Qwen3.6"]
    if llama_cpp_version_at_least(MIN_QWEN38_LLAMA_CPP_VERSION):
        chat_handlers.append("Qwen3.8")
if Qwen3ASRChatHandler is not None:
    chat_handlers.append("Qwen3-ASR")
if GLM46VChatHandler is not None:
    chat_handlers.append("GLM-4.6V")
if GLM41VChatHandler is not None:
    chat_handlers.append("GLM-4.1V")
if Step3VLChatHandler is not None:
    chat_handlers.append("Step3-VL")
if LFM2VLChatHandler is not None:
    chat_handlers.append("LFM2-VL")
if LFM25VLChatHandler is not None:
    chat_handlers.append("LFM2.5-VL")
if PaddleOCRChatHandler is not None:
    chat_handlers.append("PaddleOCR-VL-1.5")
if GraniteDoclingChatHandler is not None:
    chat_handlers.append("Granite-Docling")
if MTMDChatHandler is not None:
    chat_handlers.append("DeepSeek-OCR")

# __init__ 接受 enable_thinking 开关的 handler（Qwen3-VL 用 force_reasoning，单独处理）
ENABLE_THINKING_HANDLERS = [
    "MiniCPM-v4.5", "MiniCPM-v4.6",
    "GLM-4.6V", "Qwen3.5", "Qwen3.6", "Qwen3.8",
    "Gemma4", "Step3-VL",
]

# 不受 enable_thinking 控制、模型自身始终输出思考块的 handler
ALWAYS_THINKING_HANDLERS = ["GLM-4.1V"]

def normalize_chat_handler(chat_handler, enable_thinking=False):
    """兼容旧 workflow 中的 "X-Thinking" 选项：拆成基础名 + enable_thinking=True。"""
    if chat_handler.endswith("-Thinking"):
        base = chat_handler.removesuffix("-Thinking")
        return (base, True) if base in chat_handlers else (chat_handler, enable_thinking)
    return chat_handler, enable_thinking

def resolve_reasoning_kwargs(chat_handler, thinking_enabled):
    """为 Reasoning Budget 采样器解析各模型的思考块标记。

    reasoning_start_in_prompt: 思考模式下部分模板会在 generation prompt 末尾预注入
    起始标记（如 Qwen 系列的 '<think>\\n'），此时计数需从首个生成 token 开始。
    """
    if chat_handler == "Gemma4":
        return {
            "reasoning_start": "<|channel>thought",
            "reasoning_end": "<channel|>",
            "reasoning_start_in_prompt": False,
        }
    start_in_prompt = thinking_enabled and chat_handler in (
        "Qwen3-VL", "Qwen3.5", "Qwen3.6", "Qwen3.8",
        "MiniCPM-v4.5", "MiniCPM-v4.6",
        "GLM-4.6V", "Step3-VL",
    )
    return {
        "reasoning_start": "<think>",
        "reasoning_end": "</think>",
        "reasoning_start_in_prompt": start_in_prompt,
    }

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

class LLAMA_CPP_STORAGE:
    llm = None
    chat_handler = None
    current_config = None
    current_embedding_mode = False
    #states = {}
    messages = {}
    sys_prompts = {}

    @classmethod
    def clean_state(cls, id=-1):
        if id == -1:
            #cls.states.clear()
            cls.messages.clear()
            cls.sys_prompts.clear()
        else:
            #cls.states.pop(f"{id}", None)
            cls.messages.pop(f"{id}", None)
            cls.sys_prompts.pop(f"{id}", None)
        
    @classmethod
    def clean(cls, all=False):
        try:
            if cls.llm:
                cls.llm.close()
        except Exception:
            pass
            
        try:
            if cls.chat_handler and hasattr(cls.chat_handler, "_exit_stack"):
                cls.chat_handler._exit_stack.close()
        except Exception:
            pass
        
        cls.llm = None
        cls.chat_handler = None
        cls.current_config = None
        cls.current_embedding_mode = False
        if all:
            cls.clean_state()
        
        gc.collect()
        mm.soft_empty_cache()
    
    @classmethod
    def load_model(cls, config, embedding_mode=False):
        def get_chat_handler(chat_handler):
            match chat_handler:
                case "Qwen3.5" | "Qwen3.6" | "Qwen3.8":
                    return Qwen35ChatHandler
                case "Qwen3-VL":
                    return Qwen3VLChatHandler
                case "Qwen3-ASR":
                    return Qwen3ASRChatHandler
                case "Qwen2.5-VL"|"MinerU2.5-Pro":
                    return Qwen25VLChatHandler
                case "LLaVA-1.5":
                    return Llava15ChatHandler
                case "LLaVA-1.6":
                    return Llava16ChatHandler
                case "Moondream2":
                    return MoondreamChatHandler
                case "nanoLLaVA":
                    return NanoLlavaChatHandler
                case "llama3-Vision-Alpha":
                    return Llama3VisionAlphaChatHandler
                case "MiniCPM-v2.6":
                    return MiniCPMv26ChatHandler
                case "MiniCPM-v4.5":
                    return MiniCPMv45ChatHandler
                case "MiniCPM-v4.6":
                    return MiniCPMv46ChatHandler
                case "Step3-VL":
                    return Step3VLChatHandler
                case "PaddleOCR-VL":
                    return PaddleOCRChatHandler
                case "Gemma3":
                    return Gemma3ChatHandler
                case "Gemma4":
                    return Gemma4ChatHandler
                case "GLM-4.6V":
                    return GLM46VChatHandler
                case "GLM-4.1V":
                    return GLM41VChatHandler
                case "LFM2-VL":
                    return LFM2VLChatHandler
                case "LFM2.5-VL":
                    return LFM25VLChatHandler
                case "Granite-Docling":
                    return GraniteDoclingChatHandler
                case "DeepSeek-OCR":
                    return MTMDChatHandler
                case "PaddleOCR-VL-1.5":
                    return PaddleOCRChatHandler
                case "Step3-VL":
                    return Step3VLChatHandler
                case "None":
                    return None
                case _:
                    raise ValueError(f'Unknown model type: "{chat_handler}"')

        cls.clean(all=True)
        cls.current_config = config.copy()
        model = config["model"]
        mmproj = config["mmproj"]
        chat_handler = config["chat_handler"]
        enable_thinking = config.get("enable_thinking", False)
        n_ctx = config["n_ctx"]
        vram_limit = config["vram_limit"]
        n_cpu_moe = config.get("n_cpu_moe", 0)
        load_mtp = config.get("load_mtp", False)
        image_max_tokens = config["image_max_tokens"]
        image_min_tokens = config["image_min_tokens"]
        effective_image_min_tokens, effective_image_max_tokens, image_max_auto_raised = normalize_mtmd_image_token_limits(
            image_min_tokens, image_max_tokens
        )
        n_gpu_layers = -1
        llama_kwargs = {
            "chat_handler": None,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "embeddings": embedding_mode,
            "verbose": False,
        }
        if n_cpu_moe == -1:
            llama_kwargs["cpu_moe"] = True
        elif n_cpu_moe > 0:
            llama_kwargs["n_cpu_moe"] = n_cpu_moe
        
        model_path = os.path.join(folder_paths.models_dir, 'LLM', model)
        handler = get_chat_handler(chat_handler)
        
        if vram_limit != -1:
            gguf_layers = get_layer_count(model_path) or 32
            gguf_size = os.path.getsize(model_path)  * 1.55 / (1024 ** 3)
            gguf_layer_size = gguf_size / gguf_layers
        
        if mmproj and mmproj != "None":
            mmproj_path = os.path.join(folder_paths.models_dir, 'LLM', mmproj)
            if chat_handler == "None":
                raise ValueError('"chat_handler" cannot be None!')
            
            if vram_limit != -1:
                mmproj_size = os.path.getsize(mmproj_path)  * 1.55 / (1024 ** 3)
                n_gpu_layers = max(1, int((vram_limit - mmproj_size) / gguf_layer_size))
            
            print(f"[llama-cpp_vlm] Loading clip:  {mmproj}")
            warn_if_gemma4_unified_mmproj(chat_handler, mmproj_path)
            if image_max_auto_raised:
                print(
                    f"[llama-cpp_vlm] MTMD image_max_tokens auto-raised to {effective_image_max_tokens} "
                    f"because image_min_tokens is {effective_image_min_tokens}."
                )
            
            think_mode = enable_thinking
            kwargs = {
                "clip_model_path": mmproj_path,
                "verbose": False,
                "image_max_tokens": effective_image_max_tokens,
                "image_min_tokens": effective_image_min_tokens,
            }
            if chat_handler == "Qwen3-VL":
                kwargs["force_reasoning"] = think_mode
            elif chat_handler in ENABLE_THINKING_HANDLERS:
                kwargs["enable_thinking"] = think_mode

            if chat_handler == "Gemma4" and any(
                tag in model.lower() for tag in ["e2b", "e4b"]
            ):
                print("[llama-cpp_vlm] Warning: Gemma4 E2B/E4B does not officially support the enable_thinking toggle; behavior may be inconsistent.")

            try:
                cls.chat_handler = handler(**kwargs)
            except Exception as e:
                raise RuntimeError(f"{e}\nPlease update llama-cpp-python from 'https://github.com/JamePeng/llama-cpp-python/releases'")

        else:
            if vram_limit != -1:
                n_gpu_layers = max(1, int(vram_limit / gguf_layer_size))
            if handler is not None:
                cls.chat_handler = handler(verbose=False)
            else:
                cls.chat_handler = None
        
        llama_kwargs["chat_handler"] = cls.chat_handler
        llama_kwargs["n_gpu_layers"] = n_gpu_layers

        if mmproj and mmproj != "None" and cls.chat_handler is not None:
            image_token_budget, budget_source = resolve_mtmd_image_token_budget(
                chat_handler, effective_image_min_tokens, effective_image_max_tokens
            )
            if image_token_budget > 0:
                required_batch = align_to_multiple(image_token_budget)
                if budget_source != "handler_default" and required_batch > n_ctx:
                    raise ValueError(
                        f'"image_max_tokens"/"image_min_tokens" requires at least {required_batch} context tokens, but "n_ctx" is only {n_ctx}. '
                        'Lower the image token settings or increase n_ctx.'
                    )
                if required_batch > MTMD_DEFAULT_N_UBATCH:
                    effective_batch = min(n_ctx, required_batch)
                    llama_kwargs["n_batch"] = effective_batch
                    llama_kwargs["n_ubatch"] = effective_batch
                    print(
                        f"[llama-cpp_vlm] MTMD batch override: {budget_source}={image_token_budget}, "
                        f"n_batch={effective_batch}, n_ubatch={effective_batch}"
                    )

        print(f"[llama-cpp_vlm] Loading model: {model}")
        print(f"[llama-cpp_vlm] n_gpu_layers = {n_gpu_layers}")
        if "load_mtp" in inspect.signature(Llama.__init__).parameters:
            llama_kwargs["load_mtp"] = load_mtp
        elif load_mtp:
            raise RuntimeError('"load_mtp" is unavailable! Please upgrade your llama-cpp-python.')

        cls.llm = Llama(model_path, **llama_kwargs)
        cls.current_embedding_mode = embedding_mode

    @classmethod
    def ensure_embedding_mode(cls, embedding_mode):
        if cls.llm is None or cls.current_embedding_mode == embedding_mode:
            return
        config = cls.current_config.copy()
        print(f"[llama-cpp_vlm] Reloading model with embeddings={embedding_mode}...")
        cls.load_model(config, embedding_mode=embedding_mode)

any_type = AnyType("*")

if not hasattr(mm, "unload_all_models_backup"):
    mm.unload_all_models_backup = mm.unload_all_models
    def patched_unload_all_models(*args, **kwargs):
        LLAMA_CPP_STORAGE.clean(all=True)
        result = mm.unload_all_models_backup(*args, **kwargs)
        return result
    mm.unload_all_models = patched_unload_all_models
    print("[llama-cpp_vlm] Model cleanup hook applied!")

llm_extensions = ['.ckpt', '.pt', '.bin', '.pth', '.safetensors', '.gguf']
folder_paths.folder_names_and_paths["LLM"] = ([os.path.join(folder_paths.models_dir, "LLM")], llm_extensions)
preset_prompts = {
    "Empty - Nothing": "",
    "Normal - Describe": "Describe this @.",
    "Prompt Style - Tags": "Your task is to generate a clean list of comma-separated tags for a text-to-@ AI, based *only* on the visual information in the @. Limit the output to a maximum of 50 unique tags. Strictly describe visual elements like subject, clothing, environment, colors, lighting, and composition. Do not include abstract concepts, interpretations, marketing terms, or technical jargon (e.g., no 'SEO', 'brand-aligned', 'viral potential'). The goal is a concise list of visual descriptors. Avoid repeating tags.",
    "Prompt Style - Simple": "Analyze the @ and generate a simple, single-sentence text-to-@ prompt. Describe the main subject and the setting concisely.",
    "Prompt Style - Detailed": "Generate a detailed, artistic text-to-@ prompt based on the @. Combine the subject, their actions, the environment, lighting, and overall mood into a single, cohesive paragraph of about 2-3 sentences. Focus on key visual details.",
    "Prompt Style - Extreme Detailed": "Generate an extremely detailed and descriptive text-to-@ prompt from the @. Create a rich paragraph that elaborates on the subject's appearance, textures of clothing, specific background elements, the quality and color of light, shadows, and the overall atmosphere. Aim for a highly descriptive and immersive prompt.",
    "Prompt Style - Cinematic": "Act as a master prompt engineer. Create a highly detailed and evocative prompt for an @ generation AI. Describe the subject, their pose, the environment, the lighting, the mood, and the artistic style (e.g., photorealistic, cinematic, painterly). Weave all elements into a single, natural language paragraph, focusing on visual impact.",
    "Creative - Detailed Analysis": "Describe this @ in detail, breaking down the subject, attire, accessories, background, and composition into separate sections.",
    "Creative - Summarize Video": "Summarize the key events and narrative points in this video.",
    "Creative - Short Story": "Write a short, imaginative story inspired by this @ or video.",
    "Creative - Refine & Expand Prompt": "Refine and enhance the following user prompt for creative text-to-@ generation. Keep the meaning and keywords, make it more expressive and visually rich. Output **only the improved prompt text itself**, without any reasoning steps, thinking process, or additional commentary.",
    "Vision - *Bounding Box": 'Locate every instance that belongs to the following categories: "#". Report bbox coordinates in {"bbox_2d": [x1, y1, x2, y2], "label": "string"} JSON format as a List.'
}
preset_tags = list(preset_prompts.keys())

MTMD_DEFAULT_N_UBATCH = 512
MTMD_BATCH_ALIGN = 32
MTMD_IMAGE_TOKEN_DEFAULTS = {
    "Gemma4": 280,
    "LFM2-VL": 256,
    "LFM2.5-VL": 256,
}

def align_to_multiple(value: int, step: int = MTMD_BATCH_ALIGN):
    if value <= 0:
        return 0
    return ((value + step - 1) // step) * step

def resolve_mtmd_image_token_budget(chat_handler, image_min_tokens, image_max_tokens):
    if image_max_tokens > 0:
        return image_max_tokens, "image_max_tokens"
    if image_min_tokens > 0:
        return image_min_tokens, "image_min_tokens"
    fallback = MTMD_IMAGE_TOKEN_DEFAULTS.get(chat_handler, 0)
    if fallback > 0:
        return fallback, "handler_default"
    return 0, "llama_default"

def normalize_mtmd_image_token_limits(image_min_tokens, image_max_tokens):
    image_min_tokens = max(0, int(image_min_tokens or 0))
    image_max_tokens = max(0, int(image_max_tokens or 0))
    if image_min_tokens > 0 and image_max_tokens < image_min_tokens:
        return image_min_tokens, image_min_tokens, True
    return image_min_tokens, image_max_tokens, False

def read_gguf_field(reader, key):
    try:
        field = reader.fields.get(key)
        return field.contents() if field is not None else None
    except Exception:
        return None

def get_mmproj_info(path):
    info = {
        "vision_projector": None,
        "audio_projector": None,
        "vision_blocks": None,
        "audio_blocks": None,
    }
    try:
        from gguf import GGUFReader

        reader = GGUFReader(path)
    except Exception:
        return info

    info["vision_projector"] = read_gguf_field(reader, "clip.vision.projector_type")
    info["audio_projector"] = read_gguf_field(reader, "clip.audio.projector_type")
    info["vision_blocks"] = read_gguf_field(reader, "clip.vision.block_count")
    info["audio_blocks"] = read_gguf_field(reader, "clip.audio.block_count")
    return info

def warn_if_gemma4_unified_mmproj(chat_handler, mmproj_path):
    if chat_handler != "Gemma4":
        return
    info = get_mmproj_info(mmproj_path)
    projectors = {
        str(info.get("vision_projector") or "").lower(),
        str(info.get("audio_projector") or "").lower(),
    }
    if ("gemma4uv" in projectors or "gemma4ua" in projectors) and not llama_cpp_version_at_least(MIN_LLAMA_CPP_VERSION):
        print(
            "[llama-cpp_vlm] Warning: this Gemma4 mmproj uses projector_type "
            f"vision={info.get('vision_projector')}, audio={info.get('audio_projector')} "
            f"with block_count vision={info.get('vision_blocks')}, audio={info.get('audio_blocks')}. "
            f"Gemma4 unified projectors require llama-cpp-python >= {MIN_LLAMA_CPP_VERSION}."
        )

def format_mtmd_context_error(error):
    config = LLAMA_CPP_STORAGE.current_config or {}
    mmproj = config.get("mmproj")
    if not mmproj or mmproj == "None":
        return str(error)

    mmproj_path = os.path.join(folder_paths.models_dir, "LLM", mmproj)
    info = get_mmproj_info(mmproj_path)
    details = (
        f"mmproj={mmproj}, "
        f"vision_projector={info.get('vision_projector')}, "
        f"audio_projector={info.get('audio_projector')}, "
        f"vision_blocks={info.get('vision_blocks')}, "
        f"audio_blocks={info.get('audio_blocks')}, "
        f"image_min_tokens={config.get('image_min_tokens')}, "
        f"image_max_tokens={config.get('image_max_tokens')}"
    )
    return (
        f"{error}\n"
        f"[llama-cpp_vlm] MTMD failed to initialize the selected projector ({details}).\n"
        f"For Gemma4 unified projectors, use llama-cpp-python >= {MIN_LLAMA_CPP_VERSION} "
        "and a matching text model/mmproj pair. If you raise image_min_tokens manually, "
        "image_max_tokens must be at least the same value; this node now normalizes that "
        "automatically before loading."
    )

def create_chat_completion(messages, seed, parameters):
    try:
        return LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **parameters)
    except ValueError as e:
        if "Failed to load mtmd context" in str(e):
            raise RuntimeError(format_mtmd_context_error(e)) from e
        raise

def select_visible_output(raw_text, thinking_text, answer_text, strip_thinking):
    if strip_thinking:
        if answer_text:
            return answer_text
        if thinking_text:
            return ""
    return raw_text

def image2base64(image):
    img = Image.fromarray(image)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64

def prompt_builder_image_url(frame, max_size):
    image = frame.detach().cpu().numpy()
    if image.ndim != 3:
        raise ValueError(f"Expected an image frame with 3 dimensions, got {image.ndim}.")

    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] > 3:
        image = image[..., :3]

    pil_image = Image.fromarray(image, mode="RGB")
    if max_size > 0:
        pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=85)
    data = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"

def prompt_builder_frames(images, limit=None):
    if images is None:
        return []
    if not torch.is_tensor(images):
        raise TypeError(f"Expected IMAGE input to be a tensor, got {type(images).__name__}.")
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError(f"Expected IMAGE input with shape [B,H,W,C], got {tuple(images.shape)}.")

    count = images.shape[0]
    if limit is not None and count > limit:
        indices = np.linspace(0, count - 1, limit, dtype=int)
        return [images[index] for index in indices]
    return [images[index] for index in range(count)]

def prompt_builder_audio_url(audio):
    waveform = audio["waveform"]
    if waveform.ndim == 3:
        waveform = waveform[0]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(f"Expected AUDIO waveform with shape [B,C,S] or [C,S], got {tuple(waveform.shape)}.")
    if waveform.shape[-1] == 0:
        raise ValueError("The AUDIO input contains no samples.")

    waveform = torch.nan_to_num(waveform.detach().float().cpu()).clamp(-1.0, 1.0)
    pcm = (waveform * 32767.0).round().to(torch.int16).transpose(0, 1).contiguous().numpy()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(waveform.shape[0])
        wav_file.setsampwidth(2)
        wav_file.setframerate(max(1, int(audio["sample_rate"])))
        wav_file.writeframes(pcm.astype("<i2", copy=False).tobytes())

    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{data}"

def parse_json(json_str):
    json_output = json_str.strip()
    if json_output.startswith("```json"):
        json_output = json_output[7:]
    if json_output.startswith("```"):
        json_output = json_output[3:]
    if json_output.endswith("```"):
        json_output = json_output[:-3]
    json_output = json_output.strip()
    try:
        parsed = json.loads(json_output)
    except Exception as e:
        raise ValueError(f"Unable to load JSON data!\n{e}")
    return parsed

def scale_image(image: torch.Tensor, max_size: int = 128):
    img_np = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    
    w, h = img_pil.size
    scale = min(max_size / max(w, h), 1.0)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return np.array(img_resized)

def strip_thinking_chain(text):
    _, answer = split_thinking_chain(text)
    return answer

def split_thinking_chain(text, assume_open=False):
    """拆分思考块与答案。

    assume_open: 思考模板把 reasoning_start 预注入 prompt 时（Qwen3.5 等），
    生成文本里不会出现开头标记；若闭合标记也缺失（被 max_tokens 截断），
    整段文本应视为不完整的思考链而不是答案。
    """
    if not isinstance(text, str):
        return "", text

    if "<think>" in text and "</think>" in text:
        think_blocks = re.findall(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL)
        answer = re.sub(r"<think>\s*.*?\s*</think>\s*", "", text, flags=re.DOTALL).lstrip()
        thinking = "\n\n".join(block.strip() for block in think_blocks if block.strip())
        return thinking, answer

    if "</think>" in text:
        thinking, answer = text.split("</think>", 1)
        return thinking.replace("<think>", "").strip(), answer.lstrip()

    if "<think>" in text:
        thinking = text.split("<think>", 1)[1].strip()
        return thinking, ""

    channel_pattern = re.compile(r"<\|channel\>\s*([^\n<]*)\n?(.*?)<channel\|>", flags=re.DOTALL)
    matches = list(channel_pattern.finditer(text))
    if matches:
        thinking_parts = []
        answer_parts = []
        consumed_ranges = []
        for match in matches:
            channel_name = (match.group(1) or "").strip().lower()
            channel_body = (match.group(2) or "").strip()
            consumed_ranges.append((match.start(), match.end()))
            if not channel_body:
                continue
            if any(token in channel_name for token in ["thought", "think", "reason", "analysis"]):
                thinking_parts.append(channel_body)
            else:
                answer_parts.append(channel_body)

        leftovers = []
        cursor = 0
        for start, end in consumed_ranges:
            fragment = text[cursor:start].strip()
            if fragment:
                leftovers.append(fragment)
            cursor = end
        tail = text[cursor:].strip()
        if tail:
            leftovers.append(tail)

        thinking = "\n\n".join(part for part in thinking_parts if part)
        answer_candidates = answer_parts + leftovers
        answer = "\n\n".join(part for part in answer_candidates if part).strip()
        return thinking, answer

    open_channel = re.search(r"<\|channel\>\s*([^\n<]*)\n?(.*)$", text, flags=re.DOTALL)
    if open_channel:
        channel_name = (open_channel.group(1) or "").strip().lower()
        channel_body = (open_channel.group(2) or "").strip()
        if any(token in channel_name for token in ["thought", "think", "reason", "analysis"]):
            return channel_body, ""
        if channel_body:
            return "", channel_body

    if "<channel|>" in text:
        prefix, suffix = text.split("<channel|>", 1)
        if "<|channel>" in prefix:
            answer = prefix.split("<|channel>", 1)[0].strip()
            thinking = suffix.strip()
            if thinking or answer:
                return thinking, answer

    if assume_open:
        return text.strip(), ""

    return "", text

def qwen3bbox(image, json_data):
    img = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    bboxes = []
    if isinstance(json_data, dict):
        json_data = [json_data]
    for item in json_data:
        if not isinstance(item, dict) or "bbox_2d" not in item:
            continue
        x0, y0, x1, y1 = item["bbox_2d"]
        size = 1000.0
        x0 = x0 / size * img.width
        y0 = y0 / size * img.height
        x1 = x1 / size * img.width
        y1 = y1 / size * img.height
        bboxes.append((x0, y0, x1, y1))
    return bboxes

def draw_bbox(image, json_data, mode):
    label_colors = {}
    img = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    
    if isinstance(json_data, dict):
        json_data = [json_data]

    for item in json_data:
        if not isinstance(item, dict):
            continue
        label = item.get("label", item.get("text_content", "bbox"))
        if "bbox_2d" not in item:
            continue
        x0, y0, x1, y1 = item["bbox_2d"]
        if mode in ["Qwen3-VL", "Qwen2.5-VL"]:
            size = 1000.0
            x0 = x0 / size * img.width
            y0 = y0 / size * img.height
            x1 = x1 / size * img.width
            y1 = y1 / size * img.height
        bbox = (x0, y0, x1, y1)
        
        if label not in label_colors:
            label_colors[label] = tuple(random.randint(80, 180) for _ in range(3))
        color = label_colors[label]
        draw.rectangle(bbox, outline=color, width=4)
        text_y = max(0, y0 - 10)
        text_size = draw.textbbox((x0, text_y), str(label))
        draw.rectangle([text_size[0], text_size[1]-2, text_size[2]+4, text_size[3]+2], fill=color)
        draw.text((x0+2, text_y), str(label), fill=(255,255,255))
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

class llama_cpp_multimodal_prompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL",),
                "system_prompt": ("STRING", {
                    "default": "You are an expert prompt writer for image and video generation models. Analyze the attached references and the user's request, then produce one precise, production-ready prompt. Output only the final prompt text.",
                    "multiline": True,
                }),
                "user_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "A batch of reference images. Each image is exposed to the model in batch order.",
                }),
                "video": ("VIDEO", {
                    "tooltip": "A video reference. Frames are sampled evenly and sent as ordered visual references.",
                }),
                "audio": ("AUDIO", {
                    "tooltip": "An optional audio reference. It is encoded locally as WAV for the multimodal model.",
                }),
                "max_images": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Maximum number of IMAGE batch items to send.",
                }),
                "video_max_frames": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Maximum number of evenly sampled video frames to send.",
                }),
                "image_max_size": ("INT", {
                    "default": 1024,
                    "min": 128,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Maximum image edge sent to the model.",
                }),
                "include_video_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include the audio track from VIDEO when one is present.",
                }),
                "filter_thinking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove Gemma4/reasoning channel content from the prompt output.",
                }),
                "force_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload the llama.cpp model after generation.",
                }),
                "parameters": ("LLAMACPPARAMS",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "thinking")
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"

    def process(
        self,
        llama_model,
        system_prompt,
        user_prompt,
        seed,
        images=None,
        video=None,
        audio=None,
        max_images=16,
        video_max_frames=8,
        image_max_size=1024,
        include_video_audio=True,
        filter_thinking=True,
        force_offload=False,
        parameters=None,
    ):
        if not LLAMA_CPP_STORAGE.llm:
            LLAMA_CPP_STORAGE.load_model(llama_model)
        else:
            LLAMA_CPP_STORAGE.ensure_embedding_mode(False)

        image_frames = prompt_builder_frames(images, max_images)
        video_frames = []
        video_audio = None
        if video is not None:
            video_components = video.get_components()
            video_frames = prompt_builder_frames(video_components.images, video_max_frames)
            video_audio = video_components.audio

        media_present = bool(
            image_frames
            or video_frames
            or (include_video_audio and video_audio is not None)
            or audio is not None
        )
        current_config = LLAMA_CPP_STORAGE.current_config or {}
        if media_present and current_config.get("mmproj") in (None, "", "None"):
            raise ValueError("Multimedia input requires a loaded mmproj projector. Select the matching mmproj in the model loader.")

        media_content = []
        media_descriptions = []

        for index, frame in enumerate(image_frames, start=1):
            media_descriptions.append(f"Image{index}: an attached reference image.")
            media_content.extend([
                {"type": "text", "text": f"\nReference Image{index}:"},
                {"type": "image_url", "image_url": {"url": prompt_builder_image_url(frame, image_max_size)}},
            ])

        if video is not None:
            media_descriptions.append(
                f"Video1: an attached video represented by {len(video_frames)} evenly sampled frames in temporal order."
            )
            for index, frame in enumerate(video_frames, start=1):
                media_content.extend([
                    {"type": "text", "text": f"\nReference Video1, frame {index} of {len(video_frames)}:"},
                    {"type": "image_url", "image_url": {"url": prompt_builder_image_url(frame, image_max_size)}},
                ])

        audio_items = []
        if include_video_audio and video_audio is not None:
            audio_items.append(("Audio1", "the audio track extracted from Video1", video_audio))
        if audio is not None:
            audio_number = len(audio_items) + 1
            audio_items.append((f"Audio{audio_number}", "an attached audio reference", audio))

        for label, description, audio_input in audio_items:
            media_descriptions.append(f"{label}: {description}.")
            media_content.extend([
                {"type": "text", "text": f"\nReference {label}:"},
                {"type": "audio_url", "audio_url": {"url": prompt_builder_audio_url(audio_input)}},
            ])

        prompt_parts = []
        if media_descriptions:
            prompt_parts.append(
                "[REFERENCE MEDIA]\n"
                + "\n".join(media_descriptions)
                + "\nUse these labels when a reference needs to be named in the generated prompt."
            )
        if user_prompt.strip():
            prompt_parts.append("[USER REQUEST]\n" + user_prompt)
        user_text = "\n\n".join(prompt_parts)
        if media_content:
            user_content = [{"type": "text", "text": user_text}]
            user_content.extend(media_content)
        else:
            user_content = user_text

        if parameters is None:
            parameters = {
                "max_tokens": 2048,
                "top_k": 30,
                "top_p": 0.9,
                "min_p": 0.05,
                "typical_p": 1.0,
                "temperature": 0.4,
                "repeat_penalty": 1.0,
                "frequency_penalty": 0.0,
                "mirostat_mode": 0,
                "mirostat_eta": 0.1,
                "mirostat_tau": 5.0,
            }

        llama_parameters = parameters.copy()
        llama_parameters.pop("state_uid", None)
        llama_parameters.pop("presence_penalty", None)
        if _MTMD:
            llama_parameters.pop("present_penalty", None)

        current_handler = current_config.get("chat_handler", "")
        thinking_enabled = current_config.get("enable_thinking", False) or current_handler in ALWAYS_THINKING_HANDLERS
        thinking_mode = (filter_thinking and thinking_enabled) or (current_handler == "Gemma4" and not thinking_enabled)
        thinking_open_in_prompt = thinking_enabled and resolve_reasoning_kwargs(
            current_handler, thinking_enabled
        )["reasoning_start_in_prompt"]

        reasoning_budget = llama_parameters.get("reasoning_budget", -1)
        if not isinstance(reasoning_budget, int) or reasoning_budget < 0:
            llama_parameters.pop("reasoning_budget", None)
            llama_parameters.pop("reasoning_budget_message", None)
        else:
            if not llama_parameters.get("reasoning_budget_message"):
                llama_parameters.pop("reasoning_budget_message", None)
            for key, value in resolve_reasoning_kwargs(current_handler, thinking_enabled).items():
                llama_parameters.setdefault(key, value)

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        try:
            output = create_chat_completion(messages, seed, llama_parameters)
        finally:
            if force_offload:
                LLAMA_CPP_STORAGE.clean()

        raw_text = output["choices"][0]["message"]["content"].removeprefix(": ").lstrip()
        thinking_text, answer_text = split_thinking_chain(raw_text, assume_open=thinking_open_in_prompt)
        result = select_visible_output(raw_text, thinking_text, answer_text, thinking_mode)
        if not result and not filter_thinking:
            result = answer_text or raw_text
        result = re.sub(r"^```(?:text|markdown|plaintext)?\s*", "", result.strip(), flags=re.IGNORECASE)
        result = re.sub(r"\s*```$", "", result).strip()
        thinking_text = thinking_text.strip()
        gc.collect()
        return (result, thinking_text)

class llama_cpp_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        all_llms = folder_paths.get_filename_list("LLM")
        model_list = [f for f in all_llms if "mmproj" not in f.lower()]
        mmproj_list = ["None"]+[f for f in all_llms if "mmproj" in f.lower()]
            
        return {"required": {
            "model": (model_list,),
            "mmproj": (mmproj_list, {"default": "None"}),
            "chat_handler": (chat_handlers, {"default": "None"}),
            "n_ctx": ("INT", {
                "default": 8192,
                "min": 1024, "max": 327680, "step": 128,
                "tooltip": "Context length limit."
            }),
            "vram_limit": ("INT", {
                "default": -1,
                "min": -1, "max": 1024, "step": 1,
                "tooltip": "VRAM usage limit in GB (-1 = no limit)\nReference range; actual usage may slightly exceed."
            }),
            "image_min_tokens": ("INT", {
                "default": 0, "min": 0, "max": 4096, "step": 32,
                "tooltip": "Minimum image token budget for MTMD-based multimodal handlers. Higher values can improve detail retention but may increase VRAM usage."
            }),
            "image_max_tokens": ("INT", {
                "default": 0, "min": 0, "max": 4096, "step": 32,
                "tooltip": "Maximum image token budget for MTMD-based multimodal handlers. Values above 512 will automatically raise n_batch/n_ubatch to avoid Gemma4-style MTMD assertion failures, which also increases VRAM usage."
            }),
            "n_cpu_moe": ("INT", {
                "default": 0,
                "min": -1, "max": 256, "step": 1,
                "tooltip": "Offload MoE expert weights to CPU/RAM (for MoE models like Qwen3-VL-30B-A3B / Gemma4-26BA4B).\n0 = disabled, -1 = offload all MoE weights, N > 0 = offload the first N layers.\nAttention/router weights stay on GPU, greatly reducing VRAM usage."
            }),
                        "enable_thinking": ("BOOLEAN", {
                "default": False,
                "tooltip": "Enable the model's thinking/reasoning mode (for handlers that support it:\nMiniCPM-v4.5/4.6, Gemma4, Qwen3-VL, Qwen3.5, GLM-4.6V, Step3-VL).\nGLM-4.1V always thinks regardless of this switch.\nCombine with the \"reasoning_budget\" parameter to limit thinking length."
            }),
            },
            "optional": {
                "load_mtp": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Load MTP layers when supported by llama-cpp-python 0.3.46 or newer."
                }),
            },
        }

    RETURN_TYPES = ("LLAMACPPMODEL",)
    RETURN_NAMES = ("llama_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "llama-cpp-vlm"
    
    '''
    @classmethod
    def IS_CHANGED(s, model, mmproj, chat_handler, n_ctx, vram_limit, image_min_tokens, image_max_tokens):
        if LLAMA_CPP_STORAGE.llm is None:
            return float("NaN") 
        
        custom_config = {
            "model": model,
            "mmproj": mmproj,
            "chat_handler":chat_handler,
            "n_ctx": n_ctx,
            "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens
        }
        config_str = json.dumps(custom_config, sort_keys=True, ensure_ascii=False)
        return config_str
    '''
    @classmethod
    def VALIDATE_INPUTS(s, chat_handler):
        # 兼容旧 workflow 中的 "X-Thinking" 选项（已合并为基础名 + enable_thinking 开关）
        base, _ = normalize_chat_handler(chat_handler)
        if base in chat_handlers:
            return True
        return f'Unknown chat_handler: "{chat_handler}"'

    def loadmodel(self, model, mmproj, chat_handler, enable_thinking, n_ctx, vram_limit, n_cpu_moe, image_min_tokens, image_max_tokens, load_mtp=False):
        chat_handler, enable_thinking = normalize_chat_handler(chat_handler, enable_thinking)
        custom_config = {
            "model": model,
            "mmproj": mmproj,
            "chat_handler":chat_handler,
            "enable_thinking": enable_thinking,
            "n_ctx": n_ctx,
            "vram_limit": vram_limit,
            "n_cpu_moe": n_cpu_moe,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens,
            "load_mtp": load_mtp,
        }
        if not LLAMA_CPP_STORAGE.llm or LLAMA_CPP_STORAGE.current_config != custom_config:
            print("[llama-cpp_vlm] Loading model...")
            LLAMA_CPP_STORAGE.load_model(custom_config)
        return (custom_config,)

class llama_cpp_instruct_adv:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL",),
                "preset_prompt": (preset_tags, {"default": preset_tags[1]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": 'user_prompt\n\nFor preset hints marked with an "*", this will be used to fill the placeholder (e.g., Object names in BBox detection)\nOtherwise, this will override the preset prompts.'}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "inference_mode": (["one by one", "images", "video"], {
                    "default": "one by one",
                    "tooltip": "one by one: Process every image frame separately\nimages: Combine all image list items into one prompt\nvideo: Treat each image list item as a separate video clip"
                }),
                "max_frames": ("INT", {
                    "default": 24,
                    "min": 2,
                    "max": 1024,
                    "step": 1,
                    "tooltip": 'Number of frames to sample evenly from input video.\n(for "video" mode only)'
                }),
                "max_size": ("INT", {
                    "default": 256,
                    "min": 128,
                    "max": 16384,
                    "step": 64,
                    "tooltip": 'Max size of input images in "images" and "video" modes.'
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "force_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload the model after inference."
                }),
                "save_states": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Preserve the context of this conversation in RAM."
                }),
                "filter_thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Hide reasoning blocks in the main output. The `thinking` and `answer` outputs still keep the split result."
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
            "optional": {
                "parameters": ("LLAMACPPARAMS",),
                "images": ("IMAGE",),
                "queue_handler": (any_type, {"tooltip": "Used to control the execution order of instruct nodes."}),
            },
            
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("output", "output_list", "thinking", "answer", "state_uid")
    OUTPUT_IS_LIST = (False, True, False, False, False)
    INPUT_IS_LIST = True
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def sanitize_messages(self, messages):
        clean_messages = json.loads(json.dumps(messages))
        for msg in clean_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        item["image_url"]["url"] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAACXBIWXMAAAsTAAALEwEAmpwYAAAADElEQVQImWP4//8/AAX+Av5Y8msOAAAAAElFTkSuQmCC"
        return clean_messages
    
    def process(self, llama_model, preset_prompt, custom_prompt, system_prompt, inference_mode, max_frames, max_size, seed, force_offload, save_states, filter_thinking, unique_id, parameters=None, images=None, queue_handler=None):
        llama_model = llama_model[0] if isinstance(llama_model, list) else llama_model
        preset_prompt = preset_prompt[0] if isinstance(preset_prompt, list) else preset_prompt
        custom_prompt = custom_prompt[0] if isinstance(custom_prompt, list) else custom_prompt
        system_prompt = system_prompt[0] if isinstance(system_prompt, list) else system_prompt
        inference_mode = inference_mode[0] if isinstance(inference_mode, list) else inference_mode
        max_frames = max_frames[0] if isinstance(max_frames, list) else max_frames
        max_size = max_size[0] if isinstance(max_size, list) else max_size
        seed = seed[0] if isinstance(seed, list) else seed
        force_offload = force_offload[0] if isinstance(force_offload, list) else force_offload
        save_states = save_states[0] if isinstance(save_states, list) else save_states
        filter_thinking = filter_thinking[0] if isinstance(filter_thinking, list) else filter_thinking
        unique_id = unique_id[0] if isinstance(unique_id, list) else unique_id
        parameters = parameters[0] if isinstance(parameters, list) and parameters else parameters

        if not LLAMA_CPP_STORAGE.llm:
            LLAMA_CPP_STORAGE.load_model(llama_model)
        else:
            LLAMA_CPP_STORAGE.ensure_embedding_mode(False)
        
        if parameters is None:
            parameters = {
                "max_tokens": 1024,
                "top_k": 30,
                "top_p": 0.9,
                "min_p": 0.05,
                "typical_p": 1.0,
                "temperature": 0.8,
                "repeat_penalty": 1.0,
                "frequency_penalty": 0.0,
                "mirostat_mode": 0,
                "mirostat_eta": 0.1,
                "mirostat_tau": 5.0
            }

        _uid = parameters.get("state_uid", None)
        _parameters = parameters.copy()
        _parameters.pop("state_uid", None)
        _parameters.pop("presence_penalty", None)  # 旧版遗留键，上游已更名为 present_penalty
        if _MTMD:
            _parameters.pop("present_penalty", None)
        uid = str(unique_id).rpartition('.')[-1] if _uid in (None, -1) else str(_uid)
        current_handler = LLAMA_CPP_STORAGE.current_config.get("chat_handler", "")
        thinking_enabled = (
            LLAMA_CPP_STORAGE.current_config.get("enable_thinking", False)
            or current_handler in ALWAYS_THINKING_HANDLERS
        )
        # 非思考模式的 Gemma4 输出中仍可能带 channel 标记，始终做剥离兜底
        thinking_mode = (filter_thinking and thinking_enabled) or (current_handler == "Gemma4" and not thinking_enabled)
        # 模板预注入 reasoning_start 的 handler：截断输出应整体视为思考链
        thinking_open_in_prompt = thinking_enabled and resolve_reasoning_kwargs(
            current_handler, thinking_enabled
        )["reasoning_start_in_prompt"]

        # Reasoning Budget: -1 关闭；0 强制立即结束思考块；N>0 限制首个思考块的 token 预算
        reasoning_budget = _parameters.get("reasoning_budget", -1)
        if not isinstance(reasoning_budget, int) or reasoning_budget < 0:
            _parameters.pop("reasoning_budget", None)
            _parameters.pop("reasoning_budget_message", None)
        else:
            if not _parameters.get("reasoning_budget_message"):
                _parameters.pop("reasoning_budget_message", None)
            for key, value in resolve_reasoning_kwargs(current_handler, thinking_enabled).items():
                _parameters.setdefault(key, value)
        
        last_sys_prompt = LLAMA_CPP_STORAGE.sys_prompts.get(f"{uid}", None)
        video_input = inference_mode == "video"
        system_prompts = "请将输入的图片序列当做视频而不是静态帧序列, " + system_prompt if video_input else system_prompt
        if last_sys_prompt != system_prompts:
            messages = []
            LLAMA_CPP_STORAGE.clean_state(uid)
            LLAMA_CPP_STORAGE.sys_prompts[f"{uid}"] = system_prompts
            if system_prompts.strip():
                messages.append({"role": "system", "content": system_prompts})
        else:
            if save_states:
                try:
                    print(f"[llama-cpp_vlm] Loading state and history id={uid}...")
                    #LLAMA_CPP_STORAGE.llm.load_state(LLAMA_CPP_STORAGE.states[f"{uid}"])
                    messages = LLAMA_CPP_STORAGE.messages.get(f"{uid}", [])
                except Exception:
                    messages = []
            else:
                messages = []
        out1 = ""
        out2 = []
        thinking_out = ""
        answer_out = ""
        user_content = []
        if custom_prompt.strip() and "*" not in preset_prompt:
            user_content.append({"type": "text", "text": custom_prompt})
        else:
            p = preset_prompts[preset_prompt].replace("#", custom_prompt.strip()).replace("@", "video" if video_input else "image")
            user_content.append({"type": "text", "text": p})
            
        raw_image_list = [image for image in images if image is not None] if isinstance(images, list) else ([images] if images is not None else [])
        image_groups = []
        for image_item in raw_image_list:
            if image_item.ndim == 3:
                image_groups.append([image_item])
            elif image_item.ndim == 4:
                image_groups.append([image_item[index] for index in range(image_item.shape[0])])
            else:
                raise ValueError(f"Expected IMAGE input with 3 or 4 dimensions, got {tuple(image_item.shape)}.")

        if image_groups:
            current_config = LLAMA_CPP_STORAGE.current_config or {}
            if LLAMA_CPP_STORAGE.chat_handler is None or current_config.get("mmproj") in (None, "None"):
                raise ValueError("Image input detected, but the loaded model is not configured with a mmproj module.")

            if inference_mode == "one by one":
                frames = [frame for group in image_groups for frame in group]
                tmp_list = []
                state_user_content = user_content
                print(f"[llama-cpp_vlm] Start processing {len(frames)} images")

                for index, image in enumerate(cqdm(frames), start=1):
                    if mm.processing_interrupted():
                        raise mm.InterruptProcessingException()
                    data = image2base64(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
                    current_user_content = json.loads(json.dumps(user_content))
                    current_user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{data}"},
                    })
                    state_user_content = current_user_content
                    current_messages = messages + [{"role": "user", "content": current_user_content}]
                    output = create_chat_completion(current_messages, seed, _parameters)
                    raw_text = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
                    thinking_text, answer_text = split_thinking_chain(raw_text, assume_open=thinking_open_in_prompt)
                    text = select_visible_output(raw_text, thinking_text, answer_text, thinking_mode)
                    out2.append(text)
                    if len(frames) > 1:
                        label = f"====== Image {index} ======"
                        tmp_list.extend([label, text])
                        if thinking_text:
                            thinking_out += ("" if not thinking_out else "\n\n") + f"{label}\n{thinking_text}"
                        answer_out += ("" if not answer_out else "\n\n") + f"{label}\n{answer_text or text}"
                    else:
                        tmp_list.append(text)
                        thinking_out = thinking_text
                        answer_out = answer_text or text

                out1 = "\n\n".join(tmp_list)
                messages.append({"role": "user", "content": state_user_content})

            elif inference_mode == "images":
                frames = [frame for group in image_groups for frame in group]
                for image in frames:
                    if len(frames) > 1:
                        data = image2base64(scale_image(image, max_size))
                    else:
                        data = image2base64(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{data}"},
                    })

                messages.append({"role": "user", "content": user_content})
                output = create_chat_completion(messages, seed, _parameters)
                raw_out = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
                thinking_out, answer_out = split_thinking_chain(raw_out, assume_open=thinking_open_in_prompt)
                out1 = select_visible_output(raw_out, thinking_out, answer_out, thinking_mode)
                out2 = [out1]

            elif inference_mode == "video":
                tmp_list = []
                state_user_content = user_content
                print(f"[llama-cpp_vlm] Start processing {len(image_groups)} video clips")

                for clip_index, video_frames in enumerate(cqdm(image_groups), start=1):
                    if mm.processing_interrupted():
                        raise mm.InterruptProcessingException()
                    sample_count = min(len(video_frames), max_frames)
                    indices = np.linspace(0, len(video_frames) - 1, sample_count, dtype=int)
                    current_user_content = json.loads(json.dumps(user_content))
                    for frame_index in indices:
                        data = image2base64(scale_image(video_frames[frame_index], max_size))
                        current_user_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{data}"},
                        })
                    state_user_content = current_user_content
                    current_messages = messages + [{"role": "user", "content": current_user_content}]
                    output = create_chat_completion(current_messages, seed, _parameters)
                    raw_text = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
                    thinking_text, answer_text = split_thinking_chain(raw_text, assume_open=thinking_open_in_prompt)
                    text = select_visible_output(raw_text, thinking_text, answer_text, thinking_mode)
                    out2.append(text)
                    if len(image_groups) > 1:
                        label = f"====== Video Clip {clip_index} ======"
                        tmp_list.extend([label, text])
                        if thinking_text:
                            thinking_out += ("" if not thinking_out else "\n\n") + f"{label}\n{thinking_text}"
                        answer_out += ("" if not answer_out else "\n\n") + f"{label}\n{answer_text or text}"
                    else:
                        tmp_list.append(text)
                        thinking_out = thinking_text
                        answer_out = answer_text or text

                out1 = "\n\n".join(tmp_list)
                messages.append({"role": "user", "content": state_user_content})
        else:
            messages.append({"role": "user", "content": user_content})
            output = create_chat_completion(messages, seed, _parameters)
            raw_out = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
            thinking_out, answer_out = split_thinking_chain(raw_out, assume_open=thinking_open_in_prompt)
            out1 = select_visible_output(raw_out, thinking_out, answer_out, thinking_mode)
            out2 = [out1]

        if not answer_out:
            answer_out = out1
            
        if save_states:
            print(f"[llama-cpp_vlm] Saving state id={uid}...")
            #LLAMA_CPP_STORAGE.states[f"{uid}"] = LLAMA_CPP_STORAGE.llm.save_state()
            messages.append({"role": "assistant", "content": out1})
            clear_message = self.sanitize_messages(messages)
            LLAMA_CPP_STORAGE.messages[f"{uid}"] = clear_message
        else:
            if not LLAMA_CPP_STORAGE.messages.get(f"{uid}"):
                LLAMA_CPP_STORAGE.sys_prompts.pop(f"{uid}", None)
                
        if force_offload:
            LLAMA_CPP_STORAGE.clean()
        else:
            if LLAMA_CPP_STORAGE.current_config["chat_handler"] in ("Qwen3.5", "Qwen3.6", "Qwen3.8"):
                LLAMA_CPP_STORAGE.llm.n_tokens = 0
                LLAMA_CPP_STORAGE.llm._ctx.memory_clear(True)
                if LLAMA_CPP_STORAGE.llm.is_hybrid and LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr is not None:
                    LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr.clear()
            
        del messages
        gc.collect()
        return (out1, out2, thinking_out, answer_out, uid)

class llama_cpp_parameters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_tokens": ("INT", {"default": 1024, "min": 0, "max": 32768, "step": 1}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 1000, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "present_penalty": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Presence penalty (renamed from presence_penalty in llama-cpp-python)."
                }),
                #"tfs_z": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                #"penalty_last_n": ("INT", {"default": 64, "min": -1, "max": 8192, "step": 1}),
                "mirostat_mode": ("INT", {"default": 0, "min": 0, "max": 2, "step": 1}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "state_uid": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "Use a specific ID to save the conversation state.\n(-1 = use node's unique_id)"
                }),
            }
        }
    RETURN_TYPES = ("LLAMACPPARAMS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    def process(self, **kwargs):
        return (kwargs,)
    
class llama_cpp_clean_states:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,),
                "state_uid": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "Clear the saved state for a specific ID (-1 = clear all)"
                }),
            },
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, any, state_uid):
        print(f"[llama-cpp_vlm] Cleaning up saved states {state_uid}...")
        LLAMA_CPP_STORAGE.clean_state(state_uid)
        return (any,)

class llama_cpp_unload_model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any": (any_type,)}}
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, any):
        print("[llama-cpp_vlm] Unloading llama model...")
        LLAMA_CPP_STORAGE.clean()
        return (any,)

class json_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("STRING", {"forceInput": True}),
                "mode": (["simple","Qwen3-VL", "Qwen2.5-VL"], {"default": "simple"}),
                "label": ("STRING", {
                    "default":"",
                    "multiline": False,
                    "tooltip": "Select only the BBoxes with specific labels."
                }),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("BBOX", "IMAGE")
    RETURN_NAMES = ("bboxes", "image_list")
    OUTPUT_IS_LIST = (True, True)
    INPUT_IS_LIST = True
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, json, mode, label, image=None):
        mode_val = mode[0] if isinstance(mode, list) and mode else "simple"
        label_val = label[0] if isinstance(label, list) and label else ""

        flat_images_list = []
        original_structure = []
    
        if image is not None:
            for img_batch in image:
                if img_batch is None:
                    continue
                if img_batch.ndim == 3:
                    flat_images_list.append(img_batch.unsqueeze(0))
                    original_structure.append(1)
                elif img_batch.ndim == 4:
                    count = img_batch.shape[0]
                    original_structure.append(count)
                    for n in range(count):
                        flat_images_list.append(img_batch[n:n+1])
        
        total_images = len(flat_images_list)
        output_bboxes = []
        processed_flat_results = []
        
        for i, j in enumerate(json):
            bboxes = parse_json(j)
            if isinstance(bboxes, dict):
                bboxes = [bboxes]
            
            if label_val != "":
                bboxes = [
                    item for item in bboxes
                    if isinstance(item, dict) and (item.get("label") == label_val or item.get("text_content") == label_val)
                ]

            if total_images > 0:
                curr_idx = i if i < total_images else (total_images - 1)
                curr_img = flat_images_list[curr_idx]
                
                try:
                    res_img = draw_bbox(curr_img[0], bboxes, mode_val)
                    if res_img.ndim == 3:
                        res_img = res_img.unsqueeze(0)
                    elif res_img.ndim == 4 and res_img.shape[0] > 1:
                        res_img = res_img[0:1]
                        
                    processed_flat_results.append(res_img)
                except Exception as e:
                    print(f"Error drawing on image {curr_idx}: {e}")
                    processed_flat_results.append(curr_img)
                    
            if mode_val in ["Qwen3-VL", "Qwen2.5-VL"]:
                if total_images == 0:
                    raise ValueError("Image required for Qwen mode")
                curr_idx = i if i < total_images else (total_images - 1)
                bbox = qwen3bbox(flat_images_list[curr_idx][0], bboxes)
            else:
                bbox = [tuple(item["bbox_2d"]) for item in bboxes if isinstance(item, dict) and "bbox_2d" in item]
                
            output_bboxes.append(bbox)
            
        restructured_images_list = []
        cursor = 0
        for count in original_structure:
            chunk = processed_flat_results[cursor : cursor + count]
            if chunk:
                restructured_images_list.append(torch.cat(chunk, dim=0))
            cursor += count
            
        return (output_bboxes, restructured_images_list)

class SEG:
    def __init__(self, cropped_image, cropped_mask, confidence, crop_region, bbox, label, control_net_wrapper=None):
        self.cropped_image = cropped_image
        self.cropped_mask = cropped_mask
        self.confidence = confidence
        self.crop_region = crop_region
        self.bbox = bbox
        self.label = label
        self.control_net_wrapper = control_net_wrapper
        
    def __repr__(self):
        return (f"SEG(cropped_image={self.cropped_image}, cropped_mask=shape{self.cropped_mask.shape}, confidence={self.confidence}, bbox={self.bbox}, label='{self.label}'), control_net_wrapper={self.control_net_wrapper}")
    
class bbox_to_segs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("SEGS",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image, dilation, feather):
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)
        
        seg_list = []
        image_for_cropping = image[0] 
        
        flat_bboxes = []
        for item in bboxes:
            if isinstance(item, (list, tuple)) and item and isinstance(item[0], (list, tuple)):
                flat_bboxes.extend(item)
            else:
                flat_bboxes.append(item)

        for bbox in flat_bboxes:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                print(f"Warning: Skipping invalid bbox item: {bbox}")
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            x1_exp = x1 - dilation
            y1_exp = y1 - dilation
            x2_exp = x2 + dilation
            y2_exp = y2 + dilation
            
            crop_region = [x1_exp, y1_exp, x2_exp, y2_exp]
            crop_w = x2_exp - x1_exp
            crop_h = y2_exp - y1_exp
            
            if crop_h <= 0 or crop_w <= 0:
                print(f"Warning: Skipping bbox with invalid expanded size: {crop_region}")
                continue
            
            local_mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
            local_x1 = dilation
            local_y1 = dilation
            local_x2 = min(crop_w, local_x1 + (x2 - x1))
            local_y2 = min(crop_h, local_y1 + (y2 - y1))
            local_mask_np[local_y1:local_y2, local_x1:local_x2] = 1.0
            
            if feather > 0:
                local_mask_np = gaussian_filter(local_mask_np, sigma=feather)
                
            cropped_mask_np = local_mask_np
            cropped_img_padded = torch.zeros((crop_h, crop_w, 3), dtype=image.dtype, device=image.device)
            
            src_x_start = max(0, x1_exp)
            src_y_start = max(0, y1_exp)
            src_x_end = min(width, x2_exp)
            src_y_end = min(height, y2_exp)
            
            dst_x_start = src_x_start - x1_exp
            dst_y_start = src_y_start - y1_exp
            dst_x_end = src_x_end - x1_exp
            dst_y_end = src_y_end - y1_exp
            
            if src_x_end > src_x_start and src_y_end > src_y_start:
                source_crop = image_for_cropping[src_y_start:src_y_end, src_x_start:src_x_end, :]
                cropped_img_padded[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = source_crop
                
            cropped_image_tensor = cropped_img_padded.permute(2, 0, 1).unsqueeze(0)
            
            seg = SEG(
                cropped_image=cropped_image_tensor,
                cropped_mask=cropped_mask_np,
                confidence=np.array([0.9], dtype=np.float32),
                crop_region=crop_region,
                bbox=np.array(bbox, dtype=np.float32),
                label="bbox"
            )
            
            seg_list.append(seg)
            
        segs = (mask_shape, seg_list)
        
        return (segs,)
    
class bbox_to_mask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image, dilation, feather):
        masks = []
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)
        combined_full_mask = torch.zeros(mask_shape, dtype=torch.float32, device=image.device)
        
        flat_bboxes = []
        for item in bboxes:
            if isinstance(item, (list, tuple)) and item and isinstance(item[0], (list, tuple)):
                flat_bboxes.extend(item)
            else:
                flat_bboxes.append(item)

        for bbox in flat_bboxes:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                print(f"Warning: Skipping invalid bbox item: {bbox}")
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            x1_exp = x1 - dilation
            y1_exp = y1 - dilation
            x2_exp = x2 + dilation
            y2_exp = y2 + dilation
            crop_w = x2_exp - x1_exp
            crop_h = y2_exp - y1_exp
            
            if crop_h <= 0 or crop_w <= 0:
                continue
            
            local_mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
            local_x1 = dilation
            local_y1 = dilation
            local_x2 = min(crop_w, local_x1 + (x2 - x1))
            local_y2 = min(crop_h, local_y1 + (y2 - y1))
            local_mask_np[local_y1:local_y2, local_x1:local_x2] = 1.0
            
            if feather > 0:
                local_mask_np = gaussian_filter(local_mask_np, sigma=feather)
                
            current_full_mask_np = np.zeros(mask_shape, dtype=np.float32)
            x1_c, y1_c = max(0, x1_exp), max(0, y1_exp)
            x2_c, y2_c = min(width, x2_exp), min(height, y2_exp)
            
            if x2_c > x1_c and y2_c > y1_c:
                src_x1, src_y1 = max(0, -x1_exp), max(0, -y1_exp)
                src_x2 = src_x1 + (x2_c - x1_c)
                src_y2 = src_y1 + (y2_c - y1_c)
                current_full_mask_np[y1_c:y2_c, x1_c:x2_c] = local_mask_np[src_y1:src_y2, src_x1:src_x2]
                
            current_full_mask_tensor = torch.from_numpy(current_full_mask_np).to(image.device)
            combined_full_mask = torch.maximum(combined_full_mask, current_full_mask_tensor)
            
        masks.append(combined_full_mask.unsqueeze(0))
        return (torch.cat(masks, dim=0),)

class bboxes_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "bbox_index": ("INT", {
                    "default": 0,
                    "min": -998,
                    "max": 999,
                    "step": 1,
                    "tooltip": "BBox index in the image. Set to 999 to get all bboxes."
                }),
            }
        }
    
    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bbox",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image_index, bbox_index):
        if not bboxes:
            return ([],)

        if isinstance(bboxes[0], (list, tuple)) and bboxes[0] and isinstance(bboxes[0][0], (list, tuple)):
            image_index = min(max(0, image_index), len(bboxes) - 1)
            target_bboxes = bboxes[image_index]
        else:
            target_bboxes = bboxes

        if not target_bboxes:
            return ([],)
        if bbox_index == 999:
            return (target_bboxes,)

        bbox_index = min(max(0, bbox_index), len(target_bboxes) - 1)
        return ([target_bboxes[bbox_index]],)

# from: https://github.com/crystian/ComfyUI-Crystools
class parse_json_node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "key": ("STRING", {"default": ""}),
                "default": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = (any_type, "STRING", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("any", "string", "int", "float", "boolean")
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, input, key="", default=""):
        if isinstance(input, str):
            input_list = [input]
        else:
            input_list = input

        res_any, res_str, res_int, res_float, res_bool = [], [], [], [], []
        for json_str in input_list:
            if not key:
                val = json_str
            else:
                parsed_json = json_str.strip()
                if parsed_json.startswith("```json"):
                    parsed_json = parsed_json[7:]
                if parsed_json.startswith("```"):
                    parsed_json = parsed_json[3:]
                if parsed_json.endswith("```"):
                    parsed_json = parsed_json[:-3]
                val = get_nested_value(parsed_json.strip(), key, default)

            res_any.append(val)
            res_str.append(str(val) if val is not None else "")
            try:
                res_int.append(int(val))
            except Exception:
                res_int.append(0)
            try:
                res_float.append(float(val))
            except Exception:
                res_float.append(0.0)
            try:
                res_bool.append(val if isinstance(val, bool) else str(val).lower() == "true")
            except Exception:
                res_bool.append(False)

        if len(res_any) == 1:
            return (res_any[0], res_str[0], res_int[0], res_float[0], res_bool[0])
        return (res_any, res_str, res_int, res_float, res_bool)

def get_nested_value(data, dotted_key, default=None):
    keys = dotted_key.split('.')
    for key in keys:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                return default
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data

class remove_code_block:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "label": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, input, label=""):
        input_str = "\n".join(input) if isinstance(input, list) else input
        value = input_str.strip()
        if label and value.startswith(f"```{label}"):
            value = value[len(f"```{label}"):]
        elif value.startswith("```"):
            lines = value.split("\n", 1)
            value = lines[1] if len(lines) > 1 else lines[0].removeprefix("```")

        value = value.strip()
        if value.endswith("```"):
            value = value[:-3].strip()
        return (value,)

class PromptEnhancerPreset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (["Qwen-Image [EN]", "Qwen-Image [ZH]", "Qwen-Image 2512 [EN]", "Qwen-Image 2512 [ZH]", "Qwen-Image-Edit", "Qwen-Image-Edit 2509", "Qwen-Image-Edit 2511", "Z-Image Turbo", "Flux.2 T2I", "Flux.2 I2I", "Wan T2V [EN]", "Wan T2V [ZH]", "Wan I2V [EN]", "Wan I2V [ZH]", "Wan I2V Full-Auto [EN]", "Wan I2V Full-Auto [ZH]", "Wan FLF2V [EN]", "Wan FLF2V [ZH]"], )
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "main"
    CATEGORY = "llama-cpp-vlm"
    
    def main(self, preset):
        match preset:
            case "Qwen-Image [EN]":
                return (QWEN_IMAGE_EN,)
            case "Qwen-Image [ZH]":
                return (QWEN_IMAGE_ZH,)
            case "Qwen-Image 2512 [EN]":
                return (QWEN_IMAGE_2512_EN,)
            case "Qwen-Image 2512 [ZH]":
                return (QWEN_IMAGE_2512_ZH,)
            case "Qwen-Image-Edit":
                return (QWEN_IMAGE_EDIT,)
            case "Qwen-Image-Edit 2509":
                return (QWEN_IMAGE_EDIT_2509,)
            case "Qwen-Image-Edit 2511":
                return (QWEN_IMAGE_EDIT_2511,)
            case "Z-Image Turbo":
                return (ZIMAGE_TURBO,)
            case "Flux.2 T2I":
                return (FLUX2_T2I,)
            case "Flux.2 I2I":
                return (FLUX2_I2I,)
            case "Wan T2V [EN]":
                return (WAN_T2V_EN,)
            case "Wan T2V [ZH]":
                return (WAN_T2V_ZH,)
            case "Wan I2V [EN]":
                return (WAN_I2V_EN,)
            case "Wan I2V [ZH]":
                return (WAN_I2V_ZH,)
            case "Wan I2V Full-Auto [EN]":
                return (WAN_I2V_EMPTY_EN,)
            case "Wan I2V Full-Auto [ZH]":
                return (WAN_I2V_EMPTY_ZH,)
            case "Wan FLF2V [EN]":
                return (WAN_FLF2V_EN,)
            case "Wan FLF2V [ZH]":
                return (WAN_FLF2V_ZH,)
            case _:
                raise ValueError(f'Unknown preset: "{preset}"')

class llama_cpp_text_encoder:
    PRESETS = {
        "Boogu-Image": {
            "template": "<|im_start|>system\nYou are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n",
            "layer_idx": -1,
            "trim_template": False,
        },
        "JoyImage": {
            "template": "<|im_start|>system\n \nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "layer_idx": -1,
            "trim_template": True,
        },
        "Lumina2": {
            "template": "{}",
            "layer_idx": -2,
            "trim_template": False,
        },
        "Mage-Flow": {
            "template": "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "layer_idx": -1,
            "trim_template": True,
        },
        "MiniMax-T2V": {
            "template": "{}",
            "layer_idx": 49,
            "trim_template": False,
            "is_minimax_t2v": True,
        },
        "Qwen-Image": {
            "template": "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "layer_idx": -1,
            "trim_template": True,
        },
        "Z-Image": {
            "template": "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "layer_idx": -2,
            "trim_template": False,
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": True}),
                "type": (list(cls.PRESETS), {"default": "Z-Image"}),
                "force_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload the model after encoding.",
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "llama-cpp-vlm"

    def _extract_embeddings(self, llm, target_layer, tokens):
        model = getattr(llm, "_model", None) or getattr(llm, "model", None) or llm
        n_layers = getattr(model, "n_layer", None) or getattr(llm, "n_layer", None)
        if callable(n_layers):
            n_layers = n_layers()
        if target_layer < 0:
            if n_layers is None:
                raise RuntimeError("Unable to determine the model layer count.")
            target_layer += n_layers

        set_target_layer_ids = getattr(model, "set_target_layer_ids", None)
        target_layer_ids = getattr(model, "target_layer_ids", None)
        if callable(set_target_layer_ids):
            set_target_layer_ids([target_layer])
        elif target_layer_ids is not None:
            target_layer_ids = target_layer_ids() if callable(target_layer_ids) else target_layer_ids
            if isinstance(target_layer_ids, list):
                target_layer_ids[:] = [target_layer]

        if hasattr(llm, "n_tokens"):
            llm.n_tokens = 0
        if hasattr(llm._ctx, "kv_cache_clear"):
            llm._ctx.kv_cache_clear()
        elif hasattr(llm._ctx, "memory_clear"):
            llm._ctx.memory_clear(True)

        llm.eval(tokens)
        context = getattr(llm, "ctx", None) or getattr(llm._ctx, "ctx", None)
        n_embd = llm.n_embd() if callable(getattr(llm, "n_embd", None)) else getattr(llm, "n_embd", None)
        if context is None or n_embd is None:
            raise RuntimeError("The loaded llama-cpp-python build does not expose embedding context metadata.")

        embeddings = None
        for function_name in ("llama_get_layer_state", "llama_get_layer_embeddings", "llama_get_layer_output"):
            function = getattr(llama_cpp, function_name, None)
            if function is None:
                continue
            try:
                pointer = function(context, ctypes.c_int(target_layer))
                if pointer:
                    embeddings = np.ctypeslib.as_array(pointer, shape=(len(tokens), n_embd)).copy()
                    break
            except (TypeError, ValueError):
                continue

        if embeddings is None and hasattr(llama_cpp, "llama_get_embeddings_ith"):
            token_embeddings = []
            for index in range(len(tokens)):
                pointer = llama_cpp.llama_get_embeddings_ith(context, ctypes.c_int(index))
                if pointer:
                    token_embeddings.append(np.ctypeslib.as_array(pointer, shape=(n_embd,)).copy())
            if len(token_embeddings) == len(tokens):
                embeddings = np.stack(token_embeddings)

        if embeddings is None:
            raise RuntimeError(f"Unable to extract layer {target_layer} embeddings from this llama-cpp-python build.")

        return torch.from_numpy(embeddings).to(torch.float32).unsqueeze(0)

    def encode(self, llama_model, prompt, type, force_offload):
        if not llama_cpp_version_at_least(MIN_TEXT_ENCODER_LLAMA_CPP_VERSION):
            raise RuntimeError(
                f"Llama-cpp Text Encoder requires llama-cpp-python >= {MIN_TEXT_ENCODER_LLAMA_CPP_VERSION}; "
                f"installed version is {getattr(llama_cpp, '__version__', 'unknown')}."
            )

        if LLAMA_CPP_STORAGE.llm is None or LLAMA_CPP_STORAGE.current_config != llama_model:
            LLAMA_CPP_STORAGE.load_model(llama_model, embedding_mode=True)
        else:
            LLAMA_CPP_STORAGE.ensure_embedding_mode(True)

        config = self.PRESETS[type]
        prompt_text = config["template"].format(prompt)
        tokens = LLAMA_CPP_STORAGE.llm.tokenize(prompt_text.encode("utf-8"), add_bos=False, special=True)
        if not tokens:
            tokens = [151643]
        if config.get("is_minimax_t2v") and len(tokens) > 1 and tokens[0] in (151643, 151644, 1):
            tokens = tokens[1:]

        hidden_tensor = self._extract_embeddings(LLAMA_CPP_STORAGE.llm, config["layer_idx"], tokens)
        if config["trim_template"] and len(tokens) > 3:
            trim_index = 0
            im_start_count = 0
            for index in range(len(tokens) - 2):
                if tokens[index] != 151644:
                    continue
                im_start_count += 1
                if im_start_count == 2:
                    trim_index = index + 3 if tokens[index + 1:index + 3] == [872, 198] else index + 1
                    break
            if 0 < trim_index < hidden_tensor.shape[1]:
                hidden_tensor = hidden_tensor[:, trim_index:, :]

        pooled = torch.zeros(
            (hidden_tensor.shape[0], hidden_tensor.shape[-1]),
            dtype=hidden_tensor.dtype,
            device=hidden_tensor.device,
        )
        conditioning_metadata = {"pooled_output": pooled}
        if config.get("is_minimax_t2v"):
            conditioning_metadata["minimax_token_tags"] = torch.ones(
                hidden_tensor.shape[1], dtype=torch.long, device=hidden_tensor.device
            )

        conditioning = [[hidden_tensor, conditioning_metadata]]
        if force_offload:
            LLAMA_CPP_STORAGE.clean()
        return (conditioning,)

NODE_CLASS_MAPPINGS = {
    "llama_cpp_model_loader": llama_cpp_model_loader,
    "llama_cpp_instruct_adv": llama_cpp_instruct_adv,
    "llama_cpp_multimodal_prompt": llama_cpp_multimodal_prompt,
    "llama_cpp_parameters": llama_cpp_parameters,
    "llama_cpp_unload_model": llama_cpp_unload_model,
    "llama_cpp_clean_states": llama_cpp_clean_states,
    "parse_json_node": parse_json_node,
    "json_to_bbox": json_to_bbox,
    "bbox_to_segs": bbox_to_segs,
    "bbox_to_mask": bbox_to_mask,
    "bboxes_to_bbox": bboxes_to_bbox,
    "remove_code_block": remove_code_block,
    "PromptEnhancerPreset": PromptEnhancerPreset,
    "llama_cpp_text_encoder": llama_cpp_text_encoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_model_loader": "Llama-cpp Model Loader",
    "llama_cpp_instruct_adv": "Llama-cpp Instruct",
    "llama_cpp_multimodal_prompt": "Llama-cpp Multimodal Prompt Builder",
    "llama_cpp_parameters": "Llama-cpp Parameters",
    "llama_cpp_unload_model": "Llama-cpp Unload Model",
    "llama_cpp_clean_states": "Llama-cpp Clean States",
    "parse_json_node": "Parse JSON",
    "json_to_bbox": "JSON to BBoxes",
    "bbox_to_segs": "BBoxes to SEGS",
    "bbox_to_mask": "BBoxes to MASK",
    "bboxes_to_bbox": "BBoxes to BBox",
    "remove_code_block": "Unpack Code Block",
    "PromptEnhancerPreset": "Prompt Enhancer Preset",
    "llama_cpp_text_encoder": "Llama-cpp Text Encoder (BETA)",
}
