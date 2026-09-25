"""Remote llama-server nodes.

These nodes talk to a running ``llama-server`` (OpenAI compatible HTTP API)
instead of loading a GGUF in-process through llama-cpp-python. They exist so
that models which need a forked llama.cpp build (for example Bonsai-2 27B
PQ2_0/PTQ1_0, which needs the PrismML fork) can still be used from ComfyUI
without rebuilding llama-cpp-python.

One node (``Llama-cpp Server``) owns the whole server configuration: launch or
connect mode, model/mmproj, context, sampling and thinking settings, plus
presets. The frontend extension in ``web/js/llama_cpp_server.js`` adds the
Start/Stop button, status line and preset save/delete buttons on top of it,
talking to the HTTP routes registered at the bottom of this file.
"""

import atexit
import gc
import json
import os
import shlex
import subprocess
import threading
import time
from urllib.parse import urlparse

import numpy as np
import requests
import torch

import folder_paths
import comfy.model_management as mm

from .support.cqdm import cqdm
from .support.result_cache import RESULT_CACHE, hash_image, hash_audio, make_key
from .nodes import (
    any_type,
    image2base64,
    preset_prompts,
    preset_tags,
    prompt_builder_audio_url,
    prompt_builder_frames,
    prompt_builder_image_url,
    scale_image,
    select_visible_output,
    split_thinking_chain,
)

LOG_PREFIX = "[llama-cpp_vlm/server]"
SERVER_TYPE = "LLAMACPPSERVER"
REASONING_EFFORTS = ["default", "xhigh", "medium", "low"]
MODES = ["launch", "connect"]
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_EXTRA_ARGS = "-fa on -np 1 --jinja"
DEFAULT_TIMEOUT = 600
CONNECT_TIMEOUT = 10
ROUTE_PREFIX = "/llama_cpp_vlm/server"
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_PRESETS_PATH = os.path.join(NODE_DIR, "server_presets.json")
CUSTOM_PRESET = "Custom"

# llama.cpp treats 0xFFFFFFFF as "random seed", so keep ComfyUI seeds below it.
SEED_MODULUS = 0xFFFFFFFF

SAMPLING_KEYS = ["max_tokens", "temperature", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty"]

# Widgets a preset may set. ``model``/``mmproj`` are matched by substring on the
# frontend because the actual file names depend on the user's models folder.
PRESET_KEYS = [
    "mode", "base_url", "server_exe", "cuda_devices", "n_ctx", "n_gpu_layers", "image_max_tokens", "extra_args",
    "enable_thinking", "reasoning_effort", *SAMPLING_KEYS,
]

BUILTIN_PRESETS = {
    "Bonsai-2 27B PQ2_0 · Fast (no thinking)": {
        "model_match": "Bonsai-2-27B-PQ2_0",
        "mmproj_match": "Bonsai-2-27B-mmproj",
        "mode": "launch",
        "n_ctx": 16384, "n_gpu_layers": 99, "image_max_tokens": 1024, "extra_args": DEFAULT_EXTRA_ARGS,
        "enable_thinking": False, "reasoning_effort": "default",
        "max_tokens": 2048, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
        "repeat_penalty": 1.0, "presence_penalty": 1.5,
    },
    "Bonsai-2 27B PQ2_0 · Thinking (medium)": {
        "model_match": "Bonsai-2-27B-PQ2_0",
        "mmproj_match": "Bonsai-2-27B-mmproj",
        "mode": "launch",
        "n_ctx": 32768, "n_gpu_layers": 99, "image_max_tokens": 1024, "extra_args": DEFAULT_EXTRA_ARGS,
        "enable_thinking": True, "reasoning_effort": "medium",
        "max_tokens": 8192, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.05,
        "repeat_penalty": 1.0, "presence_penalty": 0.0,
    },
    "Bonsai-2 27B PTQ1_0 · Fast (8 GB VRAM)": {
        "model_match": "Bonsai-2-27B-PTQ1_0",
        "mmproj_match": "Bonsai-2-27B-mmproj",
        "mode": "launch",
        "n_ctx": 8192, "n_gpu_layers": 99, "image_max_tokens": 1024, "extra_args": DEFAULT_EXTRA_ARGS,
        "enable_thinking": False, "reasoning_effort": "default",
        "max_tokens": 2048, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
        "repeat_penalty": 1.0, "presence_penalty": 1.5,
    },
    "External server (connect only)": {
        "mode": "connect",
        "base_url": DEFAULT_BASE_URL,
        "enable_thinking": False, "reasoning_effort": "default",
        "max_tokens": 2048, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
        "repeat_penalty": 1.0, "presence_penalty": 0.0,
    },
}


def _log(message):
    print(f"{LOG_PREFIX} {message}")


# --------------------------------------------------------------------------- HTTP helpers

def _join_url(base_url, path):
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def normalize_base_url(base_url):
    base_url = (base_url or "").strip() or DEFAULT_BASE_URL
    if not base_url.startswith(("http://", "https://")):
        base_url = "http://" + base_url
    return base_url.rstrip("/")


def port_from_base_url(base_url):
    parsed = urlparse(normalize_base_url(base_url))
    if parsed.port:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80


def _headers(server):
    headers = {"Content-Type": "application/json"}
    api_key = (server or {}).get("api_key") or ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def probe_server(base_url, api_key="", timeout=CONNECT_TIMEOUT):
    """Return ``(healthy, props)``; raise ConnectionError when unreachable."""
    headers = _headers({"api_key": api_key})
    try:
        health = requests.get(_join_url(base_url, "/health"), headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise ConnectionError(f"Cannot reach llama-server at {base_url}: {e}") from e

    if health.status_code == 503:
        return False, {}
    if health.status_code != 200:
        raise ConnectionError(
            f"llama-server at {base_url} answered /health with HTTP {health.status_code}: {health.text[:200]}"
        )

    props = {}
    try:
        resp = requests.get(_join_url(base_url, "/props"), headers=headers, timeout=timeout)
        if resp.status_code == 200:
            props = resp.json()
    except (requests.RequestException, ValueError):
        pass
    return True, props


def wait_for_server(base_url, api_key="", timeout=600, poll=2.0, process=None):
    """Poll ``/health`` until the model is loaded or ``timeout`` seconds pass."""
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        if mm.processing_interrupted():
            raise mm.InterruptProcessingException()
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {process.returncode}.")
        try:
            healthy, props = probe_server(base_url, api_key)
            if healthy:
                return props
        except ConnectionError as e:
            last_error = e
        time.sleep(poll)
    raise TimeoutError(f"llama-server at {base_url} did not become ready within {timeout}s. Last error: {last_error}")


def describe_props(props):
    if not props:
        return "no /props info"
    model_path = props.get("model_path") or props.get("default_generation_settings", {}).get("model")
    modalities = props.get("modalities") or {}
    parts = []
    if model_path:
        parts.append(f"model={os.path.basename(str(model_path))}")
    if modalities:
        enabled = [name for name, flag in modalities.items() if flag]
        parts.append("modalities=" + (",".join(enabled) if enabled else "text-only"))
    n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
    if n_ctx:
        parts.append(f"n_ctx={n_ctx}")
    slots = props.get("total_slots")
    if slots:
        parts.append(f"slots={slots}")
    return ", ".join(parts) if parts else "no /props info"


def summarize_props(props):
    """Compact dict for the frontend status line."""
    if not props:
        return {}
    model_path = props.get("model_path") or ""
    return {
        "model": os.path.basename(str(model_path)) if model_path else "",
        "vision": bool((props.get("modalities") or {}).get("vision")),
        "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
        "slots": props.get("total_slots"),
    }


def cache_identity(server):
    """The parts of a server config that influence generated text."""
    server = server or {}
    return {
        "served_model": server.get("served_model") or server.get("base_url"),
        "enable_thinking": server.get("enable_thinking", False),
        "reasoning_effort": server.get("reasoning_effort", "default"),
        "sampling": server.get("sampling") or {},
    }


def server_supports_vision(server):
    modalities = (server or {}).get("modalities") or {}
    if not modalities:
        return True  # unknown: let the request decide
    return bool(modalities.get("vision"))


# --------------------------------------------------------------------------- chat completion

def build_payload(server, messages, seed, stream=True):
    payload = {
        "messages": messages,
        "stream": stream,
        "seed": int(seed) % SEED_MODULUS,
        # Put reasoning into ``reasoning_content`` so we do not need to regex the answer.
        "reasoning_format": "deepseek",
    }
    sampling = (server or {}).get("sampling") or {}
    for key in SAMPLING_KEYS:
        value = sampling.get(key)
        if value is None:
            continue
        if key == "max_tokens" and int(value) <= 0:
            continue
        payload[key] = value

    model = (server or {}).get("model") or ""
    if model:
        payload["model"] = model

    template_kwargs = {}
    if not (server or {}).get("enable_thinking", False):
        template_kwargs["enable_thinking"] = False
    else:
        effort = (server or {}).get("reasoning_effort", "default")
        if effort and effort != "default":
            template_kwargs["reasoning_effort"] = effort
    if template_kwargs:
        payload["chat_template_kwargs"] = template_kwargs
    return payload


def _extract_error(response):
    try:
        data = response.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                return err.get("message") or json.dumps(err)
            if err:
                return str(err)
    except ValueError:
        pass
    return response.text[:500]


def server_chat_completion(server, messages, seed):
    """Call ``/v1/chat/completions`` and return ``(content, reasoning)``.

    Streaming is used so that ComfyUI's cancel button interrupts long
    generations instead of waiting for the whole response.
    """
    base_url = server["base_url"]
    timeout = server.get("timeout", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT
    stream = server.get("stream", True)
    payload = build_payload(server, messages, seed, stream=stream)
    url = _join_url(base_url, "/v1/chat/completions")

    try:
        response = requests.post(
            url, headers=_headers(server), data=json.dumps(payload),
            stream=stream, timeout=(CONNECT_TIMEOUT, timeout),
        )
    except requests.RequestException as e:
        raise ConnectionError(f"Request to llama-server failed ({url}): {e}") from e

    if response.status_code != 200:
        raise RuntimeError(f"llama-server returned HTTP {response.status_code}: {_extract_error(response)}")

    if not stream:
        data = response.json()
        message = data["choices"][0]["message"]
        return message.get("content") or "", message.get("reasoning_content") or ""

    content_parts = []
    reasoning_parts = []
    try:
        # Decode ourselves: llama-server sends ``text/event-stream`` without a charset,
        # so requests would otherwise fall back to ISO-8859-1 and garble CJK text.
        for raw_line in response.iter_lines():
            if mm.processing_interrupted():
                raise mm.InterruptProcessingException()
            if not raw_line:
                continue
            if isinstance(raw_line, bytes):
                raw_line = raw_line.decode("utf-8", errors="replace")
            line = raw_line.strip()
            if line.startswith("error:") or line.startswith("event: error"):
                raise RuntimeError(f"llama-server stream error: {line}")
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except ValueError:
                continue
            if "error" in chunk and not chunk.get("choices"):
                err = chunk["error"]
                raise RuntimeError(f"llama-server stream error: {err.get('message', err) if isinstance(err, dict) else err}")
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning_parts.append(delta["reasoning_content"])
    finally:
        response.close()

    return "".join(content_parts), "".join(reasoning_parts)


def finalize_output(content, reasoning, filter_thinking):
    """Return ``(visible, thinking, answer)``.

    ``reasoning`` comes from ``reasoning_content`` when the server parsed it.
    If the server runs with ``--reasoning-format none`` the ``<think>`` block is
    still inside ``content`` so we fall back to splitting it here.
    """
    raw_text = (content or "").removeprefix(": ").lstrip()
    thinking_text, answer_text = split_thinking_chain(raw_text)
    if reasoning:
        thinking_text = (reasoning.strip() + ("\n\n" + thinking_text if thinking_text else "")).strip()
        if not answer_text:
            answer_text = raw_text
    visible = select_visible_output(raw_text, thinking_text, answer_text, filter_thinking)
    if not visible and not filter_thinking:
        visible = answer_text or raw_text
    return visible, thinking_text.strip(), answer_text


def strip_code_fence(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3]
    return text.strip()


# --------------------------------------------------------------------------- presets

def load_user_presets():
    if not os.path.isfile(USER_PRESETS_PATH):
        return {}
    try:
        with open(USER_PRESETS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as e:
        _log(f"Failed to read presets: {e}")
        return {}


def save_user_presets(presets):
    with open(USER_PRESETS_PATH, "w", encoding="utf-8") as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)


def all_presets():
    merged = dict(BUILTIN_PRESETS)
    merged.update(load_user_presets())
    return merged


def preset_names():
    return [CUSTOM_PRESET] + list(all_presets().keys())


# --------------------------------------------------------------------------- managed server process

class ManagedServer:
    """Owns the single llama-server subprocess started from ComfyUI."""

    def __init__(self):
        self.process = None
        self.config = None
        self.base_url = None
        self.log_path = None
        self.last_error = None
        self.lock = threading.RLock()

    # -- state -------------------------------------------------------------

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def read_log_tail(self, lines=40):
        if not self.log_path or not os.path.isfile(self.log_path):
            return ""
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().splitlines()
            return "\n".join(content[-lines:])
        except OSError:
            return ""

    def status(self, base_url=None):
        """Status dict for the frontend."""
        with self.lock:
            running = self.is_running()
            if not running and self.process is not None:
                # Process died: surface the exit code + log tail once.
                code = self.process.returncode
                self.last_error = f"llama-server exited with code {code}"
                self.process = None
                self.config = None
            info = {
                "managed": running,
                "running": running,
                "pid": self.process.pid if running else None,
                "base_url": base_url or self.base_url,
                "healthy": False,
                "props": {},
                "error": self.last_error,
                "log_tail": self.read_log_tail(8) if (running or self.last_error) else "",
            }
        target = info["base_url"]
        if target:
            try:
                healthy, props = probe_server(target, timeout=2)
                info["healthy"] = healthy
                info["props"] = summarize_props(props)
                info["reachable"] = True
            except ConnectionError:
                info["reachable"] = False
        return info

    # -- lifecycle -----------------------------------------------------------

    @staticmethod
    def build_command(config):
        exe = config["server_exe"]
        cmd = [
            exe,
            "-m", config["model_path"],
            "--host", "127.0.0.1",
            "--port", str(config["port"]),
            "-c", str(config["n_ctx"]),
            "-ngl", str(config["n_gpu_layers"]),
            "--reasoning-format", "deepseek",
        ]
        if config.get("mmproj_path"):
            cmd += ["--mmproj", config["mmproj_path"]]
        if config.get("image_max_tokens", 0) > 0:
            cmd += ["--image-max-tokens", str(config["image_max_tokens"])]
        extra = (config.get("extra_args") or "").strip()
        if extra:
            cmd += shlex.split(extra, posix=(os.name != "nt"))
        return cmd

    def start(self, config):
        """Spawn llama-server (non-blocking). Returns ``(started, message)``.

        If a server with the same config is already running nothing happens.
        If a foreign server already answers on the port it is reused.
        """
        with self.lock:
            base_url = f"http://127.0.0.1:{config['port']}"
            if self.is_running() and self.config == config:
                return False, "already running"

            if self.is_running():
                self.stop()
                self._wait_port_free(base_url)
            else:
                try:
                    healthy, props = probe_server(base_url, timeout=2)
                    self.base_url = base_url
                    self.last_error = None
                    return False, f"port {config['port']} already serves a llama-server ({describe_props(props)}); reusing it"
                except ConnectionError:
                    pass

            cmd = self.build_command(config)
            env = os.environ.copy()
            if config.get("cuda_devices"):
                env["CUDA_VISIBLE_DEVICES"] = config["cuda_devices"]

            log_dir = folder_paths.get_temp_directory()
            os.makedirs(log_dir, exist_ok=True)
            self.log_path = os.path.join(log_dir, f"llama_server_{config['port']}.log")
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)

            _log("Launching: " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
            _log(f"Server log: {self.log_path}")
            with open(self.log_path, "w", encoding="utf-8", errors="replace") as log_file:
                self.process = subprocess.Popen(
                    cmd,
                    cwd=os.path.dirname(config["server_exe"]),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
            self.config = config
            self.base_url = base_url
            self.last_error = None
            return True, f"started pid {self.process.pid}"

    def _wait_port_free(self, base_url, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                probe_server(base_url, timeout=1)
                time.sleep(0.5)
            except ConnectionError:
                return
        raise RuntimeError(f"{base_url} is still in use after stopping the previous llama-server.")

    def stop(self):
        with self.lock:
            proc = self.process
            self.process = None
            self.config = None
            if proc is None or proc.poll() is not None:
                return False
            _log(f"Stopping llama-server (pid {proc.pid})...")
            try:
                if os.name == "nt":
                    # Kill the whole tree so wrapper scripts (.cmd/.bat) do not leave the real server orphaned.
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            except Exception as e:
                _log(f"Failed to stop llama-server cleanly: {e}")
            return True

    def ensure_ready(self, config, startup_timeout=600):
        """Blocking: start if needed and wait for /health. Returns props."""
        base_url = f"http://127.0.0.1:{config['port']}"
        self.start(config)
        try:
            return wait_for_server(base_url, timeout=startup_timeout, process=self.process if self.config == config else None)
        except Exception as e:
            tail = self.read_log_tail()
            self.last_error = str(e)
            self.stop()
            raise RuntimeError(f"llama-server failed to start: {e}\n--- log tail ---\n{tail}") from e


MANAGED = ManagedServer()
atexit.register(MANAGED.stop)


def resolve_launch_config(values):
    """Validate widget values and turn them into a launch config dict."""
    server_exe = os.path.expandvars(os.path.expanduser(str(values.get("server_exe", "")).strip().strip('"')))
    if not server_exe:
        raise ValueError("server_exe is empty. Point it at llama-server(.exe) from a PrismML fork build.")
    if os.path.isdir(server_exe):
        candidate = os.path.join(server_exe, "llama-server.exe" if os.name == "nt" else "llama-server")
        if os.path.isfile(candidate):
            server_exe = candidate
    if not os.path.isfile(server_exe):
        raise FileNotFoundError(f"llama-server executable not found: {server_exe}")

    model = values.get("model")
    if not model:
        raise ValueError("No model selected.")
    model_path = folder_paths.get_full_path("LLM", model) or os.path.join(folder_paths.models_dir, "LLM", model)
    mmproj = values.get("mmproj") or "None"
    mmproj_path = None
    if mmproj != "None":
        mmproj_path = folder_paths.get_full_path("LLM", mmproj) or os.path.join(folder_paths.models_dir, "LLM", mmproj)

    return {
        "server_exe": server_exe,
        "model_path": model_path,
        "mmproj_path": mmproj_path,
        "port": port_from_base_url(values.get("base_url", DEFAULT_BASE_URL)),
        "n_ctx": int(values.get("n_ctx", 16384)),
        "n_gpu_layers": int(values.get("n_gpu_layers", 99)),
        "image_max_tokens": int(values.get("image_max_tokens", 1024)),
        "cuda_devices": str(values.get("cuda_devices", "")).strip(),
        "extra_args": str(values.get("extra_args", "")).strip(),
    }


# --------------------------------------------------------------------------- conversation storage

class SERVER_STORAGE:
    messages = {}
    sys_prompts = {}

    @classmethod
    def clean_state(cls, uid=-1):
        if uid in (-1, "-1", None):
            cls.messages.clear()
            cls.sys_prompts.clear()
        else:
            cls.messages.pop(f"{uid}", None)
            cls.sys_prompts.pop(f"{uid}", None)


# --------------------------------------------------------------------------- nodes

class llama_cpp_server:
    """Single node that owns the llama-server connection + generation config."""

    @classmethod
    def INPUT_TYPES(s):
        all_llms = folder_paths.get_filename_list("LLM")
        model_list = [f for f in all_llms if "mmproj" not in f.lower()] or ["None"]
        mmproj_list = ["None"] + [f for f in all_llms if "mmproj" in f.lower()]
        return {
            "required": {
                "preset": (preset_names(), {
                    "default": CUSTOM_PRESET,
                    "tooltip": "Apply a saved configuration to the widgets below. Use the Save/Delete preset buttons to manage your own.",
                }),
                "mode": (MODES, {
                    "default": "launch",
                    "tooltip": "launch: start llama-server from server_exe on this machine.\nconnect: only use an already running server at base_url.",
                }),
                "base_url": ("STRING", {
                    "default": DEFAULT_BASE_URL,
                    "tooltip": "Server URL. In launch mode the port is taken from here and the host is 127.0.0.1.",
                }),
                "server_exe": ("STRING", {
                    "default": "",
                    "tooltip": "Path to llama-server(.exe). For Bonsai-2 use a PrismML fork build "
                               "(github.com/PrismML-Eng/llama.cpp/releases); stock builds reject PQ2_0/PTQ1_0.",
                }),
                "model": (model_list,),
                "mmproj": (mmproj_list, {"default": "None"}),
                "n_ctx": ("INT", {
                    "default": 16384, "min": 1024, "max": 262144, "step": 1024,
                    "tooltip": "Context length (-c). Each image costs up to image_max_tokens of context.",
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Sent per request as chat_template_kwargs. Off is much faster for prompt generation.",
                }),
                "reasoning_effort": (REASONING_EFFORTS, {
                    "default": "default",
                    "tooltip": "Only used when enable_thinking is on. Qwen3.8/Bonsai-2: xhigh (model default), medium, low.",
                }),
                "max_tokens": ("INT", {"default": 2048, "min": 0, "max": 65536, "step": 64, "tooltip": "0 = let the server decide."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 1000, "step": 1}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "n_gpu_layers": ("INT", {"default": 99, "min": 0, "max": 999, "step": 1, "tooltip": "-ngl. 99 = everything on GPU, 0 = CPU."}),
                "image_max_tokens": ("INT", {
                    "default": 1024, "min": 0, "max": 8192, "step": 64,
                    "tooltip": "--image-max-tokens. 0 = uncapped (better OCR, much more VRAM/context per image).",
                }),
                "cuda_devices": ("STRING", {
                    "default": "",
                    "tooltip": "CUDA_VISIBLE_DEVICES for the server process, e.g. \"1\" to keep it off ComfyUI's GPU.",
                }),
                "extra_args": ("STRING", {"default": DEFAULT_EXTRA_ARGS, "tooltip": "Extra llama-server arguments appended verbatim."}),
                "api_key": ("STRING", {"default": "", "tooltip": "Bearer token if the server uses --api-key."}),
                "timeout": ("INT", {"default": DEFAULT_TIMEOUT, "min": 10, "max": 7200, "step": 10, "tooltip": "Seconds to wait for one generation."}),
                "startup_timeout": ("INT", {"default": 600, "min": 30, "max": 3600, "step": 10, "tooltip": "Seconds to wait for the model to load in launch mode."}),
                "cache_size": ("INT", {
                    "default": 64, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "Number of inference results kept in the persistent result cache shared by all llama-cpp nodes (0 = disable). Oldest results are evicted automatically.",
                }),
            },
        }

    RETURN_TYPES = (SERVER_TYPE,)
    RETURN_NAMES = ("llama_server",)
    FUNCTION = "configure"
    CATEGORY = "llama-cpp-vlm/server"

    @classmethod
    def VALIDATE_INPUTS(s, preset):
        # Presets can be created from the frontend without restarting ComfyUI.
        return True

    def configure(self, preset, mode, base_url, server_exe, model, mmproj, n_ctx, enable_thinking, reasoning_effort,
                  max_tokens, temperature, top_p, top_k, min_p, repeat_penalty, presence_penalty, n_gpu_layers,
                  image_max_tokens, cuda_devices, extra_args, api_key, timeout, startup_timeout, cache_size=64):
        RESULT_CACHE.set_limit(cache_size)
        base_url = normalize_base_url(base_url)
        values = {
            "server_exe": server_exe, "model": model, "mmproj": mmproj, "base_url": base_url,
            "n_ctx": n_ctx, "n_gpu_layers": n_gpu_layers, "image_max_tokens": image_max_tokens,
            "cuda_devices": cuda_devices, "extra_args": extra_args,
        }

        if mode == "launch":
            config = resolve_launch_config(values)
            base_url = f"http://127.0.0.1:{config['port']}"
            props = MANAGED.ensure_ready(config, startup_timeout=startup_timeout)
            _log(f"llama-server ready: {describe_props(props)}")
        else:
            healthy, props = probe_server(base_url, api_key.strip())
            if not healthy:
                props = wait_for_server(base_url, api_key.strip(), timeout=startup_timeout)
            _log(f"Connected to {base_url}: {describe_props(props)}")

        server = {
            "base_url": base_url,
            "api_key": api_key.strip(),
            "model": "",
            "enable_thinking": enable_thinking,
            "reasoning_effort": reasoning_effort,
            "timeout": timeout,
            "stream": True,
            "modalities": props.get("modalities") or {},
            # Identity used by the result cache: which model is actually served.
            "served_model": os.path.basename(str(props.get("model_path") or "")) or base_url,
            "sampling": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "repeat_penalty": repeat_penalty,
                "presence_penalty": presence_penalty,
            },
        }
        return (server,)


class llama_cpp_server_multimodal_prompt:
    """Remote twin of ``llama_cpp_multimodal_prompt``."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llama_server": (SERVER_TYPE,),
                "system_prompt": ("STRING", {
                    "default": "You are an expert prompt writer for image and video generation models. Analyze the attached references and the user's request, then produce one precise, production-ready prompt. Output only the final prompt text.",
                    "multiline": True,
                }),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "A batch of reference images. Each image is exposed to the model in batch order."}),
                "video": ("VIDEO", {"tooltip": "A video reference. Frames are sampled evenly and sent as ordered visual references."}),
                "audio": ("AUDIO", {"tooltip": "Optional audio reference (only for servers whose model supports audio)."}),
                "max_images": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "video_max_frames": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
                "image_max_size": ("INT", {
                    "default": 1024, "min": 128, "max": 4096, "step": 64,
                    "tooltip": "Maximum image edge sent to the server. The server may downscale further via --image-max-tokens.",
                }),
                "include_video_audio": ("BOOLEAN", {"default": False}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Keep reasoning out of the prompt output."}),
                "use_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reuse the previous result when server model, sampling, prompts, seed and every image/frame are unchanged. Cache size is set on the server node.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "thinking")
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm/server"

    def process(self, llama_server, system_prompt, user_prompt, seed, images=None, video=None, audio=None,
                max_images=16, video_max_frames=8, image_max_size=1024, include_video_audio=False, filter_thinking=True,
                use_cache=True):
        image_frames = prompt_builder_frames(images, max_images)
        video_frames = []
        video_audio = None
        if video is not None:
            components = video.get_components()
            video_frames = prompt_builder_frames(components.images, video_max_frames)
            video_audio = components.audio

        cache_key = None
        if use_cache and RESULT_CACHE.enabled:
            cache_key = make_key(
                "server_multimodal_prompt", cache_identity(llama_server), system_prompt, user_prompt, seed,
                [hash_image(f) for f in image_frames], [hash_image(f) for f in video_frames],
                hash_audio(video_audio) if include_video_audio else None, hash_audio(audio),
                image_max_size, filter_thinking,
            )
            cached = RESULT_CACHE.get(cache_key)
            if cached is not None:
                _log("Result cache hit, skipping inference.")
                return tuple(cached)

        if (image_frames or video_frames) and not server_supports_vision(llama_server):
            raise ValueError("The connected llama-server reports no vision support. Select an mmproj / start it with --mmproj.")

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
            audio_items.append((f"Audio{len(audio_items) + 1}", "an attached audio reference", audio))
        for label, description, audio_input in audio_items:
            media_descriptions.append(f"{label}: {description}.")
            media_content.extend([
                {"type": "text", "text": f"\nReference {label}:"},
                {"type": "audio_url", "audio_url": {"url": prompt_builder_audio_url(audio_input)}},
            ])

        prompt_parts = []
        if media_descriptions:
            prompt_parts.append(
                "[REFERENCE MEDIA]\n" + "\n".join(media_descriptions)
                + "\nUse these labels when a reference needs to be named in the generated prompt."
            )
        if user_prompt.strip():
            prompt_parts.append("[USER REQUEST]\n" + user_prompt)
        user_text = "\n\n".join(prompt_parts)
        user_content = [{"type": "text", "text": user_text}] + media_content if media_content else user_text

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        content, reasoning = server_chat_completion(llama_server, messages, seed)
        result, thinking_text, _ = finalize_output(content, reasoning, filter_thinking)
        result = strip_code_fence(result)
        if cache_key is not None:
            RESULT_CACHE.put(cache_key, [result, thinking_text])
        gc.collect()
        return (result, thinking_text)


class llama_cpp_server_instruct:
    """Remote twin of ``llama_cpp_instruct_adv``."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llama_server": (SERVER_TYPE,),
                "preset_prompt": (preset_tags, {"default": preset_tags[1]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": 'user_prompt\n\nFor preset hints marked with an "*", this will be used to fill the placeholder (e.g., Object names in BBox detection)\nOtherwise, this will override the preset prompts.'}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "inference_mode": (["one by one", "images", "video"], {
                    "default": "one by one",
                    "tooltip": "one by one: Process every image frame separately\nimages: Combine all image list items into one prompt\nvideo: Treat each image list item as a separate video clip",
                }),
                "max_frames": ("INT", {"default": 24, "min": 2, "max": 1024, "step": 1, "tooltip": 'Frames sampled evenly from each clip ("video" mode).'}),
                "max_size": ("INT", {"default": 256, "min": 128, "max": 16384, "step": 64, "tooltip": 'Max image edge in "images" and "video" modes.'}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "save_states": ("BOOLEAN", {"default": False, "tooltip": "Keep this node's conversation history in RAM (clear it with the button on the server node)."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Hide reasoning in the main output. The `thinking` and `answer` outputs still keep the split result."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
            "optional": {
                "images": ("IMAGE",),
                "queue_handler": (any_type, {"tooltip": "Used to control the execution order of instruct nodes."}),
                "use_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reuse the previous result when server model, sampling, prompts, seed and every image are unchanged (ignored when save_states is on). Cache size is set on the server node.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("output", "output_list", "thinking", "answer", "state_uid")
    OUTPUT_IS_LIST = (False, True, False, False, False)
    INPUT_IS_LIST = True
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm/server"

    @staticmethod
    def sanitize_messages(messages):
        clean = json.loads(json.dumps(messages))
        for msg in clean:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        item["image_url"]["url"] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAACXBIWXMAAAsTAAALEwEAmpwYAAAADElEQVQImWP4//8/AAX+Av5Y8msOAAAAAElFTkSuQmCC"
        return clean

    @staticmethod
    def _first(value):
        return value[0] if isinstance(value, list) else value

    def process(self, llama_server, preset_prompt, custom_prompt, system_prompt, inference_mode, max_frames, max_size,
                seed, save_states, filter_thinking, unique_id, images=None, queue_handler=None, use_cache=True):
        first = self._first
        use_cache = first(use_cache)
        llama_server = first(llama_server)
        preset_prompt = first(preset_prompt)
        custom_prompt = first(custom_prompt)
        system_prompt = first(system_prompt)
        inference_mode = first(inference_mode)
        max_frames = first(max_frames)
        max_size = first(max_size)
        seed = first(seed)
        save_states = first(save_states)
        filter_thinking = first(filter_thinking)
        unique_id = first(unique_id)
        uid = str(unique_id).rpartition(".")[-1]

        video_input = inference_mode == "video"
        system_prompts = "请将输入的图片序列当做视频而不是静态帧序列, " + system_prompt if video_input else system_prompt
        last_sys_prompt = SERVER_STORAGE.sys_prompts.get(f"{uid}", None)
        if last_sys_prompt != system_prompts:
            messages = []
            SERVER_STORAGE.clean_state(uid)
            SERVER_STORAGE.sys_prompts[f"{uid}"] = system_prompts
            if system_prompts.strip():
                messages.append({"role": "system", "content": system_prompts})
        elif save_states:
            _log(f"Loading history id={uid}...")
            messages = list(SERVER_STORAGE.messages.get(f"{uid}", []))
        else:
            messages = []
            if system_prompts.strip():
                messages.append({"role": "system", "content": system_prompts})

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

        cache_key = None
        if use_cache and not save_states and RESULT_CACHE.enabled:
            cache_key = make_key(
                "server_instruct", cache_identity(llama_server), preset_prompt, custom_prompt, system_prompt,
                inference_mode, max_frames, max_size, seed, filter_thinking,
                [[hash_image(frame) for frame in group] for group in image_groups],
            )
            cached = RESULT_CACHE.get(cache_key)
            if cached is not None:
                _log("Result cache hit, skipping inference.")
                out1, out2, thinking_out, answer_out = cached
                return (out1, list(out2), thinking_out, answer_out, uid)

        if image_groups and not server_supports_vision(llama_server):
            raise ValueError("Image input detected, but the connected llama-server has no vision projector (mmproj).")

        def run(current_messages):
            content, reasoning = server_chat_completion(llama_server, current_messages, seed)
            return finalize_output(content, reasoning, filter_thinking)

        def image_part(image):
            data = image2base64(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}

        def scaled_image_part(image):
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image2base64(scale_image(image, max_size))}"}}

        def run_batch(items, label_prefix, build_content):
            tmp_list, thinking_acc, answer_acc, state_content = [], "", "", user_content
            for index, item in enumerate(cqdm(items), start=1):
                if mm.processing_interrupted():
                    raise mm.InterruptProcessingException()
                current_user_content = json.loads(json.dumps(user_content)) + build_content(item)
                state_content = current_user_content
                text, thinking_text, answer_text = run(messages + [{"role": "user", "content": current_user_content}])
                out2.append(text)
                if len(items) > 1:
                    label = f"====== {label_prefix} {index} ======"
                    tmp_list.extend([label, text])
                    if thinking_text:
                        thinking_acc += ("" if not thinking_acc else "\n\n") + f"{label}\n{thinking_text}"
                    answer_acc += ("" if not answer_acc else "\n\n") + f"{label}\n{answer_text or text}"
                else:
                    tmp_list.append(text)
                    thinking_acc = thinking_text
                    answer_acc = answer_text or text
            messages.append({"role": "user", "content": state_content})
            return "\n\n".join(tmp_list), thinking_acc, answer_acc

        out2 = []
        if image_groups and inference_mode == "one by one":
            frames = [frame for group in image_groups for frame in group]
            _log(f"Start processing {len(frames)} images")
            out1, thinking_out, answer_out = run_batch(frames, "Image", lambda image: [image_part(image)])

        elif image_groups and inference_mode == "images":
            frames = [frame for group in image_groups for frame in group]
            for image in frames:
                user_content.append(scaled_image_part(image) if len(frames) > 1 else image_part(image))
            messages.append({"role": "user", "content": user_content})
            out1, thinking_out, answer_out = run(messages)
            out2 = [out1]

        elif image_groups and inference_mode == "video":
            _log(f"Start processing {len(image_groups)} video clips")

            def clip_content(video_frames):
                sample_count = min(len(video_frames), max_frames)
                indices = np.linspace(0, len(video_frames) - 1, sample_count, dtype=int)
                return [scaled_image_part(video_frames[i]) for i in indices]

            out1, thinking_out, answer_out = run_batch(image_groups, "Video Clip", clip_content)

        else:
            messages.append({"role": "user", "content": user_content})
            out1, thinking_out, answer_out = run(messages)
            out2 = [out1]

        if not answer_out:
            answer_out = out1

        if save_states:
            _log(f"Saving history id={uid}...")
            messages.append({"role": "assistant", "content": out1})
            SERVER_STORAGE.messages[f"{uid}"] = self.sanitize_messages(messages)
        elif not SERVER_STORAGE.messages.get(f"{uid}"):
            SERVER_STORAGE.sys_prompts.pop(f"{uid}", None)

        if cache_key is not None:
            RESULT_CACHE.put(cache_key, [out1, list(out2), thinking_out, answer_out])
        del messages
        gc.collect()
        return (out1, out2, thinking_out, answer_out, uid)


# --------------------------------------------------------------------------- HTTP routes for the frontend

def register_routes(routes):
    """Register the frontend API on an aiohttp RouteTableDef."""
    from aiohttp import web

    def ok(data=None, **extra):
        payload = {"ok": True}
        if data:
            payload.update(data)
        payload.update(extra)
        return web.json_response(payload)

    def fail(message, status=400):
        return web.json_response({"ok": False, "error": str(message)}, status=status)

    @routes.get(f"{ROUTE_PREFIX}/status")
    async def status(request):
        base_url = request.query.get("base_url") or None
        if base_url:
            base_url = normalize_base_url(base_url)
        return ok(MANAGED.status(base_url), cache=RESULT_CACHE.stats())

    @routes.post(f"{ROUTE_PREFIX}/start")
    async def start(request):
        try:
            values = await request.json()
            config = resolve_launch_config(values)
            started, message = MANAGED.start(config)
        except Exception as e:
            return fail(e)
        return ok(started=started, message=message, base_url=MANAGED.base_url)

    @routes.post(f"{ROUTE_PREFIX}/stop")
    async def stop(request):
        stopped = MANAGED.stop()
        return ok(stopped=stopped)

    @routes.post(f"{ROUTE_PREFIX}/clear_states")
    async def clear_states(request):
        SERVER_STORAGE.clean_state(-1)
        return ok()

    @routes.post(f"{ROUTE_PREFIX}/clear_cache")
    async def clear_cache(request):
        RESULT_CACHE.clear()
        return ok(cache=RESULT_CACHE.stats())

    @routes.get(f"{ROUTE_PREFIX}/presets")
    async def get_presets(request):
        return ok(builtin=BUILTIN_PRESETS, user=load_user_presets(), keys=PRESET_KEYS)

    @routes.post(f"{ROUTE_PREFIX}/presets")
    async def save_preset(request):
        try:
            body = await request.json()
            name = str(body.get("name", "")).strip()
            values = body.get("values") or {}
            if not name or name == CUSTOM_PRESET:
                return fail("Preset name is empty or reserved.")
            if name in BUILTIN_PRESETS:
                return fail("Built-in presets cannot be overwritten; choose another name.")
            preset = {k: values[k] for k in PRESET_KEYS if k in values}
            for key in ("model", "mmproj"):
                if values.get(key) and values[key] != "None":
                    preset[f"{key}_match"] = values[key]
            presets = load_user_presets()
            presets[name] = preset
            save_user_presets(presets)
        except Exception as e:
            return fail(e)
        return ok(names=preset_names())

    @routes.post(f"{ROUTE_PREFIX}/presets/delete")
    async def delete_preset(request):
        try:
            body = await request.json()
            name = str(body.get("name", "")).strip()
            if name in BUILTIN_PRESETS:
                return fail("Built-in presets cannot be deleted.")
            presets = load_user_presets()
            if name not in presets:
                return fail(f"Unknown preset: {name}", status=404)
            presets.pop(name)
            save_user_presets(presets)
        except Exception as e:
            return fail(e)
        return ok(names=preset_names())


try:
    from server import PromptServer

    if getattr(PromptServer, "instance", None) is not None:
        register_routes(PromptServer.instance.routes)
except Exception as e:  # pragma: no cover - only when imported outside ComfyUI
    _log(f"HTTP routes not registered: {e}")


NODE_CLASS_MAPPINGS = {
    "llama_cpp_server": llama_cpp_server,
    "llama_cpp_server_instruct": llama_cpp_server_instruct,
    "llama_cpp_server_multimodal_prompt": llama_cpp_server_multimodal_prompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_server": "Llama-cpp Server",
    "llama_cpp_server_instruct": "Llama-cpp Server Instruct",
    "llama_cpp_server_multimodal_prompt": "Llama-cpp Server Multimodal Prompt Builder",
}
