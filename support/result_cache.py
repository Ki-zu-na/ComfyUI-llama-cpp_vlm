"""Persistent LRU cache for inference results.

Both the in-process (llama-cpp-python) nodes and the remote llama-server nodes
use this to skip a generation when every input that influences the output is
unchanged: model/server identity, sampling parameters, prompts, seed and the
exact pixels of every image/frame sent to the model.

The cache is keyed by a SHA-256 over a canonical JSON of those parts and holds
at most ``max_entries`` results (least recently used entries are evicted). It
is persisted as JSON next to the node so hits survive ComfyUI restarts.
"""

import hashlib
import json
import os
import threading
import time

import numpy as np

DEFAULT_MAX_ENTRIES = 64


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def hash_image(tensor):
    """Hash an image tensor ([H,W,C] float 0..1) by its 8-bit pixels."""
    try:
        import torch

        if torch.is_tensor(tensor):
            tensor = tensor.detach().cpu().numpy()
    except ImportError:
        pass
    array = np.clip(np.asarray(tensor) * 255.0, 0, 255).astype(np.uint8)
    array = np.ascontiguousarray(array)
    return f"{array.shape}:{hash_bytes(array.tobytes())}"


def hash_audio(audio):
    """Hash a ComfyUI AUDIO dict ({"waveform": tensor, "sample_rate": int})."""
    if audio is None:
        return None
    waveform = audio["waveform"]
    try:
        import torch

        if torch.is_tensor(waveform):
            waveform = waveform.detach().float().cpu().numpy()
    except ImportError:
        pass
    array = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32))
    return f"{array.shape}@{audio.get('sample_rate')}:{hash_bytes(array.tobytes())}"


def make_key(*parts):
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hash_bytes(payload.encode("utf-8"))


class ResultCache:
    def __init__(self, path, max_entries=DEFAULT_MAX_ENTRIES):
        self.path = path
        self.max_entries = max_entries
        self._entries = {}  # key -> {"value": ..., "time": ...}; insertion order == LRU order
        self._lock = threading.RLock()
        self._loaded = False
        self.hits = 0
        self.misses = 0

    # -- persistence ---------------------------------------------------------

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("entries", []):
                if isinstance(item, dict) and "key" in item:
                    self._entries[item["key"]] = {"value": item.get("value"), "time": item.get("time", 0)}
            if isinstance(data.get("max_entries"), int) and data["max_entries"] > 0:
                self.max_entries = data["max_entries"]
        except (OSError, ValueError) as e:
            print(f"[llama-cpp_vlm] Failed to load result cache: {e}")
            self._entries = {}

    def _save(self):
        data = {
            "max_entries": self.max_entries,
            "entries": [{"key": k, "value": v["value"], "time": v["time"]} for k, v in self._entries.items()],
        }
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"[llama-cpp_vlm] Failed to save result cache: {e}")

    # -- API -------------------------------------------------------------------

    @property
    def enabled(self):
        return self.max_entries > 0

    def set_limit(self, max_entries):
        with self._lock:
            self._ensure_loaded()
            max_entries = max(0, int(max_entries or 0))
            if max_entries == self.max_entries:
                return
            self.max_entries = max_entries
            self._evict()
            self._save()

    def _evict(self):
        while len(self._entries) > max(0, self.max_entries):
            oldest = next(iter(self._entries))
            self._entries.pop(oldest)

    def get(self, key):
        with self._lock:
            self._ensure_loaded()
            if not self.enabled or key not in self._entries:
                self.misses += 1
                return None
            entry = self._entries.pop(key)
            entry["time"] = time.time()
            self._entries[key] = entry  # move to the most-recent end
            self.hits += 1
            return entry["value"]

    def put(self, key, value):
        with self._lock:
            self._ensure_loaded()
            if not self.enabled:
                return
            self._entries.pop(key, None)
            self._entries[key] = {"value": value, "time": time.time()}
            self._evict()
            self._save()

    def clear(self):
        with self._lock:
            self._ensure_loaded()
            self._entries.clear()
            self._save()

    def stats(self):
        with self._lock:
            self._ensure_loaded()
            return {"entries": len(self._entries), "max_entries": self.max_entries, "hits": self.hits, "misses": self.misses}


RESULT_CACHE = ResultCache(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "result_cache.json"))
