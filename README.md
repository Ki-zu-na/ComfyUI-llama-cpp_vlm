# ComfyUI-llama-cpp  
Run LLM/VLM models natively in ComfyUI based on llama.cpp  
**[[📃中文版](./README_zh.md)]** 

## Preview  
![](./img/preview.jpg)

## Installation  

#### Install the node:  
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
#For CUDA 13, please run:
#python -m pip install -r ComfyUI-llama-cpp/requirements_cu131.txt
```

#### Download models:  
- Place your model files in the `ComfyUI/models/LLM` folder.  

	> If you need a VLM model to process image input, don't forget to download the `mmproj` weights.

## Remote llama-server nodes (Bonsai-2 / forked llama.cpp)

Some GGUFs need a patched llama.cpp that llama-cpp-python does not ship. The most common case is
**Bonsai-2 27B** (`PQ2_0` / `PTQ1_0` from `prism-ml/Ternary-Bonsai-2-27B-gguf`), which only loads in the
[PrismML fork](https://github.com/PrismML-Eng/llama.cpp/releases). For these models use the
`llama-cpp-vlm/server` nodes, which drive a `llama-server` process over its OpenAI compatible HTTP API:

| Node | Purpose |
|---|---|
| **Llama-cpp Server** | One node for everything: preset selector, launch/connect mode, model + mmproj, context, thinking switch, sampling. Buttons: **Start/Stop server**, **Save/Delete preset**, **Clear chat history**, plus a live status line. |
| **Llama-cpp Server Multimodal Prompt Builder** | Same as *Llama-cpp Multimodal Prompt Builder* (IMAGE / VIDEO / AUDIO inputs); sampling comes from the server node. |
| **Llama-cpp Server Instruct** | Same as *Llama-cpp Instruct* (one by one / images / video modes). |

Quick start for Bonsai-2:

1. Download `Ternary-Bonsai-2-27B-PQ2_0.gguf` and `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` into `ComfyUI/models/LLM/` (a sub-folder is fine).
2. Download a PrismML fork build (`llama-prism-<tag>-bin-win-cuda-13.3-x64.zip` plus the matching `cudart-*.zip`) and unzip both into one folder.
3. Add **Llama-cpp Server**, set `server_exe` to that `llama-server.exe`, pick the preset *Bonsai-2 27B PQ2_0 · Fast (no thinking)* and press **▶ Start server**.
4. Connect its output to **Llama-cpp Server Multimodal Prompt Builder**. Running the workflow also starts the server automatically when it is not up yet.

`connect` mode skips launching and only talks to `base_url`, so a server started by hand (or on another
machine) works too. `cuda_devices` (advanced) sets `CUDA_VISIBLE_DEVICES` for the server so it can live on a
different GPU than ComfyUI. User presets are stored in `server_presets.json` next to the node.

## Credits  
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai  
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
