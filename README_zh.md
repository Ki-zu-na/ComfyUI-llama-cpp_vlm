# ComfyUI-llama-cpp
在 ComfyUI 中基于 llama.cpp 框架原生运行 LLM & VLM 模型。  
**[[📃English](./README.md)]**   

## 预览
![](./img/preview.jpg) 

## 安装步骤

#### 安装节点:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
#CUDA 13用户请执行:
#python -m pip install -r ComfyUI-llama-cpp/requirements_cu131.txt
```

### 模型路径:
- 请将下载的 `.gguf` 模型放置在 `ComfyUI/models/LLM` 目录中.  

	> 在使用VLM模型进行图像推理之前, 请确保已经下载并选择了主模型对应的`mmproj`权重文件.

## 远程 llama-server 节点（Bonsai-2 / llama.cpp 分支）

有些 GGUF 需要 llama-cpp-python 没有集成的 llama.cpp 补丁，最典型的是 **Bonsai-2 27B**
（`prism-ml/Ternary-Bonsai-2-27B-gguf` 的 `PQ2_0` / `PTQ1_0`），它只能在
[PrismML 分支](https://github.com/PrismML-Eng/llama.cpp/releases) 里加载。这类模型请使用
`llama-cpp-vlm/server` 分类下的节点，它们通过 OpenAI 兼容的 HTTP 接口驱动一个 `llama-server` 进程：

| 节点 | 作用 |
|---|---|
| **Llama-cpp Server** | 一个节点搞定全部：预设选择、launch/connect 模式、模型 + mmproj、上下文、思考开关、采样参数。按钮：**启动/停止服务**、**保存/删除预设**、**清理对话历史**，并有实时状态行。 |
| **Llama-cpp Server Multimodal Prompt Builder** | 与 *Llama-cpp Multimodal Prompt Builder* 一致（IMAGE / VIDEO / AUDIO 输入），采样参数直接来自服务节点。 |
| **Llama-cpp Server Instruct** | 与 *Llama-cpp Instruct* 一致（one by one / images / video 模式）。 |

Bonsai-2 快速上手：

1. 把 `Ternary-Bonsai-2-27B-PQ2_0.gguf` 和 `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` 放到 `ComfyUI/models/LLM/`（可以放子目录）。
2. 下载 PrismML 分支预编译包（`llama-prism-<tag>-bin-win-cuda-13.3-x64.zip` 以及配套的 `cudart-*.zip`），解压到同一目录。
3. 添加 **Llama-cpp Server** 节点，`server_exe` 填该目录下的 `llama-server.exe`，预设选 *Bonsai-2 27B PQ2_0 · Fast (no thinking)*，点 **▶ Start server**。
4. 把输出连到 **Llama-cpp Server Multimodal Prompt Builder**。直接运行工作流时如果服务没起来也会自动启动。

`connect` 模式不启动进程，只连接 `base_url`，手动启动或在其他机器上的服务也能用。高级项 `cuda_devices`
会给服务进程设置 `CUDA_VISIBLE_DEVICES`，可让它和 ComfyUI 分别占用不同显卡。用户预设保存在节点目录下的 `server_presets.json`。

## 致谢
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
