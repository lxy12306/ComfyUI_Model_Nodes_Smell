# ComfyUI Qwen2.5VL

## 更新日志
- 支持 Qwen2.5VL_IMAGE 节点
- 支持 Qwen2.5VL_VIDEO 节点
- 支持 JoyCaptionbetaone 节点

## 目录结构

```
自定义目录/LLM/
├── QwenVL/                # Qwen2.5VL相关节点及模型配置
│   ├── nodes.py
│   ├── json/
│   │   └── models.json
│   └── ...
├── JoyCaption/            # JoyCaption相关节点
│   └── node.py
├── Common/                # 通用选项节点
│   └── option.py
├── __init__.py            # 节点注册入口
├── README.md
└── requirements.txt
```

## 安装方法

1. **克隆本仓库到 `ComfyUI/custom_nodes/` 目录下：**
    ```bash
    git clone https://github.com/lxy12306/ComfyUI_Model_Nodes_Smell.git
    ```

2. **进入插件目录：**
    ```bash
    cd ComfyUI_Model_Nodes_Smell
    ```

3. **安装依赖：**
    ```bash
    pip install -r requirements.txt
    pip install git+https://github.com/huggingface/transformers
    # 需要最新版transformers，官方包尚未发布
    ```

4. **下载模型（任选其一，放到合适的模型目录（自定义目录/LLM））：**
    - [Qwen2.5-VL-7B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct)
    - [Qwen2.5-VL-3B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct)
    - [Qwen2.5-VL-72B-Instruct-AWQ](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-72B-Instruct-AWQ)
    - [llama-joycaption-beta-one-hf-llava](https://modelscope.cn/models/fancyfeast/llama-joycaption-beta-one-hf-llava)

## 使用说明

- 启动 ComfyUI 后，相关节点会自动出现在节点面板中。
- 可在 `QwenVL`、`JoyCaption`、`Common` 分类下找到对应节点。

## 其他说明

- 节点注册入口为 `__init__.py`，自动加载各模块节点。

如有问题欢迎提交 issue。