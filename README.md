# ComfyUI Qwen2.5VL
## Update
Suppport Qwen2.5VL_IMAGE in ComfyUi
Suppport Qwen2.5VL_VEDIO in ComfyUi

## Installation

To install ComfyUI_Model_Nodes_Smell in `ComfyUI\custom_nodes\`, follow these steps:

1. *Clone the repository*:
    ```bash
    git clone https://github.com/lxy12306/ComfyUI_Model_Nodes_Smell.git
    ```

2. *Navigate to the cloned directory*:
    ```bash
    cd ComfyUI_Model_Nodes_Smell
    ```

3. *Install the required dependencies*:
    ```bash
    pip install -r requirements.txt
    git+https://github.com/huggingface/transformers #Need the latest transformers; this version has not yet released the Python package.
    ```

4. *Download the model (choose one of the following)*:
    ```bash
    https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct
    https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct
    https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-72B-Instruct-AWQ
    ```