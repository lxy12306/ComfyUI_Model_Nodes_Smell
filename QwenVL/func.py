import logging
import os
import torch
import gc
import inspect

try:
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    QWEN_AVAILABLE = True
except ImportError:
    logging.warning("Transformers package not found. Please install with: pip install transformers torch")
    QWEN_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cuda_vram_last = 0
cpu_vram_last = 0

from ..Common.common_func import *
from ..Common.image_func import *
from ..Common.common_model import *

current_dir = os.path.dirname(os.path.abspath(__file__))
current_json_dir = os.path.join(current_dir, 'json')

def load_system_prompt(file_name, en=True):
    json_data = read_json_file(current_json_dir, file_name)
    if json_data == {} :
        return None

    if en:
        json_data = json_data["en"]
    else :
        json_data = json_data["中文"]
    keys = ["None"]
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            keys.append(key)

    return keys, json_data

class Qwen2_5VL(CommonModel):
    def __init__(self):
        super().__init__()
        global cuda_vram_last
        global cpu_vram_last

        if torch.cuda.is_available() :
            import pynvml
            pynvml.nvmlInit()
            self.deviceHandle = pynvml.nvmlDeviceGetHandleByIndex(0)
            if (cuda_vram_last == 0):
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.deviceHandle)
                self.vram = int((mem.total - mem.used) * 0.95 / (1024 ** 2))
                cuda_vram_last = self.vram
            else :
                self.vram = cuda_vram_last
            logging.info(f"{self.device} 可使用内存: {self.vram :.2f} MB")
        else :
            import psutil
            self.deviceHandle = -1
            if cpu_vram_last == 0:
                mem = psutil.virtual_memory()
                self.vram = int((mem.total - mem.used) * 0.95 / (1024 ** 2))
                cpu_vram_last = self.vram
            else :
                self.vram = cpu_vram_last
            logging.info(f"{self.device} 可使用内存: {self.vram :.2f} MB")

    def clear(self):
        global cuda_vram_last
        global cpu_vram_last
        super().clear()
        cuda_vram_last = 0
        cpu_vram_last = 0

    def select_system_prompt(self, key, json_data):
        data = json_data.get(key, {})
        instruction = data.get("instruction", "")
        rules = json_data.get("rules", [])

        if instruction and rules:
            return instruction + "\n\n" + "\n".join([f"- {rule}" for rule in rules])
        elif instruction:
            return instruction

    def load_model(self, base_model_dir, model_name, use_shared_gpu_mem, precision="float16", quantization=None):

        if not QWEN_AVAILABLE:
            raise ImportError("Transformers package not found. Please install with: pip install transformers torch")

        if self.model is not None:
            # Check if we need to reinitialize
            if getattr(self, "current_model_name", None) == model_name and \
               getattr(self, "current_precision", None) == precision:
                logger.info(f"Model {model_name} already loaded, skipping")

                # If we're using CUDA, make sure the model is on CUDA
                if self.device == "cuda" and hasattr(self.model, "device") and self.model.device.type != "cuda":
                    logger.info(f"Moving model from {self.model.device} to cuda:0")
                    try:
                        self.model = self.model.to("cuda:0")
                    except Exception as e:
                        logger.warning(f"Failed to move model to CUDA: {e}")
                        # We'll try again later in analyze_sequence
                return
            else:
                # Clean up old model before loading a new one
                logger.info(f"Unloading previous model {getattr(self, 'current_model_name', 'unknown')}")
                self.clear()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()

        model_id = f"qwen/{model_name}"
        # put downloaded model to model/LLM dir
        self.model_checkpoint = os.path.join(
            base_model_dir, "LLM", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            raise FileNotFoundError(
                f"模型检查点 '{self.model_checkpoint}' 不存在。请使用以下命令下载模型:\n"
                f"modelscope download --model Qwen/{model_name} --local_dir {self.model_checkpoint}"
            )

        if precision == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif precision == "float32":
            dtype = torch.float32
        else:
            dtype = torch.float16

        can_use_flash_attn = False
        if self.device == "cuda":
            try:
                from flash_attn import flash_attn_func
                can_use_flash_attn = True
                logger.info("Flash attention available, will use for better performance")
            except ImportError:
                pass

        if self.processor is None:
            min_pixels = 256 * 256
            max_pixels = 512 * 512 * 2

            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        model_kwargs = {
            "torch_dtype": dtype,
            "local_files_only": True,
        }
        if can_use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        if use_shared_gpu_mem and self.deviceHandle != -1:
            self.vram *= 2
            logger.info(f"Set maximum memory to {self.vram} MB")
            model_kwargs["max_memory"] = {0: f"{self.vram}MB"}

        if quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,)
        elif quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True,)

        # Configure device mapping based on hardware
        if self.device == "cuda":
            # Always use a single GPU setup to avoid device conflicts
            logger.info("Using single GPU configuration to avoid device conflicts")
            # Don't use device_map at all - load to a specific device instead
            model_kwargs.pop("device_map", None)
        else:
            # For CPU or MPS, use the device directly
            logger.info(f"Using {self.device} device")
            model_kwargs["device_map"] = {"": self.device}

        if "AWQ" in model_name:
            logger.info("Loading AWQ quantized model")
            model_kwargs.pop("quantization_config", None)
            model_kwargs["device_map"] = {"": self.device}

        if self.model is None:
            # Load the model on the available device(s)

            # 获取函数的签名
            signature = inspect.signature(Qwen2_5_VLForConditionalGeneration.from_pretrained)

            # 列出所有参数
            for param in signature.parameters.values():
                logger.info(f"参数名: {param.name}, 默认值: {param.default}, 类型: {param.annotation}")

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                **model_kwargs
            )

        self.current_model_name = model_name
        self.current_precision = precision

        # For CUDA, explicitly move model to cuda:0 after loading
        if self.device == "cuda" and not hasattr(self.model, "device_map"):
                logger.info("Moving model to cuda:0")
                self.model = self.model.to("cuda:0")