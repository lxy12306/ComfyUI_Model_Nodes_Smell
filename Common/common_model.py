import os
import gc
import logging
import inspect
from huggingface_hub import snapshot_download

from .common_func import *

import folder_paths

class CommonModelProcessor:
    def __init__(self):
        self.input_init()

    @classmethod
    def input_init(cls):
        if not hasattr(cls, 'input_init_ok'):
            current_file = inspect.getfile(cls)
            cls.current_dir = os.path.dirname(current_file)
            cls.current_json_dir = os.path.join(cls.current_dir, 'json')
            cls.device_list = ["cpu"]
            if torch.cuda.is_available():
                cls.device_list.append("cuda")
            cls.input_init_ok = True

class CommonModel:
    def __init__(self):
        self.input_init()
        if torch.cuda.is_available() :
            self.device = torch.device("cuda")
            self.device_name = "cuda"
        else:
            self.device = torch.device("cpu")
            self.device_name = "cpu"
        self.bf16_support = is_bf16_supported(self.device)
        self.current_precision = None
        self.current_quantization = None
        self.current_model_name = None
        self.model_dir = ''
        self.exDir = ''
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        
        self.quantization_list = []
        self.model_list = []

        self.__logger = None
        self.input_init_ok = True
        self.been_clear = False

    @classmethod
    def input_init(cls):
        if not hasattr(cls, 'input_init_ok'):
            current_file = inspect.getfile(cls)
            cls.current_dir = os.path.dirname(current_file)
            cls.current_json_dir = os.path.join(cls.current_dir, 'json')
            cls.device_list = ["cpu"]
            if torch.cuda.is_available():
                cls.device_list.append("cuda")

    def logger(self):
        if self.__logger is None:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.__logger = logging.getLogger(type(self).__name__)

        return self.__logger
    
    def bak_input_info(self):
        self.input_info = {
            "base_model_dir": self.model_dir,
            "model": self.current_model_name,
            "precision": self.current_precision,
            "quantization": self.current_quantization,
            "device": self.device_name,
        }
    
    def reload_for_bak(self):
        if self.been_clear and hasattr(self, 'input_info'):
            model_dir = self.input_info.get("base_model_dir", folder_paths.models_dir)
            model = self.input_info.get("model", None)
            precision = self.input_info.get("precision", "float16")
            quantization = self.input_info.get("quantization", "none")
            name = self.input_info.get("device", "cpu")

            self.load_model(
                base_model_dir=model_dir,
                model=model,
                precision=precision,
                quantization=quantization,
                device=name
            )
            self.been_clear = False

    def clear(self):
        self.bak_input_info()
        del self.model
        del self.processor
        self.current_precision = None
        self.current_quantization = None
        self.current_model_name = None
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.been_clear = True

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def reload_model_precheck(self, model, precision, quantization, device):
        self.device = torch.device(device)
        if self.model is not None:
            # Check if we need to reinitialize
            if self.current_model_name == model and \
               self.current_precision == precision and \
               self.current_quantization == quantization:
                self.logger().info(f"Model {model} already loaded, skipping")

                # If we're using CUDA, make sure the model is on CUDA
                if self.device.type == "cuda" and hasattr(self.model, "device") and self.model.device.type != "cuda":
                    self.logger().info(f"Moving model from {self.model.device} to cuda:0")
                    try:
                        self.model = self.model.to("cuda:0")
                    except Exception as e:
                        self.logger().warning(f"Failed to move model to CUDA: {e}")
                        # We'll try again later in analyze_sequence
                return
            else:
                # Clean up old model before loading a new one
                self.logger().info(f"Unloading previous model {getattr(self, 'current_model_name', 'unknown')}")
                self.clear()
                self.current_model_name = model
                self.current_precision = precision
                self.current_quantization = quantization
                self.been_clear = False
        else:
            self.current_model_name = model
            self.current_precision = precision
            self.current_quantization = quantization
    
    def load_model(self, base_model_dir, model, precision, quantization, device):
        self.reload_model_precheck(model, precision, quantization, device)
        self.model_dir = base_model_dir

    def get_dtype(self):
        if self.current_precision == "bfloat16" and self.bf16_support:
            return torch.bfloat16
        elif self.current_precision == "float32":
            return torch.float32
        else:
            return torch.float16

    def load_hg_model(
        self,
        resume_download: bool = True,
        proxies: dict = None
    ) -> str:
        """
        检查指定本地目录下是否存在模型，
        不存在则从Hugging Face下载，
        如下载失败则抛出异常。

        :param resume_download: 是否启用断点续传
        :param proxies: 代理设置，例如 {'http': 'http://...', 'https': 'http://...'}
        :return: 下载或加载的模型本地路径
        """
        print(f"当前模型名称: {self.current_model_name} {self.model_dir} {self.exDir}",)
        model_checkpoint = os.path.join(self.model_dir, self.exDir, os.path.basename(self.current_model_name))
        print(f"模型检查点路径: {model_checkpoint}")

        if not os.path.exists(model_checkpoint):
            print(f"本地未找到模型 '{model_checkpoint}'，尝试从 Hugging Face 下载...")
            try:
                snapshot_download(
                    repo_id=self.current_model_name,
                    local_dir=model_checkpoint,
                    resume_download=resume_download,
                    proxies=proxies,
                    # 如果有需要，可指定分支、标签或提交ID
                    # revision="main",
                )
                print(f"模型已成功下载到 '{model_checkpoint}'")
            except Exception as e:
                raise RuntimeError(f"从 Hugging Face 下载模型失败，错误信息: {e}")
        else:
            print(f"已检测到本地模型目录: {model_checkpoint}")

        return model_checkpoint

    def get_options(self, options):
        ret = {}

        if options is None:
            return ret

        enablers = ['enable_max_new_tokens', 'enable_min_new_tokens',
                    'enable_repeat_penalty', 'enable_temperature',
                    'enable_top_k', 'enable_top_p']

        for enabler in enablers:
            if options[enabler]:
                key = enabler.replace("enable_", "")
                ret[key] = options[key]

        return ret
    
    @classmethod
    def load_model_input(cls, model_list=[], quantization_list=[], device_list=[]):
        return {
            "required": {
                "base_model_dir": ("STRING", {"default": folder_paths.models_dir, "multiline": False}),
                "model": (cls.model_list,),
                "precision": (["bfloat16", "float16", "float32"],),
                "quantization": (cls.quantization_list,),
                "device": (cls.device_list,),
            },
        }