import os
import torch
import inspect

import folder_paths

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info


from ..Common.common_func import *
from ..Common.image_func import *


current_dir = os.path.dirname(os.path.abspath(__file__))
models_json_file_path = os.path.join(current_dir, 'json', 'models.json')
cuda_vram_last = 0
cpu_vram_last = 0


class Qwen2_5VL_IMAGE:
    def __init__(self):
        global cuda_vram_last
        global cpu_vram_last
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        if torch.cuda.is_available() :
            import pynvml
            pynvml.nvmlInit()
            self.device = torch.device("cuda")
            self.device_name = "cuda"
            self.deviceHandle = pynvml.nvmlDeviceGetHandleByIndex(0)
            if (cuda_vram_last == 0):
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.deviceHandle)
                self.vram = int((mem.total - mem.used) * 0.95 / (1024 ** 2))
                cuda_vram_last = self.vram
            else :
                self.vram = cuda_vram_last

            print(f"{self.device} 可使用内存: {self.vram :.2f} MB")
        else :
            import psutil

            self.device = torch.device("cpu")
            self.device_name = "cpu"
            self.deviceHandle = -1
            if cpu_vram_last == 0:
                mem = psutil.virtual_memory()
                self.vram = int((mem.total - mem.used) * 0.95 / (1024 ** 2))
                cpu_vram_last = self.vram
            else :
                self.vram = cpu_vram_last
            print(f"{self.device} 可使用内存: {self.vram :.2f} MB")

        self.bf16_support = is_bf16_supported(self.device)

    @classmethod
    def INPUT_TYPES(s):
        models = read_json_value(models_json_file_path, "models", list)
        return {
            "required": {
                "model_dir": ("STRING", {"default": os.path.join(folder_paths.models_dir, "LLM"), "multiline": False}),
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (models,),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),  # add quantization type selection
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1}),  # add seed parameter, default is -1
            },
            "optional": {
                "image": ("IMAGE",),
                "options": ("MODLE_OPTIONS",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"


    def get_options(self, options):
        ret = None

        if options is None:
            return ret

        enablers = ['enable_max_new_tokens', 'enable_min_new_tokens',
                    'enable_repeat_penalty', 'enable_temperature',
                    'enable_top_k', 'enable_top_p']

        for enabler in enablers:
            if options[enabler]:
                if ret is None:
                    ret = {}
                key = enabler.replace("enable_", "")
                ret[key] = options[key]

        return ret


    def inference(self, model_dir, text, model, quantization, keep_model_loaded, seed, options=None, image=None):
        if seed != -1:
            torch.manual_seed(seed)

        real_options = self.get_options(options)

        model_id = f"qwen/{model}"
        # put downloaded model to model/LLM dir
        self.model_checkpoint = os.path.join(
            model_dir, os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            raise FileNotFoundError(
                f"模型检查点 '{self.model_checkpoint}' 不存在。请使用以下命令下载模型:\n"
                f"modelscope download --model Qwen/{model} --local_dir {self.model_checkpoint}"
            )

        if self.processor is None:
            # Define min_pixels and max_pixels:
            # Images will be resized to maintain their aspect ratio
            # within the range of min_pixels and max_pixels.
            min_pixels = 256*256
            max_pixels = 1024*1024*4

            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        if self.model is None:
            # Load the model on the available device(s)
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None

        if model == "Qwen2.5-VL-72B-Instruct-AWQ" :
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_checkpoint, torch_dtype="auto", device_map={"": self.device_name})
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
            )

        with torch.no_grad():
            if torch.is_tensor(image):
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                    raise ValueError(image.shape)

                pil_image = tensor_to_pil(image)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": pil_image,
                            },
                            {"type": "text", "text": text},
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                        ],
                    }
                ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            print(real_options)

            try:
                generated_ids = self.model.generate(
                    **inputs, **real_options
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                result = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            except Exception as e:
                return (f"Error during model inference: {str(e)}",)

            if not keep_model_loaded:
                del self.processor
                del self.model
                self.processor = None
                self.model = None
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            return result


class Qwen2_5VL_VEDIO:
    def __init__(self):
        global cuda_vram_last
        global cpu_vram_last
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        if torch.cuda.is_available() :
            import pynvml
            pynvml.nvmlInit()
            self.device = torch.device("cuda")
            self.device_name = "cuda"
            self.deviceHandle = pynvml.nvmlDeviceGetHandleByIndex(0)
            if (cuda_vram_last == 0):
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.deviceHandle)
                self.vram = int((mem.total - mem.used) * 0.95 / (1024 ** 2))
                cuda_vram_last = self.vram
            else :
                self.vram = cuda_vram_last

            print(f"{self.device} 可使用内存: {self.vram :.2f} MB")
        else :
            import psutil

            self.device = torch.device("cpu")
            self.device_name = "cpu"
            self.deviceHandle = -1
            if cpu_vram_last == 0:
                mem = psutil.virtual_memory()
                self.vram = int((mem.total - mem.used) * 0.95 / (1024 ** 2))
                cpu_vram_last = self.vram
            else :
                self.vram = cpu_vram_last
            print(f"{self.device} 可使用内存: {self.vram :.2f} MB")

        self.bf16_support = is_bf16_supported(self.device)

    @classmethod
    def INPUT_TYPES(s):
        models = read_json_value(models_json_file_path, "models", list)
        return {
            "required": {
                "base_model_dir": ("STRING", {"default": os.path.join(folder_paths.models_dir, "LLM"), "multiline": False}),
                "text": ("STRING", {"default": "Describe this video. What does the video show at the beginning, then what does the video show, and what is the camera movement like?", "multiline": True}),
                "model": (models,),
                "quantization": (["none", "4bit", "8bit"],),
                "images": ("IMAGE",),
                "image_analysis_count": ("INT", {"default": 4, "min":1, "max":32 }),
                "use_shared_gpu_mem": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1}),  # add seed parameter, default is -1
            },
            "optional": {
                "options": ("MODLE_OPTIONS",),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("response","analysis_images")
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"
    def get_options(self, options):
        ret = None

        if options is None:
            return ret

        enablers = ['enable_max_new_tokens', 'enable_min_new_tokens',
                    'enable_repeat_penalty', 'enable_temperature',
                    'enable_top_k', 'enable_top_p']

        for enabler in enablers:
            if options[enabler]:
                if ret is None:
                    ret = {}
                key = enabler.replace("enable_", "")
                ret[key] = options[key]

        return ret

    def inference(self, base_model_dir, text, model, quantization,
                  images, image_analysis_count,
                  use_shared_gpu_mem, keep_model_loaded, seed, options=None,):
        global cuda_vram_last
        global cpu_vram_last
        if seed != -1:
            torch.manual_seed(seed)

        prefix = "Describe this video. "
        if not text.startswith(prefix):
            text = prefix + text

        real_options = self.get_options(options)

        model_id = f"qwen/{model}"

        self.model_checkpoint = os.path.join(os.path.join(base_model_dir, "LLM"), os.path.basename(model_id))

        if not os.path.exists(self.model_checkpoint):
            raise FileNotFoundError(
                f"模型检查点 '{self.model_checkpoint}' 不存在。请使用以下命令下载模型:\n"
                f"modelscope download --model Qwen/{model} --local_dir {self.model_checkpoint}"
            )

        if self.processor is None:
            min_pixels = 256 * 256
            max_pixels = 512 * 512 * 2

            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        if self.model is None:
            # Load the model on the available device(s)
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None
            # 获取函数的签名
            signature = inspect.signature(Qwen2_5_VLForConditionalGeneration.from_pretrained)

            # 列出所有参数
            for param in signature.parameters.values():
                print(f"参数名: {param.name}, 默认值: {param.default}, 类型: {param.annotation}")

            # 获取当前 GPU 的总内存

            if use_shared_gpu_mem and self.deviceHandle != -1:
                self.vram *= 2
            print(f"Set maximum memory to {self.vram} MB")
            max_memory = {
                0: f"{self.vram}MB"
            }
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map= "auto",
                quantization_config=quantization_config,
                local_files_only=True,
                max_memory=max_memory,
            )

        extracted_images = None
        with torch.no_grad():
            if torch.is_tensor(images):
                extracted_images = extract_images(images, image_analysis_count)
                tmp_saved_vedio = tmp_save_vedio(extracted_images, current_dir)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": tmp_saved_vedio[0],
                                "fps": image_analysis_count,
                            },
                            {"type": "text", "text": text},
                        ],
                    }
                ]
                print(f"{messages}")
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                        ],
                    }
                ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            ).to(self.device_name)

            print(inputs)

            try:
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs, **real_options
                    )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    result = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
            except Exception as e:
                return (f"Error during model inference: {str(e)}",)

            if not keep_model_loaded:
                del self.processor
                del self.model
                self.processor = None
                self.model = None
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                cuda_vram_last = 0
                cpu_vram_last = 0

            return (result, extracted_images)

NODE_CLASS_MAPPINGS = {
    "Qwen2_5VL_IMAGE": Qwen2_5VL_IMAGE,
    "Qwen2_5VL_VEDIO": Qwen2_5VL_VEDIO,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2_5VL_IMAGE": "Smell Qwen2_5VL IMAGE",
    "Qwen2_5VL_VEDIO": "Smell Qwen2_5VL VEDIO"
}