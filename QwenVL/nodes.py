import os
import torch
import folder_paths


from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info


from ..Common.common_func import *

class Qwen2_5VL_IMAGE:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_dir": ("STRING", {"default": os.path.join(folder_paths.models_dir, "LLM"), "multiline": False}),
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "Qwen2.5-VL-3B-Instruct",
                        "Qwen2.5-VL-7B-Instruct",
                    ],
                    {"default": "Qwen2.5-VL-3B-Instruct"},
                ),
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

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
            )

        with torch.no_grad():
            if torch.is_tensor(image):
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
        
NODE_CLASS_MAPPINGS = {
    "Qwen2_5VL_IMAGE": Qwen2_5VL_IMAGE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2_5VL_IMAGE": "Smell Qwen2_5VL IMAGE",
}