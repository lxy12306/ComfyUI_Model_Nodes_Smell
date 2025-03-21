import os
import torch
import inspect
import gc
import folder_paths

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info


from ..Common.common_func import *
from ..Common.image_func import *
from .func import *


current_dir = os.path.dirname(os.path.abspath(__file__))
models_json_file_path = os.path.join(current_dir, 'json', 'models.json')



class Qwen2_5VL_IMAGE(Qwen2_5VL):
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        models = read_json_value(models_json_file_path, "models", list)
        system_prompt_list, _ = load_system_prompt(file_name="image_role.json")
        return {
            "required": {
                "base_model_dir": ("STRING", {"default": folder_paths.models_dir, "multiline": False}),
                "system_prompt_choice": (system_prompt_list,),
                "user_system_prompt": ("STRING", {"default": "", "multiline": True}),
                "user_prompt": ("STRING", {"default": "Describe this picture.", "multiline": True}),
                "model": (models,),
                "precision": (["bfloat16", "float16", "float32"],),
                "quantization": (["none", "4bit", "8bit"],),
                "use_shared_gpu_mem": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "options": ("MODLE_OPTIONS",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"


    def inference(self, base_model_dir, system_prompt_choice, user_system_prompt, user_prompt, model, use_shared_gpu_mem, precision, quantization, keep_model_loaded, seed, options=None, image=None):
        if seed != -1:
            torch.manual_seed(seed)
        self.system_prompt_list, self.system_prompt_info = load_system_prompt(file_name="image_role.json")
        real_options = self.get_options(options)
        self.load_model(base_model_dir, model, use_shared_gpu_mem, precision, quantization)
        if system_prompt_choice != "None":
            system_prompt = self.select_system_prompt(system_prompt_choice, self.system_prompt_info)
        else:
            system_prompt = user_system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": []}
        ]
        messages[1]["content"].append({
                        "type": "text",
                        "text": user_prompt,
                        })

        with torch.no_grad():
            if torch.is_tensor(image):
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                    raise ValueError(image.shape)

                pil_image = tensor_to_pil(image)
                messages[1]["content"].append({
                                "type": "image",
                                "image": pil_image,
                            })

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
                self.clear()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()

            return result


class Qwen2_5VL_VEDIO(Qwen2_5VL):
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        models = read_json_value(models_json_file_path, "models", list)
        system_prompt_list, _ = load_system_prompt(file_name="vedio_role.json")
        return {
            "required": {
                "base_model_dir": ("STRING", {"default": folder_paths.models_dir, "multiline": False}),
                "system_prompt_choice": (system_prompt_list,),
                "user_system_prompt": ("STRING", {"default": "", "multiline": True}),
                "user_prompt": ("STRING", {"default": "Describe this video.", "multiline": True}),
                "model": (models,),
                "precision": (["bfloat16", "float16", "float32"],),
                "quantization": (["none", "4bit", "8bit"],),
                "images": ("IMAGE",),
                "image_analysis_count": ("INT", {"default": 4, "min":1, "max":32 }),
                "use_shared_gpu_mem": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1}),
            },
            "optional": {
                "options": ("MODLE_OPTIONS",),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("response","analysis_images")
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"

    def inference(self, base_model_dir, system_prompt_choice, user_system_prompt, user_prompt,
                  model, precision, quantization,
                  images, image_analysis_count,
                  use_shared_gpu_mem, keep_model_loaded, seed, options=None,):
        if seed != -1:
            torch.manual_seed(seed)
        self.system_prompt_list, self.system_prompt_info = load_system_prompt(file_name="vedio_role.json")
        real_options = self.get_options(options)
        self.load_model(base_model_dir, model, use_shared_gpu_mem, precision, quantization)
        if system_prompt_choice != "None":
            system_prompt = self.select_system_prompt(system_prompt_choice, self.system_prompt_info)
        else:
            system_prompt = user_system_prompt

        prefix = "Describe this video. "
        if not user_prompt.startswith(prefix):
            user_prompt = prefix + user_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": []}
        ]
        messages[1]["content"].append({
                        "type": "text",
                        "text": user_prompt,
                        })

        extracted_images = None
        with torch.no_grad():
            if torch.is_tensor(images):
                extracted_images = extract_images(images, image_analysis_count)

                #tmp_saved_vedio_image =  tmp_save_images(extracted_images, current_dir)
                #messages[1]["content"].append({
                #        "type": "video",
                #        "video": tmp_saved_vedio_image,
                #        })
                tmp_saved_vedio = tmp_save_vedio(extracted_images, current_dir)
                messages[1]["content"].append({
                        "type": "video",
                        "video": tmp_saved_vedio[0],
                        "fps": 1,
                        })

                print(f"{messages}")
            else:
                raise ValueError("输入的图片异常")

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
                self.clear()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()

            return (result, extracted_images)

NODE_CLASS_MAPPINGS = {
    "Qwen2_5VL_IMAGE": Qwen2_5VL_IMAGE,
    "Qwen2_5VL_VEDIO": Qwen2_5VL_VEDIO,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2_5VL_IMAGE": "Smell Qwen2_5VL IMAGE",
    "Qwen2_5VL_VEDIO": "Smell Qwen2_5VL VEDIO"
}