from transformers import AutoProcessor, LlavaForConditionalGeneration

from tqdm import tqdm
from comfy.utils import ProgressBar

from ..Common.common_func import *
from ..Common.image_func import *
from ..Common.common_model import *



class JoyCaptionBetaOne(CommonModel):

    def __init__(self):
        super().__init__()
        self.exDir = "LLM"


    @classmethod
    def INPUT_TYPES(cls):
        cls.input_init()

        cls.quantization_configs = {
            "nf4": {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                    "bnb_4bit_use_double_quant": True,
                },
            "int8": {
                "load_in_8bit": True,
            },
        }
        cls.current_model_json_file = os.path.join(cls.current_json_dir, 'joycaption_beta_one.json')
        cls.model_list = read_json_value(cls.current_model_json_file, "MODEL_LIST", list)
        cls.quantization_list = list(cls.quantization_configs.keys())
        cls.quantization_list.append("none")
        print(f"JoyCaptionBetaOne INPUT_TYPES: {cls.model_list}, {cls.quantization_list}, {cls.device_list}")

        return cls.load_model_input(cls.model_list, cls.quantization_list, cls.device_list)
    
    RETURN_TYPES = ("JoyCaptionBetaOne", )
    RETURN_NAMES = ("joycaption_beta_one_pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "🌱SmellLargeModel/JoyCaptionBetaOne"
    DESCRIPTION = "导入 JoyCaption Beta One 模型"
    
    def clear(self):
        super().clear()

    def load_model(self, base_model_dir, model, precision, quantization, device):
        super().load_model(base_model_dir, model, precision, quantization, device)

        self.model_checkpoint = self.load_hg_model(
            resume_download=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_checkpoint,
            trust_remote_code=True,
            local_files_only=True
        )

        model_kwargs = {
            "torch_dtype": self.get_dtype(),
            "local_files_only": True,
            "trust_remote_code": True,
        }

        # Configure device mapping based on hardware
        if self.device == "cuda":
            # Always use a single GPU setup to avoid device conflicts
            self.logger().info("Using single GPU configuration to avoid device conflicts")
            # Don't use device_map at all - load to a specific device instead
            model_kwargs.pop("device_map", None)
        else:
            # For CPU or MPS, use the device directly
            self.logger().info(f"Using {self.device} device")
            model_kwargs["device_map"] = {"": self.device}
        
        if quantization != "none":
            quantization_config = self.quantization_configs.get(quantization, {})
            from transformers import BitsAndBytesConfig
            qnt_config = BitsAndBytesConfig(
                **quantization_config,
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                # Transformer's Siglip implementation has bugs when quantized, so skip those.
            )
            model_kwargs["quantization_config"] = qnt_config
            model_kwargs["torch_dtype"] = "auto"

        self.logger().info(f"Loading model from {model_kwargs}")
        self.model = LlavaForConditionalGeneration.from_pretrained(
                     self.model_checkpoint,
                     **model_kwargs
                     )
        self.model.eval()

        return (self,)

    @torch.inference_mode()
    def generate(self, image:Image, system:str, prompt:str, options:dict) -> str:
        options = self.get_options(options)
        if options.get('max_new_tokens', None) is None:
            options['max_new_tokens'] = 2048
        self.logger().info(f"Generating caption with options: {options}")
        convo = [
            {
                "role": "system",
                "content": system.strip(),
            },
            {
                "role": "user",
                "content": prompt.strip(),
            },
        ]

        # Format the conversation
        convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        assert isinstance(convo_string, str)

        # Process the inputs
        inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt").to(self.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(self.get_dtype())

        do_sample = options.get('temperature', 1) > 0

        # Generate the captions
        generate_ids = self.model.generate(
            **inputs,
            do_sample=do_sample,
            suppress_tokens=None,
            use_cache=True,
            **options,
        )[0]

        # Trim off the prompt
        generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

        # Decode the caption
        caption = self.processor.tokenizer.decode(generate_ids, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
        return caption.strip()

class JoyCaptionBetaOneProcessor(CommonModelProcessor):
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        cls.input_init()
        cls.current_model_json_file = os.path.join(cls.current_json_dir, 'joycaption_beta_one.json')
        cls.caption_type = read_json_value(cls.current_model_json_file, 'CAPTION_TYPE_MAP', dict)
        cls.caption_type_list = list(cls.caption_type.keys())
        cls.caption_length_list = ["any", "very short", "short", "medium-length", "long", "very long"] + [str(i) for i in range(20, 261,10)]

        return {
            "required": {
                "image": ("IMAGE",),
                "joycaption_beta_one_pipeline": ("JoyCaptionBetaOne",),
                "caption_type": (cls.caption_type_list,),
                "caption_length": (cls.caption_length_list,),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "options": ("MODLE_OPTIONS",),
                "extra_options": ("Extra_Options",)
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process_image"
    CATEGORY = "🌱SmellLargeModel/JoyCaptionBetaOne"
    DESCRIPTION = "图片推理"

    def build_prompt(self, caption_type_key: str, caption_length: str | int, extra_options: list[str] = None,
                     user_prompt: str = "") -> str:

        # 如果提供了user_prompt，直接使用它
        if user_prompt and user_prompt.strip():
            prompt = user_prompt.strip()
        else:
            # 选择正确的模板行
            if caption_length == "any":
                map_idx = 0
            elif isinstance(caption_length, str) and caption_length.isdigit():
                map_idx = 1  # 数字字数模板
            else:
                map_idx = 2  # 长度描述符模板

            prompt = self.caption_type[caption_type_key][map_idx]

        if extra_options is not None:
            extra, name_input = extra_options
            prompt += " " + " ".join(extra)
        else:
            name_input = "{NAME}"

        return prompt.format(
            name=name_input,
            length=caption_length,
            word_count=caption_length)
    
    def process_image(self, image, joycaption_beta_one_pipeline, caption_type, caption_length, keep_model_loaded, user_prompt, options=None, extra_options=None):
        joycaption_beta_one_pipeline.reload_for_bak()
        ret_text = []
        system_prompt = "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions."
        prompt = self.build_prompt(caption_type, caption_length, extra_options, user_prompt)
        
        comfy_pbar = ProgressBar(image.shape[0])
        tqdm_pbar = tqdm(total=image.shape[0], desc="Processing Images")

        for img in image:
            img = tensor_to_pil(img.unsqueeze(0)).convert('RGB')
            caption = joycaption_beta_one_pipeline.generate(
                image=img,
                system=system_prompt,
                prompt=prompt,
                options=options
            )
            ret_text.append(caption)
            comfy_pbar.update(1)
            tqdm_pbar.update(1)
        
        if not keep_model_loaded:
            joycaption_beta_one_pipeline.clear()
        
        return (ret_text,)
    
class JoyCaptionBetaOneExtraOptions:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        current_file = inspect.getfile(cls)
        cls.current_dir = os.path.dirname(current_file)
        cls.current_json_dir = os.path.join(cls.current_dir, 'json')
        cls.current_model_json_file = os.path.join(cls.current_json_dir, 'joycaption_beta_one.json')
        cls.extra_promt_map = read_json_value(cls.current_model_json_file, 'EXTRA_PROMT_MAP', dict)

        options = list(cls.extra_promt_map.keys())
        options.insert(0, "None")  # Add "None" as the first option
        required = {}
        for i in range(len(options)):
            required[f"option_{i}"] = (options,)
        return {
            "required": required
        }

    RETURN_TYPES = ("Extra_Options",)
    RETURN_NAMES = ("extra_options",)
    FUNCTION = "run"
    CATEGORY = "🌱SmellLargeModel/JoyCaptionBetaOne"
    DESCRIPTION = "JoyCaptionBetaOneExtraOptions"

    def run(self, **kwargs):
        options_selected = list(kwargs.values())
        values = []
        for selected in options_selected:
            if selected != "None":
                values.append(self.extra_promt_map[selected])
        return (values, )