class CommonOptions:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enable_num_ctx": ("BOOLEAN", {"default": False}),
                "num_ctx": ("INT", {"default": 2048, "min": 0, "step": 1}),

                "enable_max_new_tokens" : ("BOOLEAN", {"default": True}),
                "max_new_tokens":("INT", {"default":512, "min": 128, "max": 2048, "step": 1}),

                "enable_min_new_tokens" : ("BOOLEAN", {"default": True}),
                "min_new_tokens":("INT", {"default":128, "min": 64, "max": 512, "step": 1}),

                "enable_repeat_last_n": ("BOOLEAN", {"default": False}),
                "repeat_last_n": ("INT", {"default": 64, "min": -1, "max": 64, "step": 1}),
                #如果希望模型在生成时重复最后的 3 个标记，你可以设置 repeat_last_n 为 3。这样，模型在生成新内容时会考虑到最近的 3 个标记，从而可能会生成更连贯的文本。

                "enable_repeat_penalty": ("BOOLEAN", {"default": False}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0, "max": 2, "step": 0.05}),
                #重复惩罚：repetition_penalty 的值用于调整模型对重复词汇的惩罚程度。较高的惩罚值会使模型在生成时更倾向于避免重复使用相同的词汇，从而提高生成文本的多样性和自然性。
                #值的范围：
                #    通常，repetition_penalty 的值大于 1.0 时，模型会对重复的词汇施加惩罚。例如，设置为 1.2 意味着模型在生成重复词汇时会受到更大的惩罚。
                #    如果设置为 1.0，则没有惩罚，模型可以自由地重复使用词汇。
                #    如果设置为小于 1.0，模型可能会更倾向于生成重复的词汇，这通常不是我们想要的效果。

                "enable_temperature": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),

                "enable_top_k": ("BOOLEAN", {"default": False}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1}),

                "enable_top_p": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05}),

                "debug": ("BOOLEAN", {"default": False}), # this is for nodes code usage only
            },
        }

    RETURN_TYPES = ("MODLE_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "options"
    CATEGORY = "Smell_Model"

    def options(self, **kargs):
        kargs_without_debug = {key: value for key, value in kargs.items() if key != 'debug'}  
        if kargs['debug']:
            print("--- options dump\n")
            print(kargs_without_debug)
            print("---------------------------------------------------------")

        return (kargs_without_debug,)
    
NODE_CLASS_MAPPINGS = {
    "CommonOptions": CommonOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CommonOptions": "Smell CommonOptions",
}