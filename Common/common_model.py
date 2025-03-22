from .common_func import *

class CommonModel:
    def __init__(self):
        if torch.cuda.is_available() :
            self.device = torch.device("cuda")
            self.device_name = "cuda"
        else:
            self.device = torch.device("cpu")
            self.device_name = "cpu"
        self.bf16_support = is_bf16_supported(self.device)
        self.current_precision = None
        self.current_model_name = None
        self.model_checkpoint = None
        self.processor = None
        self.model = None

    def clear(self):
        del self.model
        del self.processor
        self.current_precision = None
        self.current_model_name = None
        self.model_checkpoint = None
        self.processor = None
        self.model = None

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