import torch
import json
import os
import logging


def is_bf16_supported(device):
    """
    检查给定设备是否支持 BF16。

    参数：
        device (torch.device): 要检查 BF16 支持的设备。

    返回：
        bool: 如果支持 BF16，则返回 True；否则返回 False。
    """
    return (
        torch.cuda.is_available() and
        torch.cuda.get_device_capability(device)[0] >= 8
    )

def read_json_file(dir_name, file_name):
        # Try to load existing file with different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'gbk']
        file_path = os.path.join(
            dir_name, file_name
        )
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    data = json.loads(content)
                    return data
            except Exception:
                continue

        logging.error(f"Error: Failed to load {file_path} with any supported encoding")
        return {}



def read_json_value(file_path, key, expected_type=None):
    """
    从 JSON 文件中读取指定键的值并检查其类型。

    参数：
        file_path (str): JSON 文件的路径。
        key (str): 需要检索的键。
        expected_type (type, 可选): 值的预期类型,如 list, dict, str, int。如果提供，该函数将检查值是否与此类型匹配。

    返回：
        如果与预期类型匹配，则返回与指定键关联的值；如果未找到该键或类型不匹配，则返回 None。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            value = data.get(key, None)  # Get the value for the specified key
            if expected_type is not None and not isinstance(value, expected_type):
                return None  # Return None if the type does not match
            return value  # Return the value if type matches or expected_type is None
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from the file: {file_path}")
        print(f"具体错误信息: {e}")
        return None