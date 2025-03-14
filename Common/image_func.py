import os
import shutil
import torch
from PIL import Image
import numpy as np
import cv2


def tensor_to_pil(image_tensor: torch.Tensor, batch_index: int = 0 )-> Image:
    """
    将形状为 [batch, channels, height, width] 的张量在指定的 batch_index 转换为 PIL 图像。

    参数：
        image_tensor (torch.Tensor): 输入的图像张量。
        batch_index (int): 要转换的图像在批次中的索引，默认为 0。

    返回：
        Image: 转换后的 PIL 图像。
    """
    # 从张量中提取指定索引的图像并增加一个维度
    image_tensor = image_tensor[batch_index]

    if image_tensor.shape[0] > 4:
        image_tensor = image_tensor.permute(2, 0, 1)

    # 将张量值缩放到 [0, 255] 范围
    i = 255.0 * image_tensor.cpu().numpy()
    # 确保通道数正确并调整维度顺序
    if image_tensor.shape[0] == 1:  # 单通道图像（灰度图像）
        img = Image.fromarray(np.clip(i.squeeze(), 0, 255).astype(np.uint8), mode='L')
    elif image_tensor.shape[0] == 3:  # RGB 图像
        img = Image.fromarray(np.clip(i.transpose(1, 2, 0), 0, 255).astype(np.uint8))
    else:
        raise ValueError(f"输入的张量必须是形状为 [batch, channels, height, width] 的四维张量，{image_tensor.shape} 且通道数应为 1 或 3。")

    return img


def tensor_to_video(image_tensor: torch.Tensor, output_path: str, fps: int = 30):
    """
    将形状为 [batch, channels, height, width] 的张量转换为 MP4 视频。

    参数：
        image_tensor (torch.Tensor): 输入的图像张量，形状为 [batch, channels, height, width]。
        output_path (str): 输出视频文件的路径。
        fps (int): 视频的帧率，默认为 30。
    """
    # 获取张量的形状
    batch_size, channels, height, width = image_tensor.shape

    # 确保通道数为 3（RGB）
    if channels != 3:
        raise ValueError("输入的张量必须具有 3 个通道（RGB）。")

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 遍历每一帧并写入视频
    for i in range(batch_size):

        # 将张量转换为 NumPy 数组并调整通道顺序
        frame = image_tensor[i].permute(1, 2, 0).cpu().numpy()  # 从 [channels, height, width] 到 [height, width, channels]

        # 确保像素值在 [0, 255] 范围内
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

        # 将 RGB 转换为 BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 写入帧到视频
        video_writer.write(frame)

    # 释放视频写入对象
    video_writer.release()
    print(f"视频已保存到 {output_path}")

def extract_images(images: torch.Tensor, image_analysis_count: int) -> torch.Tensor:
    """
    根据 image_analysis_count 从输入张量中提取特定图像。

    参数：
        images (torch.Tensor): 输入图像张量，形状为 (batch_size, C, H, W)。
        image_analysis_count (int): 要提取的图像数量。

    返回：
        list: 提取的图像张量列表。
    """
    batch_size = images.shape[0]

    if batch_size < 2:
        raise ValueError("批次大小必须至少为 2，以提取第一张和最后一张图像。")

    # 确保提取第一张和最后一张图像
    extracted_indices = [0, batch_size - 1]  # 第一张和最后一张的索引

    # 计算还需提取多少张图像
    additional_count = image_analysis_count - 2  # 减去第一张和最后一张图像

    if additional_count > 0:
        # 计算步长，以均匀分布额外提取的图像
        step = (batch_size - 2) // (additional_count + 1)  # +1 是为了包括第一张和最后一张图像

        # 从中间提取额外的图像
        for i in range(1, additional_count + 1):
            index = i * step
            extracted_indices.append(index)

    # 排序索引以保持顺序
    extracted_indices = sorted(extracted_indices)

    # 使用计算出的索引提取图像
    extracted_images = images[extracted_indices]

    return extracted_images

def tmp_save_images(extracted_images: torch.Tensor, output_dir: str) -> list:
    """
    将提取的图像张量数组保存为 PNG 格式，并返回这些图像的全路径数组。

    参数：
        extracted_images (torch.Tensor): 提取的图像张量，形状为 (N, C, H, W)。
        output_dir (str): 保存图像的目录路径。

    返回：
        list: 保存的 PNG 图像的全路径数组。
    """
    # 创建输出目录（如果不存在）
    output_dir = os.path.join(output_dir, 'tmp')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    shape = extracted_images.shape  # 获取张量的形状

    # 检查张量的维度
    if len(shape) == 4:
        N, *dims = shape  # 解包形状
        if dims[2] <= 4:
            extracted_images = extracted_images.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"张量维度不正确，应该是 4 维，实际维度: {len(shape)}")
    # 存储保存的图像路径
    saved_image_paths = []

    # 遍历提取的图像并保存
    for i in range(extracted_images.shape[0]):
        # 将张量转换为 PIL 图像
        image = tensor_to_pil(extracted_images[i].unsqueeze(0))
        # 构建保存路径
        file_path = os.path.join(output_dir, f'tmp_image_{i}.png')

        # 保存图像
        image.save(file_path)
        saved_image_paths.append(file_path)  # 添加到路径列表

    return saved_image_paths


def tmp_save_vedio(extracted_images: torch.Tensor, output_dir: str) -> list:
    """
    将提取的图像张量数组保存为 PNG 格式，并返回这些图像的全路径数组。

    参数：
        extracted_images (torch.Tensor): 提取的图像张量，形状为 (N, C, H, W)。
        output_dir (str): 保存图像的目录路径。

    返回：
        list: 保存的 PNG 图像的全路径数组。
    """
    # 创建输出目录（如果不存在）
    output_dir = os.path.join(output_dir, 'tmp')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    shape = extracted_images.shape  # 获取张量的形状

    # 检查张量的维度
    if len(shape) == 4:
        N, *dims = shape  # 解包形状
        if dims[2] <= 4:
            extracted_images = extracted_images.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"张量维度不正确，应该是 4 维，实际维度: {len(shape)}")

    # 遍历提取的图像并保存为mp4
    file_path = os.path.join(output_dir, f'tmp_vedio.mp4')
    tensor_to_video(extracted_images, file_path, shape[0])

    return [file_path]