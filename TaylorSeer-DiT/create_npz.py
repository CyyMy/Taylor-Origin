"""
将图片文件夹转换为 .npz 文件

用法:
    python create_npz.py --input_dir <图片目录> --output <输出.npz路径> [--num_images <数量>]

示例:
    python create_npz.py --input_dir ./samples/my_images --output ./samples/my_images.npz
    python create_npz.py --input_dir D:/Datasets/imagenet/val --output imagenet_val.npz --num_images 50000
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def create_npz_from_folder(input_dir, output_path, num_images=None, image_size=None):
    """
    从文件夹读取图片并打包成 .npz 文件

    Args:
        input_dir: 图片所在目录
        output_path: 输出 .npz 文件路径
        num_images: 要处理的图片数量，None 表示处理所有图片
        image_size: 目标图片尺寸 (H, W)，None 表示保持原尺寸
    """

    # 收集所有图片文件
    print(f"扫描目录: {input_dir}")
    image_files = []

    # 支持的图片格式
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    # 如果是按编号命名的文件夹（如 000000.png, 000001.png...）
    if num_images is not None:
        for i in range(num_images):
            # 尝试不同的命名格式
            for ext in ['.png', '.jpg', '.jpeg']:
                path = os.path.join(input_dir, f"{i:06d}{ext}")
                if os.path.exists(path):
                    image_files.append(path)
                    break
            else:
                print(f"警告: 找不到第 {i} 张图片")
    else:
        # 递归扫描所有图片
        for root, dirs, files in os.walk(input_dir):
            for file in sorted(files):
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    image_files.append(os.path.join(root, file))

    if not image_files:
        raise ValueError(f"在 {input_dir} 中没有找到图片文件")

    print(f"找到 {len(image_files)} 张图片")

    # 读取并转换图片
    samples = []
    for img_path in tqdm(image_files, desc="读取图片"):
        try:
            img = Image.open(img_path)

            # 转换为 RGB（如果是 RGBA 或灰度图）
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 调整尺寸（如果指定）
            if image_size is not None:
                img = img.resize((image_size[1], image_size[0]), Image.LANCZOS)

            # 转换为 numpy 数组
            img_np = np.asarray(img).astype(np.uint8)
            samples.append(img_np)

        except Exception as e:
            print(f"警告: 无法读取 {img_path}: {e}")
            continue

    if not samples:
        raise ValueError("没有成功读取任何图片")

    # 堆叠成一个大数组
    print("堆叠数组...")
    samples = np.stack(samples)

    # 检查形状
    print(f"数组形状: {samples.shape}")
    assert len(samples.shape) == 4, f"期望 4 维数组 (N, H, W, 3)，得到 {samples.shape}"
    assert samples.shape[3] == 3, f"期望 RGB 3 通道，得到 {samples.shape[3]} 通道"

    # 保存为 .npz
    print(f"保存到: {output_path}")
    np.savez(output_path, arr_0=samples)

    # 显示文件大小
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"完成! 文件大小: {file_size_mb:.2f} MB")
    print(f"数组形状: {samples.shape}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='将图片文件夹转换为 .npz 文件')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='图片所在目录')
    parser.add_argument('--output', type=str, required=True,
                        help='输出 .npz 文件路径')
    parser.add_argument('--num_images', type=int, default=None,
                        help='要处理的图片数量（默认处理所有）')
    parser.add_argument('--image_size', type=int, nargs=2, default=None,
                        help='目标图片尺寸 H W，例如: --image_size 256 256')

    args = parser.parse_args()

    # 检查输入目录
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"输入目录不存在: {args.input_dir}")

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 转换图片尺寸参数
    image_size = tuple(args.image_size) if args.image_size else None

    # 执行转换
    create_npz_from_folder(
        input_dir=args.input_dir,
        output_path=args.output,
        num_images=args.num_images,
        image_size=image_size
    )


if __name__ == "__main__":
    main()
