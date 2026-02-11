#!/usr/bin/env python3
"""
图片转 ICO 转换脚本
支持 WebP、PNG、JPG、JPEG、BMP、GIF、TIFF 等多种格式
"""

from PIL import Image
import sys
import os
import argparse


# 支持的输入格式
SUPPORTED_FORMATS = ('.webp', '.png', '.jpg', '.jpeg', 
                     '.bmp', '.gif', '.tiff', '.tif')


def image_to_ico(input_path, output_path=None, sizes=None, 
                 input_format=None, quality=95):
    """
    将图片转换为 ICO 格式
    
    Args:
        input_path: 输入的图片文件路径
        output_path: 输出的 ICO 文件路径（可选，默认与输入同名）
        sizes: 包含的尺寸列表，默认 [(16,16), (32,32), (48,48), (256,256)]
        input_format: 强制指定输入格式（可选，用于标准输入等情况）
        quality: 输出质量（1-100），仅对有损格式有意义
    """
    if sizes is None:
        sizes = [(16, 16), (32, 32), (48, 48), (256, 256)]
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ 错误：文件 '{input_path}' 不存在")
        return False
    
    # 自动检测或验证格式
    file_ext = os.path.splitext(input_path)[1].lower()
    
    if input_format:
        # 用户强制指定了格式
        input_format = input_format.lower()
        if not input_format.startswith('.'):
            input_format = '.' + input_format
    else:
        # 自动检测
        input_format = file_ext
    
    # 验证格式是否支持
    if input_format not in SUPPORTED_FORMATS:
        print(f"❌ 不支持的格式 '{input_format}'")
        print(f"支持的格式: {', '.join(SUPPORTED_FORMATS)}")
        return False
    
    # 如果没有指定输出路径，自动生成
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = base_name + ".ico"
    
    try:
        # 打开图片
        with Image.open(input_path) as img:
            # 处理动画 GIF/WebP（取第一帧）
            if getattr(img, "is_animated", False):
                print(f"⚠️  检测到动画文件，仅转换第一帧")
                img.seek(0)
            
            # 转换为 RGBA 模式（支持透明通道）
            if img.mode in ('RGBA', 'LA', 'P'):
                # 如果有透明通道或调色板，转换为 RGBA
                img = img.convert('RGBA')
            elif img.mode != 'RGB':
                # 其他模式先转 RGB，再转 RGBA
                img = img.convert('RGB').convert('RGBA')
            
            # 创建不同尺寸的图标
            icons = []
            for size in sizes:
                # 使用高质量缩放（LANCZOS 是 Pillow 9.0+ 的推荐方式）
                try:
                    resized = img.resize(size, Image.Resampling.LANCZOS)
                except AttributeError:
                    # 兼容旧版 Pillow
                    resized = img.resize(size, Image.ANTIALIAS)
                icons.append(resized)
            
            # 保存为 ICO 格式
            icons[0].save(
                output_path,
                format='ICO',
                sizes=sizes,
                append_images=icons[1:]
            )
            
            # 获取文件大小
            output_size = os.path.getsize(output_path)
            
            print(f"✅ 转换成功！")
            print(f"   输入: {input_path} ({input_format[1:].upper()})")
            print(f"   输出: {output_path}")
            print(f"   文件大小: {output_size / 1024:.1f} KB")
            print(f"   包含尺寸: {', '.join([f'{w}×{h}' for w, h in sizes])}")
            return True
            
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False


def batch_convert(directory, formats=None, recursive=False):
    """
    批量转换目录中的图片文件
    
    Args:
        directory: 目标目录
        formats: 指定要转换的格式列表，None 表示所有支持的格式
        recursive: 是否递归处理子目录
    """
    if formats is None:
        formats = SUPPORTED_FORMATS
    else:
        # 确保格式统一
        formats = tuple(f.lower() if f.startswith('.') else f'.{f.lower()}' 
                       for f in formats)
    
    # 收集所有匹配的文件
    files_to_convert = []
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(formats):
                    files_to_convert.append(os.path.join(root, file))
    else:
        files_to_convert = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.lower().endswith(formats) and os.path.isfile(os.path.join(directory, f))
        ]
    
    if not files_to_convert:
        print(f"⚠️  目录 '{directory}' 中没有找到匹配的图片文件")
        print(f"支持的格式: {', '.join(formats)}")
        return
    
    print(f"📁 找到 {len(files_to_convert)} 个文件，开始批量转换...\n")
    
    success_count = 0
    failed_files = []
    
    for file_path in files_to_convert:
        if image_to_ico(file_path):
            success_count += 1
            print()  # 空行分隔
        else:
            failed_files.append(file_path)
    
    # 总结报告
    print("=" * 50)
    print(f"📊 批量转换完成")
    print(f"   成功: {success_count}/{len(files_to_convert)}")
    print(f"   失败: {len(failed_files)}")
    
    if failed_files:
        print(f"\n❌ 失败的文件:")
        for f in failed_files:
            print(f"   - {f}")


def main():
    parser = argparse.ArgumentParser(
        description='图片转 ICO 工具 - 支持 WebP、PNG、JPG、BMP、GIF、TIFF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s image.webp                    # WebP 转 ICO
  %(prog)s photo.jpg -o icon.ico         # JPG 转 ICO，指定输出名
  %(prog)s logo.png --sizes 32 64 128    # PNG 转 ICO，自定义尺寸
  %(prog)s ./images -b                   # 批量转换目录所有图片
  %(prog)s ./images -b -f webp png       # 批量转换，仅处理 webp 和 png
  %(prog)s pic.jpg -f jpg                # 强制指定输入格式（用于管道等）
        """
    )
    
    parser.add_argument('input', nargs='?', 
                        help='输入的图片文件或目录路径')
    parser.add_argument('-o', '--output', 
                        help='输出 ICO 文件路径（单文件模式）')
    parser.add_argument('-b', '--batch', action='store_true',
                        help='批量转换模式（输入为目录）')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='递归处理子目录（批量模式有效）')
    parser.add_argument('-f', '--formats', nargs='+',
                        choices=['webp', 'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'tif'],
                        help='指定要处理的格式（批量模式有效，默认全部）')
    parser.add_argument('--sizes', nargs='+', type=int, 
                        default=[16, 32, 48, 256],
                        metavar='SIZE',
                        help='ICO 包含的尺寸，默认: 16 32 48 256')
    parser.add_argument('--format', dest='input_format',
                        choices=['webp', 'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'tif'],
                        help='强制指定输入格式（通常自动检测）')
    parser.add_argument('-q', '--quality', type=int, default=95,
                        help='输出质量 1-100（默认: 95）')
    parser.add_argument('-l', '--list-formats', action='store_true',
                        help='列出所有支持的格式')
    
    args = parser.parse_args()
    
    # 列出支持的格式
    if args.list_formats:
        print("支持的输入格式:")
        for fmt in SUPPORTED_FORMATS:
            print(f"  - {fmt[1:].upper()}")
        print("\n输出格式: ICO")
        return
    
    # 如果没有输入参数，显示帮助
    if not args.input:
        parser.print_help()
        return 1
    
    # 解析尺寸参数
    sizes = [(s, s) for s in args.sizes]
    
    # 处理输入格式参数
    input_format = None
    if args.input_format:
        input_format = f".{args.input_format.lower()}"
    
    # 批量模式
    if args.batch or os.path.isdir(args.input):
        target_formats = None
        if args.formats:
            target_formats = [f".{f.lower()}" for f in args.formats]
        batch_convert(args.input, target_formats, args.recursive)
    else:
        # 单文件模式
        image_to_ico(args.input, args.output, sizes, input_format, args.quality)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
    
"""
# 查看支持的格式
python img2ico.py -l

# 各种格式转 ICO
python img2ico.py photo.jpg
python img2ico.py image.png -o app.ico
python img2ico.py anim.gif --sizes 32 64 128

# 批量转换（所有格式）
python img2ico.py ./images -b

# 批量转换（仅 webp 和 png）
python img2ico.py ./images -b -f webp png

# 递归处理子目录
python img2ico.py ./assets -b -r

# 强制指定格式（用于特殊场景）
python img2ico.py - -f webp < image.webp  # 从标准输入读取
"""