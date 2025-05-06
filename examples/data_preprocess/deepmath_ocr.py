"""
Preprocess dataset for deepmath task.
"""

import os
import subprocess
import tempfile

from PIL import Image, ImageChops, ImageOps
from datasets import load_dataset, Sequence
from datasets import Image as ImageFeature
import argparse

from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True

def make_prefix(template_type):
    # NOTE: also need to change reward_score/deepmath.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. 
User: solve the problem: <image>
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system
You are a helpful assistant. The User asks a question, and you solve it. You should first think about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags, respectively. i.e., <think> reasoning process here </think> <answer> answer here </answer><|im_end|>
<|im_start|>user
solve the problem: <image><|im_end|>
<|im_start|>assistant
Let me solve this step by step.\n<think>"""
    return prefix


def convert_to_image(question: str,
                     save_path: str,
                     dpi: int = 150,
                     fontsize: int = 16,
                     padding_pixels: float = 33,
                     width_inch: float = 5.0) -> Image.Image:
    """
    参数：
      question: 包含普通文字和 LaTeX 数学（用 $...$ 或 $$...$$ 包裹）的字符串。
      dpi:        渲染分辨率，默认 300。
      fontsize:   字体大小。
      padding_inches: 图像四周空白（英寸）。
      width_inch: 图像固定宽度（英寸），高度自动根据内容调整。

    返回：
      Pillow Image 对象，模式为 RGB，背景为白色。
    """
    with tempfile.TemporaryDirectory() as d:
        tex_path = os.path.join(d, "expr.tex")
        dvi_path = os.path.join(d, "expr.dvi")
        png_path = os.path.join(d, "expr.png")

        # 1. 写 LaTeX 源文件
        with open(tex_path, "w") as f:
            text = r"\documentclass[" + str(fontsize) + r"""pt]{article}
\usepackage{amsmath}
\pagestyle{empty}
\begin{document}""" + "\n" + question + "\n" + r"""\end{document}"""
            f.write(text)

        # 2. 生成 DVI
        subprocess.run([
            "/Library/TeX/texbin/latex", "-interaction=nonstopmode",
            "-output-directory", d, tex_path
        ], check=True)

        # 3. DVI 转 PNG（白色背景）
        subprocess.run([
            "/Library/TeX/texbin/dvipng",
            "-T", "tight",
            "-D", str(dpi),
            "--bdpi", str(dpi),
            "-Q", "4",
            "--truecolor",
            "-bg", "White",
            "-o", png_path,
            dvi_path
        ], cwd=d, check=True)

        if not os.path.exists(png_path):
            raise FileNotFoundError(f"未找到生成的 {png_path}")

        # 4. 打开图像，裁剪空白区域
        img = Image.open(png_path).convert("RGB")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg)

        gray = diff.convert("L")
        bw = gray.point(lambda x: 255 if x > 10 else 0)
        bbox = bw.getbbox()
        cropped = img.crop(bbox)

        # 5. 添加上下左右各约 1 行的白色边距
        padded = ImageOps.expand(cropped, border=padding_pixels, fill="white")

        # 6. 缩放到 A4 宽度一半（单位像素）
        target_w = int(width_inch * dpi)
        scale = target_w / padded.width
        target_h = int(padded.height * scale)
        final = padded.resize((target_w, target_h), Image.LANCZOS)

        # 7. 保存最终图像
        final.save(save_path)
        print("image:", isinstance(final, Image.Image))
        return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/Users/kuan/code/math-ocr-zero/data/deepmath-ocr-1000')
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--train_size', type=int, default=10000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')

    args = parser.parse_args()

    data_source = 'DeepMath-103K'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    raw_dataset = load_dataset('zwhe99/DeepMath-103K', split='train')

    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))


    def make_map_fn(split):
        def process_fn(example, idx):
            try:
                question = make_prefix(template_type=args.template_type)
                final_answer = example['final_answer']
                save_dir = os.path.join(args.local_dir, split)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, f"{idx}.png")
                image = convert_to_image(example['question'], save_path)  # 可能抛出异常
            except Exception as e:
                print(f"Error processing example at index {idx}: {e}")
                return
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "images": [image],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": final_answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }

            return data
        return process_fn


    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    train_dataset = train_dataset.filter(lambda x: x is not None)
    test_dataset = test_dataset.filter(lambda x: x is not None)
    train_dataset = train_dataset.cast_column("images", Sequence(ImageFeature()))
    train_dataset = train_dataset.cast_column("images", Sequence(ImageFeature()))
    print('train dataset:', len(train_dataset))
    print('test dataset:', len(test_dataset))

    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
