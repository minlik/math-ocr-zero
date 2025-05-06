import subprocess
import os
import tempfile
from PIL import Image, ImageOps, ImageChops

# A4 宽度的一半：105mm ≈ 4.133in
HALF_A4_INCH = 105 / 25.4
DPI = 150
LINE_HEIGHT_PX = int(16 / 72 * DPI)  # 16pt ≈ 1行，转换为像素（≈ 22 px）

with tempfile.TemporaryDirectory() as d:
    tex_path = os.path.join(d, "expr.tex")
    dvi_path = os.path.join(d, "expr.dvi")
    png_path = os.path.join(d, "expr.png")
    final_png = os.path.join(os.getcwd(), "limit_formula.png")

    # 1. 写 LaTeX 源文件
    with open(tex_path, "w") as f:
        f.write(r"""\documentclass[20pt]{article}
\usepackage{amsmath}
\pagestyle{empty}
\begin{document}
Evaluate the limit:\[
  \lim_{x \to \infty} \sqrt{x}\,\bigl(\sqrt[3]{x+1} - \sqrt[3]{x-1}\bigr)
\]
\end{document}
""")

    # 2. 生成 DVI
    subprocess.run([
        "/Library/TeX/texbin/latex", "-interaction=nonstopmode",
        "-output-directory", d, tex_path
    ], check=True)

    # 3. DVI 转 PNG（白色背景）
    subprocess.run([
        "/Library/TeX/texbin/dvipng",
        "-T", "tight",
        "-D", str(DPI),
        "--bdpi", str(DPI),
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
    bg = Image.new("RGB", img.size, (255,255,255))
    diff = ImageChops.difference(img, bg)

    gray = diff.convert("L")
    bw = gray.point(lambda x: 255 if x > 10 else 0)

    bbox = bw.getbbox()
    print(bbox)


    if bbox is None:
        raise ValueError("无法识别文字区域")

    cropped = img.crop(bbox)

    # 5. 添加上下左右各约 1 行的白色边距
    padded = ImageOps.expand(cropped, border=LINE_HEIGHT_PX, fill="white")

    # 6. 缩放到 A4 宽度一半（单位像素）
    target_w = int(HALF_A4_INCH * DPI)
    scale = target_w / padded.width
    target_h = int(padded.height * scale)
    final = padded.resize((target_w, target_h), Image.LANCZOS)

    # 7. 保存最终图像
    final.save(final_png)
    print(f"渲染完毕，已保存为：{final_png} ({target_w}×{target_h}px，{DPI} dpi)")
