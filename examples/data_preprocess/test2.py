from PIL import Image
from datasets import load_dataset, Sequence
from datasets import Image as ImageFeature

train_data_path = '/Users/kuan/code/math-ocr-zero/data/deepmath-ocr-1000/'
# train_data_path = 'hiyouga/geometry3k'
train_dataset = load_dataset(train_data_path, split='train')
# train_dataset = train_dataset.cast_column("images", Sequence(ImageFeature()))
# train_dataset.set_format(type="pil", columns=["images"])

data = train_dataset[0]
image = data['images'][0]
print("image:", isinstance(image, Image.Image))