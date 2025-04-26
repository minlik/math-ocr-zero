from datasets import load_dataset


local_dir = '/root/data/data/deepmath-1000/'

ds = load_dataset(local_dir, split='train')

print(ds[:10])