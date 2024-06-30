from datasets import load_dataset
import os

# データセットをダウンロード
ds = load_dataset("marcelarosalesj/inria-person")

# データセットの保存先ディレクトリ
os.makedirs('data/positive', exist_ok=True)
os.makedirs('data/negative', exist_ok=True)

# 画像の保存
positive_count = 0
negative_count = 0

for i, example in enumerate(ds['train']):
    if example['label'] == 1 and positive_count < 500:  # 人間が写っている画像
        image = example['image']
        image.save(f'data/positive/img_{positive_count}.jpg')
        positive_count += 1
    elif example['label'] == 0 and negative_count < 500:  # 人間が写っていない画像
        image = example['image']
        image.save(f'data/negative/img_{negative_count}.jpg')
        negative_count += 1

    if positive_count >= 500 and negative_count >= 500:
        break

print("Downloaded and saved 500 positive and 500 negative images.")
