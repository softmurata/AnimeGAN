import subprocess
import random
import os

real_dataset_dir = './dataset/real/'
train_split_ratio = 0.8


real_images = os.listdir(real_dataset_dir)

real_image_num = len(real_images)

train_dataset_num = int(real_image_num * train_split_ratio)

train_real_images = random.sample(real_images, train_dataset_num)
test_real_images = [r for r in real_images if r not in train_real_images]

train_real_dataset_dir = real_dataset_dir + 'train/'
test_real_dataset_dir = real_dataset_dir + 'test/'

os.makedirs(train_real_dataset_dir, exist_ok=True)
os.makedirs(test_real_dataset_dir, exist_ok=True)

for train_r in train_real_images:
    train_r_path = train_real_dataset_dir + train_r
    args = 'cp -f {} {}'.format(real_dataset_dir + train_r, train_r_path)
    subprocess.call(args, shell=True)

for test_r in test_real_images:
    test_r_path = test_real_dataset_dir + test_r
    args = 'cp -f {} {}'.format(real_dataset_dir + test_r, test_r_path)
    subprocess.call(args, shell=True)
