import argparse
import cv2
import numpy as np
import os
# smoothing and gray scale

def apply_smoothing_and_grayscale(img_size, anime_name):
    # get anime images
    anime_dataset_dir = './dataset/{}/anime/'.format(anime_name)
    anime_smooth_dir = './dataset/{}/anime_smooth/'.format(anime_name)
    anime_gray_dir = './dataset/{}/anime_gray/'.format(anime_name)

    os.makedirs(anime_smooth_dir, exist_ok=True)
    os.makedirs(anime_gray_dir, exist_ok=True)

    anime_images = sorted(os.listdir(anime_dataset_dir))
    
    # smoothing parameters
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size)).astype(np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)


    for anime_image_path in anime_images:
        # load images
        bgr_image = cv2.imread(anime_dataset_dir + anime_image_path)
        gray_image = cv2.imread(anime_dataset_dir + anime_image_path, 0)

        # resize
        bgr_image = cv2.resize(bgr_image, (img_size, img_size))
        pad_image = np.pad(bgr_image, ((2, 2), (2, 2), (0, 0)), mode='reflect')
        gray_image = cv2.resize(gray_image, (img_size, img_size))

        # get edges
        edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
        dilation = cv2.dilate(edges, kernel)

        gauss_image = np.copy(bgr_image)
        idx = np.where(dilation!=0)

        for i in range(np.sum(dilation != 0)):
            gauss_image[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(pad_image[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))

            gauss_image[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(pad_image[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            
            gauss_image[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(pad_image[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

        # save
        smooth_img_path = anime_smooth_dir + anime_image_path
        cv2.imwrite(smooth_img_path, gauss_image)
        gray_img_path = anime_gray_dir + anime_image_path
        cv2.imwrite(gray_img_path, gray_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--anime_name', type=str, default='sentochihiro')
    args = parser.parse_args()

    apply_smoothing_and_grayscale(args.img_size, args.anime_name)

