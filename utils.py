
def rgb2yuv(rgb):
    # rgb => pillow version
    # rgb => yuv
    img = (img + 1) / 2
    img_yuv = img.convert('YCbCr')

    return img_yuv