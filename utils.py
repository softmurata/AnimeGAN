import torch
from skimage.color import rgb2yuv

def _convert(input_, type_):
    return {'float': input_.float(), 'double': input_.double()}.get(type_, input_)


def _generic_transform_sk_4d(transform, in_type='', out_type=''):
    def apply_transform(input_):
        to_squeeze = (input_.dim()==3)
        device = input_.device
        input_ = input_.cpu()  # cpu mode
        input_ = _convert(input_, in_type)
        
        if to_squeeze:
            input_ = input_.unsqueeze(0)
        input_ = input_.permute(0, 2, 3, 1).detach().numpy()  # (b, c, h, w) -> (b, h, w, c)
        transformed = transform(input_)
        # to tensor
        output = torch.from_numpy(transformed).float().permute(0, 3, 1, 2)  # (b, h, w, c) -> (b, c, h, w)
        if to_squeeze:
           output = output.squeeze(0)
        output = _convert(output, out_type)
        # for cuda
        output = output.to(device)
        return output
    return apply_transform

def convert_rgb2yuv(rgb):
    rgb_to_yuv = _generic_transform_sk_4d(rgb2yuv)
    return rgb_to_yuv(rgb)



"""
def rgb2yuv(rgb):
    # rgb => pillow version
    # rgb => yuv
    img = (rgb + 1) / 2
    img = Image.fromarray(img.cpu().numpy().transpose(1, 2, 0))
    img_yuv = img.convert('YCbCr')
    

    return img_yuv

"""
