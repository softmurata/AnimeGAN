from torchvision import datasets
from torch.utils.data import Dataset

def data_load(path, transform, batch_size, shuffle=True, drop_last=True):
    dataset = datasets.ImageFolder(path, transform)

    data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=2,
                                          drop_last=drop_last)

    return data_loader


class ImageDataset(Dataset):

    def __init__(self, dataset_path, args):

        
        pass

    def load_images(self):
        pass


    def __len__(self):
        pass

    def __getitem__(self, idx):



        return data

