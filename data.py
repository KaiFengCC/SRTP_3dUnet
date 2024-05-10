from torch.utils.data import Dataset
from torchvision import transforms
from utils import *

transform = transforms.Compose([
    transforms.ToTensor()
])


class ImageDataSet(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'masks'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segmentName = self.name[index]
        segmentPath = os.path.join(self.path, 'masks', segmentName)
        imagePath = os.path.join(self.path, 'imgs', segmentName)
        segmentImage = change_image_size(segmentPath)
        image = change_image_size(imagePath)
        return transform(image).unsqueeze(0).float(), transform(segmentImage).unsqueeze(0).float()


class ValidationImageDataSet(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'testimgs'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segmentName = self.name[index]
        #segmentPath = os.path.join(self.path, 'validationmasks', segmentName)
        imagePath = os.path.join(self.path, 'testimgs', segmentName)
        #segmentImage = change_image_size(segmentPath)
        image = change_image_size(imagePath)
        return transform(image).unsqueeze(0).float()


if __name__ == '__main__':
    # sort nii files
    filepath = r'/home/jiangtongling/myfiles/BE_new/data/source1'
    imagepath = r'/home/jiangtongling/myfiles/BE_new/data/imgs'
    maskpath = r'/home/jiangtongling/myfiles/BE_new/data/masks'
    sort_nii_image(filepath, imagepath, maskpath)

    # Dataset test
    MyDataSet = ImageDataSet(r'/home/jiangtongling/fyp/3dUnet/data')
