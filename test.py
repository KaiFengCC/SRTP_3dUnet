import os
from os import mkdir
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data import *
from utils import *
from Net import *
import numpy as np
import sys


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
if torch.cuda.is_available():
    device = 'cuda'
    print("Using cuda")
else:
    device = 'cpu'
    print("Using CPU")

weightPath = sys.argv[1]
#weightPath = r'G:/LAB/brainExtraction/0.1/params_3/unet.pth'#
filePath = sys.argv[2]
#filePath = r'G:/LAB/brainExtraction/0.1/data/source2'#
if __name__ =='__main__':
    dataPath = filePath[0:filePath.rfind('/')]
    print('Your data path: ' + dataPath)
    imagePath = dataPath + '/testimgs'
    maskPath = dataPath + '/testmasks'
    resultPath = dataPath + '/results'
    if not os.path.exists(imagePath):
        mkdir(imagePath)
    if not os.path.exists(maskPath):
        mkdir(maskPath)
    if not os.path.exists(resultPath):
        mkdir(resultPath)

    sort_nii_image(filePath, imagePath, maskPath)
    validationDataset = ValidationImageDataSet(dataPath)

    for i in range(len(validationDataset)):
         print(validationDataset.name[i])

    batchSize = 1

    net = UNet().to(device)

    if os.path.exists(weightPath):
        #net.load_state_dict(torch.load(weightPath, map_location=torch.device(device)))
        net.load_state_dict(torch.load(weightPath, map_location = 'cpu'))
        print("Loading Weight Successful")
    else:
        print("Loading Weight Failed")


    for i in range(len(validationDataset)):
        image = validationDataset[i]
        # print(image)
        image = torch.unsqueeze(image, 0)
        image = image.to(device)
        with torch.no_grad():
            outImage = net(image)
        result = outImage.clone().detach().cpu().numpy()
        result = result.squeeze(0).squeeze(0)
        # mask_resized = itpl.zoom(result, (152 / 160, 72 / 80, 152 / 160), mode='constant')
        mask_ = result.transpose([2, 0, 1])
        origin_nii = sitk.ReadImage(filePath + '/' + validationDataset.name[i])#r'D:\0a\brainExtraction\data\source\subj011.nii')
        origin = origin_nii.GetOrigin()
        direction = origin_nii.GetDirection()
        space = origin_nii.GetSpacing()
        origin_arr = sitk.GetArrayFromImage(origin_nii)
        # print(np.shape(origin_arr))
        savedImg = sitk.GetImageFromArray(mask_)
        savedImg.SetOrigin(origin)
        savedImg.SetDirection(direction)
        savedImg.SetSpacing(space)
        sitk.WriteImage(savedImg, resultPath + '/' + validationDataset.name[i][0:validationDataset.name[i].find('.')] + '_mask.nii')#r'D:\0a\brainExtraction\data\results\subj011_mask.nii')
    print('Brain Extraction Done')

    

