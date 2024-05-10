import os
import SimpleITK as sitk
from scipy.ndimage import interpolation as itpl
import torch
import numpy as np

#imagePath = r'D:\fyp\3Dunet\data\imgs'
#maskPath = r'D:\fyp\3Dunet\data\masks'


def sort_nii_image(niifile, imagePath, maskPath):
    for root, dirs, files in os.walk(niifile):
        for file in files:
            # read nii files
            img_path = os.path.join(root, file)
            img = sitk.ReadImage(img_path)
            origin = img.GetOrigin()
            direction = img.GetDirection()
            space = img.GetSpacing()
            img_arr = sitk.GetArrayFromImage(img).transpose([2, 0, 1])
            print('shape:', np.shape(img_arr))
            Img = sitk.GetImageFromArray(img_arr)
            if "mask" in file:
                #new_nii_dir = os.path.join(maskPath, '{}.nii'.format(file[0:5]))
                filename = file[0: file.find('_')] + '.nii'
                new_nii_dir = os.path.join(maskPath, filename)
                Img.SetOrigin(origin)
                Img.SetDirection(direction)
                Img.SetSpacing(space)
                sitk.WriteImage(Img, new_nii_dir)
            else:
                #new_nii_dir = os.path.join(imagePath, '{}.nii'.format(file[0:5]))
                filename = file
                new_nii_dir = os.path.join(imagePath, file)
                Img.SetOrigin(origin)
                Img.SetDirection(direction)
                Img.SetSpacing(space)
                sitk.WriteImage(Img, new_nii_dir)



def change_image_size(path):
    image = sitk.ReadImage(path)
    image_arr = sitk.GetArrayFromImage(image)
    #image_resized_arr = itpl.zoom(image_arr, (80 / 72, 160 / 152, 160 / 152), mode='constant')
    #image_resized = sitk.GetImageFromArray(image_resized_arr)
    #print(type(image_resized_arr))
    return np.int16(image_arr)


def gray2RGB(img):
    out_img = torch.cat((img, img, img), 0)
    return out_img


if __name__ == '__main__':
    pass
