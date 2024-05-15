import numpy as np
import SimpleITK as sitk
import os

def calculate_dice_coefficient(nii1_path, nii2_path):

    img1 = sitk.ReadImage(nii1_path)
    img2 = sitk.ReadImage(nii2_path)


    img1_arr = sitk.GetArrayFromImage(img1)
    img2_arr = sitk.GetArrayFromImage(img2)

    if img1_arr.shape != img2_arr.shape:
        img2_arr = img2_arr.transpose((1, 2, 0))

    intersection = np.sum(img1_arr * img2_arr)
    dice_coefficient = 2. * intersection / (np.sum(img1_arr) + np.sum(img2_arr))

    return dice_coefficient


oriData_dir = 'oriData'
segData_dir = 'segData'

sum = 0
for filename in os.listdir(oriData_dir):
    if filename.endswith('.nii'):
        nii1_path = os.path.join(oriData_dir, filename)
        nii2_path = os.path.join(segData_dir, filename.replace('.nii', '_mask.nii'))
        dice_coefficient = calculate_dice_coefficient(nii1_path, nii2_path)
        sum = sum + dice_coefficient
        print(f'Dice coefficient for {filename}: {dice_coefficient}')

print('avg_dice_coefficient: ', sum / len(os.listdir(oriData_dir)))