# coding=gbk
import tkinter as tk
from tkinter import filedialog
import torch
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from Net import UNet  # ȷ����������ģ���ļ���ģ������ȷ��·��������
from data import *
from utils import *

def load_model(model_path):
    model = UNet()  # ȷ��UNet����Ҫ���ص�ģ�͵���ȷ����
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def segment_nifti(model, nifti_path, device='cpu'):
    # ���ز�����ͼ��
    image = change_image_size(nifti_path)
    # ȷ��ͼ����numpy���飬��ת��Ϊfloat���ͣ���Ϊ�����PyTorchģ�Ͷ����������������͹�����
    image = image.astype(np.float32)
    # ���һ����ά������ͨ��������Ѿ���������Ҫ��һ��
    if len(image.shape) == 3:
        image = image[np.newaxis, :]
    # ���һ����ά����������
    image = image[np.newaxis, :]
    # ��numpy����ת��ΪPyTorch��Tensor
    image = torch.from_numpy(image).to(device)
    
    with torch.no_grad():
        outImage = model(image)
    result = outImage.clone().detach().cpu().numpy()
    result = result.squeeze(0).squeeze(0)
    mask_ = result.transpose([2, 0, 1])
    origin_nii = sitk.ReadImage(nifti_path)
    print(origin_nii.__dict__)
    origin = origin_nii.GetOrigin()
    direction = origin_nii.GetDirection()
    space = origin_nii.GetSpacing()
    size = origin_nii.GetSize()
    print(size)
    savedImg = sitk.GetImageFromArray(mask_)
    savedImg.SetOrigin(origin)
    savedImg.SetDirection(direction)
    savedImg.SetSpacing(space)
    
    return savedImg

def save_nifti(output, original_nifti_path, output_path):
    original_nii = nib.load(original_nifti_path)
    new_nii = nib.Nifti1Image(output, original_nii.affine, original_nii.header)
    nib.save(new_nii, output_path)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("NIfTI Segmenter")
        self.model = None  # �����洢ģ��

        self.btn_load_model = tk.Button(root, text="Load Model", command=self.load_model).pack()
        self.btn_load_nifti = tk.Button(root, text="Load NIfTI File", command=self.load_nifti).pack()
        self.btn_segment = tk.Button(root, text="Segment NIfTI File", command=self.segment).pack()
        self.lbl_status = tk.Label(root, text="Status: Ready").pack()

    def load_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if model_path:
            self.model = load_model(model_path)
            self.update_status(f"Loaded model: {model_path}")

    def load_nifti(self):
        self.nii_path = filedialog.askopenfilename(filetypes=[("NIfTI File", "*.nii"), ("NIfTI File", "*.nii.gz")])
        if self.nii_path:
            self.update_status(f"Loaded NIfTI: {self.nii_path}")

    def segment(self):
        if not self.model or not hasattr(self, 'nii_path'):
            self.update_status("Error: Load a model and a NIfTI file first!")
            return

        self.update_status("Segmenting...")
        try:
            output = sitk.GetArrayFromImage(segment_nifti(self.model, self.nii_path))

            output_path = self.nii_path.replace(".nii", "_segmented.nii").replace(".gz", "")
            save_nifti(output, self.nii_path, output_path)
            self.update_status(f"Segmentation complete: {output_path}")
        except Exception as e:
            self.update_status(f"An error occurred: {e}")
            print(e)

    def update_status(self, message):
        tk.Label(self.root, text=f"Status: {message}").pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()



