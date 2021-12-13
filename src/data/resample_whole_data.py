import os
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

project_dir = Path(__file__).resolve().parents[2]
dotenv_path = project_dir / ".env"

path_data_nii = project_dir / "data/interim/nii_raw"
path_output = project_dir / "data/interim/nii_resampled/"

path_output.mkdir(parents=True, exist_ok=True)
# cores = None
cores = 30


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    patient_list = [
        f.name.split("__")[0] for f in path_data_nii.rglob("*LUNG*")
    ]
    resampler = Resampler(
        path_data_nii,
        path_output,
        spacing=(1, 1, 1),
        interp_order=3,
        mask_smoothing=False,
        smoothing_radius=3,
    )
    if cores:
        with Pool(cores) as p:
            # tqdm(p.imap(resampler, patient_list), total=len(patient_list))
            p.map(resampler, patient_list)
    else:
        for p in tqdm(patient_list):
            resampler(p)


def split_lung_mask(lung_sitk):
    lung = sitk.GetArrayFromImage(lung_sitk)
    lung1 = np.zeros_like(lung, dtype=int)
    lung2 = np.zeros_like(lung, dtype=int)
    lung1[lung == 1] = 1
    lung2[lung == 2] = 1
    lung1 = sitk.GetImageFromArray(lung1)
    lung1.SetOrigin(lung_sitk.GetOrigin())
    lung1.SetSpacing(lung_sitk.GetSpacing())
    lung1.SetDirection(lung_sitk.GetDirection())
    lung2 = sitk.GetImageFromArray(lung2)
    lung2.SetOrigin(lung_sitk.GetOrigin())
    lung2.SetSpacing(lung_sitk.GetSpacing())
    lung2.SetDirection(lung_sitk.GetDirection())
    return lung1, lung2


def get_bb_mask_voxel(mask_sitk):
    mask = sitk.GetArrayFromImage(mask_sitk)
    positions = np.where(mask != 0)
    z_min = np.min(positions[0])
    y_min = np.min(positions[1])
    x_min = np.min(positions[2])
    z_max = np.max(positions[0])
    y_max = np.max(positions[1])
    x_max = np.max(positions[2])
    return x_min, y_min, z_min, x_max, y_max, z_max


def get_bb_mask_mm(mask_sitk):
    x_min, y_min, z_min, x_max, y_max, z_max = get_bb_mask_voxel(mask_sitk)
    return (*mask_sitk.TransformIndexToPhysicalPoint(
        [int(x_min), int(y_min), int(z_min)]),
            *mask_sitk.TransformIndexToPhysicalPoint(
                [int(x_max), int(y_max), int(z_max)]))


class Resampler():
    def __init__(
        self,
        path_nii,
        output_path,
        spacing=(1, 1, 1),
        interp_order=3,
        mask_smoothing=False,
        smoothing_radius=3,
    ):
        self.path_nii = path_nii
        self.output_path = output_path
        self.spacing = spacing
        self.interp_order = interp_order
        self.mask_smoothing = mask_smoothing
        self.smoothing_radius = smoothing_radius

    def __call__(
        self,
        patient_name,
    ):
        print(f"resampling patient {patient_name}")
        # t1 = time.time()
        ct_sitk = sitk.ReadImage(
            str((self.path_nii / (patient_name + "__CT.nii.gz")).resolve()))
        pt_sitk = sitk.ReadImage(
            str((self.path_nii / (patient_name + "__PT.nii.gz")).resolve()))
        mask_gtvt_sitk = sitk.ReadImage(
            str((self.path_nii /
                 (patient_name + "__GTV_T__RTSTRUCT__CT.nii.gz")).resolve()))
        mask_gtvl_sitk = sitk.ReadImage(
            str((self.path_nii /
                 (patient_name + "__GTV_L__RTSTRUCT__CT.nii.gz")).resolve()))
        mask_lung_sitk = sitk.ReadImage(
            str((self.path_nii /
                 (patient_name + "__LUNG__SEG__CT.nii.gz")).resolve()))
        # print(f"Time reading the files for patient {patient} : {time.time()-t1}")
        # t1 = time.time()
        output_shape = (np.array(ct_sitk.GetSize()) / np.array(self.spacing) *
                        np.array(ct_sitk.GetSpacing()))
        resampler = sitk.ResampleImageFilter()
        if self.interp_order == 3:
            resampler.SetInterpolator(sitk.sitkBSpline)
        # compute center
        # bb_gtvt = get_bb_mask_mm(mask_gtvt_sitk)
        # bb_gtvl = get_bb_mask_mm(mask_gtvl_sitk)
        # z_max = np.max([bb_gtvt[-1], bb_gtvl[-1]])
        # z_min = np.min([bb_gtvt[2], bb_gtvl[2]])
        bb_lung = get_bb_mask_mm(mask_lung_sitk)
        z_max = bb_lung[5]
        z_min = bb_lung[2]
        origin = np.array(ct_sitk.GetOrigin())
        origin[2] = z_min
        z_shape = int((z_max - z_min) / self.spacing[2])

        resampler.SetOutputOrigin(origin)
        resampler.SetOutputSpacing(self.spacing)
        resampler.SetSize(
            (int(output_shape[0]), int(output_shape[1]), z_shape))

        ct_sitk = resampler.Execute(ct_sitk)
        pt_sitk = resampler.Execute(pt_sitk)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        mask_gtvt_sitk = resampler.Execute(mask_gtvt_sitk)
        mask_gtvl_sitk = resampler.Execute(mask_gtvl_sitk)
        mask_lung_sitk = resampler.Execute(mask_lung_sitk)

        sitk.WriteImage(
            ct_sitk,
            str((self.output_path / (patient_name + "__CT.nii.gz")).resolve()))
        sitk.WriteImage(
            pt_sitk,
            str((self.output_path / (patient_name + "__PT.nii.gz")).resolve()))
        sitk.WriteImage(
            mask_gtvt_sitk,
            str((self.output_path /
                 (patient_name + "__GTV_T__RTSTRUCT__CT.nii.gz")).resolve()))
        sitk.WriteImage(
            mask_gtvl_sitk,
            str((self.output_path /
                 (patient_name + "__GTV_L__RTSTRUCT__CT.nii.gz")).resolve()))
        sitk.WriteImage(
            mask_lung_sitk,
            str((self.output_path /
                 (patient_name + "__LUNG__SEG__CT.nii.gz")).resolve()))


if __name__ == '__main__':
    main()
