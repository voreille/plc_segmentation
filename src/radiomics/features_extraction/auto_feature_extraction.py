from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
from radiomics import featureextractor
from scipy.ndimage import (distance_transform_edt, binary_dilation,
                           binary_erosion)

from src.model_dino.train_dino import DEBUG

project_dir = Path(__file__).resolve().parents[3]
params_ct = project_dir / "src/radiomics/parameters/param_CT.yaml"
params_pt = project_dir / "src/radiomics/parameters/param_PT.yaml"
data_path = project_dir / "data/interim/nii_resampled"

DEBUG = True


def main():

    patients_list = [
        p.name.split("__")[0] for p in data_path.rglob("*__PT.nii.gz")
    ]
    patients_list = [p for p in patients_list if p != "PatientLC_63"]

    results_df = pd.DataFrame()

    extractor = Extractor(param_ct=params_ct,
                          param_pt=params_pt,
                          data_path=data_path)

    if DEBUG:
        patients_list = ["PatientLC_4"]
        results = [extractor(p) for p in patients_list]
        return 0
    else:
        with Pool(32) as p:
            results = p.map(extractor, patients_list)

    results_df = pd.concat(results, ignore_index=True)

    results_df.to_csv(project_dir /
                      "data/processed/radiomics/extracted_features_auto.csv")


class Extractor():

    def __init__(self, param_ct, param_pt, data_path):
        self.extractor_ct = featureextractor.RadiomicsFeatureExtractor(
            str(param_ct))
        self.extractor_pt = featureextractor.RadiomicsFeatureExtractor(
            str(param_pt))
        self.data_path = data_path

    def __call__(self, patient_id):
        print(f"extracting patient {patient_id} - START")
        image_pt = sitk.ReadImage(
            str(self.data_path / f"{patient_id}__PT.nii.gz"))
        image_ct = sitk.ReadImage(
            str(self.data_path / f"{patient_id}__CT.nii.gz"))
        mask_gtvt = sitk.ReadImage(
            str(self.data_path / f"{patient_id}__GTV_T__RTSTRUCT__CT.nii.gz"))
        mask_lung = sitk.ReadImage(
            str(self.data_path / f"{patient_id}__LUNG__SEG__CT.nii.gz"))
        try:
            peritumoral_mask = get_peritumoral_mask(image_pt, mask_gtvt,
                                                    mask_lung)
        except RuntimeError as e:
            print(f"{e} for the patient {patient_id}")
        if DEBUG:
            sitk.WriteImage(peritumoral_mask, f"{patient_id}__AUTO_SEG.nii.gz")
            return pd.DataFrame()
        results = self.extractor_ct.execute(image_ct, peritumoral_mask)
        results = {k: i for k, i in results.items() if "iagnostics" not in k}
        results_df = pd.DataFrame()
        results_df = results_df.append(
            {
                "patient_id": patient_id,
                "modality": "CT",
                "voi": "autoGTV_L",
                **results,
            },
            ignore_index=True)

        results = self.extractor_pt.execute(image_pt, peritumoral_mask)
        results = {k: i for k, i in results.items() if "iagnostics" not in k}
        results_df = results_df.append(
            {
                "patient_id": patient_id,
                "modality": "PT",
                "voi": "autoGTV_L",
                **results,
            },
            ignore_index=True)
        print(f"extracting patient {patient_id} - DONE ")
        return results_df


def sphere(r):
    output = np.zeros(3 * (2 * r + 1, ))
    x = np.linspace(-r, r, 2 * r + 1)
    x, y, z = np.meshgrid(x, x, x, indexing="xy")
    rho = np.sqrt(x**2 + y**2 + z**2)
    output[rho <= r] = 1
    return output


def get_peritumoral_mask(
    image_pt,
    mask_gtvt,
    mask_lung,
    dilation_radius=5,
):

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(mask_gtvt)
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing((1, 1, 1))
    resampler.SetInterpolator(sitk.sitkLinear)

    image_pt = resampler.Execute(image_pt)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_gtvt = resampler.Execute(mask_gtvt)
    mask_lung = resampler.Execute(mask_lung)

    gtvt = sitk.GetArrayFromImage(mask_gtvt)
    lung = (sitk.GetArrayFromImage(mask_lung) != 0) & (gtvt == 0)
    gtvt_dilated = binary_dilation(gtvt, structure=sphere(dilation_radius))
    new_mask = ((gtvt_dilated != 0) & (lung != 0)).astype(np.uint32)

    output = sitk.GetImageFromArray(new_mask.astype(np.uint32))
    output.SetOrigin(mask_gtvt.GetOrigin())
    return output


def get_peritumoral_mask_old(
    image_pt,
    mask_gtvt,
    mask_lung,
    dilation_radius=40,
    voi_radius=15,
):

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(mask_gtvt)
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing((1, 1, 1))
    resampler.SetInterpolator(sitk.sitkLinear)

    image_pt = resampler.Execute(image_pt)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_gtvt = resampler.Execute(mask_gtvt)
    mask_lung = resampler.Execute(mask_lung)

    pt = sitk.GetArrayFromImage(image_pt)
    gtvt = sitk.GetArrayFromImage(mask_gtvt)
    gtvt_dilated = gtvt
    structure = np.zeros((3, 3, 3))
    structure[:, 1, 1] = 1
    structure[1, :, 1] = 1
    structure[1, 1, :] = 1
    for _ in range(dilation_radius):
        gtvt_dilated = binary_dilation(gtvt_dilated, structure=structure)
    lung = (sitk.GetArrayFromImage(mask_lung) != 0) & (gtvt == 0)
    lung_eroded = lung
    for _ in range(dilation_radius // 4):
        lung_eroded = binary_erosion(lung_eroded, structure=structure)
    pt = pt * (gtvt_dilated != 0) * (lung_eroded != 0)
    position_suvmax = np.where(pt == np.max(pt))

    distances = distance_transform_edt((pt != 0))
    ind = np.argmin(distances[position_suvmax])

    position_suvmax = [position_suvmax[k][ind] for k in range(3)]
    new_mask = np.zeros_like(gtvt_dilated)
    new_mask[position_suvmax[0] - voi_radius:position_suvmax[0] + voi_radius +
             1, position_suvmax[1] - voi_radius:position_suvmax[1] +
             voi_radius + 1,
             position_suvmax[2] - voi_radius:position_suvmax[2] + voi_radius +
             1, ] = sphere(voi_radius)

    gtvt_dilated = gtvt
    for _ in range(dilation_radius // 4):
        gtvt_dilated = binary_dilation(gtvt_dilated, structure=structure)

    new_mask[(new_mask != 0) & (lung_eroded == 0)] = 0
    new_mask[(new_mask != 0) & (gtvt_dilated == 1)] = 0
    output = sitk.GetImageFromArray(new_mask.astype(np.uint32))
    # output = sitk.GetImageFromArray(lung_eroded.astype(np.uint32))
    output.SetOrigin(mask_gtvt.GetOrigin())
    return output


def find_closest_position(distances, suvmax_position, radius):
    # start from suvmax and straightest path until distance >= radius and contains suvmaxposition
    # convex hull if dilation is not sufficient, to do before computing suvmax position
    pass


if __name__ == '__main__':
    main()