from pathlib import Path

import numpy as np
import h5py
from tqdm import tqdm
import SimpleITK as sitk

project_dir = Path(__file__).resolve().parents[2]
dotenv_path = project_dir / ".env"

path_data_nii = project_dir / "data/interim/nii_resampled"
path_mask_lung_nii = project_dir / "data/interim/nii_resampled"

only_gtv_slice = True
standardize_method_pet = "only-lung-slices"

path_output = project_dir / "data/processed/hdf5_2d_pet_standardized_lung_slices"
path_output.mkdir(parents=True, exist_ok=True)


def main():
    patient_list = [
        f.name.split("__")[0] for f in path_mask_lung_nii.rglob("*LUNG*")
    ]

    path_file = ((path_output / 'data.hdf5').resolve())
    if path_file.exists():
        path_file.unlink()  # delete file if exists
    hdf5_file = h5py.File(path_file, 'a')
    for patient in tqdm(patient_list):

        image, mask = parse_image(
            patient,
            path_data_nii,
            path_mask_lung_nii,
            only_gtv_slices=only_gtv_slice,
            standardize_method_pet=standardize_method_pet)

        hdf5_file.create_group(f"{patient}")
        hdf5_file.create_dataset(f"{patient}/image",
                                 data=image,
                                 dtype="float32")
        hdf5_file.create_dataset(f"{patient}/mask", data=mask, dtype="uint16")

    hdf5_file.close()


def to_np(x):
    return np.squeeze(np.transpose(sitk.GetArrayFromImage(x), (2, 1, 0)))


def get_bb_mask_voxel(mask):
    positions = np.where(mask != 0)
    x_min = np.min(positions[0])
    y_min = np.min(positions[1])
    z_min = np.min(positions[2])
    x_max = np.max(positions[0])
    y_max = np.max(positions[1])
    z_max = np.max(positions[2])
    return x_min, y_min, z_min, x_max, y_max, z_max


def get_bb_mask_sitk_voxel(mask_sitk):
    mask = to_np(sitk.GetArrayFromImage(mask_sitk))
    positions = np.where(mask != 0)
    x_min = np.min(positions[0])
    y_min = np.min(positions[1])
    z_min = np.min(positions[2])
    x_max = np.max(positions[0])
    y_max = np.max(positions[1])
    z_max = np.max(positions[2])
    return x_min, y_min, z_min, x_max, y_max, z_max


def get_bb_mask_mm(mask_sitk):
    x_min, y_min, z_min, x_max, y_max, z_max = get_bb_mask_sitk_voxel(
        mask_sitk)
    return (*mask_sitk.TransformIndexToPhysicalPoint(
        [int(x_min), int(y_min), int(z_min)]),
            *mask_sitk.TransformIndexToPhysicalPoint(
                [int(x_max), int(y_max), int(z_max)]))


def slice_volumes(*args, s1=0, s2=-1):
    output = []
    for im in args:
        output.append(im[:, :, s1:s2 + 1])

    return output


def standardize_within_mask(image, mask):
    values = image[mask != 0]
    std = np.std(values)
    m = np.mean(values)
    return (image - m) / std


def standardize_containing_slices(image, mask):
    bb = get_bb_mask_voxel(mask)
    std = np.std(image[:, :, bb[2]:bb[-1]])
    m = np.mean(image[:, :, bb[2]:bb[-1]])
    return (image - m) / std


def standardize(image):
    std = np.std(image)
    m = np.mean(image)
    return (image - m) / std


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


def parse_image(patient_name,
                path_nii,
                path_lung_mask_nii,
                mask_smoothing=False,
                smoothing_radius=3,
                only_gtv_slices=False,
                standardize_method_pet="full-image"):
    """Parse the raw data of HECKTOR 2020

    Args:
        folder_name ([Path]): the path of the folder containing 
        the 3 sitk images (ct, pt and mask)
    """
    # t1 = time.time()
    ct_sitk = sitk.ReadImage(
        str((path_nii / (patient_name + "__CT.nii.gz")).resolve()))
    pt_sitk = sitk.ReadImage(
        str((path_nii / (patient_name + "__PT.nii.gz")).resolve()))
    mask_gtvt_sitk = sitk.ReadImage(
        str((path_nii /
             (patient_name + "__GTV_T__RTSTRUCT__CT.nii.gz")).resolve()))
    mask_gtvl_sitk = sitk.ReadImage(
        str((path_nii /
             (patient_name + "__GTV_L__RTSTRUCT__CT.nii.gz")).resolve()))
    mask_lung_sitk = sitk.ReadImage(
        str((path_lung_mask_nii /
             (patient_name + "__LUNG__SEG__CT.nii.gz")).resolve()))

    mask_lung1_sitk, mask_lung2_sitk = split_lung_mask(mask_lung_sitk)
    if mask_smoothing:
        smoother = sitk.BinaryMedianImageFilter()
        smoother.SetRadius(int(smoothing_radius))
        mask_gtvt_sitk = smoother.Execute(mask_gtvt_sitk)
        mask_gtvl_sitk = smoother.Execute(mask_gtvl_sitk)
        mask_lung1_sitk = smoother.Execute(mask_lung1_sitk)
        mask_lung2_sitk = smoother.Execute(mask_lung2_sitk)
    mask_gtvt = to_np(mask_gtvt_sitk)
    mask_gtvl = to_np(mask_gtvl_sitk)
    mask_lung1 = to_np(mask_lung1_sitk)
    mask_lung2 = to_np(mask_lung2_sitk)

    ct = to_np(ct_sitk)
    pt = to_np(pt_sitk)
    if standardize_method_pet == "within-lung":
        pt = standardize_within_mask(pt, mask_lung1 + mask_lung2)
    elif standardize_method_pet == "full-image":
        pt = standardize(pt)
    elif standardize_method_pet == "only-lung-slices":
        pt = standardize_containing_slices(pt, mask_lung1 + mask_lung2)
    elif standardize_method_pet == "None":
        pass
    else:
        raise ValueError(
            f"the method {standardize_method_pet} is not defined.")

    if only_gtv_slices:
        bb_gtvt = get_bb_mask_voxel(mask_gtvt)
        bb_gtvl = get_bb_mask_voxel(mask_gtvl)
        z_max = np.max([bb_gtvt[-1], bb_gtvl[-1]])
        z_min = np.min([bb_gtvt[2], bb_gtvl[2]])
    else:
        bb_lung = get_bb_mask_voxel(mask_lung1 + mask_lung2)
        z_max = bb_lung[-1]
        z_min = bb_lung[2]

    ct, pt, mask_gtvt, mask_gtvl, mask_lung1, mask_lung2 = slice_volumes(
        ct,
        pt,
        mask_gtvt,
        mask_gtvl,
        mask_lung1,
        mask_lung2,
        s1=z_min,
        s2=z_max,
    )

    image = np.stack([ct, pt, np.zeros_like(ct)], axis=-1)
    mask = np.stack([mask_gtvt, mask_gtvl, mask_lung1, mask_lung2], axis=-1)

    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0

    return image, mask.astype(np.uint8)


if __name__ == '__main__':
    main()
