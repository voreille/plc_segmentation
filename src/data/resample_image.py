from pathlib import Path

import numpy as np
from tqdm import tqdm

import SimpleITK as sitk

input_dir = "/home/val/python_wkspce/plc_seg/data/additional_cases_ct_lung"
output_dir = "/home/val/python_wkspce/plc_seg/data/additional_cases_ct_lung_2mm"
spacing = (2, 2, 2)


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    files = [f for f in Path(input_dir).rglob("*.nii.gz")]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)
    for f in tqdm(files):
        image = sitk.ReadImage(str(f))
        original_spacing = np.array(image.GetSpacing())
        original_size = np.array(image.GetSize())
        output_size = np.round(original_size * original_spacing /
                              np.array(spacing)).astype(int)
        resampler.SetReferenceImage(image)
        resampler.SetOutputSpacing(spacing)
        resampler.SetSize([int(k) for k in output_size])
        image = resampler.Execute(image)
        output_file = Path(output_dir) / (f.name.split(".")[0] + ".nii")
        sitk.WriteImage(image, str(output_file.resolve()))


if __name__ == '__main__':
    main()
