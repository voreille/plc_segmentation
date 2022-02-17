from pathlib import Path
from itertools import product
from multiprocessing import Pool
from sklearn import feature_extraction

from tqdm import tqdm
import pandas as pd
import numpy as np
from okapy.dicomconverter.converter import ExtractorConverter

project_dir = Path(__file__).resolve().parents[2]

data_path = project_dir / "data/interim/nii_resampled"

params_ct = str(project_dir / "src/radiomics/param_CT.yaml")
params_pt = str(project_dir / "src/radiomics/param_PT.yaml")

vois = ["GTV_L", "GTV_T", "GTV_N"]


def main():
    feature_extractor = ExtractorConverter.from_params(
        project_dir / "src/radiomics/params_okapy.yaml")
    df = feature_extractor("/home/valentin/python_wkspce/plc_segmentation/data/raw/dicom/case_1")
    print("yo")


if __name__ == '__main__':
    main()