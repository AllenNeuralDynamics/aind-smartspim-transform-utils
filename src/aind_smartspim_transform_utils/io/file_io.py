#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 13:59:08 2025

@author: nicholas.lusk
"""

import json
import os
from glob import glob

import ants
import boto3
import numpy as np
import pandas as pd
from imlib.IO.cells import get_cells
from tqdm import tqdm


def _check_transforms(transforms_list: list):
    """
    There are many versions of registration. This check informs the user if
    the dataset was directly registered or if it used template-based

    Parameters
    ----------
    transforms_list : list
        DESCRIPTION.

    Returns
    -------
    None.

    """


def _read_json_as_dict(filepath: str) -> dict:
    """
    Loads json file

    Parameters
    ----------
    filepath : str
        path to json file

    Returns
    -------
    data: dict
        loaded json formated as a dict

    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist.")

    with open(filepath, "r") as fp:
        data = json.load(fp)

    return data


def _load_data_from_s3(
    dataset_paths: list, bucket: str
) -> dict:  # pragma: no cover
    client = boto3.client("s3")

    for path in dataset_paths:
        path = path.split(bucket)[-1][1:]
        try:
            res = client.get_object(Bucket=bucket, Key=path)
            data = json.loads(res["Body"].read())
        except:
            pass

    if "data" not in locals():
        file = os.path.basename(dataset_paths[0])
        raise FileNotFoundError(
            f"Could not locate {file}. Please check provided path"
        )

    return data


def _load_data_from_local(dataset_paths: list) -> dict:  # pragma: no cover
    for path in dataset_paths:
        try:
            data = _read_json_as_dict(path)
        except:
            pass

    if "data" not in locals():
        file = os.path.basename(dataset_paths[0])
        raise FileNotFoundError(
            f"Could not locate {file}. Please check provided path"
        )

    return data


def _download_data_from_s3(data_folder: str, files: list, dest: str):
    """
    Downloads data files from s3 bucket and saves them to the provided
    path. If you are working in a Code Ocean Capsule it is recommended
    that you set your destination to '/scratch/'.

    Parameters
    ----------
    data_folder : str
        folder on s3 where the data files you wish to download are located
    files : list
        list of files to download from the specified folder
    dest : str
        the local directory where the data will be stored

    Returns
    -------
    None.

    """

    client = boto3.client("s3")

    for file in files:
        fname = os.path.join(dest, file)
        s3_object_key = f"{data_folder}/{file}"

        meta_data = client.head_object(
            Bucket="aind-open-data", Key=s3_object_key
        )
        total_len = int(meta_data.get("ContentLength", 0))
        bar = "{percentage:.1f}%|{bar:25} | {rate_fmt} | {desc}"

        with tqdm(
            total=total_len,
            desc=s3_object_key,
            bar_format=bar,
            unit="B",
            unit_scale=True,
        ) as pbar:
            with open(fname, "wb") as f:
                client.download_fileobj(
                    "aind-open-data", s3_object_key, f, Callback=pbar.update
                )


def _read_xml_as_df(fpath: str) -> pd.DataFrame:  # pragma: no cover
    file_cells = get_cells(fpath)

    cells = []

    for cell in file_cells:
        cells.append(
            [
                cell.x,
                cell.y,
                cell.z,
            ]
        )

    df = pd.DataFrame(cells, columns=["x", "y", "z"])

    return df


def get_transforms(dataset_path: str, channel: str, dest=None) -> list:
    """
    Get the paths to the transforms for the provided dataset as well as the
    required acquisition information for transforming points

    Parameters
    ----------
    dataset_path : str
        path to the transforms created during registration

    channel : str
        channel that was registered

    dest : str Optional
        destination for saving transforms if loaded from s3

    Returns
    -------
    transforms : list
        list of all transform files from registration

    """

    transforms = {}
    dataset_path = str(dataset_path)

    if "s3://" in dataset_path:
        if not os.path.exists(dest):
            raise FileExistsError(f"Provided directory {dest} does not exist")

        data_folder = os.path.join(
            dataset_path.split("/")[-1], "image_atlas_alignment", channel
        )

        files = [
            "ls_to_template_SyN_0GenericAffine.mat",
            "ls_to_template_SyN_1InverseWarp.nii.gz",
            "ls_to_template_SyN_1Warp.nii.gz",
        ]

        _download_data_from_s3(data_folder, files, dest)

        transforms_path = dest

    else:
        transforms_path = os.path.join(
            dataset_path, "image_atlas_alignment", channel
        )

    try:
        transforms["points_to_ccf"] = [
            glob(os.path.join(transforms_path, "*SyN_0GenericAffine.mat"))[0],
            glob(os.path.join(transforms_path, "*InverseWarp.nii.gz"))[0],
        ]
    except:
        raise FileNotFoundError(
            "Could not find files needed for moving points from light sheet to CCF"
        )

    try:
        transforms["points_from_ccf"] = [
            glob(os.path.join(transforms_path, "*SyN_1Warp.nii.gz"))[0],
            glob(os.path.join(transforms_path, "*SyN_0GenericAffine.mat"))[0],
        ]
    except:
        raise FileNotFoundError(
            "Could not find files needed for moving points from CCF to light sheet"
        )

    return transforms


def get_acquisition(dataset_path: str, bucket=None) -> dict:
    """
    Loads the acquisition metadata from processing manifest that is needed for
    properly registering the data. Dataset location can be local or on s3.

    local example: /data/SmartSPIM_696515_2024-04-17_17-18-37_stitched_2025-05-14_05-41-57
    s3 example: s3://aind-open-data/SmartSPIM_696515_2024-04-17_17-18-37_stitched_2025-05-14_05-41-57

    Parameters
    ----------
    dataset_path : str
        the root directory of the stitched dataset that you want to process
    bucket : str Optional
        the bucket that contains stitched dataset that you want to process.
        Only needed if you are loading from s3

    Returns
    -------
    dict
        information on how the data was acquired

    """

    dataset_path = str(dataset_path)

    try:
        raw_path = dataset_path
    except:
        raise ValueError(
            f"Please provided stitched folder path. Provided path was {dataset_path}"
        )

    acquisition_paths = [
        f"{raw_path}/acquisition.json",
    ]

    if "s3://" in dataset_path:
        return _load_data_from_s3(acquisition_paths, bucket)
    else:
        return _load_data_from_local(acquisition_paths)


def get_image_metadata(dataset_path: str, channel: str, bucket=None) -> dict:
    """
    Loads metadata on the zarr file. Information on the image shape is required
    for transforming points

    Parameters
    ----------
    dataset_path : str
        the root directory of the stitched dataset that you want to process
    channel : dict
        channel that was used for registration
    bucket : str Optional
        the bucket that contains stitched dataset that you want to process.
        Only needed if you are loading from s3

    Returns
    -------
    dict
        metadata on the image that was used during registration

    """

    zarr_path = os.path.join(
        str(dataset_path),
        "image_tile_fusing/OMEZarr",
        f"{channel}.zarr",
        "0",
        ".zarray",
    )

    if "s3://" in zarr_path:
        return _load_data_from_s3([zarr_path], bucket)
    else:
        return _load_data_from_local([zarr_path])


def load_cell_locations(  # pragma: no cover
    dataset: str, channel: str
) -> pd.DataFrame:
    cell_path = os.path.join(dataset, "image_cell_segmentation", channel)

    files = [
        "detected_cells.csv",
        "classified_cells.xml",
        "detected_cells.xml",
    ]

    found = False

    for file in files:
        fpath = os.path.join(cell_path, file)

        if os.path.exists(fpath):
            found = True
            print(f"Found file {file} in path. Loading cells from here")

            if "csv" in file:
                df = pd.read_csv(fpath, index_col=0)
            else:
                df = _read_xml_as_df(fpath)

    if not found:
        raise FileNotFoundError(
            f"Could not find a cell location file for {dataset}"
        )

    return df


def load_ants_nifti(filepath: str) -> dict:
    """
    Loads an ants image object and returns image information


    Parameters
    ----------
    filepath : str
        location of the ants nii.gz file

    Returns
    -------
    image: ants.image
        ants object

    description: dict
        dictionary with descriptive information related to the ants image

    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist.")

    ants_img = ants.image_read(filepath)

    description = {
        "orientation": ants_img.orientation,
        "dims": ants_img.dimension,
        "scale": ants_img.spacing,
        "origin": ants_img.origin,
        "direction": ants_img.direction[np.where(ants_img.direction != 0)],
    }

    return ants_img, description
