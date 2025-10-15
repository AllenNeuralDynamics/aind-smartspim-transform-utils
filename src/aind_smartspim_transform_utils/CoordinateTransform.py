#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 16:21:46 2025

@author: nicholas.lusk
"""

import os
from glob import glob

import numpy as np
import pandas as pd

from aind_smartspim_transform_utils import base_dir
from aind_smartspim_transform_utils.io import file_io as fio
from aind_smartspim_transform_utils.utils import utils


def _check_path(name: str, sub_folder: str):  # pragma: no cover
    fpath = os.path.join(base_dir, name, sub_folder)

    if not os.path.exists(fpath):
        os.makedirs(fpath)
        return False

    return True


def _get_ccf_transforms(name: str) -> dict:
    """
    Loads static transforms that move points between the CCFv3 and
    SmartSPIM-LCA template.

    Parameters
    ----------
    name : str
        name of the template you want to load. Currently 'SmartSPIM-LCA'
        is the only option

    Returns
    -------
    transforms: dict
        the transforms needed for moving pts forward or backward

    """

    transforms = {}

    if name.lower() == "smartspim_lca":
        file_check = _check_path(name, "transforms")

        data_folder = "SmartSPIM-template_2024-05-16_11-26-14"
        files = [
            "spim_template_to_ccf_syn_0GenericAffine_25.mat",
            "spim_template_to_ccf_syn_1Warp_25.nii.gz",
            "spim_template_to_ccf_syn_1InverseWarp_25.nii.gz",
        ]
        dest = os.path.join(base_dir, name.lower(), "transforms")

        if not file_check:
            fio._download_data_from_s3(data_folder, files, dest)

        root = os.path.join(base_dir, name, "transforms")

        transforms["points_to_ccf"] = [
            glob(os.path.join(root, "*.mat"))[0],
            glob(os.path.join(root, "*1InverseWarp_25.nii.gz"))[0],
        ]

        transforms["points_from_ccf"] = [
            glob(os.path.join(root, "*1Warp_25.nii.gz"))[0],
            glob(os.path.join(root, "*.mat"))[0],
        ]
    else:
        raise ValueError(
            f"name: {name} is not a currently available transformation"
        )

    return transforms


def _get_ccf_template(name):
    """
    loads the nifti file for the ccf you need based on the nape you provide
    to the transfrom function. If you currently do not have that file it will
    download it from S3

    Parameters
    ----------
    name : str
        The name of the transform process you are using

    Returns
    -------
    ants_ccf : ants.array
        The template volume loaded as an ants object
    ccf_info : dict
        dictionary describing the template. Includes the orientation, axes
        directions, dimensions, scale, and origin of the ants.array

    """

    if name.lower() == "smartspim_lca":
        file_check = _check_path(name, "ccf")

        data_folder = "SmartSPIM-template_2024-05-16_11-26-14"
        files = ["ccf_average_template_25.nii.gz"]
        dest = os.path.join(base_dir, name.lower(), "ccf")

        if not file_check:
            fio._download_data_from_s3(data_folder, files, dest)

        root = os.path.join(base_dir, name, "ccf")

        ants_ccf, ccf_info = fio.load_ants_nifti(
            f"{root}/ccf_average_template_25.nii.gz"
        )
    else:
        raise ValueError(f"name: {name} is not a currently available ccf")

    return ants_ccf, ccf_info


def _get_ls_template(name):
    """
    loads the nifti file for the lightsheet template you need based on the
    name you provide to the transfrom function. If you currently do not have
    that file it will download it from S3

    Parameters
    ----------
    name : str
        The name of the transform process you are using

    Returns
    -------
    ants_template: ants.array
        The template volume loaded as an ants object
    template_info : dict
        dictionary describing the template. Includes the orientation, axes
        directions, dimensions, scale, and origin of the ants.array
    """

    if name.lower() == "smartspim_lca":
        file_check = _check_path(name, "template")

        data_folder = "SmartSPIM-template_2024-05-16_11-26-14"
        files = ["smartspim_lca_template_25.nii.gz"]
        dest = os.path.join(base_dir, name.lower(), "template")

        if not file_check:
            fio._download_data_from_s3(data_folder, files, dest)

        root = os.path.join(base_dir, name, "template")

        ants_template, template_info = fio.load_ants_nifti(
            f"{root}/smartspim_lca_template_25.nii.gz"
        )
    else:
        raise ValueError(f"name: {name} is not a currently available ccf")

    return ants_template, template_info


def _fetch_zarr_data(  # pragma: no cover
    dataset_path: str, channel: str, level: int
) -> list:
    zarr_path = os.path.join(
        dataset_path,
        "image_tile_fusing/OMEZarr",
        channel + ".zarr",
        str(level),
        ".zarray",
    )

    return fio._read_json_as_dict(zarr_path)


def _get_estimated_downsample(
    voxel_resolution: list,
    registration_res: tuple = (16.0, 14.4, 14.4),
) -> int:
    """
    Get the estimated multiscale based on the provided
    voxel resolution. This is used for image stitching.

    e.g., if the original resolution is (1.8. 1.8, 2.0)
    in XYZ order, and you provide (3.6, 3.6, 4.0) as
    image resolution, then the picked resolution will be
    1.

    Parameters
    ----------
    voxel_resolution: List[float]
        Image original resolution. This would be the resolution
        in the multiscale "0".
    registration_res: Tuple[float]
        Approximated resolution that was used for registration
        in the computation of the transforms. Default: (16.0, 14.4, 14.4)
    """

    downsample_versions = []
    for idx in range(len(voxel_resolution)):
        downsample_versions.append(
            registration_res[idx] // float(voxel_resolution[idx])
        )

    downsample_res = int(min(downsample_versions))
    return round(np.log2(downsample_res))


def _parse_acquisition_data(acquisition_dict: dict):
    """
    Retrieves the relevant imaging information from the acquisition.json
    that is required for transforming points


    Parameters
    ----------
    acquisition_dict : dict
        The data from loading the acquisition.json

    Returns
    -------
    orientation : dict
        Dictionary containing the axes names (i.e. X, Y, Z), the imaging
        resolution of each axis, the dimension order of the axes, and the
        direction of each axis

    """

    orientation = acquisition_dict["axes"]

    scales = {}
    for scale, axis in zip(
        acquisition_dict["tiles"][0]["coordinate_transformations"][1]["scale"],
        ["X", "Y", "Z"],
    ):
        scales[axis] = scale

    for c, axis in enumerate(orientation):
        for s, res in scales.items():
            if s == axis["name"]:
                axis["resolution"] = res
                orientation[c] = axis

    channels = []

    for tile in acquisition_dict["tiles"]:
        channel = tile["file_name"].split("/")[0]
        if channel not in channels:
            channels.append(channel)

    acquisition = {
        "orientation": orientation,
        "registration": _get_estimated_downsample(
            [s[1] for s in sorted(scales.items(), reverse=True)]
        ),
        "channels": channels,
    }

    return acquisition


def get_dataset_transforms(dataset_path: str) -> dict:
    """
    Loads the dynamic transforms for a given dataset. dataset path can either
    be a local location or the S3 bucket location for a given dataset

    Parameters
    ----------
    dataset_path : str
        location of the transforms and acquisition.json for a given dataset
        if there is no acquisition.json will only register to template

    Returns
    -------

    transforms: dict
        the transforms needed for moving pts forward or backward

    """

    transforms = {}

    if not os.path.exists(dataset_path):
        raise FileExistsError(f"{dataset_path} does not exist.")

    transforms = fio.get_transforms(dataset_path)

    return transforms


class CoordinateTransform:
    """
    Class for transforming pts between light sheet and CCFv3 space

    Currently there is only one option for name
        - smartspim_lca

    Parameters
    ----------

    name: str
        The name of the transforms that you want to use

    dataset_transforms: list
        A list of the dataset specific transforms you want to use

    acquisition: dict
        metadata for your dataset loaded from the acquisition.json

    image_metadata: dict
    """

    def __init__(  # pragma: no cover
        self,
        name: str,
        dataset_transforms: list,
        acquisition: dict,
        image_metadata: dict,
    ):
        self.ccf_transforms = _get_ccf_transforms(name)
        self.ccf_template, self.ccf_template_info = _get_ccf_template(name)
        self.ls_template, self.ls_template_info = _get_ls_template(name)

        self.dataset_transforms = dataset_transforms
        self.acquisition = _parse_acquisition_data(acquisition)
        self.zarr_shape = image_metadata["shape"]

    def forward_transform(
        self,
        points: pd.DataFrame,
        ccf_res=25,
    ) -> np.array:
        """
        Moves points from light sheet state space into CCFv3 space

        Parameters
        ----------
        coordinates : np.array
            array of points in raw light sheet space
        input_image: da.array
            dask array of the image that the points were annotated on
        ccf_res: int
            The resolution of the ccf used in registration

        Returns
        -------
        transformed_pts : np.array
            array of points in CCFv3 space

        """

        # order columns to align with imaging
        col_order = ["", "", ""]
        for dim in self.acquisition["orientation"]:
            col_order[dim["dimension"]] = dim["name"].lower()

        points = points[col_order]
        reg_ds = self.acquisition["registration"]

        # downsample points to registration resolution
        points_ds = points.values / 2**reg_ds

        # get dimensions of registered image for orienting points
        input_shape = self.zarr_shape
        if len(input_shape) == 5:
            input_shape = input_shape[2:]

        reg_dims = [dim / 2**reg_ds for dim in input_shape]

        # flip axis based on the template orientation relative to input image
        orient = utils.get_orientation(self.acquisition["orientation"])

        _, swapped, mat = utils.get_orientation_transform(
            orient, self.ls_template_info["orientation"]
        )

        for idx, dim_orient in enumerate(mat.sum(axis=1)):
            if dim_orient < 0:
                points_ds[:, idx] = reg_dims[idx] - points_ds[:, idx]

        image_res = [
            float(dim["resolution"]) for dim in self.acquisition["orientation"]
        ]

        # scale points and orient axes to template
        scaling = utils.calculate_scaling(
            image_res=image_res,
            downsample=2**reg_ds,
            ccf_res=ccf_res,
            direction="forward",
        )

        scaled_pts = utils.scale_points(points_ds, scaling)
        orient_pts = scaled_pts[:, swapped]

        # convert points into ccf space
        ants_pts = utils.convert_to_ants_space(
            self.ls_template_info, orient_pts
        )

        template_pts = utils.apply_transforms_to_points(
            ants_pts,
            self.dataset_transforms["points_to_ccf"],
            invert=(True, False),
        )

        ccf_pts = utils.apply_transforms_to_points(
            template_pts,
            self.ccf_transforms["points_to_ccf"],
            invert=(True, False),
        )

        ccf_pts = utils.convert_from_ants_space(
            self.ccf_template_info, ccf_pts
        )

        _, swapped, _ = utils.get_orientation_transform(
            self.ls_template_info["orientation"],
            self.ccf_template_info["orientation"],
        )

        transformed_pts = ccf_pts[:, swapped]
        transformed_df = pd.DataFrame(
            transformed_pts, columns=["AP", "DV", "ML"]
        )

        return transformed_df

    def reverse_transform(
        self,
        points: pd.DataFrame,
        ccf_res=25,
    ) -> np.array:
        """
        Moves points from CCFv3 space into light sheet state space.

        Parameters
        ----------
        points : pd.DataFrame
            array of points in CCFv3 space. Input columns for dataframe must
            have column names defined as 'ML', 'AP', and 'DV'
        ccf_res: int
            The resolution of the ccf used in registration

        Returns
        -------
        transformed_pts : np.array
            array of points in light sheet space
        """

        # make sure points are ordered correctly
        cff_order = ["AP", "DV", "ML"]
        points = points[cff_order].values
        reg_ds = self.acquisition["registration"]

        # orient points for transformation
        _, swapped, _ = utils.get_orientation_transform(
            self.ccf_template_info["orientation"],
            self.ls_template_info["orientation"],
        )

        ccf_pts = points[:, swapped]
        ordered_cols = [cff_order[c] for c in swapped]

        # convert points into raw space
        ants_pts = utils.convert_to_ants_space(self.ccf_template_info, ccf_pts)

        template_pts = utils.apply_transforms_to_points(
            ants_pts,
            self.ccf_transforms["points_from_ccf"],
            invert=(False, False),
        )

        raw_pts = utils.apply_transforms_to_points(
            template_pts,
            self.dataset_transforms["points_from_ccf"],
            invert=(False, False),
        )

        raw_pts = utils.convert_from_ants_space(self.ls_template_info, raw_pts)

        # orient axes to original image
        orient = utils.get_orientation(self.acquisition["orientation"])

        _, swapped, mat = utils.get_orientation_transform(
            self.ls_template_info["orientation"], orient
        )

        orient_pts = raw_pts[:, swapped]
        ordered_cols = [ordered_cols[c] for c in swapped]

        # scale points
        image_res = [
            float(dim["resolution"]) for dim in self.acquisition["orientation"]
        ]

        scaling = utils.calculate_scaling(
            image_res=image_res,
            downsample=2**reg_ds,
            ccf_res=ccf_res,
            direction="reverse",
        )

        scaled_pts = utils.scale_points(orient_pts, scaling)

        # get dimensions of registered image for orienting points
        input_shape = self.zarr_shape
        if len(input_shape) == 5:
            input_shape = input_shape[2:]

        reg_dims = [dim / 2**reg_ds for dim in input_shape]

        for idx, dim_orient in enumerate(mat.sum(axis=0)):
            if dim_orient < 0:
                scaled_pts[:, idx] = reg_dims[idx] - scaled_pts[:, idx]

        # upsample points from registration to raw image space
        transformed_pts = scaled_pts * 2**reg_ds

        transformed_df = pd.DataFrame(transformed_pts, columns=ordered_cols)

        return transformed_df
