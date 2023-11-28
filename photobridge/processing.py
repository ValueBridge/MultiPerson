"""
Module with processing functions
"""

import cv2
import icecream
import numpy as np
import torch

import lib.utils.img_utils


def get_pose_estimator_input(img_patch, FLAGS):
    img_patch = img_patch.copy()
    img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
    img_patch_resize_256 = cv2.resize(img_patch, (FLAGS.proc_side, FLAGS.proc_side))

    img_pe_input = torch.from_numpy(img_patch_resize_256.copy())
    img_pe_input = img_pe_input.unsqueeze(0)
    img_pe_input = img_pe_input.permute(0, 3, 1, 2).float()
    img_pe_input /= 255
    img_pe_input = img_pe_input.cuda()

    imshape = torch.FloatTensor([FLAGS.proc_side, FLAGS.proc_side])
    fov_degrees = FLAGS.fov_degrees
    fov_radians = fov_degrees * torch.FloatTensor([np.pi / 180])
    larger_size = torch.max(imshape)
    focal_length = larger_size / (torch.tan(fov_radians/2)*2)
    intrinsic = torch.FloatTensor(
        [[
            [focal_length, 0, imshape[1]/2],
            [0, focal_length, imshape[0]/2],
            [0, 0, 1],
        ]]
    )
    intrinsic = intrinsic.cuda()
    return img_patch_resize_256, img_pe_input, intrinsic


def get_adjusted_mesh_rendering(pose_estimator, joints_coordinates, mesh_rendering, pose_estimation_flags):
    """
    Given joints coordinates and mesh rendering, adjust mesh rendering to overlap with joints coordinates

    Args:
        pose_estimator: pose estimation network
        joints_coordinates (np.ndarray): 24x2 array of joints coordinates
        mesh_rendering (np.ndarray): 3D mesh rendering
        pose_estimation_flags (config): configuration data for pose estimation
    """

    mesh_mask = (mesh_rendering[:, :, -1] > 0)

    # Get coordinates of mesh box
    mesh_box = [
        np.where(mesh_mask)[1].min(),
        np.where(mesh_mask)[0].min(),
        np.where(mesh_mask)[1].max(),
        np.where(mesh_mask)[0].max()
    ]

    center = (mesh_box[0] + mesh_box[2]) / 2, (mesh_box[1] + mesh_box[3]) / 2
    crop_width = mesh_box[2] - mesh_box[0]
    crop_height = mesh_box[3] - mesh_box[1]
    crop_size = max(crop_width, crop_height)*1.2

    transform = lib.utils.img_utils.gen_trans_from_patch_cv(
        center_x=center[0],
        center_y=center[1],
        src_width=crop_size,
        src_height=crop_size,
        dst_width=256,
        dst_height=256,
        scale=1,
        rotation_in_degrees=0,
        inv=False)

    inverse_transform = lib.utils.img_utils.gen_trans_from_patch_cv(
        center_x=center[0],
        center_y=center[1],
        src_width=crop_size,
        src_height=crop_size,
        dst_width=256,
        dst_height=256,
        scale=1,
        rotation_in_degrees=0,
        inv=True)

    img_patch = cv2.warpAffine(mesh_rendering, transform, (256, 256), flags=cv2.INTER_LINEAR)

    with torch.no_grad():

        _, img_pe_input, intrinsic = get_pose_estimator_input(img_patch, pose_estimation_flags)

        # Detect pose on mesh image
        mesh_joint_coordinates_in_image_patch, _, _, _, edges_from_mesh = pose_estimator(
            img_pe_input, intrinsic, intrinsic)

    mesh_joint_coordinates_in_image_patch = mesh_joint_coordinates_in_image_patch[0].cpu().numpy()

    # Translate mesh joint cooridinates to global coordinates

    # Add a column of ones to j2d
    mesh_joint_coordinates_in_image_patch = np.concatenate(
        [mesh_joint_coordinates_in_image_patch, np.ones((mesh_joint_coordinates_in_image_patch.shape[0], 1))],
        axis=1)

    # Convert j2d to original image coordinates using inverse_transform
    mesh_joints_in_original_image_coordinates = np.matmul(mesh_joint_coordinates_in_image_patch, inverse_transform.T)

    # Compute mean shift between original joints estimate and joints estimate from mesh for
    # joints 0, 6 and 12 (pelvis, chest and collar bone)
    shift = np.mean(
        [
            mesh_joints_in_original_image_coordinates[0] - joints_coordinates[0],
            mesh_joints_in_original_image_coordinates[6] - joints_coordinates[6],
            mesh_joints_in_original_image_coordinates[12] - joints_coordinates[12]
        ], axis=0
    )

    # Compute transformation between original joints estimate and joints estimate from mesh
    destination_points = np.array([
        joints_coordinates[0, :],
        joints_coordinates[6, :],
        joints_coordinates[12, :],
    ], dtype=np.float32)

    source_points = np.array([
        joints_coordinates[0, :] + shift,
        joints_coordinates[6, :]  + shift,
        joints_coordinates[12, :]  + shift,
    ], dtype=np.float32)

    mesh_transformation_matrix = cv2.getAffineTransform(source_points, destination_points)

    transformed_mesh_rendering = cv2.warpAffine(
        mesh_rendering, mesh_transformation_matrix,
        (mesh_rendering.shape[1], mesh_rendering.shape[0]),
        flags=cv2.INTER_LINEAR)

    return transformed_mesh_rendering


def get_mesh_overlay(image, mesh_rendering):
    """
    Return image with mesh rendering overlaid over the image

    Args:
        image (np.ndarray): image to draw mesh rendering on
        mesh_rendering (np.ndarray): mesh rendering
    """

    mask = (mesh_rendering[:, :, -1] > 0)[:, :, np.newaxis]
    return np.clip(0, 255, ((1 - mask) * image) + mesh_rendering)