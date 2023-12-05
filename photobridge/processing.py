"""
Module with processing functions
"""

import typing

import cv2
import icecream
import matplotlib.pyplot as plt
import numpy as np
import torch

import lib.utils.img_utils
import lib.utils.output_utils


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


def get_pose_transformation_matrix(source_joints: np.ndarray, destination_joints: np.ndarray) -> np.ndarray:
    """
    Get a transformation matrix to transform source_joints to destination_joints

    Args:
        source_joints (np.ndarray): matrix of source joints coordinates
        destination_joints (np.ndarray): matrix of destination joints coordinates

    Returns:
        np.ndarray: transformation matrix
    """

    joints_indices = [
        15,  # head
        6,  # chest
        9  # chest
    ]

    # joints_indices = [
    #     15,  # head
    #     16,  # chest
    #     17  # chest
    # ]

    # Compute transformation between original joints estimate and joints estimate from mesh.
    # Use head and shoulders for reference points
    source_points = np.array([
        source_joints[joints_indices[0], :],
        source_joints[joints_indices[1], :],
        source_joints[joints_indices[2], :]
    ], dtype=np.float32)

    destination_points = np.array([
        destination_joints[joints_indices[0], :],
        destination_joints[joints_indices[1], :],
        destination_joints[joints_indices[2], :]
    ], dtype=np.float32)

    manual_mesh_transformation_matrix = np.zeros((2, 3), dtype=np.float32)

    mean_shift = np.mean(destination_points - source_points, axis=0)

    # Compute x scale change by looking at ratio of distances between shoulders
    x_scale = (
        np.linalg.norm(destination_joints[17, :] - destination_joints[16, :]) /
        np.linalg.norm(source_joints[17, :] - source_joints[16, :])
    )

    # Compute y scale change by looking at rion of distances between head and chest
    y_scale = (
        np.linalg.norm(destination_joints[6, :] - destination_joints[15, :]) /
        np.linalg.norm(source_joints[6, :] - source_joints[15, :])
    )

    # Fix scale
    manual_mesh_transformation_matrix[0, 0] = x_scale
    manual_mesh_transformation_matrix[1, 1] = y_scale

    manual_mesh_transformation_matrix[0, 2] = mean_shift[0]
    manual_mesh_transformation_matrix[1, 2] = mean_shift[1]

    return manual_mesh_transformation_matrix


def get_ransac_pose_transformation_matrix(source_joints: np.ndarray, destination_joints: np.ndarray) -> np.ndarray:
    """
    Get a transformation matrix to transform source_joints to destination_joints

    Args:
        source_joints (np.ndarray): matrix of source joints coordinates
        destination_joints (np.ndarray): matrix of destination joints coordinates

    Returns:
        np.ndarray: transformation matrix
    """

    # Upper body joint indices
    joints_indices = [6, 9, 12, 15, 16, 17]

    source_points = source_joints[joints_indices]
    destination_points = destination_joints[joints_indices]

    homography = cv2.findHomography(source_points, destination_points, cv2.LMEDS, 5)[0]

    return homography


def get_adjusted_mesh_rendering(pose_estimator, joints_coordinates, mesh_rendering, pose_estimation_flags, image):
    """
    Given joints coordinates and mesh rendering, adjust mesh rendering to overlap with joints coordinates

    Args:
        pose_estimator: pose estimation network
        joints_coordinates (np.ndarray): 24x2 array of joints coordinates
        mesh_rendering (np.ndarray): 3D mesh rendering
        pose_estimation_flags (config): configuration data for pose estimation
        image (np.ndarray): image containing target person
    """

    mesh_mask = (mesh_rendering[:, :, -1] > 0)

    # Get coordinates of mesh box
    mesh_box = [
        np.where(mesh_mask)[1].min(),
        np.where(mesh_mask)[0].min(),
        np.where(mesh_mask)[1].max(),
        np.where(mesh_mask)[0].max()
    ]

    mesh_joints_estimation, _ = get_pose_estimation(
        image=mesh_rendering,
        bounding_box=mesh_box,
        pose_estimator=pose_estimator,
        pose_estimation_flags=pose_estimation_flags)

    # mesh_transformation_matrix = get_pose_transformation_matrix(
    #     source_joints=mesh_joints_estimation, destination_joints=joints_coordinates)

    # transformed_mesh_rendering = cv2.warpAffine(
    #     mesh_rendering, mesh_transformation_matrix,
    #     (mesh_rendering.shape[1], mesh_rendering.shape[0]),
    #     flags=cv2.INTER_LINEAR)

    mesh_transformation_matrix = get_ransac_pose_transformation_matrix(
        source_joints=mesh_joints_estimation, destination_joints=joints_coordinates)

    transformed_mesh_rendering = cv2.warpPerspective(
        mesh_rendering, mesh_transformation_matrix,
        (mesh_rendering.shape[1], mesh_rendering.shape[0]),
        flags=cv2.INTER_LINEAR)

    # return transformed_mesh_rendering
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


def get_pose_estimation(
        image: np.ndarray, bounding_box: np.ndarray, pose_estimator, pose_estimation_flags
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Get pose estimation for person in bounding box

    Args:
        image (np.ndarray): image in which to detect pose
        bounding_box (np.ndarray): bounding box of person for which to detect pose
        pose_estimator: pose estimation network
        pose_estimation_flags: pose estimation configuration

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: tuple (2D joints coorinates, edges information)
    """

    center = (bounding_box[0] + bounding_box[2]) / 2, (bounding_box[1] + bounding_box[3]) / 2
    crop_width = bounding_box[2] - bounding_box[0]
    crop_height = bounding_box[3] - bounding_box[1]
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

    img_patch = cv2.warpAffine(image.astype(np.uint8), transform, (256, 256), flags=cv2.INTER_LINEAR)

    with torch.no_grad():

        _, img_pe_input, intrinsic = get_pose_estimator_input(img_patch, pose_estimation_flags)

        # Detect pose on mesh image
        joint_coordinates_in_image_patch, _, _, _, edges = pose_estimator(img_pe_input, intrinsic, intrinsic)

    joint_coordinates_in_image_patch = joint_coordinates_in_image_patch[0].cpu().numpy()

    # Translate mesh joint cooridinates to global coordinates

    # Add a column of ones to j2d
    joint_coordinates_in_image_patch = np.concatenate(
        [joint_coordinates_in_image_patch, np.ones((joint_coordinates_in_image_patch.shape[0], 1))],
        axis=1)

    # Convert j2d to original image coordinates using inverse_transform
    joints_in_original_image_coordinates = np.matmul(joint_coordinates_in_image_patch, inverse_transform.T)

    return joints_in_original_image_coordinates, edges


def draw_upper_body_joints(image, joints_coordinates, thickness=8):
    """
    Draw only some selected joints on image

    Returns:
        np.ndarray: image with joints drawn on
    """

    annotated_image = image.copy()

    joints_indices = [
        6, 9, 12, 15, 16, 17
    ]

    color_map = plt.get_cmap('rainbow')
    colors = [255 * np.array(color) for color in color_map(np.linspace(0, 1, len(joints_indices)))]

    # Draw joints
    for index, joint_id in enumerate(joints_indices):

        joint = joints_coordinates[joint_id, :]

        cv2.circle(
            annotated_image,
            (int(joint[0]), int(joint[1])),
            thickness,
            colors[index],
            -1)

    return annotated_image


def split_boxes_cv2(img, relative_boxes, class_names):

    img = np.copy(img)

    width = img.shape[1]
    height = img.shape[0]

    n_person = 0
    refined_boxes = []
    inverse_transforms = []
    cropped_images = []

    for i in range(len(relative_boxes)):

        box = relative_boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        if len(box) >= 7 and class_names:

            cls_id = box[6]

            if class_names[cls_id] == "person":

                center = (x1+x2)/2, (y1+y2)/2
                crop_width = x2-x1
                crop_height = y2-y1
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

                img_patch = cv2.warpAffine(img, transform, (256, 256), flags=cv2.INTER_LINEAR)

                cropped_images.append(img_patch)
                n_person += 1

                refined_boxes.append([center[0], center[1], crop_size, crop_size])
                inverse_transforms.append(inverse_transform)

    return cropped_images, refined_boxes, inverse_transforms
