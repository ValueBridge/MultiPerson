"""
Module with processing functions
"""

import itertools
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


def get_pose_transformation_matrix(
    source_joints: np.ndarray, destination_joints: np.ndarray, image_shape: tuple) -> np.ndarray:
    """
    Get a transformation matrix to transform source_joints to destination_joints

    Args:
        source_joints (np.ndarray): matrix of source joints coordinates
        destination_joints (np.ndarray): matrix of destination joints coordinates
        image_shape (tuple): image shape (height, width)

    Returns:
        np.ndarray: transformation matrix
    """

    indices = list(get_optimal_transformation_mapping_indices(
        source_points=destination_joints,
        destination_points=source_joints,
        image_shape=image_shape
    ))

    return cv2.getAffineTransform(
        source_joints[indices].astype(np.float32),
        destination_joints[indices].astype(np.float32)
    )


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

    mesh_transformation_matrix = get_pose_transformation_matrix(
        source_joints=mesh_joints_estimation,
        destination_joints=joints_coordinates,
        image_shape=image.shape[:2])

    transformed_mesh_rendering = cv2.warpAffine(
        mesh_rendering, mesh_transformation_matrix,
        (mesh_rendering.shape[1], mesh_rendering.shape[0]),
        flags=cv2.INTER_LINEAR)

    # mesh_transformation_matrix = get_ransac_pose_transformation_matrix(
    #     source_joints=mesh_joints_estimation, destination_joints=joints_coordinates)

    # transformed_mesh_rendering = cv2.warpPerspective(
    #     mesh_rendering, mesh_transformation_matrix,
    #     (mesh_rendering.shape[1], mesh_rendering.shape[0]),
    #     flags=cv2.INTER_LINEAR)

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
    mask = np.clip(0, 255, ((1 - mask) * image) + mesh_rendering).astype(np.uint8)

    # Create a semi-transparent image by blending the original image and the mask
    return cv2.addWeighted(image, 0.6, mask, 0.4, 0)




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


def get_optimal_transformation_mapping_indices(
        source_points: np.ndarray, destination_points: np.ndarray, image_shape: tuple) -> typing.Set[int]:
    """
    Given a set of source points and matching destination points, return a 3 elements list of indices
    for points that are judged to be best candidates for computing a transformation matrix between
    source and destination points.

    Criteria for choosing the best points are:
    - both source and destination points must be inside the image
    - the distance between triplet source points is maximized

    Args:
        source_points (np.ndarray): 2D array of source points (x, y)
        destination_points (np.ndarray): 2D array of destination points (x, y)
        image_shape (tuple): image shape (height, width)

    Returns:
        typing.Set[int]: set of indices to use for computing transformation matrix
        between source and destination points
    """

    # Compute combinations of unique points indices
    indices_triplets = np.array(list(itertools.combinations(range(len(source_points)), 3)))

    # Compute max horizontal distance between points from each triplet
    horizontal_distances = np.max(np.abs(np.array([
        source_points[indices_triplets[:, 0], 0] - source_points[indices_triplets[:, 1], 0],
        source_points[indices_triplets[:, 0], 0] - source_points[indices_triplets[:, 2], 0],
        source_points[indices_triplets[:, 1], 0] - source_points[indices_triplets[:, 2], 0]
    ])).T, axis=1)

    # Compute max vertical distances between points from each triplet
    vertical_distances = np.max(np.abs(np.array([
        source_points[indices_triplets[:, 0], 1] - source_points[indices_triplets[:, 1], 1],
        source_points[indices_triplets[:, 0], 1] - source_points[indices_triplets[:, 2], 1],
        source_points[indices_triplets[:, 1], 1] - source_points[indices_triplets[:, 2], 1]
    ])).T, axis=1)

    max_areas = horizontal_distances * vertical_distances

    # We want to drop from consideration any indices for which source or destination points fall outside of the image

    invalid_source_points_flags = np.logical_or(
        np.logical_or(source_points[:, 0] < 0, source_points[:, 0] >= image_shape[1]),
        np.logical_or(source_points[:, 1] < 0, source_points[:, 1] >= image_shape[0])
    )

    invalid_destination_points_flags = np.logical_or(
        np.logical_or(destination_points[:, 0] < 0, destination_points[:, 0] >= image_shape[1]),
        np.logical_or(destination_points[:, 1] < 0, destination_points[:, 1] >= image_shape[0])
    )

    invalid_points_indices = np.unique(np.concatenate([
        np.where(invalid_source_points_flags)[0],
        np.where(invalid_destination_points_flags)[0]
    ]))

    invalid_triplets_flags = np.isin(indices_triplets, invalid_points_indices).any(axis=1)

    # Set max area to -1 for triplets with invalid points, so they are not considered
    max_areas[invalid_triplets_flags] = -1

    # Return indices of points in combination with largest area
    return set(indices_triplets[np.argmax(max_areas)])