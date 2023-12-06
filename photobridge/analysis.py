""""
Module with logic for analyzing experiments
"""

import cv2
import numpy as np
import torch

import lib.utils.output_utils
import photobridge.processing


class MeshAlignmentAnalyzer:
    """
    Class for analyzing mesh alignment
    """

    def __init__(
        self, log_dir, image, mesh_render, path_index, person_id,
        pose_estimator, img_patch, flags, inverse_transform, max_num_person,
        demo_cfg, renderer, smplTR, smpl_layer, feature_extractor, ik_net,
        mesh_color, person_box):

        self.log_dir = log_dir
        self.image = image
        self.mesh_remesh_rendernder = mesh_render
        self.path_index = path_index
        self.person_id = person_id
        self.pose_estimator = pose_estimator
        self.img_patch = img_patch
        self.flags = flags
        self.inverse_transform = inverse_transform
        self.max_num_person = max_num_person
        self.demo_cfg = demo_cfg
        self.renderer = renderer
        self.smplTR = smplTR
        self.smpl_layer = smpl_layer
        self.feature_extractor = feature_extractor
        self.ik_net = ik_net
        self.mesh_color = mesh_color
        self.person_box = person_box

        self.joint_marker_size = int(min(image.shape[:2]) / 150)
        self.pose_line_thickness = int(min(image.shape[:2]) / 250)
        self.path_stem = f"{self.log_dir}/{str(path_index).zfill(2)}_{str(person_id).zfill(2)}_"

        self.orig_height = self.image.shape[0]
        self.orig_width = self.image.shape[1]

    def _log_pose_estimation(self, j2d_in_original_image_coordinates, edges):

        pose_estimation_overlay = lib.utils.output_utils.draw_2d_joint(
            self.image, j2d_in_original_image_coordinates, edges,
            thickness=self.pose_line_thickness
        )

        pose_estimation_overlay = photobridge.processing.draw_upper_body_joints(
            pose_estimation_overlay, j2d_in_original_image_coordinates,
            thickness=self.joint_marker_size
        )

        cv2.imwrite(
            self.path_stem + "_b_pose_estimation_overlay.jpg",
            pose_estimation_overlay)

    def _get_mesh_rendering(self, j3d_abs):

        feature_dump = torch.zeros(1, self.max_num_person, 2048).float().cuda()
        rot6d_dump = torch.zeros(1, self.max_num_person, 24, 6).float().cuda()
        betas_dump = torch.zeros(1, self.max_num_person, 10).float().cuda()

        img_ik_input = lib.utils.input_utils.get_ik_input(self.img_patch, self.demo_cfg, self.flags)
        j3ds_abs_meter = j3d_abs / 1000
        ik_net_output = self.ik_net(img_ik_input, j3ds_abs_meter)
        rot6d_ik_net = ik_net_output.pred_rot6d
        betas_ik_net = ik_net_output.pred_shape

        img_fe_input = lib.utils.input_utils.get_feature_extractor_input(self.img_patch)

        img_feature = self.feature_extractor.extract(img_fe_input)

        feature_dump[0][self.person_id] = img_feature[0]
        rot6d_dump[0][self.person_id] = rot6d_ik_net[0]
        betas_dump[0][self.person_id] = betas_ik_net[0]

        with torch.no_grad():

            refined_rot6d, refined_betas, refined_cam = self.smplTR(feature_dump, rot6d_dump, betas_dump)

            axis_angle, rot6d, betas, cameras, vertices_batch, faces_batch = lib.utils.output_utils.process_output(
                self.smpl_layer, refined_rot6d, refined_betas, refined_cam)

        # Get mesh rendering for target person
        return lib.utils.output_utils.get_mesh_rendering(
            renderer=self.renderer,
            camera=cameras[self.person_id:self.person_id+1].detach().cpu().numpy(),
            image_width=self.orig_width,
            image_height=self.orig_height,
            vertices=vertices_batch[self.person_id],
            color=self.mesh_color,
            person_box=self.person_box)

    def _draw_annotated_mesh_overlay(self, mesh_rendering):

        mesh_mask = (mesh_rendering[:, :, -1] > 0)

        # Get coordinates of mesh box
        mesh_box = [
            np.where(mesh_mask)[1].min(),
            np.where(mesh_mask)[0].min(),
            np.where(mesh_mask)[1].max(),
            np.where(mesh_mask)[0].max()
        ]

        mesh_pose_estimation, mesh_pose_estimation_edges = photobridge.processing.get_pose_estimation(
            image=mesh_rendering,
            bounding_box=mesh_box,
            pose_estimator=self.pose_estimator,
            pose_estimation_flags=self.flags)

        mesh_overlay = photobridge.processing.get_mesh_overlay(self.image, mesh_rendering)

        mesh_overlay = lib.utils.output_utils.draw_2d_joint(
            img=mesh_overlay.astype(np.uint8),
            j2d=mesh_pose_estimation,
            edges=mesh_pose_estimation_edges,
            thickness=self.pose_line_thickness
        )

        mesh_overlay = photobridge.processing.draw_upper_body_joints(
            image=mesh_overlay,
            joints_coordinates=mesh_pose_estimation,
            thickness=self.joint_marker_size
        )

        cv2.imwrite(
            self.path_stem + "_c_original_mesh_overlay.jpg",
            mesh_overlay)

    def log_analysis(self):

        cv2.imwrite(
            self.path_stem + "_a_original_image.jpg",
            self.image)

        _, img_pe_input, intrinsic = photobridge.processing.get_pose_estimator_input(self.img_patch, self.flags)

        j2d_in_local_coordinates, _, j3d_abs, _, edges = self.pose_estimator(img_pe_input, intrinsic, intrinsic)

        j2d_in_local_coordinates = j2d_in_local_coordinates.detach().cpu().numpy()[0]

        j2d_in_local_coordinates = np.concatenate(
            [j2d_in_local_coordinates, np.ones((j2d_in_local_coordinates.shape[0], 1))],
            axis=1)

        # Convert j2d to original image coordinates using inverse_transform
        j2d_in_original_image_coordinates = np.matmul(
            j2d_in_local_coordinates,
            self.inverse_transform.T)

        self._log_pose_estimation(
            j2d_in_original_image_coordinates=j2d_in_original_image_coordinates,
            edges=edges)

        mesh_rendering = self._get_mesh_rendering(j3d_abs=j3d_abs)

        self._draw_annotated_mesh_overlay(mesh_rendering=mesh_rendering)

        transformed_mesh_rendering = photobridge.processing.get_adjusted_mesh_rendering(
            pose_estimator=self.pose_estimator,
            joints_coordinates=j2d_in_original_image_coordinates,
            mesh_rendering=mesh_rendering,
            pose_estimation_flags=self.flags,
            image=self.image)

        cv2.imwrite(
            self.path_stem + "_d_transformed_mesh_overlay.jpg",
            photobridge.processing.get_mesh_overlay(self.image, transformed_mesh_rendering)
        )