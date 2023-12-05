"""
Module with visualization commands
"""

import invoke


@invoke.task
def visualize_mesh_predictions(_context, config_path):
    """
    Visualize mesh predictions

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    # Tell pyopengl to use headless display mode
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

    import glob

    import cv2
    import icecream
    import matplotlib.pyplot as plt
    import numpy as np
    import shutil
    import torch

    import YOLOv4.tool.torch_utils
    import YOLOv4.tool.utils

    import lib.utils.file_utils
    import lib.utils.img_utils
    import lib.utils.input_utils
    import lib.utils.model_utils
    import lib.utils.output_utils
    import lib.utils.renderer

    import photobridge.processing
    import photobridge.utilities

    shutil.rmtree("/tmp", ignore_errors=True)

    np.set_printoptions(suppress=True)

    config = photobridge.utilities.read_yaml(config_path)

    image_path = config.demo.image_path
    demo_cfg = lib.utils.file_utils.update_config("configs/demo.yaml")

    for folder in demo_cfg.SAVE_FOLDER.values():
        shutil.rmtree(folder, ignore_errors=True)

    for path in glob.glob("/tmp/*.jpg"):
        os.remove(path)

    namesfile = demo_cfg.YOLO.namesfile
    height = demo_cfg.YOLO.target_height
    width = demo_cfg.YOLO.target_width

    FLAGS = demo_cfg.PoseEstimator.FLAGS
    max_num_person = demo_cfg.SmplTR.max_num_person

    yolo, pose_estimator, ik_net, feature_extractor, smpl_layer, smplTR = \
        lib.utils.model_utils.create_all_network(demo_cfg)

    split_images_folder, pose_results_folder, \
    mesh_results_folder = lib.utils.file_utils.make_folder_v2(demo_cfg)

    original_img = cv2.imread(image_path)
    orig_height, orig_width = original_img.shape[:2]

    # Inference input size is 416*416 does not mean training size is the same
    # Training size could be 608*608 or even other sizes
    # Optional inference sizes:
    #   Hight in {320, 416, 512, 608, ... 320 + 96 * n}
    #   Width in {320, 416, 512, 608, ... 320 + 96 * m}
    yolo_input_img = cv2.resize(original_img, (width, height))
    yolo_input_img = cv2.cvtColor(yolo_input_img, cv2.COLOR_BGR2RGB)

    # This 'for' loop is for speed check
    # Because the first iteration is usually longer
    for i in range(2):
        relative_category_boxes = YOLOv4.tool.torch_utils.do_detect(yolo, yolo_input_img, 0.4, 0.6, use_cuda=True)

    class_names = YOLOv4.tool.utils.load_class_names(namesfile)

    img_patch_list, transformed_people_center_boxes, inverse_transforms = photobridge.processing.split_boxes_cv2(
        img=original_img,
        relative_boxes=relative_category_boxes[0],
        class_names=class_names)

    transformed_people_center_boxes = np.array(transformed_people_center_boxes)

    num_person = len(img_patch_list)
    num_person = min(num_person, max_num_person)

    feature_dump = torch.zeros(1, max_num_person, 2048).float().cuda()
    rot6d_dump = torch.zeros(1, max_num_person, 24, 6).float().cuda()
    betas_dump = torch.zeros(1, max_num_person, 10).float().cuda()

    j2ds_per_person = []
    edges_per_person = []

    for person_id, img_patch in enumerate(img_patch_list[:num_person]):

        img_plot, img_pe_input, intrinsic = photobridge.processing.get_pose_estimator_input(img_patch, FLAGS)

        cv2.imwrite(f"/tmp/img_patch_{person_id}.jpg", img_patch)
        cv2.imwrite(f"/tmp/img_plot_{person_id}.jpg", img_plot)

        with torch.no_grad():

            j2d, j3d, j3d_abs, skeleton_indices, edges = pose_estimator(img_pe_input, intrinsic, intrinsic)

            j2ds_per_person.append(np.squeeze(j2d.detach().cpu().numpy()))
            edges_per_person.append(edges)

        lib.utils.output_utils.save_3d_joints(j3d_abs, edges, pose_results_folder, person_id)
        lib.utils.output_utils.save_2d_joints(img_plot, j2d, edges, pose_results_folder, person_id)

        img_ik_input = lib.utils.input_utils.get_ik_input(img_patch, demo_cfg, FLAGS)
        j3ds_abs_meter = j3d_abs / 1000
        ik_net_output = ik_net(img_ik_input, j3ds_abs_meter)
        rot6d_ik_net = ik_net_output.pred_rot6d
        betas_ik_net = ik_net_output.pred_shape

        img_fe_input = lib.utils.input_utils.get_feature_extractor_input(img_patch)

        img_feature = feature_extractor.extract(img_fe_input)

        feature_dump[0][person_id] = img_feature[0]
        rot6d_dump[0][person_id] = rot6d_ik_net[0]
        betas_dump[0][person_id] = betas_ik_net[0]

    with torch.no_grad():

        refined_rot6d, refined_betas, refined_cam = smplTR(feature_dump, rot6d_dump, betas_dump)

        axis_angle, rot6d, betas, cameras, vertices_batch, faces_batch = lib.utils.output_utils.process_output(
            smpl_layer, refined_rot6d, refined_betas, refined_cam)

    num_person = transformed_people_center_boxes.shape[0]

    lib.utils.output_utils.save_mesh_obj(vertices_batch, faces_batch, num_person, mesh_results_folder)

    renderer = lib.utils.renderer.HeadlessRenderer(
        smpl=smpl_layer, resolution=(orig_width, orig_height), orig_img=True)

    color_map = plt.get_cmap('rainbow')
    colors = [color_map(i) for i in np.linspace(0, 1, num_person + 2)]
    colors = [(c[2], c[1], c[0]) for c in colors]

    # Draw markers around head, pelvis and chest
    for person_id in range(num_person):

        j2d_in_local_coordinates = j2ds_per_person[person_id]
        edges = edges_per_person[person_id]

        # Add a column of ones to j2d
        j2d_in_local_coordinates = np.concatenate(
            [j2d_in_local_coordinates, np.ones((j2d_in_local_coordinates.shape[0], 1))],
            axis=1)

        # Convert j2d to original image coordinates using inverse_transform
        j2d_in_original_image_coordinates = np.matmul(j2d_in_local_coordinates, inverse_transforms[person_id].T)

        pose_estimation_overlay = lib.utils.output_utils.draw_2d_joint(
            original_img, j2d_in_original_image_coordinates, edges, thickness=8)

        pose_estimation_overlay = photobridge.processing.draw_upper_body_joints(
            pose_estimation_overlay, j2d_in_original_image_coordinates, thickness=32)

        cv2.imwrite(f"/tmp/pose_estimation_overlay_{person_id}.jpg", pose_estimation_overlay)

        # Get mesh rendering for target person
        mesh_rendering = lib.utils.output_utils.get_mesh_rendering(
            renderer=renderer,
            camera=cameras[person_id:person_id+1].detach().cpu().numpy(),
            image_width=orig_width,
            image_height=orig_height,
            vertices=vertices_batch[person_id],
            color=colors[person_id],
            person_box=transformed_people_center_boxes[person_id])

        mesh_overlay = photobridge.processing.get_mesh_overlay(original_img, mesh_rendering)

        mesh_mask = (mesh_rendering[:, :, -1] > 0)

        # Get coordinates of mesh box
        mesh_box = [
            np.where(mesh_mask)[1].min(),
            np.where(mesh_mask)[0].min(),
            np.where(mesh_mask)[1].max(),
            np.where(mesh_mask)[0].max()
        ]

        mesh_pose_estimation, mesh_pose_estimation_edges = photobridge.processing.get_pose_estimation(
            image=mesh_overlay,
            bounding_box=mesh_box,
            pose_estimator=pose_estimator,
            pose_estimation_flags=FLAGS)

        mesh_overlay = lib.utils.output_utils.draw_2d_joint(
            mesh_overlay.astype(np.uint8), mesh_pose_estimation, mesh_pose_estimation_edges, thickness=8)

        mesh_overlay = photobridge.processing.draw_upper_body_joints(mesh_overlay, mesh_pose_estimation, thickness=32)

        cv2.imwrite(f"/tmp/overlay_original_mesh_rendering_{person_id}.jpg", mesh_overlay)

        transformed_mesh_rendering = photobridge.processing.get_adjusted_mesh_rendering(
            pose_estimator=pose_estimator,
            joints_coordinates=j2d_in_original_image_coordinates,
            mesh_rendering=mesh_rendering,
            pose_estimation_flags=FLAGS,
            image=original_img)

        cv2.imwrite(
            f"/tmp/overlay_transformed_mesh_rendering_{person_id}.jpg",
            photobridge.processing.get_mesh_overlay(original_img, transformed_mesh_rendering)
        )


@invoke.task
def visualize_mesh_predictions_over_batch_of_data(_context, config_path):
    """
    Visualize mesh predictions over batch of data from a folder

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    # Tell pyopengl to use headless display mode
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

    import glob

    import cv2
    import icecream
    import matplotlib.pyplot as plt
    import numpy as np
    import shutil
    import torch
    import tqdm

    import YOLOv4.tool.torch_utils
    import YOLOv4.tool.utils

    import lib.utils.file_utils
    import lib.utils.img_utils
    import lib.utils.input_utils
    import lib.utils.model_utils
    import lib.utils.output_utils
    import lib.utils.renderer

    import photobridge.processing
    import photobridge.utilities

    import photobridge.utilities

    for path in glob.glob("/tmp/*.jpg"):
        os.remove(path)

    config = photobridge.utilities.read_yaml(config_path)

    demo_cfg = lib.utils.file_utils.update_config("configs/demo.yaml")

    yolo, pose_estimator, ik_net, feature_extractor, smpl_layer, smplTR = \
        lib.utils.model_utils.create_all_network(demo_cfg)

    namesfile = demo_cfg.YOLO.namesfile
    height = demo_cfg.YOLO.target_height
    width = demo_cfg.YOLO.target_width

    FLAGS = demo_cfg.PoseEstimator.FLAGS
    max_num_person = demo_cfg.SmplTR.max_num_person

    class_names = YOLOv4.tool.utils.load_class_names(namesfile)

    color_map = plt.get_cmap('rainbow')
    colors = [color_map(i) for i in np.linspace(0, 1, max_num_person + 2)]
    colors = [(c[2], c[1], c[0]) for c in colors]

    for path_index, image_path in enumerate(tqdm.tqdm(
            sorted(glob.glob(os.path.join(config.demo.images_directory,"*.jpg"))))):

        original_img = cv2.imread(image_path)
        orig_height, orig_width = original_img.shape[:2]

        renderer = lib.utils.renderer.HeadlessRenderer(
            smpl=smpl_layer, resolution=(orig_width, orig_height), orig_img=True)

        yolo_input_img = cv2.resize(original_img, (width, height))
        yolo_input_img = cv2.cvtColor(yolo_input_img, cv2.COLOR_BGR2RGB)

        # This 'for' loop is for speed check
        # Because the first iteration is usually longer
        for i in range(2):
            relative_category_boxes = YOLOv4.tool.torch_utils.do_detect(
                yolo, yolo_input_img, 0.4, 0.6, use_cuda=True, verbose=False)

        img_patch_list, transformed_people_center_boxes, inverse_transforms = photobridge.processing.split_boxes_cv2(
            img=original_img,
            relative_boxes=relative_category_boxes[0],
            class_names=class_names)

        num_person = min(len(img_patch_list), max_num_person)

        joint_marker_size = int(min([orig_width, orig_height]) / 150)
        pose_line_thickness = int(min([orig_width, orig_height]) / 250)

        for person_id, img_patch in enumerate(img_patch_list[:num_person]):

            path_stem = f"/tmp/{str(path_index).zfill(2)}_{str(person_id).zfill(2)}_"

            cv2.imwrite(
                path_stem + "_a_original_image.jpg",
                original_img)

            _, img_pe_input, intrinsic = photobridge.processing.get_pose_estimator_input(img_patch, FLAGS)

            j2d_in_local_coordinates, _, j3d_abs, _, edges = pose_estimator(img_pe_input, intrinsic, intrinsic)

            j2d_in_local_coordinates = np.squeeze(j2d_in_local_coordinates.detach().cpu().numpy())

            # Add a column of ones to j2d
            j2d_in_local_coordinates = np.concatenate(
                [j2d_in_local_coordinates, np.ones((j2d_in_local_coordinates.shape[0], 1))],
                axis=1)

            # Convert j2d to original image coordinates using inverse_transform
            j2d_in_original_image_coordinates = np.matmul(j2d_in_local_coordinates, inverse_transforms[person_id].T)

            pose_estimation_overlay = lib.utils.output_utils.draw_2d_joint(
                original_img, j2d_in_original_image_coordinates, edges,
                thickness=pose_line_thickness
            )

            pose_estimation_overlay = photobridge.processing.draw_upper_body_joints(
                pose_estimation_overlay, j2d_in_original_image_coordinates,
                thickness=joint_marker_size
            )

            cv2.imwrite(
                path_stem + "_b_pose_estimation_overlay.jpg",
                pose_estimation_overlay)

            feature_dump = torch.zeros(1, max_num_person, 2048).float().cuda()
            rot6d_dump = torch.zeros(1, max_num_person, 24, 6).float().cuda()
            betas_dump = torch.zeros(1, max_num_person, 10).float().cuda()

            img_ik_input = lib.utils.input_utils.get_ik_input(img_patch, demo_cfg, FLAGS)
            j3ds_abs_meter = j3d_abs / 1000
            ik_net_output = ik_net(img_ik_input, j3ds_abs_meter)
            rot6d_ik_net = ik_net_output.pred_rot6d
            betas_ik_net = ik_net_output.pred_shape

            img_fe_input = lib.utils.input_utils.get_feature_extractor_input(img_patch)

            img_feature = feature_extractor.extract(img_fe_input)

            feature_dump[0][person_id] = img_feature[0]
            rot6d_dump[0][person_id] = rot6d_ik_net[0]
            betas_dump[0][person_id] = betas_ik_net[0]

            with torch.no_grad():

                refined_rot6d, refined_betas, refined_cam = smplTR(feature_dump, rot6d_dump, betas_dump)

                axis_angle, rot6d, betas, cameras, vertices_batch, faces_batch = lib.utils.output_utils.process_output(
                    smpl_layer, refined_rot6d, refined_betas, refined_cam)

            # Get mesh rendering for target person
            mesh_rendering = lib.utils.output_utils.get_mesh_rendering(
                renderer=renderer,
                camera=cameras[person_id:person_id+1].detach().cpu().numpy(),
                image_width=orig_width,
                image_height=orig_height,
                vertices=vertices_batch[person_id],
                color=colors[person_id],
                person_box=transformed_people_center_boxes[person_id])

            mesh_overlay = photobridge.processing.get_mesh_overlay(original_img, mesh_rendering)

            mesh_mask = (mesh_rendering[:, :, -1] > 0)

            # Get coordinates of mesh box
            mesh_box = [
                np.where(mesh_mask)[1].min(),
                np.where(mesh_mask)[0].min(),
                np.where(mesh_mask)[1].max(),
                np.where(mesh_mask)[0].max()
            ]

            mesh_pose_estimation, mesh_pose_estimation_edges = photobridge.processing.get_pose_estimation(
                image=mesh_overlay,
                bounding_box=mesh_box,
                pose_estimator=pose_estimator,
                pose_estimation_flags=FLAGS)

            mesh_overlay = lib.utils.output_utils.draw_2d_joint(
                mesh_overlay.astype(np.uint8), mesh_pose_estimation, mesh_pose_estimation_edges,
                thickness=pose_line_thickness
            )

            mesh_overlay = photobridge.processing.draw_upper_body_joints(
                mesh_overlay, mesh_pose_estimation,
                thickness=joint_marker_size
            )

            mesh_mask = (mesh_rendering[:, :, -1] > 0)

            # Get coordinates of mesh box
            mesh_box = [
                np.where(mesh_mask)[1].min(),
                np.where(mesh_mask)[0].min(),
                np.where(mesh_mask)[1].max(),
                np.where(mesh_mask)[0].max()
            ]

            mesh_pose_estimation, mesh_pose_estimation_edges = photobridge.processing.get_pose_estimation(
                image=mesh_overlay,
                bounding_box=mesh_box,
                pose_estimator=pose_estimator,
                pose_estimation_flags=FLAGS)

            mesh_overlay = lib.utils.output_utils.draw_2d_joint(
                mesh_overlay.astype(np.uint8), mesh_pose_estimation, mesh_pose_estimation_edges,
                thickness=pose_line_thickness
            )

            mesh_overlay = photobridge.processing.draw_upper_body_joints(
                mesh_overlay, mesh_pose_estimation,
                thickness=joint_marker_size)

            cv2.imwrite(
                path_stem + "_c_original_mesh_overlay.jpg",
                mesh_overlay)

            transformed_mesh_rendering = photobridge.processing.get_adjusted_mesh_rendering(
                pose_estimator=pose_estimator,
                joints_coordinates=j2d_in_original_image_coordinates,
                mesh_rendering=mesh_rendering,
                pose_estimation_flags=FLAGS,
                image=original_img)

            transformed_mesh_rendering = photobridge.processing.get_mesh_overlay(
                original_img, transformed_mesh_rendering)

            cv2.imwrite(
                path_stem + "_d_transformed_mesh_overlay.jpg",
                transformed_mesh_rendering
            )