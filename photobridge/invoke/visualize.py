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

    import cv2
    import icecream
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

    import photobridge.utilities

    shutil.rmtree("/tmp", ignore_errors=True)

    config = photobridge.utilities.read_yaml(config_path)

    image_path = config.demo.image_path
    demo_config_path = "configs/demo.yaml"

    demo_cfg = lib.utils.file_utils.update_config(demo_config_path)

    for folder in demo_cfg.SAVE_FOLDER.values():
        shutil.rmtree(folder, ignore_errors=True)

    namesfile = demo_cfg.YOLO.namesfile
    height = demo_cfg.YOLO.target_height
    width = demo_cfg.YOLO.target_width
    n_classes = demo_cfg.YOLO.n_classes

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

    img_patch_list, transformed_people_center_boxes, inverse_transforms = lib.utils.img_utils.split_boxes_cv2(
        img=original_img,
        relative_boxes=relative_category_boxes[0],
        save_folder=split_images_folder,
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

        img_plot, img_pe_input, intrinsic = lib.utils.input_utils.get_pose_estimator_input(img_patch, FLAGS)

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

    lib.utils.output_utils.save_mesh_rendering_v2(
        renderer=renderer,
        vertices_batch=vertices_batch,
        boxes=transformed_people_center_boxes,
        cameras=cameras,
        orig_height=orig_height,
        orig_width=orig_width,
        num_person=num_person,
        mesh_results_folder=mesh_results_folder,
        original_image=original_img,
        inverse_transforms=inverse_transforms,
        j2ds_per_person=j2ds_per_person,
        edges_per_person=edges_per_person)