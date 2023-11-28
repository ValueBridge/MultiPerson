import os.path as osp

import cv2
import icecream
import matplotlib.pyplot as plt
import numpy as np
import pickle
import trimesh

from lib.utils.img_utils import convert_crop_cam_to_orig_img
from lib.utils.pose_utils import (
    matrix_to_axis_angle,
    rot6d_to_rotmat,
)


def process_output(smpl_layer, rot6d, betas, cam):
    rot6d = rot6d.reshape(-1, 144)
    betas = betas.reshape(-1, 10)
    cam = cam.reshape(-1, 3)
    rotmat = rot6d_to_rotmat(rot6d)
    rotmat = rotmat.reshape(-1, 24, 3, 3)
    axis_angle = matrix_to_axis_angle(rotmat)
    axis_angle = axis_angle.reshape(-1, 24*3)
    smpl_output_est = smpl_layer(poses=axis_angle, betas=betas)

    verts = smpl_output_est["verts"].cpu().numpy()
    faces = smpl_layer.faces_tensor.cpu().numpy()

    return axis_angle, rot6d, betas, cam, verts, faces

def save_mesh_obj(verts, faces, num_person, mesh_results_folder):
    for person_id, vert in enumerate(verts[:num_person]):
        mesh = trimesh.Trimesh(vert, faces)
        mesh.export(osp.join(mesh_results_folder, f"mesh{person_id}.obj"))

def save_mesh_rendering(renderer, verts, boxes, cam, orig_height, orig_width, num_person, mesh_results_folder):
    orig_img = np.ones((orig_height, orig_width, 3))*255
    render_img = None

    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_person + 2)]
    colors = [(c[2], c[1], c[0]) for c in colors]
    for person_id in range(num_person):
        orig_cam = convert_crop_cam_to_orig_img(
            camera=cam[person_id:person_id+1].detach().cpu().numpy(),
            bbox=boxes[person_id:person_id+1],
            img_width=orig_width,
            img_height=orig_height
        )
        if render_img is None:
            render_img = renderer.render(
                orig_img,
                verts[person_id],
                cam=orig_cam[0],
                color=colors[person_id],
            )
        else:
            render_img = renderer.render(
                render_img,
                verts[person_id],
                cam=orig_cam[0],
                color=colors[person_id],
            )
    cv2.imwrite(osp.join(mesh_results_folder, f"mesh.jpg"), render_img)


def get_mesh_rendering(renderer, camera, image_width, image_height, vertices, color, person_box) -> np.ndarray:

    orig_cam = convert_crop_cam_to_orig_img(
        camera=camera,
        bbox=np.array([person_box]),
        img_width=image_width,
        img_height=image_height
    )

    render_img = renderer.render(
        verts=vertices,
        cam=orig_cam[0],
        color=color,
    )

    return render_img


def save_mesh_rendering_v2(
        renderer, vertices_batch, boxes, cameras, orig_height, orig_width, num_person, mesh_results_folder,
        original_image, inverse_transforms, j2ds_per_person, edges_per_person):

    color_map = plt.get_cmap('rainbow')
    colors = [color_map(i) for i in np.linspace(0, 1, num_person + 2)]
    colors = [(c[2], c[1], c[0]) for c in colors]

    people_renders = []

    for person_id in range(num_person):

        render_img = get_mesh_rendering(
            renderer=renderer,
            camera=cameras[person_id:person_id+1].detach().cpu().numpy(),
            image_width=orig_width,
            image_height=orig_height,
            vertices=vertices_batch[person_id],
            color=colors[person_id],
            person_box=boxes[person_id])

        people_renders.append(render_img)

        cv2.imwrite(osp.join(mesh_results_folder, f"mesh_{person_id}.jpg"), render_img)

    # Compose all renders in such a way that later renders are on top of previous ones
    composite_renders_image = np.zeros_like(original_image)

    for person_render in people_renders:

        mask = (person_render[:, :, -1] > 0)[:, :, np.newaxis]
        composite_renders_image = np.clip(0, 255, ((1 - mask) *composite_renders_image) + person_render)

    cv2.imwrite(osp.join(mesh_results_folder, f"mesh.jpg"), composite_renders_image)

    # Overlay composite_renders_image on original_image
    renders_mask = mask = (composite_renders_image[:, :, -1] > 0)[:, :, np.newaxis]

    overlay_image = np.clip(0, 255, (renders_mask * composite_renders_image) + ((1 - renders_mask) * original_image))
    overlay_image = overlay_image.astype(np.uint8)

    # Draw 2d joints on overlay_image
    for person_id in range(num_person):

        j2d = j2ds_per_person[person_id]
        edges = edges_per_person[person_id]

        # Add a column of ones to j2d
        j2d = np.concatenate([j2d, np.ones((j2d.shape[0], 1))], axis=1)

        # Convert j2d to original image coordinates using inverse_transform
        j2d_transformed = np.matmul(j2d, inverse_transforms[person_id].T)

        overlay_image = draw_2d_joint(overlay_image, j2d_transformed, edges, thickness=8)

    cv2.imwrite(osp.join(mesh_results_folder, f"mesh_on_original_image.jpg"), overlay_image)


def save_mesh_pkl(axis_angle, betas, cam, num_person, mesh_results_folder):
    for person_id in range(num_person):
        data = {
            "thetas": axis_angle[person_id].detach().cpu().numpy(),
            "betas": betas[person_id].detach().cpu().numpy(),
            "cam": cam[person_id].detach().cpu().numpy()
        }
        with open(osp.join(mesh_results_folder, f"smpl_{person_id}.pkl"), "wb") as f:
            pickle.dump(data, f)

def save_3d_joints(j3d, edges, pose_results_folder, person_id):
    j3d = j3d[0].cpu().numpy()
    j3d[:, 1], j3d[:, 2] = j3d[:, 2], -j3d[:, 1]

    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(edges) + 2)]
    colors = [(c[2], c[1], c[0]) for c in colors]
    fig = plt.figure()
    pose_ax = fig.add_subplot(1, 1, 1, projection='3d')
    pose_ax.view_init(5, -85)
    pose_ax.set_xlim3d(-1500, 1500)
    pose_ax.set_zlim3d(-1500, 1500)
    pose_ax.set_ylim3d(0, 3000)

    for j in j3d:
        pose_ax.scatter(j[0], j[1], j[2], c='r', s=2)

    for l, edge in enumerate(edges):
        pose_ax.plot(
            [j3d[edge[0]][0], j3d[edge[1]][0]],
            [j3d[edge[0]][1], j3d[edge[1]][1]],
            [j3d[edge[0]][2], j3d[edge[1]][2]],
            c=colors[l]
        )

    plt.savefig(osp.join(pose_results_folder, f"image{person_id}_3d.jpg"))


def draw_2d_joint(img, j2d, edges, thickness):

    img = np.copy(img)

    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(edges) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    for j in j2d:
        cv2.circle(img, (int(j[0]), int(j[1])), 2, (0, 0, 255), -1)

    for l, edge in enumerate(edges):
        cv2.line(img,
            [int(j2d[edge[0]][0]), int(j2d[edge[0]][1])],
            [int(j2d[edge[1]][0]), int(j2d[edge[1]][1])],
            colors[l],
            thickness
        )

    return img

def save_2d_joints(img, j2d, edges, pose_results_folder, person_id):

    j2d = j2d[0].cpu().numpy()

    img = draw_2d_joint(img, j2d, edges, thickness=4)
    cv2.imwrite(osp.join(pose_results_folder, f"image{person_id}_2d.jpg"), img[..., ::-1])