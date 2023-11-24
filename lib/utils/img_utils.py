import os.path as osp

import cv2
import icecream
import numpy as np

from torchvision import transforms


def convert_crop_cam_to_orig_img(camera, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param camera (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, width_offset, heigh_offset).
        Note: width_offset and heigh_offset should have the same value
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return: original camera
    '''
    center_x, center_y, offset = bbox[:,0], bbox[:,1], bbox[:,2]

    half_target_image_width, half_target_image_height = img_width / 2., img_height / 2.

    scale_x = camera[:,0] * (offset / img_width)
    scale_y = camera[:,0] * (offset / img_height)

    x_offset_in_original_image_coordinates = center_x - half_target_image_width
    y_offset_in_original_image_coordinates = center_y - half_target_image_height

    translation_x = (x_offset_in_original_image_coordinates / (half_target_image_width * scale_x)) + camera[:,1]
    translation_y = (y_offset_in_original_image_coordinates / (half_target_image_height * scale_y)) + camera[:,2]

    orig_cam = np.stack([scale_x, scale_y, translation_x, translation_y]).T

    return orig_cam

def split_boxes_cv2(img, relative_boxes, save_folder, class_names=None, frame_idx=0):

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

            cls_conf = box[5]
            cls_id = box[6]

            print('%s: %f' % (class_names[cls_id], cls_conf))

            if class_names[cls_id] == "person":

                center = (x1+x2)/2, (y1+y2)/2
                crop_width = x2-x1
                crop_height = y2-y1
                crop_size = max(crop_width, crop_height)*1.2

                transform = gen_trans_from_patch_cv(
                    center_x=center[0],
                    center_y=center[1],
                    src_width=crop_size,
                    src_height=crop_size,
                    dst_width=256,
                    dst_height=256,
                    scale=1,
                    rotation_in_degrees=0,
                    inv=False)

                inverse_transform = gen_trans_from_patch_cv(
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

                cv2.imwrite(
                    osp.join(save_folder, f"frame{frame_idx}_image{n_person}.jpg"),
                    img_patch)

                cropped_images.append(img_patch)
                n_person += 1

                refined_boxes.append([center[0], center[1], crop_size, crop_size])
                inverse_transforms.append(inverse_transform)

    return cropped_images, refined_boxes, inverse_transforms

def process_bbox(bbox):
    # sanitize bboxes
    x1, y1, x2, y2 = bbox

    # aspect ratio preserving bbox
    w = x2-x1
    h = y2-y1

    c_x = x1 + w/2.
    c_y = y1 + h/2.

    aspect_ratio = 1

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    return bbox

def get_default_transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return transform

def convert_cvimg_to_tensor(image):
    transform = get_default_transform()
    image = transform(image)
    return image

def gen_trans_from_patch_cv(
        center_x, center_y, src_width, src_height, dst_width, dst_height, scale, rotation_in_degrees, inv=False):

    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([center_x, center_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rotation_in_degrees / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        transform = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        transform = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return transform

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)