"""
Module with utilities
"""

import logging
import os
import shutil
import typing
import uuid

import box
import cv2
import numpy as np
import ruamel.yaml


def read_yaml(path: str):
    """Read content of yaml file from path

    :param path: path to yaml file
    :type path: str
    :return: yaml file content, usually a dictionary
    """

    yaml = ruamel.yaml.YAML(typ='unsafe')

    with open(path, encoding="utf-8") as file:

        return box.Box(yaml.load(file))


def create_empty_directory(path: str):
    """
    Create empty directory at path, removing previous content if it exists

    Args:
        path (str): target path
    """

    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


class ImagesLogger(logging.Logger):
    """
    Logger that adds ability to log images to html
    """

    def __init__(self, name: str, images_directory: str, images_html_path_prefix: str) -> None:
        """
        Constructor

        Args:
            name (str): logger's name
            images_directory (str): path to directory in which images should be stored
            images_html_path_prefix (str): prefix for images paths in html
        """
        super().__init__(name)

        self.images_directory = images_directory
        self.images_html_path_prefix = images_html_path_prefix

    def log_images(self, title: str, images: typing.List[np.array]):
        """
        Log images as html img tags

        Args:
            title (str): title for header placed above images
            images (typing.List[np.array]): list of images to log
        """

        self.info("<h2>{}</h2>".format(title))

        for image in images:

            image_id = uuid.uuid4()

            image_path_on_drive = os.path.join(self.images_directory, "{}.jpg".format(image_id))
            image_path_in_html = os.path.join(self.images_html_path_prefix, "{}.jpg".format(image_id))

            cv2.imwrite(image_path_on_drive, image)

            self.info("<img src='{}'>".format(image_path_in_html))

        self.info("<br>")
