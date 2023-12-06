"""
Tests for photobridge.processing module
"""

import numpy as np

import photobridge.processing


class TestGetOptimal_TransformationMappingIndices:
    """
    Tests for get_optimal_transformation_mapping_indices
    """

    def test_for_source_and_destination_points_have_only_three_points(self):
        """"
        Simple test for case when source and destination points have only three points
        """

        source_points = np.array([
            [0, 0],
            [0, 1],
            [1, 0]
        ])

        destination_points = np.array([
            [10, 0],
            [10, 1],
            [11, 0]
        ])

        expected = {0, 1, 2}

        actual = photobridge.processing.get_optimal_transformation_mapping_indices(
            source_points=source_points,
            destination_points=destination_points,
            image_shape=(100, 100)
        )

        assert expected == actual

    def test_when_all_points_are_in_image(self):
        """"
        Test when all points are in image
        """

        # Indices 0, 1 and 3 are furthest away from the center of the image
        source_points = np.array([
            [20, 30],
            [5, 4],
            [40, 45],
            [80, 70],
            [55, 55]
        ])

        destination_points = np.array([
            [20, 30],
            [5, 4],
            [80, 70],
            [40, 45],
            [55, 55]
        ])

        expected = {0, 1, 3}

        actual = photobridge.processing.get_optimal_transformation_mapping_indices(
            source_points=source_points,
            destination_points=destination_points,
            image_shape=(100, 100)
        )

        assert expected == actual

    def test_when_some_points_are_outsid_of_image(self):
        """"
        Test when all points are in image
        """

        # Indices 0, 1 and 3 are furthest away from the center of the image
        source_points = np.array([
            [20, 30],
            [-50, -40],  # point outside of image
            [40, 45],
            [80, 70],
            [95, 55],
            [60, 70]
        ])

        destination_points = np.array([
            [20, 30],
            [5, 4],
            [80, 70],
            [400, 45],  # point outside of image
            [55, 55],
            [40, 20]
        ])

        expected = {0, 4, 5}

        actual = photobridge.processing.get_optimal_transformation_mapping_indices(
            source_points=source_points,
            destination_points=destination_points,
            image_shape=(100, 100)
        )

        assert expected == actual