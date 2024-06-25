from itertools import permutations

import numpy as np
import pytest

from napari._vispy._tests.utils import vispy_image_scene_size
from napari._vispy.layers.image import VispyImageLayer
from napari.components.dims import Dims
from napari.layers import Image


@pytest.mark.parametrize('order', permutations((0, 1, 2)))
def test_3d_slice_of_2d_image_with_order(order):
    """See https://github.com/napari/napari/issues/4926

    We define a non-isotropic shape and scale that combined properly
    with any order should make a small square when displayed in 3D.
    """
    image = Image(np.zeros((4, 2)), scale=(1, 2))
    vispy_image = VispyImageLayer(image)

    image._slice_dims(Dims(ndim=3, ndisplay=3, order=order))

    scene_size = vispy_image_scene_size(vispy_image)
    np.testing.assert_array_equal((4, 4, 1), scene_size)


@pytest.mark.parametrize('order', permutations((0, 1, 2)))
def test_2d_slice_of_3d_image_with_order(order):
    """See https://github.com/napari/napari/issues/4926

    We define a non-isotropic shape and scale that combined properly
    with any order should make a small square when displayed in 2D.
    """
    image = Image(np.zeros((8, 4, 2)), scale=(1, 2, 4))
    vispy_image = VispyImageLayer(image)

    image._slice_dims(Dims(ndim=3, ndisplay=2, order=order))

    scene_size = vispy_image_scene_size(vispy_image)
    np.testing.assert_array_equal((8, 8, 0), scene_size)


@pytest.mark.parametrize('order', permutations((0, 1, 2)))
def test_3d_slice_of_3d_image_with_order(order):
    """See https://github.com/napari/napari/issues/4926

    We define a non-isotropic shape and scale that combined properly
    with any order should make a small cube when displayed in 3D.
    """
    image = Image(np.zeros((8, 4, 2)), scale=(1, 2, 4))
    vispy_image = VispyImageLayer(image)

    image._slice_dims(Dims(ndim=3, ndisplay=3, order=order))

    scene_size = vispy_image_scene_size(vispy_image)
    np.testing.assert_array_equal((8, 8, 8), scene_size)


@pytest.mark.parametrize('order', permutations((0, 1, 2, 3)))
def test_3d_slice_of_4d_image_with_order(order):
    """See https://github.com/napari/napari/issues/4926

    We define a non-isotropic shape and scale that combined properly
    with any order should make a small cube when displayed in 3D.
    """
    image = Image(np.zeros((16, 8, 4, 2)), scale=(1, 2, 4, 8))
    vispy_image = VispyImageLayer(image)

    image._slice_dims(Dims(ndim=4, ndisplay=3, order=order))

    scene_size = vispy_image_scene_size(vispy_image)
    np.testing.assert_array_equal((16, 16, 16), scene_size)


def test_no_float32_texture_support(monkeypatch):
    """Ensure Image node can be created if OpenGL driver lacks float textures.

    See #3988, #3990, #6652.
    """
    monkeypatch.setattr(
        'napari._vispy.layers.image.get_gl_extensions', lambda: ''
    )
    image = Image(np.zeros((16, 8, 4, 2), dtype='uint8'), scale=(1, 2, 4, 8))
    VispyImageLayer(image)


def test_multiscale_pixel_offset_2d():
    """See https://github.com/napari/napari/issues/6320"""
    image = Image([np.zeros((8, 8, 8)), np.zeros((4, 4, 4))])
    # Simulate being zoomed out and rendering the highest resolution level.
    image.corner_pixels = np.array([[0, 0, 0], [8, 8, 8]])
    image._data_level = 0
    image._slice_dims(Dims(ndim=3, ndisplay=2, point=(1, 0, 0)))

    vispy_image = VispyImageLayer(image)

    data = vispy_image.node._data
    assert data.shape == (8, 8)

    transform = vispy_image.node.transform
    top_left = transform.map((0, 0))
    bottom_right = transform.map((7, 7))

    np.testing.assert_array_equal(top_left[:2], [-0.5, -0.5])
    np.testing.assert_array_equal(bottom_right[:2], [6.5, 6.5])


def test_multiscale_pixel_offset_2d_scale():
    """See https://github.com/napari/napari/issues/6320"""
    image = Image([np.zeros((8, 8, 8)), np.zeros((4, 4, 4))], scale=[2, 2, 2])
    # Simulate being zoomed out and rendering the highest resolution level.
    image.corner_pixels = np.array([[0, 0, 0], [8, 8, 8]])
    image._data_level = 0
    image._slice_dims(Dims(ndim=3, ndisplay=2, point=(1, 0, 0)))

    vispy_image = VispyImageLayer(image)

    data = vispy_image.node._data
    assert data.shape == (8, 8)

    transform = vispy_image.node.transform
    top_left = transform.map((0, 0))
    bottom_right = transform.map((7, 7))

    np.testing.assert_array_equal(top_left[:2], [-1, -1])
    np.testing.assert_array_equal(bottom_right[:2], [13, 13])


def test_multiscale_pixel_offset_2d_zoomed():
    """See https://github.com/napari/napari/issues/6320"""
    image = Image([np.zeros((8, 8, 8)), np.zeros((4, 4, 4))])
    # Simulate being zoomed in and rendering the highest resolution level.
    image.corner_pixels = np.array([[2, 2, 2], [6, 6, 6]])
    image._data_level = 0
    image._slice_dims(Dims(ndim=3, ndisplay=2, point=(1, 0, 0)))

    vispy_image = VispyImageLayer(image)

    data = vispy_image.node._data
    assert data.shape == (5, 5)

    transform = vispy_image.node.transform
    top_left = transform.map((0, 0))
    bottom_right = transform.map((4, 4))

    np.testing.assert_array_equal(top_left[:2], [1.5, 1.5])
    np.testing.assert_array_equal(bottom_right[:2], [5.5, 5.5])


def test_multiscale_pixel_offset_2d_scale_zoomed():
    """See https://github.com/napari/napari/issues/6320"""
    image = Image([np.zeros((8, 8, 8)), np.zeros((4, 4, 4))], scale=[2, 2, 2])
    # Simulate being zoomed in and rendering the highest resolution level.
    image.corner_pixels = np.array([[2, 2, 2], [6, 6, 6]])
    image._data_level = 0
    image._slice_dims(Dims(ndim=3, ndisplay=2, point=(1, 0, 0)))

    vispy_image = VispyImageLayer(image)

    data = vispy_image.node._data
    assert data.shape == (5, 5)

    transform = vispy_image.node.transform
    top_left = transform.map((0, 0))
    bottom_right = transform.map((4, 4))

    np.testing.assert_array_equal(top_left[:2], [3, 3])
    np.testing.assert_array_equal(bottom_right[:2], [11, 11])


def test_multiscale_pixel_offset_2d_lowres():
    """See https://github.com/napari/napari/issues/6320"""
    image = Image([np.zeros((8, 8, 8)), np.zeros((4, 4, 4))])
    # Simulate being zoomed out and rendering the lowest resolution level.
    image.corner_pixels = np.array([[0, 0, 0], [3, 3, 3]])
    image._data_level = 1
    image._slice_dims(Dims(ndim=3, ndisplay=2, point=(1, 0, 0)))

    vispy_image = VispyImageLayer(image)

    data = vispy_image.node._data
    assert data.shape == (4, 4)

    transform = vispy_image.node.transform
    top_left = transform.map((0, 0))
    bottom_right = transform.map((3, 3))

    # TODO: explain why this is (0, 0) and not (-0.5, -0.5) which
    # is where I would expect to start drawing.
    np.testing.assert_array_equal(top_left[:2], [0, 0])
    np.testing.assert_array_equal(bottom_right[:2], [6, 6])


def test_multiscale_pixel_offset_2d_lowres_zoomed():
    """See https://github.com/napari/napari/issues/6320"""
    image = Image([np.zeros((8, 8, 8)), np.zeros((4, 4, 4))])
    # Simulate being zoomed in and rendering the lowest resolution level.
    image.corner_pixels = np.array([[1, 1, 1], [2, 2, 2]])
    image._data_level = 1
    image._slice_dims(Dims(ndim=3, ndisplay=2, point=(1, 0, 0)))

    vispy_image = VispyImageLayer(image)

    data = vispy_image.node._data
    assert data.shape == (2, 2)

    transform = vispy_image.node.transform
    top_left = transform.map((0, 0))
    bottom_right = transform.map((1, 1))

    np.testing.assert_array_equal(top_left[:2], [2, 2])
    np.testing.assert_array_equal(bottom_right[:2], [4, 4])


def test_multiscale_pixel_offset_3d():
    """See https://github.com/napari/napari/issues/6320"""
    image = Image([np.zeros((8, 8, 8)), np.zeros((4, 4, 4))])
    # Simulate being zoomed out and rendering the lowest resolution level.
    image.corner_pixels = np.array([[0, 0, 0], [4, 4, 4]])
    image._data_level = 1
    image._slice_dims(Dims(ndim=3, ndisplay=3))

    vispy_image = VispyImageLayer(image)

    data = vispy_image.node._last_data
    assert data.shape == (4, 4, 4)

    transform = vispy_image.node.transform
    top_left = transform.map((0, 0, 0))
    bottom_right = transform.map((3, 3, 3))

    # TODO: explain why this is (0.5, 0.5, 0.5) and not (-0.5, 0.5, 0.5)
    # where I would expect to start drawing.
    np.testing.assert_array_equal(top_left[:3], [0.5, 0.5, 0.5])
    np.testing.assert_array_equal(bottom_right[:3], [6.5, 6.5, 6.5])


def test_multiscale_pixel_offset_3d_scale():
    """See https://github.com/napari/napari/issues/6320"""
    image = Image([np.zeros((8, 8, 8)), np.zeros((4, 4, 4))], scale=(2, 2, 2))
    # Simulate being zoomed out and rendering the lowest resolution level.
    image.corner_pixels = np.array([[0, 0, 0], [4, 4, 4]])
    image._data_level = 1
    image._slice_dims(Dims(ndim=3, ndisplay=3))

    vispy_image = VispyImageLayer(image)

    data = vispy_image.node._last_data
    assert data.shape == (4, 4, 4)

    transform = vispy_image.node.transform
    top_left = transform.map((0, 0, 0))
    bottom_right = transform.map((3, 3, 3))

    np.testing.assert_array_equal(top_left[:3], [1, 1, 1])
    np.testing.assert_array_equal(bottom_right[:3], [13, 13, 13])
