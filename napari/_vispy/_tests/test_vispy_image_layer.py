from itertools import permutations

import numpy as np
import numpy.testing as npt
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


@pytest.fixture()
def im_layer() -> Image:
    return Image(np.zeros((10, 10)))


@pytest.fixture()
def pyramid_layer() -> Image:
    return Image([np.zeros((20, 20)), np.zeros((10, 10))])


def test_base_create(im_layer):
    VispyImageLayer(im_layer)


def set_translate(layer):
    layer.translate = (10, 10)


def set_affine_translate(layer):
    layer.affine.translate = (10, 10)
    layer.events.affine()


def set_rotate(layer):
    layer.rotate = 90


def set_affine_rotate(layer):
    layer.affine.rotate = 90
    layer.events.affine()


def no_op(layer):
    pass


@pytest.mark.parametrize(
    ('translate', 'exp_translate'),
    [
        (set_translate, (10, 10)),
        (set_affine_translate, (10, 10)),
        (no_op, (0, 0)),
    ],
    ids=('translate', 'affine_translate', 'no_op'),
)
@pytest.mark.parametrize(
    ('rotate', 'exp_rotate'),
    [
        (set_rotate, ((0, -1), (1, 0))),
        (set_affine_rotate, ((0, -1), (1, 0))),
        (no_op, ((1, 0), (0, 1))),
    ],
    ids=('rotate', 'affine_rotate', 'no_op'),
)
def test_transforming_child_node(
    im_layer, translate, exp_translate, rotate, exp_rotate
):
    layer = VispyImageLayer(im_layer)
    npt.assert_array_almost_equal(
        layer.node.transform.matrix[-1][:2], (-0.5, -0.5)
    )
    npt.assert_array_almost_equal(
        layer.node.transform.matrix[:2, :2], ((1, 0), (0, 1))
    )
    rotate(im_layer)
    translate(im_layer)
    npt.assert_array_almost_equal(
        layer.node.children[0].transform.matrix[:2, :2], ((1, 0), (0, 1))
    )
    npt.assert_array_almost_equal(
        layer.node.children[0].transform.matrix[-1][:2], (0.5, 0.5)
    )
    npt.assert_array_almost_equal(
        layer.node.transform.matrix[:2, :2], exp_rotate
    )
    if translate == set_translate and rotate == set_affine_rotate:
        npt.assert_array_almost_equal(
            layer.node.transform.matrix[-1][:2],
            np.dot(
                np.linalg.inv(exp_rotate),
                np.array([-0.5, -0.5]) + exp_translate,
            ),
        )
    else:
        npt.assert_array_almost_equal(
            layer.node.transform.matrix[-1][:2],
            np.dot(np.linalg.inv(exp_rotate), (-0.5, -0.5)) + exp_translate,
            # np.dot(np.linalg.inv(im_layer.affine.rotate), exp_translate)
        )


def test_transforming_child_node_pyramid(pyramid_layer):
    layer = VispyImageLayer(pyramid_layer)
    corner_pixels_world = np.array([[0, 0], [20, 20]])
    npt.assert_array_almost_equal(
        layer.node.transform.matrix[-1][:2], (-0.5, -0.5)
    )
    npt.assert_array_almost_equal(
        layer.node.children[0].transform.matrix[-1][:2], (0.5, 0.5)
    )
    pyramid_layer.translate = (-10, -10)
    pyramid_layer._update_draw(
        scale_factor=1,
        corner_pixels_displayed=corner_pixels_world,
        shape_threshold=(10, 10),
    )

    npt.assert_array_almost_equal(
        layer.node.transform.matrix[-1][:2], (-0.5, -0.5)
    )
    npt.assert_array_almost_equal(
        layer.node.children[0].transform.matrix[-1][:2], (-9.5, -9.5)
    )


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

    np.testing.assert_array_equal(top_left[:2], [-0.5, -0.5])
    np.testing.assert_array_equal(bottom_right[:2], [5.5, 5.5])


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

    np.testing.assert_array_equal(top_left[:2], [1.5, 1.5])
    np.testing.assert_array_equal(bottom_right[:2], [3.5, 3.5])


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

    # TODO: explain why this is (0.5, 0.5, 0.5) and not (-0.5, -0.5, -0.5)
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
