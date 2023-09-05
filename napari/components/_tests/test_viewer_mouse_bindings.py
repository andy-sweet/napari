import numpy as np
import pytest

from napari.components import ViewerModel
from napari.utils._test_utils import read_only_mouse_event
from napari.utils.interactions import mouse_wheel_callbacks


class WheelEvent:
    def __init__(self, inverted) -> None:
        self._inverted = inverted

    def inverted(self):
        return self._inverted


@pytest.mark.parametrize(
    "modifiers, native, expected_dim",
    [
        ([], WheelEvent(True), [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]]),
        (
            ["Control"],
            WheelEvent(False),
            [[5, 5, 5], [4, 5, 5], [3, 5, 5], [0, 5, 5]],
        ),
        (
            ["Control"],
            WheelEvent(True),
            [[5, 5, 5], [6, 5, 5], [7, 5, 5], [9, 5, 5]],
        ),
    ],
)
def test_paint(modifiers, native, expected_dim):
    """Test painting labels with circle/square brush."""
    viewer = ViewerModel()
    data = np.random.random((10, 10, 10))
    viewer.add_image(data)
    viewer.dims.last_used = 0
    viewer.dims.set_point(axis=0, value=5)
    viewer.dims.set_point(axis=1, value=5)
    viewer.dims.set_point(axis=2, value=5)

    # Simulate tiny scroll
    event = read_only_mouse_event(
        delta=[0, 0.6], modifiers=modifiers, native=native, type="wheel"
    )
    mouse_wheel_callbacks(viewer, event)
    assert np.equal(viewer.dims.point, expected_dim[0]).all()

    # Simulate tiny scroll
    event = read_only_mouse_event(
        delta=[0, 0.6], modifiers=modifiers, native=native, type="wheel"
    )

    mouse_wheel_callbacks(viewer, event)
    assert np.equal(viewer.dims.point, expected_dim[1]).all()

    # Simulate tiny scroll
    event = read_only_mouse_event(
        delta=[0, 0.9], modifiers=modifiers, native=native, type="wheel"
    )
    mouse_wheel_callbacks(viewer, event)
    assert np.equal(viewer.dims.point, expected_dim[2]).all()

    # Simulate large scroll
    event = read_only_mouse_event(
        delta=[0, 3], modifiers=modifiers, native=native, type="wheel"
    )
    mouse_wheel_callbacks(viewer, event)
    assert np.equal(viewer.dims.point, expected_dim[3]).all()
