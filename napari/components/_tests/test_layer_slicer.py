import time
from concurrent.futures import Future, wait
from dataclasses import dataclass
from threading import RLock, current_thread, main_thread
from typing import Any, Dict

import numpy as np
import pytest

from napari._tests.utils import DEFAULT_TIMEOUT_SECS, LockableData
from napari.components import Dims
from napari.components._layer_slicer import _LayerSlicer
from napari.layers import Image, Layer, Points, Shapes
from napari.layers.base._slice import _SliceRequest, _SliceResponse


class SliceObserver:
    """Used to observe the ready event from the layer slicer."""

    def __init__(self):
        self._last_response: Dict[Layer, _SliceResponse] = {}
        self._lock = RLock()

    def on_slice_ready(self, event) -> None:
        responses = event.value
        with self._lock:
            for layer, response in responses.items():
                self._last_response[layer] = response

    def get(self, layer: Layer) -> _SliceResponse:
        with self._lock:
            return self._last_response.get(layer)


@pytest.fixture()
def layer_slicer():
    layer_slicer = _LayerSlicer()
    layer_slicer._force_sync = False
    yield layer_slicer
    layer_slicer.shutdown()


@pytest.fixture()
def slice_observer(layer_slicer: _LayerSlicer):
    slice_observer = SliceObserver()
    layer_slicer.events.ready.connect(slice_observer.on_slice_ready)
    return slice_observer


def make_lockable_image() -> Image:
    np.random.seed(0)
    data = np.random.rand(3, 2)
    lockable_data = LockableData(data)
    return Image(data=lockable_data, multiscale=False, rgb=False)


def make_shapes() -> Shapes:
    np.random.seed(0)
    data = np.random.rand(3, 4, 2)
    return Shapes(data)


def test_submit_with_one_async_layer_no_block(layer_slicer, slice_observer):
    layer = make_lockable_image()

    future = layer_slicer.submit(layers=[layer], dims=Dims())

    result = _wait_for_result(future)
    assert result[layer] is slice_observer.get(layer)


def test_submit_with_multiple_async_layer_no_block(
    layer_slicer, slice_observer
):
    layer1 = make_lockable_image()
    layer2 = make_lockable_image()

    future = layer_slicer.submit(layers=[layer1, layer2], dims=Dims())

    result = _wait_for_result(future)
    assert result[layer1] is slice_observer.get(layer1)
    assert result[layer2] is slice_observer.get(layer2)


def test_submit_with_one_sync_layer(layer_slicer, slice_observer):
    layer = make_shapes()

    future = layer_slicer.submit(layers=[layer], dims=Dims())

    assert future is None
    assert slice_observer.get(layer) is not None


def test_submit_with_multiple_sync_layer(layer_slicer, slice_observer):
    layer1 = make_shapes()
    layer2 = make_shapes()

    future = layer_slicer.submit(layers=[layer1, layer2], dims=Dims())

    assert future is None
    assert slice_observer.get(layer1) is not None
    assert slice_observer.get(layer2) is not None


def test_submit_with_mixed_layers(layer_slicer, slice_observer):
    layer1 = make_lockable_image()
    layer2 = make_shapes()

    future = layer_slicer.submit(layers=[layer1, layer2], dims=Dims())

    result = _wait_for_result(future)
    assert slice_observer.get(layer1) is result[layer1]
    assert layer2 not in result
    assert slice_observer.get(layer2) is not None


def test_submit_lock_blocking(layer_slicer, slice_observer):
    layer = make_lockable_image()

    with layer.data.lock:
        blocked = layer_slicer.submit(layers=[layer], dims=Dims())
        assert not blocked.done()

    result = _wait_for_result(blocked)
    assert slice_observer.get(layer) is result[layer]


def test_submit_multiple_calls_cancels_pending(layer_slicer):
    dims = Dims()
    layer = make_lockable_image()

    with layer.data.lock:
        blocked = layer_slicer.submit(layers=[layer], dims=dims)
        _wait_until_running(blocked)
        pending = layer_slicer.submit(layers=[layer], dims=dims)
        assert not pending.running()
        layer_slicer.submit(layers=[layer], dims=dims)
        assert not blocked.done()

    assert pending.cancelled()


def test_submit_mixed_allows_sync_to_run(layer_slicer, slice_observer):
    """ensure that a blocked async slice doesn't block sync slicing"""
    dims = Dims()
    layer1 = make_lockable_image()
    layer2 = make_shapes()
    with layer1.data.lock:
        blocked = layer_slicer.submit(layers=[layer1], dims=dims)
        layer_slicer.submit(layers=[layer2], dims=dims)
        assert slice_observer.get(layer2) is not None
        assert not blocked.done()

    result = _wait_for_result(blocked)
    assert slice_observer.get(layer1) is result[layer1]


def test_submit_mixed_allows_sync_to_run_one_slicer_call(
    layer_slicer, slice_observer
):
    """ensure that a blocked async slice doesn't block sync slicing"""
    dims = Dims()
    layer1 = make_lockable_image()
    layer2 = make_shapes()
    with layer1.data.lock:
        blocked = layer_slicer.submit(layers=[layer1, layer2], dims=dims)
        sync_slice_1 = slice_observer.get(layer2)
        assert sync_slice_1 is not None
        layer_slicer.submit(layers=[layer2], dims=dims)
        sync_slice_2 = slice_observer.get(layer2)
        assert sync_slice_2 is not None
        assert sync_slice_2 is not sync_slice_1
        assert not blocked.done()

    result = _wait_for_result(blocked)
    assert slice_observer.get(layer1) is result[layer1]


def test_submit_with_multiple_async_layer_with_all_locked(
    layer_slicer,
    slice_observer,
):
    """ensure that if only all layers are locked, none continue"""
    layer1 = make_lockable_image()
    layer2 = make_lockable_image()

    with layer1.data.lock, layer2.data.lock:
        blocked = layer_slicer.submit(layers=[layer1, layer2], dims=Dims())
        assert not blocked.done()
        assert slice_observer.get(layer1) is None
        assert slice_observer.get(layer2) is None

    result = _wait_for_result(blocked)
    assert slice_observer.get(layer1) is result[layer1]
    assert slice_observer.get(layer2) is result[layer2]


def test_submit_exception_main_thread(layer_slicer):
    """Exception is raised on the main thread from an error on the main
    thread immediately when the task is created."""

    class ErrorOnRequest(Image):
        def _make_slice_request(self, dims: Dims) -> _SliceRequest:
            raise RuntimeError('_make_slice_request')

    layer = ErrorOnRequest(np.zeros((3, 2)))
    with pytest.raises(RuntimeError, match='_make_slice_request'):
        layer_slicer.submit(layers=[layer], dims=Dims())


def test_submit_exception_subthread_on_result(layer_slicer):
    """Exception is raised on the main thread from an error on a subthread
    only after result is called, not upon submission of the task."""

    @dataclass(frozen=True)
    class ErroringSliceRequest:
        id: int

        def supports_async() -> bool:
            return True

        def __call__(self) -> _SliceResponse:
            assert current_thread() != main_thread()
            raise RuntimeError('FakeSliceRequestError')

    class ErrorOnSlice(Image):
        def _make_slice_request(self, dims: Dims) -> ErroringSliceRequest:
            return ErroringSliceRequest(id=0)

    layer = ErrorOnSlice(np.zeros((3, 2)))
    future = layer_slicer.submit(layers=[layer], dims=Dims())

    # First wait for the future to be done without triggering the
    # exception, then get the result to trigger it.
    done, _ = wait([future], timeout=DEFAULT_TIMEOUT_SECS)
    assert done, 'Test future did not complete within timeout.'
    with pytest.raises(RuntimeError, match='FakeSliceRequestError'):
        _wait_for_result(future)


def test_wait_until_idle(layer_slicer, single_threaded_executor):
    dims = Dims()
    layer = make_lockable_image()

    with layer.data.lock:
        slice_future = layer_slicer.submit(layers=[layer], dims=dims)
        _wait_until_running(slice_future)
        # The slice task has started, but has not finished yet
        # because we are holding the layer's slicing lock.
        assert len(layer_slicer._layers_to_task) > 0
        # We can't call wait_until_idle on this thread because we're
        # holding the layer's slice lock, so submit it to be executed
        # on another thread and also wait for it to start.
        wait_future = single_threaded_executor.submit(
            layer_slicer.wait_until_idle,
            timeout=DEFAULT_TIMEOUT_SECS,
        )
        _wait_until_running(wait_future)

    _wait_for_result(wait_future)
    assert len(layer_slicer._layers_to_task) == 0


def test_force_sync_on_sync_layer(layer_slicer, slice_observer):
    layer = make_shapes()

    with layer_slicer.force_sync():
        assert layer_slicer._force_sync
        future = layer_slicer.submit(layers=[layer], dims=Dims())

    assert future is None
    assert slice_observer.get(layer) is not None
    assert not layer_slicer._force_sync


def test_force_sync_on_async_layer(layer_slicer, slice_observer):
    layer = make_lockable_image()

    with layer_slicer.force_sync():
        assert layer_slicer._force_sync
        future = layer_slicer.submit(layers=[layer], dims=Dims())

    assert future is None
    assert slice_observer.get(layer) is not None
    assert not layer_slicer._force_sync


def test_submit_with_one_3d_image(layer_slicer):
    np.random.seed(0)
    data = np.random.rand(8, 7, 6)
    lockable_data = LockableData(data)
    layer = Image(data=lockable_data, multiscale=False)
    dims = Dims(
        ndim=3,
        ndisplay=2,
        range=((0, 8, 1), (0, 7, 1), (0, 6, 1)),
        point=(2, 0, 0),
    )

    with lockable_data.lock:
        future = layer_slicer.submit(layers=[layer], dims=dims)
        assert not future.done()
    layer_result = _wait_for_result(future)[layer]
    np.testing.assert_equal(layer_result.image.view, data[2, :, :])


def test_submit_with_one_3d_points(layer_slicer):
    """ensure that async slicing of points does not block"""
    np.random.seed(0)
    num_points = 2
    data = np.rint(2.0 * np.random.rand(num_points, 3))
    layer = Points(data=data)

    # Note: We are directly accessing and locking the _data of layer. This
    #       forces a block to ensure that the async slicing call returns
    #       before slicing is complete.
    lockable_internal_data = LockableData(layer._data)
    layer._data = lockable_internal_data
    dims = Dims(
        ndim=3,
        ndisplay=2,
        range=((0, 3, 1), (0, 3, 1), (0, 3, 1)),
        point=(1, 0, 0),
    )

    with lockable_internal_data.lock:
        future = layer_slicer.submit(layers=[layer], dims=dims)
        assert not future.done()


def test_submit_after_shutdown_raises():
    layer_slicer = _LayerSlicer()
    layer_slicer._force_sync = False
    layer_slicer.shutdown()
    with pytest.raises(RuntimeError):
        layer_slicer.submit(layers=[make_lockable_image()], dims=Dims())


def _wait_until_running(future: Future):
    """Waits until the given future is running using a default finite timeout."""
    sleep_secs = 0.01
    total_sleep_secs = 0
    while not future.running():
        time.sleep(sleep_secs)
        total_sleep_secs += sleep_secs
        if total_sleep_secs > DEFAULT_TIMEOUT_SECS:
            raise TimeoutError(
                f'Future did not start running after a timeout of {DEFAULT_TIMEOUT_SECS} seconds.'
            )


def _wait_for_result(future: Future) -> Any:
    """Waits until the given future is finished using a default finite timeout, and returns its result."""
    return future.result(timeout=DEFAULT_TIMEOUT_SECS)
