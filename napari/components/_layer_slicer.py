from __future__ import annotations

from concurrent.futures import Future
from typing import ContextManager, Iterable, List, Optional, Protocol

from napari.components import Dims
from napari.layers import Layer
from napari.utils.events.event import Event

_ASYNC_SLICING = False


class _ReadyEmitterGroup(Protocol):
    ready: Event  # with value of type Optional[_SliceResponse]


class _LayerSlicer(Protocol):
    """Used to generate slices of napari layers.

    In general, this is not thread safe. It should only be created and
    used on the main thread.

    Events
    ------
    ready
        Emitted after an asynchronous slicing tasks is done with a dict value
        that maps from a layer to its slice response. Note that this may be
        emitted on the main or a non-main thread. If usage of this event relies
        on something happening on the main thread (e.g. a widget update), other
        actions should be taken to ensure that the callback is executed on the
        main thread (e.g. by decorating the callback with superqt's
        `@ensure_main_thread`).

    Examples
    --------
    slicer = _provide_layer_slicer()
    slicer.events.ready.connect(on_slice_ready)
    slicer.submit(viewer.layers, viewer.dims)
    """

    events: _ReadyEmitterGroup

    def submit(
        self, layers: Iterable[Layer], dims: Dims, *, force: bool
    ) -> List[Future[dict]]:
        """Slices the given layers with the given dims.

        This should only be called from the main thread.

        Some implementations may create and use other threads to perform slicing.
        Any asynchronous tasks running on these other threads are returned as futures.
        When a task completes, the ready event is emitted.

        Parameters
        ----------
        layers: iterable of layers
            The layers to slice.
        dims: Dims
            The dimensions values associated with the view to be sliced.
        force: bool
            True if slicing should be forced even if this infers that it is not needed
            (e.g. due to caching or some other state or configuration).
            False otherwise.

        Returns
        -------
        list of futures of dicts
            Each future corresponds to an async slicing task that was submitted.
            This can be empty either because this slicer is empty or because
            there is no work to be done.
        """

    def force_sync(self) -> ContextManager[None]:
        """Context manager to temporarily force slicing to be synchronous.

        Examples
        --------
        >>> layer_slicer = _provide_layer_slicer()
        >>> with layer_slice.force_sync():
        >>>     layer_slicer.submit(layers=viewer.layers, dims=viewer.dims)
        """

    def wait_until_idle(self, timeout: Optional[float] = None) -> None:
        """Wait for all slicing tasks to complete before returning.

        Parameters
        ----------
        timeout: float or None
            (Optional) time in seconds to wait before raising TimeoutError. If set as None,
            there is no limit to the wait time. Defaults to None

        Raises
        ------
        TimeoutError: when the timeout limit has been exceeded and the task is
            not yet complete
        """

    def shutdown(self) -> None:
        """Shuts this down, preventing any new slice tasks from being submitted."""


def _provide_layer_slicer() -> _LayerSlicer:
    """Provides the layer slicer that napari will use."""
    if _ASYNC_SLICING:
        from napari.components._async_layer_slicer import _AsyncLayerSlicer

        return _AsyncLayerSlicer()
    else:
        from napari.components._sync_layer_slicer import _SyncLayerSlicer

        return _SyncLayerSlicer()
