from __future__ import annotations

import logging
from concurrent.futures import Executor, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from threading import RLock
from typing import Dict, Iterable, Optional, Tuple

from napari.components import Dims
from napari.layers import Layer
from napari.utils.events.event import EmitterGroup, Event

logger = logging.getLogger("napari.components._layer_slicer")


class _LayerSlicer:
    """
    High level class to control the creation of a slice (via a slice request),
    submit it (synchronously or asynchronously) to a thread pool, and emit the
    results when complete.

    Events
    ------
    ready
        emitted after slicing is done with a dict value that maps from layer
        to slice response. Note that this may be emitted on the main or
        a non-main thread. If usage of this event relies on something happening
        on the main thread, actions should be taken to ensure that the callback
        is also executed on the main thread (e.g. by decorating the callback
        with `@ensure_main_thread`).
    """

    def __init__(self):
        """
        Attributes
        ----------
        _executor : concurrent.futures.ThreadPoolExecutor
            manager for the slicing threading
        _force_sync: bool
            if true, forces slicing to execute synchronously
        _layers_to_task : dict
            task storage for cancellation logic
        _lock_layers_to_task : threading.RLock
            lock to guard against changes to `_layers_to_task` when finding,
            adding, or removing tasks.
        """
        self.events = EmitterGroup(source=self, ready=Event)
        self._executor: Executor = ThreadPoolExecutor(max_workers=1)
        self._force_sync = True
        self._layers_to_task: Dict[Tuple[Layer], Future] = {}
        self._lock_layers_to_task = RLock()

    @contextmanager
    def force_sync(self):
        """Context manager to temporarily force slicing to be synchronous.

        This should only be used from the main thread.

        >>> layer_slicer = _LayerSlicer()
        >>> layer = Image(...)  # an async-ready layer
        >>> with layer_slice.force_sync():
        >>>     layer_slicer.submit(layers=[layer], dims=Dims())
        """
        prev = self._force_sync
        self._force_sync = True
        try:
            yield None
        finally:
            self._force_sync = prev

    def wait_until_idle(self, timeout: Optional[float] = None) -> None:
        """Wait for all slicing tasks to complete before returning.

        Attributes
        ----------
        timeout: float or None
            (Optional) time in seconds to wait before raising TimeoutError. If set as None,
            there is no limit to the wait time. Defaults to None

        Raises
        ------
        TimeoutError: when the timeout limit has been exceeded and the task is
            not yet complete
        """
        futures = self._layers_to_task.values()
        _, not_done_futures = wait(futures, timeout=timeout)

        if len(not_done_futures) > 0:
            raise TimeoutError(
                f'Slicing {len(not_done_futures)} tasks did not complete within timeout ({timeout}s).'
            )

    def submit(
        self,
        layers: Iterable[Layer],
        dims: Dims,
        force: bool = False,
    ) -> Optional[Future[dict]]:
        """Slices the given layers with the given dims.

        Submitting multiple layers at one generates multiple requests, but only ONE task.

        This will attempt to cancel all pending slicing tasks that can be entirely
        replaced the new ones. If multiple layers are sliced, any task that contains
        only one of those layers can safely be cancelled. If a single layer is sliced,
        it will wait for any existing tasks that include that layer AND another layer,
        In other words, it will only cancel if the new task will replace the
        slices of all the layers in the pending task.

        Parameters
        ----------
        layers: iterable of layers
            The layers to slice.
        dims: Dims
            The dimensions values associated with the view to be sliced.
        force: bool
            True if slicing should be performed regardless if some cache thinks
            it can be skipped, False otherwise.

        Returns
        -------
        future of dict or none
            A future with a result that maps from a layer to an async layer
            slice response. Or none if no async slicing tasks were submitted.
        """
        if existing_task := self._find_existing_task(layers):
            logger.debug('Cancelling task for %s', layers)
            existing_task.cancel()

        async_requests = {}
        sync_requests = {}
        for layer in [layer for layer in layers if layer.visible]:
            if request := layer._make_slice_request(dims, force):
                if request.supports_async() and not self._force_sync:
                    async_requests[layer] = request
                else:
                    sync_requests[layer] = request

        # Submit async requests first to get them started.
        async_task = None
        if len(async_requests) > 0:
            async_task = self._executor.submit(
                self._slice_layers, async_requests
            )
            # Store the async task before adding the done callback to keep the
            # done callback logic simpler.
            with self._lock_layers_to_task:
                self._layers_to_task[tuple(async_requests)] = async_task
            async_task.add_done_callback(self._on_slice_done)

        # Then execute the sync tasks to work concurrently with the async ones.
        for request in sync_requests:
            request()

        return async_task

    def shutdown(self) -> None:
        """Shuts this down, preventing any new slice tasks from being submitted.

        This should only be called from the main thread.
        """
        # Replace with cancel_futures=True in shutdown when we drop support
        # for Python 3.8
        with self._lock_layers_to_task:
            tasks = tuple(self._layers_to_task.values())
        for task in tasks:
            task.cancel()
        self._executor.shutdown(wait=True)

    def _slice_layers(self, requests: Dict) -> Dict:
        """
        Iterates through a dictionary of request objects and call the slice
        on each individual layer. Can be called from the main or slicing thread.

        Attributes
        ----------
        requests: dict[Layer, SliceRequest]
            Dictionary of request objects to be used for constructing the slice

        Returns
        -------
        dict[Layer, SliceResponse]: which contains the results of the slice
        """
        return {layer: request() for layer, request in requests.items()}

    def _on_slice_done(self, task: Future[Dict]) -> None:
        """
        This is the "done_callback" which is added to each task.
        Can be called from the main or slicing thread.
        """
        if not self._try_to_remove_task(task):
            logger.debug('Task not found')
            return

        if task.cancelled():
            logger.debug('Cancelled task')
            return
        result = task.result()
        self.events.ready(Event('ready', value=result))

    def _try_to_remove_task(self, task: Future[Dict]) -> bool:
        """
        Attempt to remove task, return false if task not found, return true
        if task is found and removed from layers_to_task dict.

        This function provides a lock to ensure that the layers_to_task dict
        is unmodified during this process.
        """
        with self._lock_layers_to_task:
            for k_layers, v_task in self._layers_to_task.items():
                if v_task == task:
                    del self._layers_to_task[k_layers]
                    return True
        return False

    def _find_existing_task(
        self, layers: Iterable[Layer]
    ) -> Optional[Future[Dict]]:
        """Find the task associated with a list of layers. Returns the first
        task found for which the layers of the task are a subset of the input
        layers.

        This function provides a lock to ensure that the layers_to_task dict
        is unmodified during this process.
        """
        with self._lock_layers_to_task:
            layer_set = set(layers)
            for task_layers, task in self._layers_to_task.items():
                if set(task_layers).issubset(layer_set):
                    logger.debug(f'Found existing task for {task_layers}')
                    return task
        return None
