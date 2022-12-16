from __future__ import annotations

import logging
from concurrent.futures import Executor, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from threading import RLock
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from napari.components import Dims
from napari.layers import Layer
from napari.utils.events.event import EmitterGroup, Event

logger = logging.getLogger("napari.components._layer_slicer")


# Layers that can be asynchronously sliced must be able to make
# a slice request that can be called and will produce a slice
# response. The request and response types will vary per layer
# type, which means that the values of the dictionary result of
# ``_slice_layers`` cannot be fixed to a single type.

_SliceResponse = TypeVar('_SliceResponse')
_SliceRequest = Callable[[], _SliceResponse]


@runtime_checkable
class _AsyncSliceable(Protocol[_SliceResponse]):
    def _make_slice_request(self, dims: Dims) -> _SliceRequest[_SliceResponse]:
        ...

    def _update_slice_response(self, response: _SliceResponse) -> None:
        ...


class _AsyncLayerSlicer:
    """A layer slicer that performs slicing asynchronously.

    Attributes
    ----------
    _executor : ThreadPoolExecutor
        For executing slice tasks off the main thread.
    _force_sync: bool
        True if slicing should be forced to be synchronous.
    _layers_to_task : dict
        Stores the currently running slice tasks.
    _lock_layers_to_task : RLock
        Guards against changes to `_layers_to_task` when finding,
        adding, or removing tasks.
    """

    def __init__(self):
        self.events = EmitterGroup(source=self, ready=Event)
        self._executor: Executor = ThreadPoolExecutor(max_workers=1)
        self._force_sync = False
        self._layers_to_task: Dict[Tuple[Layer], Future] = {}
        self._lock_layers_to_task = RLock()

    @contextmanager
    def force_sync(self):
        prev = self._force_sync
        self._force_sync = True
        try:
            yield None
        finally:
            self._force_sync = prev

    def wait_until_idle(self, timeout: Optional[float] = None) -> None:
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
    ) -> List[Future[dict]]:
        # Cancel any tasks that are slicing a subset of the layers
        # being sliced now. This allows us to slice arbitrary sets of
        # layers with some sensible and not too complex cancellation
        # policy.
        if existing_task := self._find_existing_task(layers):
            logger.debug('Cancelling task for %s', layers)
            existing_task.cancel()

        # Not all layer types will initially be asynchronously sliceable.
        # The following logic gives us a way to handle those in the short
        # term as we develop, and also in the long term if there are cases
        # when we want to perform sync slicing anyway.
        requests = {}
        sync_layers = []
        for layer in layers:
            if isinstance(layer, _AsyncSliceable) and not self._force_sync:
                requests[layer] = layer._make_slice_request(dims)
            else:
                sync_layers.append(layer)

        # create one task for all requests
        tasks = []
        if len(requests) > 0:
            task = self._executor.submit(self._slice_layers, requests)

            # store task for cancellation logic
            # this is purposefully done before adding the done callback to ensure
            # that the task is added before the done callback can be executed
            with self._lock_layers_to_task:
                self._layers_to_task[tuple(requests)] = task

            task.add_done_callback(self._on_slice_done)

            tasks.append(task)

        # slice the sync layers after async submission so that async
        # tasks can potentially run concurrently
        for layer in sync_layers:
            if force:
                layer.refresh()
            else:
                layer._slice_dims(dims.point, dims.ndisplay, dims.order)

        return tasks

    def shutdown(self) -> None:
        # Replace with cancel_futures=True in shutdown when we drop support
        # for Python 3.8
        with self._lock_layers_to_task:
            tasks = tuple(self._layers_to_task.values())
        for task in tasks:
            task.cancel()
        self._executor.shutdown(wait=True)

    def _slice_layers(self, requests: Dict) -> Dict:
        return {layer: request() for layer, request in requests.items()}

    def _on_slice_done(self, task: Future[Dict]) -> None:
        if not self._try_to_remove_task(task):
            logger.debug('Task not found')
            return

        if task.cancelled():
            logger.debug('Cancelled task')
            return
        result = task.result()
        self.events.ready(Event('ready', value=result))

    def _try_to_remove_task(self, task: Future[Dict]) -> bool:
        """Tries to remove a task from the layers_to_task dict.

        Returns True if the task was found, False otherwise.
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
        """Find the task associated with a list of layers.

        Returns the first task found for which the layers of the task are
        a subset of the input layers.
        """
        with self._lock_layers_to_task:
            layer_set = set(layers)
            for task_layers, task in self._layers_to_task.items():
                if set(task_layers).issubset(layer_set):
                    logger.debug(f'Found existing task for {task_layers}')
                    return task
        return None
