from __future__ import annotations

from concurrent.futures import Future
from contextlib import contextmanager
from typing import Iterable, List, Optional

from napari.components import Dims
from napari.layers import Layer
from napari.utils.events.event import EmitterGroup, Event


class _SyncLayerSlicer:
    """Slices layers synchronously."""

    def __init__(self):
        self.events = EmitterGroup(source=self, ready=Event)

    def submit(
        self,
        layers: Iterable[Layer],
        dims: Dims,
        *,
        force: bool = False,
    ) -> List[Future[dict]]:
        for layer in layers:
            if force:
                layer.refresh()
            else:
                layer._slice_dims(dims.point, dims.ndisplay, dims.order)
        return []

    def shutdown(self) -> None:
        # This has no executor to shutdown or tasks to cancel.
        pass

    @contextmanager
    def force_sync(self):
        # This is always synchronous.
        yield
        pass

    def wait_until_idle(self, timeout: Optional[float] = None) -> None:
        # This is always idle.
        pass
