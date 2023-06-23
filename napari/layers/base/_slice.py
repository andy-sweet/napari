from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING, Protocol

from napari.layers.utils._slice_input import _SliceInput

if TYPE_CHECKING:
    from napari.layers import Layer

# We use an incrementing non-negative integer to uniquely identify
# slices that is unbounded based on Python 3's int.
_request_ids = count()


def _next_request_id() -> int:
    """Returns the next integer identifier associated with a slice."""
    return next(_request_ids)


class _SliceResponse(Protocol):
    """Captures the output of slicing.

    The attributes of this vary per layer type due to different data structures,
    but some basic attributes are shared across all layers.

    Attributes
    ----------
    request_id : int
        The unique identifier associated with the request that generated this.
    dims : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    """

    request_id: int
    dims: _SliceInput


class _SliceRequest(Protocol):
    """Callable that captures all state needed to slice a layer.

    The attributes of this vary per layer type due to different data structures,
    but some basic attributes are shared across all layers.

    Attributes
    ----------
    id : int
        The unique identifier associated with this request.
        This is unique across all layer types for the liftime of a process.
    dims : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    """

    id: int
    dims: _SliceInput

    def supports_async(self) -> bool:
        """Returns True if this can be safely called on a separate thread."""

    def __call__(self) -> _SliceResponse:
        """Executes this request to return a slice response."""


@dataclass(frozen=True)
class _LayerSliceResponse:
    """The base layer slice response that only contains basic attributes.

    A request that generates this response is expected to use the old approach
    to slicing using `Layer.set_view_slice` to update the layer's slice state
    through side-effects on the main thread.
    """

    request_id: int
    dims: _SliceInput


@dataclass(frozen=True)
class _LayerSliceRequest:
    """The base layer slice request that should only be executed on the main thread."""

    layer: 'Layer'
    dims: _SliceInput
    id: int = field(default_factory=_next_request_id)

    @property
    def supports_async(self) -> bool:
        return False

    def __call__(self) -> _LayerSliceResponse:
        self.layer.set_view_slice()
        return _LayerSliceResponse(request_id=self.id, dims=self.dims)
