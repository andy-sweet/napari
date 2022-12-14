from dataclasses import dataclass
from typing import Any

from napari.layers.utils._slice_input import _SliceInput


@dataclass(frozen=True)
class _VectorsSliceResponse:
    """Contains all the output data of slicing a vectors layer.

    Currently, vectors layers do not support async slicing, so do not
    have an output. Instead executing an instance of the corresponding
    request type performs slicing through side-effects and should be
    executed synchronously instead.
    """


@dataclass(frozen=True)
class _VectorsSliceRequest:
    """A callable that stores all the input data needed to slice a vector layer.

    This should be treated a deeply immutable structure, even though some
    fields can be modified in place. It is like a function that has captured
    all its inputs already.

    In general, the calling an instance of this may take a long time, so you may
    want to run it off the main thread.

    Attributes
    ----------
    layer : Vectors
        The vectors layer to slice.
    """

    layer: Any  # Avoid cyclic import with Vectors
    dims: _SliceInput

    def supports_async(self) -> bool:
        return False

    def __call__(self) -> _VectorsSliceResponse:
        self.layer._slice_input = self.dims
        self.layer.set_view_slice()
        return _VectorsSliceResponse()
