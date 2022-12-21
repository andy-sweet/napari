from dataclasses import dataclass
from typing import Protocol

from napari.layers.utils._slice_input import _SliceInput


class Layer(Protocol):
    _slice_input: _SliceInput

    def set_view_slice(self) -> None:
        ...


@dataclass(frozen=True)
class _LayerSliceResponse:
    pass


@dataclass(frozen=True)
class _LayerSliceRequest:
    layer: Layer
    dims: _SliceInput

    def supports_async(self) -> bool:
        return False

    def __call__(self) -> _LayerSliceResponse:
        self.layer._slice_input = self.dims
        self.layer.set_view_slice()
