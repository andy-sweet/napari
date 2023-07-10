from typing import TYPE_CHECKING, Tuple

import numpy as np
from qtpy.QtWidgets import QGridLayout, QLabel, QWidget

from napari._qt.containers import QtListView
from napari._qt.containers.qt_axis_model import AxisList, AxisModel
from napari._qt.widgets.qt_tooltip import QtToolTipLabel
from napari.components import Dims
from napari.utils.events import SelectableEventedList
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.viewer import Viewer


def set_dims_order(dims: Dims, order: Tuple[int, ...]):
    if type(order[0]) == AxisModel:
        order = [a.axis for a in order]
    dims.order = order


def _array_in_range(arr: np.ndarray, low: int, high: int) -> bool:
    return (arr >= low) & (arr < high)


def move_indices(axes_list: SelectableEventedList, order: Tuple[int, ...]):
    with axes_list.events.blocker_all():
        if tuple(axes_list) == tuple(order):
            return
        axes = [a.axis for a in axes_list]
        ax_to_existing_position = {a: ix for ix, a in enumerate(axes)}
        move_list = np.asarray(
            [(ax_to_existing_position[order[i]], i) for i in range(len(order))]
        )
        for src, dst in move_list:
            axes_list.move(src, dst)
            move_list[_array_in_range(move_list[:, 0], dst, src)] += 1
        # remove the elements from the back if order has changed length
        while len(axes_list) > len(order):
            axes_list.pop()


class QtDimsSorter(QWidget):
    """
    Modified from:
    https://github.com/jni/zarpaint/blob/main/zarpaint/_dims_chooser.py
    """
    def __init__(self, viewer: 'Viewer', parent=None) -> None:
        super().__init__(parent=parent)
        dims = viewer.dims
        self.axis_list = AxisList.from_dims(dims)

        view = QtListView(self.axis_list)
        view.setSizeAdjustPolicy(QtListView.AdjustToContents)

        layout = QGridLayout()
        self.setLayout(layout)

        widget_tooltip = QtToolTipLabel(self)
        widget_tooltip.setObjectName('help_label')
        widget_tooltip.setToolTip(trans._('Drag dimensions to reorder, uncheck to lock dimension in place.'))

        widget_title = QLabel(trans._('Dims. Ordering'), self)

        self.layout().addWidget(widget_title, 0, 0)
        self.layout().addWidget(widget_tooltip, 0, 1)
        self.layout().addWidget(view, 1, 0)

        # connect axes_list and dims
        self.axis_list.events.reordered.connect(
            lambda event: set_dims_order(dims, event.value), 
            until=self.parent().finished
        )
        dims.events.order.connect(
            lambda event: move_indices(self.axis_list, event.value), 
            until=self.parent().finished
        )
