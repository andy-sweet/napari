import typing

from qtpy.QtCore import QModelIndex, QSize, Qt
from qtpy.QtGui import QImage

from ..._vispy.thumbnail import VispyThumbnail
from ...layers import Layer
from .qt_list_model import QtListModel

ThumbnailRole = Qt.UserRole + 2


class QtLayerListModel(QtListModel[Layer]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layer_to_thumbnail = dict()

    def _get_thumbnail(self, layer: Layer) -> QImage:
        # Ideally would reuse VispyThumbnail object, but this causes OpenGL
        # issues possibly related to the active canvas.
        # if layer not in self._layer_to_thumbnail:
        self._layer_to_thumbnail[layer] = VispyThumbnail(layer)
        thumbnail = self._layer_to_thumbnail[layer]
        image = thumbnail.get_image()
        return QImage(
            image,
            image.shape[1],
            image.shape[0],
            QImage.Format_RGBA8888,
        )

    def data(self, index: QModelIndex, role: Qt.ItemDataRole):
        """Return data stored under ``role`` for the item at ``index``."""
        if not index.isValid():
            return None
        layer = self.getItem(index)
        if role == Qt.ItemDataRole.DisplayRole:  # used for item text
            return layer.name
        if role == Qt.ItemDataRole.TextAlignmentRole:  # alignment of the text
            return Qt.AlignCenter
        if role == Qt.ItemDataRole.EditRole:
            # used to populate line edit when editing
            return layer.name
        if role == Qt.ItemDataRole.ToolTipRole:  # for tooltip
            return layer.name
        if (
            role == Qt.ItemDataRole.CheckStateRole
        ):  # the "checked" state of this item
            return Qt.Checked if layer.visible else Qt.Unchecked
        if role == Qt.ItemDataRole.SizeHintRole:  # determines size of item
            return QSize(200, 34)
        if role == ThumbnailRole:  # return the thumbnail
            return self._get_thumbnail(layer)
        # normally you'd put the icon in DecorationRole, but we do that in the
        # # LayerDelegate which is aware of the theme.
        # if role == Qt.ItemDataRole.DecorationRole:  # icon to show
        #     pass
        return super().data(index, role)

    def setData(
        self,
        index: QModelIndex,
        value: typing.Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if role == Qt.ItemDataRole.CheckStateRole:
            self.getItem(index).visible = value
        elif role == Qt.ItemDataRole.EditRole:
            self.getItem(index).name = value
            role = Qt.ItemDataRole.DisplayRole
        else:
            return super().setData(index, value, role=role)

        self.dataChanged.emit(index, index, [role])
        return True

    def _process_event(self, event):
        # The model needs to emit `dataChanged` whenever data has changed
        # for a given index, so that views can update themselves.
        # Here we convert native events to the dataChanged signal.
        if not hasattr(event, 'index'):
            return
        role = {
            'thumbnail': ThumbnailRole,
            'visible': Qt.ItemDataRole.CheckStateRole,
            'name': Qt.ItemDataRole.DisplayRole,
        }.get(event.type)
        roles = [role] if role is not None else []
        row = self.index(event.index)
        self.dataChanged.emit(row, row, roles)
