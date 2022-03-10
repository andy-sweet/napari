import warnings
from typing import List, Tuple

import numpy as np
from pydantic import PositiveInt, validator

from napari.layers.utils.string_encoding import (
    ConstantStringEncoding,
    StringEncoding,
    validate_string_encoding,
)

from ...utils.colormaps.standardize_color import transform_color
from ...utils.events import EventedModel
from ...utils.events.custom_types import Array
from ...utils.translations import trans
from ..base._base_constants import Blending
from ._text_constants import Anchor
from ._text_utils import get_text_anchors


class Text(EventedModel):
    """Manages properties related to text displayed in conjunction with the layer.

    Attributes
    ----------
    string : StringEncoding
        Encodes the strings to be displayed from layer features.
    visible : bool
        True if the text should be displayed, false otherwise.
    size : float
        Font size of the text, which must be positive. Default value is 12.
    color : array
        Font color for the text as an [R, G, B, A] array. Can also be expressed
        as a string on construction or setting.
    blending : Blending
        The blending mode that determines how RGB and alpha values of the layer
        visual get mixed. Allowed values are 'translucent' and 'additive'.
        Note that 'opaque` blending is not allowed, as it colors the bounding box
        surrounding the text, and if given, 'translucent' will be used instead.
    anchor : Anchor
        The location of the text origin relative to the bounding box.
        Should be 'center', 'upper_left', 'upper_right', 'lower_left', or 'lower_right'.
    translation : np.ndarray
        Offset from the anchor point.
    rotation : float
        Angle of the text elements around the anchor point. Default value is 0.
    """

    string: StringEncoding = ConstantStringEncoding(constant='')
    visible: bool = True
    size: PositiveInt = 12
    color: Array[float, (4,)] = 'cyan'
    blending: Blending = Blending.TRANSLUCENT
    anchor: Anchor = Anchor.CENTER
    # Use a scalar default translation to broadcast to any dimensionality.
    translation: Array[float] = 0
    rotation: float = 0

    def _update(self, features) -> None:
        self.string._update(features)

    def _refresh(self, features) -> None:
        self.string._clear()
        self._update(features)

    def _delete(self, indices) -> None:
        self.string._delete(indices)

    def _copy(self, indices: List[int]) -> dict:
        """Copies all encoded values at the given indices."""
        string = self.string._values
        if string.ndim > 0:
            string = string[indices]
        return {'string': string}

    def _paste(self, values: dict) -> None:
        """Pastes encoded values to the end of the existing values."""
        self.string._append(values['string'])

    def _update_safely(self, **kwargs) -> None:
        """Updates this in-place from a layer.

        This will effectively overwrite all existing state, but in-place
        so that there is no need for any external components to reconnect
        to any useful events. For this reason, only fields that change in
        value will emit their corresponding events.

        Parameters
        ----------
        See :meth:`TextManager._from_layer`.
        """
        # Create a new instance from the input to populate all fields.
        new_text = Text(**kwargs)

        # Update a copy of this so that any associated errors are raised
        # before actually making the update. This does not need to be a
        # deep copy because update will only try to reassign fields and
        # should not mutate any existing fields in-place.
        current_text = self.copy()
        current_text.update(new_text)

        # If we got here, then there were no errors, so update for real.
        # Connected callbacks may raise errors, but those are bugs.
        self.update(new_text)

    def _compute_text_coords(
        self, view_data: np.ndarray, ndisplay: int
    ) -> Tuple[np.ndarray, str, str]:
        """Calculate the coordinates for each text element in view

        Parameters
        ----------
        view_data : np.ndarray
            The in view data from the layer
        ndisplay : int
            The number of dimensions being displayed in the viewer

        Returns
        -------
        text_coords : np.ndarray
            The coordinates of the text elements
        anchor_x : str
            The vispy text anchor for the x axis
        anchor_y : str
            The vispy text anchor for the y axis
        """
        anchor_coords, anchor_x, anchor_y = get_text_anchors(
            view_data, ndisplay, self.anchor
        )
        text_coords = anchor_coords + self.translation
        return text_coords, anchor_x, anchor_y

    @validator('string', pre=True, always=True)
    def _check_string(cls, string):
        return validate_string_encoding(string)

    @validator('color', pre=True, always=True)
    def _check_color(cls, color):
        return transform_color(color)[0]

    @validator('blending', pre=True, always=True)
    def _check_blending_mode(cls, blending):
        blending_mode = Blending(blending)

        # The opaque blending mode is not allowed for text.
        # See: https://github.com/napari/napari/pull/600#issuecomment-554142225
        if blending_mode == Blending.OPAQUE:
            blending_mode = Blending.TRANSLUCENT
            warnings.warn(
                trans._(
                    'opaque blending mode is not allowed for text. setting to translucent.',
                    deferred=True,
                ),
                category=RuntimeWarning,
            )

        return blending_mode
