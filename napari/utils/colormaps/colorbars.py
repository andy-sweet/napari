import numpy as np


def make_colorbar(cmap, size=(18, 28), horizontal=True):
    """Make a colorbar from a colormap.

    Parameters
    ----------
    cmap : vispy.color.Colormap
        Colormap to create colorbar with.
    size : 2-tuple
        Shape of colorbar.
    horizontal : bool
        If True this is oriented horizontally, otherwise vertically.

    Returns
    -------
    cbar : array
        Array of colorbar in uint8.
    """

    if horizontal:
        input = np.linspace(0, 1, size[1])
        bar = np.tile(np.expand_dims(input, 1), size[0]).transpose((1, 0))
    else:
        input = np.linspace(0, 1, size[0])
        bar = np.tile(np.expand_dims(input, 1), size[1])

    color_array = cmap.map(bar.ravel())
    cbar = color_array.reshape(bar.shape + (4,))

    return np.round(255 * cbar).astype(np.uint8).copy(order='C')
