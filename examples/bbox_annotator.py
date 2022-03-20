from typing import List
from magicgui.widgets import ComboBox, Container
import napari
from napari.layers import Shapes
import numpy as np
import pandas as pd
import toolz as tz
from skimage import data

from napari.utils.events.event import Event


# set up the categorical annotation values and text display properties
box_annotations = ['person', 'sky', 'camera']
text_feature = 'box_label'
features = pd.DataFrame({
    text_feature: pd.Series([], dtype=pd.CategoricalDtype(box_annotations))
})
text_color = 'green'
text_size = 20


@tz.curry
def set_shapes_feature_default(layer: Shapes, feature: str, value: str):
    """Set the default value of a feature on a Shapes layer."""
    layer.feature_defaults[feature] = value
    layer.events.feature_defaults(changed=layer.feature_defaults[[feature]])


@tz.curry
def set_selected_shapes_features_to_default(layer: Shapes, feature: str, event: Event):
    """Sets the features values of the selected shapes to their defaults.
    This is a side-effect of the deprecated current_properties setter,
    but does not occur when modifying feature_defaults."""
    if feature in event.changed:
        indices = list(layer.selected_data)
        layer.features[feature][indices] = event.changed[feature][0]
        layer.events.features(changed=layer.features[feature][indices])


@tz.curry
def set_menu_value_to_new_feature_default(menu: ComboBox, feature: str, event: Event):
    """Updates the menu value when the associated feature default changes."""
    if feature in event.changed and event.changed[feature][0] != menu.value:
        menu.value = event.changed[feature][0]


# create the GUI for selecting the values
def create_label_menu(shapes_layer: Shapes, label_feature: str, labels: List[str]):
    """Create a label menu widget that can be added to the napari viewer dock

    Parameters
    ----------
    shapes_layer : napari.layers.Shapes
        a napari shapes layer
    label_feature : str
        the name of the shapes feature to use the displayed text
    labels : List[str]
        list of the possible text labels values.

    Returns
    -------
    label_widget : magicgui.widgets.Container
        the container widget with the label combobox
    """
    # Create the label selection menu
    label_menu = ComboBox(label='text label', choices=labels)
    label_widget = Container(widgets=[label_menu])
    label_menu.changed.connect(set_shapes_feature_default(shapes_layer, label_feature))

    shapes_layer.events.feature_defaults.connect(
        set_menu_value_to_new_feature_default(label_menu, label_feature))
    shapes_layer.events.feature_defaults.connect(
        set_selected_shapes_features_to_default(shapes_layer, label_feature))
    shapes_layer.events.features.connect(shapes_layer.refresh_text)

    return label_widget


# create a stack with the camera image shifted in each slice
n_slices = 5
base_image = data.camera()
image = np.zeros((n_slices, base_image.shape[0], base_image.shape[1]), dtype=base_image.dtype)
for slice_idx in range(n_slices):
    shift = 1 + 10 * slice_idx
    image[slice_idx, ...] = np.pad(base_image, ((0, 0), (shift, 0)), mode='constant')[:, :-shift]


# create a viewer with a fake t+2D image
viewer = napari.view_image(image)

# create an empty shapes layer initialized with
# text set to display the box label
text_kwargs = {
    'text': text_feature,
    'size': text_size,
    'color': text_color
}
shapes = viewer.add_shapes(
    face_color='black',
    features=features,
    text=text_kwargs,
    ndim=3
)

# create the label section gui
label_widget = create_label_menu(
    shapes_layer=shapes,
    label_feature=text_feature,
    labels=box_annotations
)
# add the label selection gui to the viewer as a dock widget
viewer.window.add_dock_widget(label_widget, area='right', name='label_widget')

# set the shapes layer mode to adding rectangles
shapes.mode = 'add_rectangle'

napari.run()
