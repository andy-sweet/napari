"""
Add points with text
====================

Display a points layer on top of an image layer using the add_points and
add_image APIs
"""

import numpy as np
import napari


# add the image
viewer = napari.view_image(np.zeros((400, 400)))
# add the points
points = np.array([[100, 100], [200, 300], [333, 111]])

# create features for each point
features = {
    'confidence': np.array([1, 0.5, 0]),
    'good_point': np.array([True, False, False]),
}

style = {
    'face_color': {'feature': 'good_point', 'colormap': ['blue', 'green']},
    'edge_color': {'feature': 'confidence', 'colormap': 'gray'},
}

text = {
    'string': 'Confidence is {confidence:.2f}',
    'size': 20,
    'color': 'green',
    'translation': np.array([-30, 0]),
}

# create a points layer where the face_color is set by the good_point feature
# and the edge_color is set via a color map (grayscale) on the confidence
# feature.
points_layer = viewer.add_points(
    points,
    features=features,
    text=text,
    size=20,
    edge_width=7,
    edge_width_is_relative=False,
    style=style,
)

if __name__ == '__main__':
    napari.run()
