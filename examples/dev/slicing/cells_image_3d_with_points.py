import numpy as np
import napari
from skimage.data import cells3d

image_data = np.squeeze(cells3d()[:, 1, :, :])
points_data = np.array([
    [23, 200.00786761, 172.78269126],
    [23, 77.21676614, 113.60960391],
    [26, 131.66711883, 37.76804124],
    [33, 220.84346175, 79.43922952],
    [33, 156.11421595, 111.10933261],
    [33, 141.94601194, 188.06212697],
    [31, 48.88035811, 227.51085187],
    [36, 97.49674444, 157.78106348],
    [37, 48.88035811, 182.22816061],
    [38, 220.56565383, 250.56890939],
])

viewer = napari.Viewer()
viewer.add_image(image_data, contrast_limits=(0, 32000))
viewer.add_points(points_data, face_color=[0.6, 0, 0], edge_color='transparent')

if __name__ == '__main__':
    napari.run()
