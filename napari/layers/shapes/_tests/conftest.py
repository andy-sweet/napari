import itertools

import numpy as np
import pytest


@pytest.fixture
def single_four_corner() -> list[np.ndarray]:
    return [
        np.array(
            [
                [6.5957985, 7.1852345],
                [2.87612, 2.0895412],
                [1.973858, 10.891626],
                [11.863327, 14.566921],
            ],
            dtype=np.float32,
        )
    ]


@pytest.fixture
def ten_four_corner() -> list[np.ndarray]:
    return [
        np.array(
            [
                [17.50093, 0.59361523],
                [18.020409, 18.638197],
                [14.526781, 14.467463],
                [9.254028, 18.727674],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [7.278204, 5.099733],
                [18.39801, 19.622711],
                [19.648762, 9.590886],
                [11.369405, 1.8619729],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [18.940228, 9.202899],
                [14.660338, 12.172355],
                [3.0547895, 10.981356],
                [1.7724335, 10.323669],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [3.7452614, 1.2272043],
                [17.442095, 16.222443],
                [15.570847, 12.4132],
                [8.985382, 4.0625296],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [9.748284, 4.3844504],
                [7.651263, 16.566586],
                [15.874295, 14.766316],
                [6.1144648, 9.241833],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [11.028608, 19.633142],
                [16.842916, 19.298588],
                [1.1768825, 3.404611],
                [17.53738, 14.9563875],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [12.355914, 17.984932],
                [6.8080034, 17.108805],
                [4.950972, 8.947148],
                [6.0480304, 11.312822],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [9.633244, 1.0103394],
                [17.489294, 6.927368],
                [16.326223, 0.42313808],
                [11.682402, 4.273177],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [6.587939, 3.057046],
                [9.089723, 10.495151],
                [2.997726, 0.76716715],
                [14.590194, 19.604425],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [8.401338, 11.062277],
                [4.8243365, 11.910896],
                [18.186176, 10.091279],
                [6.704347, 5.4453797],
            ],
            dtype=np.float32,
        ),
    ]


@pytest.fixture
def twenty_four_corner() -> list[np.ndarray]:
    return [
        np.array(
            [
                [3.915969, 7.8135843],
                [8.393104, 18.920172],
                [14.069965, 3.7124484],
                [3.4207442, 10.474266],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [13.093829, 10.628601],
                [13.130707, 6.009158],
                [13.47422, 19.55084],
                [11.290086, 1.8060791],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [15.458012, 17.083729],
                [18.065348, 13.092547],
                [6.021884, 7.284472],
                [14.465336, 14.867735],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [8.356629, 11.802527],
                [10.49842, 10.772692],
                [1.9225302, 15.309828],
                [2.7022123, 5.877228],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [0.0472871, 6.1385994],
                [13.490301, 6.684538],
                [16.91262, 9.216312],
                [6.5908933, 17.177036],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [13.509915, 3.2456365],
                [7.561683, 16.553562],
                [0.92989004, 5.3060174],
                [2.2870188, 19.118221],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [15.1053095, 4.8142157],
                [0.8535362, 14.753784],
                [10.039892, 17.245684],
                [19.919659, 12.106256],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [3.1561623, 6.8480477],
                [4.6244307, 13.403254],
                [15.039071, 0.5711754],
                [6.0181017, 10.039466],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [5.8035297, 1.1744947],
                [9.812645, 17.92537],
                [2.8984253, 1.9077939],
                [4.551977, 15.917308],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [19.832373, 12.937964],
                [5.653778, 3.7855823],
                [19.92601, 8.508228],
                [11.112172, 18.182554],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [3.234961, 5.392519],
                [9.195639, 11.232752],
                [7.030445, 2.1683385],
                [15.3207245, 11.744694],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [13.027798, 4.9237623],
                [1.8797916, 15.584881],
                [4.975614, 0.21063913],
                [15.884403, 2.5303805],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [14.246391, 17.44541],
                [16.53483, 16.567772],
                [15.474697, 9.554304],
                [0.9726791, 0.38446364],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [15.69831, 4.518636],
                [11.4159565, 16.977335],
                [10.6060295, 3.4504724],
                [11.813847, 19.47234],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [13.263935, 4.0687637],
                [14.042292, 4.803806],
                [2.8870966, 3.990361],
                [19.149946, 9.0334635],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [11.913461, 17.762383],
                [7.0338235, 0.88027215],
                [8.051452, 12.320653],
                [19.27673, 4.0772495],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [11.44268, 7.145645],
                [5.5853043, 4.4647007],
                [11.857039, 6.1128635],
                [19.915642, 9.154013],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [6.1400304, 8.206569],
                [7.1725554, 5.842869],
                [7.1434703, 2.830281],
                [16.632149, 0.33894876],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [15.774475, 6.7334757],
                [8.424132, 17.798683],
                [2.3209176, 2.4583323],
                [11.784108, 5.82434],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [10.666719, 19.523237],
                [11.11067, 5.5118046],
                [7.602155, 16.304794],
                [3.6093395, 19.374704],
            ],
            dtype=np.float32,
        ),
    ]


@pytest.fixture
def single_two_corners() -> list[np.ndarray]:
    return [
        np.array(
            [[19.630909, 4.094504], [6.050901, 4.739327]], dtype=np.float32
        )
    ]


@pytest.fixture
def ten_two_corners() -> list[np.ndarray]:
    return [
        np.array(
            [[13.568503, 3.6805813], [19.67907, 2.7589796]], dtype=np.float32
        ),
        np.array(
            [[0.771308, 7.7354484], [4.8191123, 2.6181173]], dtype=np.float32
        ),
        np.array(
            [[15.471089, 18.194923], [9.326611, 6.637912]], dtype=np.float32
        ),
        np.array(
            [[3.8954687, 1.0359402], [11.732754, 2.2416632]], dtype=np.float32
        ),
        np.array(
            [[4.812966, 1.8201057], [8.283552, 10.809121]], dtype=np.float32
        ),
        np.array(
            [[18.000807, 17.218533], [8.915606, 16.599905]], dtype=np.float32
        ),
        np.array(
            [[18.782663, 7.1298103], [18.03142, 16.10713]], dtype=np.float32
        ),
        np.array(
            [[8.764693, 9.151221], [0.23011725, 11.056407]], dtype=np.float32
        ),
        np.array(
            [[5.1388326, 14.443498], [4.202188, 14.370723]], dtype=np.float32
        ),
        np.array(
            [[18.97423, 14.230529], [9.725716, 5.771379]], dtype=np.float32
        ),
    ]


@pytest.fixture(
    params=[
        'single_four_corner',
        'ten_four_corner',
        'single_two_corners',
        'ten_two_corners',
    ]
)
def two_and_four_corners(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=['single_two_corners', 'ten_two_corners'])
def two_corners(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def polygons():
    return [
        np.array(
            [
                [4.3472557, 19.092707],
                [12.385868, 9.441998],
                [11.030792, 7.6805634],
                [5.285246, 10.27396],
                [2.7348402, 3.5166616],
                [3.7063575, 6.323633],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [4.4513884, 1.3679211],
                [6.0738254, 18.56807],
                [6.5996447, 12.0634985],
                [0.18547149, 17.87575],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [12.3054905, 9.864174],
                [19.879269, 1.6925181],
                [14.934044, 12.565186],
                [15.530991, 19.44206],
                [11.512451, 14.002723],
                [16.356024, 18.321203],
                [16.751368, 2.385293],
                [0.87240964, 2.6350331],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [7.100853, 2.7780702],
                [4.5400105, 19.321873],
                [16.720345, 5.207758],
                [4.631408, 15.734205],
                [16.353329, 11.723293],
                [2.6555562, 9.734464],
                [19.614532, 15.008726],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [5.2782, 3.0509028],
                [11.8710575, 14.706998],
                [7.360063, 11.472012],
                [2.41257, 14.442611],
                [16.697138, 14.617584],
                [7.5786357, 6.9946175],
                [14.939011, 5.434313],
                [0.37513006, 13.553444],
                [2.1920238, 9.2616825],
                [14.726601, 1.4248564],
            ],
            dtype=np.float32,
        ),
        np.array(
            [[2.2108445, 8.827738], [4.913185, 4.5486608]], dtype=np.float32
        ),
        np.array(
            [
                [13.347389, 0.15637287],
                [4.0802207, 2.606402],
                [9.446134, 3.0769732],
                [18.885344, 6.4319353],
                [7.6073093, 7.5026965],
                [5.6101704, 9.132152],
                [11.165885, 16.490229],
                [16.560513, 1.9102397],
                [12.203619, 9.196948],
                [10.25309, 4.242064],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [18.879221, 10.407496],
                [12.269095, 14.931961],
                [13.592372, 7.0003676],
                [4.6837134, 8.655116],
                [13.341547, 12.509719],
                [18.894463, 4.2734575],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [4.0373907, 8.009811],
                [18.000036, 7.472767],
                [0.10347261, 16.277378],
                [10.2881565, 19.298697],
                [6.6568017, 18.500061],
                [10.836271, 19.892216],
                [16.511509, 19.442808],
                [16.949703, 14.658965],
                [7.315138, 18.384237],
                [17.577011, 15.8563175],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [5.0307846, 13.863499],
                [12.189665, 6.997516],
                [8.854627, 15.04776],
                [12.708408, 14.246945],
                [6.940458, 17.575687],
                [6.773896, 16.208294],
                [19.53695, 4.133974],
                [0.81575483, 15.194465],
            ],
            dtype=np.float32,
        ),
    ]


def generate_self_intersecting_polygon(n, reverse, radius=1):
    assert n % 2 == 1
    angles = np.linspace(0, 4 * np.pi, n, endpoint=False)
    if reverse:
        angles = angles[::-1]
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def generate_regular_polygon(n, reverse, radius=1):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if reverse:
        angles = angles[::-1]
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def rotation_matrix(angle):
    return np.array(
        [
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle))],
        ]
    )


ANGLES = [0, 5, 75, 95, 355]


@pytest.fixture(
    params=itertools.product(ANGLES, [3, 4, 7, 12, 15, 20], [False, True])
)
def regular_polygon(request):
    angle, n_vertex, reverse = request.param
    rot = rotation_matrix(angle)
    poly = generate_regular_polygon(n_vertex, reverse)
    return np.dot(poly, rot)


@pytest.fixture(
    params=itertools.product(ANGLES, [5, 7, 15, 21], [False, True])
)
def self_intersecting_polygon(request):
    angle, n_vertex, reverse = request.param
    rot = rotation_matrix(angle)
    poly = generate_self_intersecting_polygon(n_vertex, reverse)
    return np.dot(poly, rot)


@pytest.fixture(params=[0, 1, 2, 3, 4])
def non_convex_poly(request):
    poly = np.array([[0, 0], [2, 0], [1, 1], [2, 2], [0, 2]])
    return np.roll(poly, request.param, axis=0)


@pytest.fixture
def line():
    return np.array([[0, 0], [5, 5], [7, 7], [10, 10]])


@pytest.fixture
def line_two_point():
    return np.array([[0, 0], [10, 10]])


@pytest.fixture
def poly_hole():
    return np.array(
        [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
            [0, 0],
            [2, 5],
            [5, 8],
            [8, 5],
            [5, 2],
            [2, 5],
        ]
    )
