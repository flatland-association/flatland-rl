import numpy as np


def position_to_coordinate(depth, positions):
    """Converts coordinates to positions:
         [ (0,0) (0,1) ..  (0,w-1)
           (1,0) (1,1)     (1,w-1)
           ...
           (d-1,0) (d-1,1)     (d-1,w-1)
          ]

         -->

         [ 0      d    ..  (w-1)*d
           1      d+1
           ...
           d-1    2d-1     w*d-1
         ]

    :param depth:
    :param positions:
    :return:
    """
    coords = ()
    for p in positions:
        coords = coords + ((int(p) % depth, int(p) // depth),)  # changed x_dim to y_dim
    return coords


def coordinate_to_position(depth, coords):
    """
    Converts positions to coordinates:
         [ 0      d    ..  (w-1)*d
           1      d+1
           ...
           d-1    2d-1     w*d-1
         ]
         -->
         [ (0,0) (0,1) ..  (0,w-1)
           (1,0) (1,1)     (1,w-1)
           ...
           (d-1,0) (d-1,1)     (d-1,w-1)
          ]

    :param depth:
    :param coords:
    :return:
    """
    position = np.empty(len(coords), dtype=int)
    idx = 0
    for t in coords:
        position[idx] = int(t[1] * depth + t[0])
        idx += 1
    return position


def distance_on_rail(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
