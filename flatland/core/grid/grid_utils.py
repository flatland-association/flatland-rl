from typing import Tuple, Callable, List, Type

import numpy as np

Vector2D: Type = Tuple[float, float]
IntVector2D: Type = Tuple[int, int]

IntVector2DArray: Type = List[IntVector2D]
IntVector2DArrayArray: Type = List[List[IntVector2D]]

Vector2DArray: Type = List[Vector2D]
Vector2DArrayArray: Type = List[List[Vector2D]]

IntVector2DDistance: Type = Callable[[IntVector2D, IntVector2D], float]


class Vec2dOperations:

    @staticmethod
    def is_equal(node_a: Vector2D, node_b: Vector2D) -> bool:
        """
        vector operation : node_a + node_b

        :param node_a: tuple with coordinate (x,y) or 2d vector
        :param node_b: tuple with coordinate (x,y) or 2d vector

        :return:
            check if node_a and nobe_b are equal
        """
        return node_a[0] == node_b[0] and node_a[1] == node_b[1]

    @staticmethod
    def subtract(node_a: Vector2D, node_b: Vector2D) -> Vector2D:
        """
        vector operation : node_a - node_b

        :param node_a: tuple with coordinate (x,y) or 2d vector
        :param node_b: tuple with coordinate (x,y) or 2d vector

        :return:
            tuple with coordinate (x,y) or 2d vector
        """
        return node_a[0] - node_b[0], node_a[1] - node_b[1]

    @staticmethod
    def add(node_a: Vector2D, node_b: Vector2D) -> Vector2D:
        """
        vector operation : node_a + node_b

        :param node_a: tuple with coordinate (x,y) or 2d vector
        :param node_b: tuple with coordinate (x,y) or 2d vector

        :return: tuple with coordinate (x,y) or 2d vector
        """
        return node_a[0] + node_b[0], node_a[1] + node_b[1]

    @staticmethod
    def make_orthogonal(node: Vector2D) -> Vector2D:
        """
        vector operation : rotates the 2D vector +90°

        :param node: tuple with coordinate (x,y) or 2d vector

        :return: tuple with coordinate (x,y) or 2d vector
        """
        return node[1], -node[0]

    @staticmethod
    def get_norm(node: Vector2D) -> float:
        """
        calculates the euclidean norm of the 2d vector
        [see: https://lyfat.wordpress.com/2012/05/22/euclidean-vs-chebyshev-vs-manhattan-distance/]

        :param node: tuple with coordinate (x,y) or 2d vector

        :return:
            tuple with coordinate (x,y) or 2d vector
        """
        return np.sqrt(node[0] * node[0] + node[1] * node[1])

    @staticmethod
    def get_euclidean_distance(node_a: Vector2D, node_b: Vector2D) -> float:
        """
        calculates the euclidean norm of the 2d vector

        Parameters
        ----------
        node_a
            tuple with coordinate (x,y) or 2d vector
        node_b
            tuple with coordinate (x,y) or 2d vector

        Returns
        -------
        float
            Euclidean distance
        """
        return Vec2dOperations.get_norm(Vec2dOperations.subtract(node_b, node_a))

    @staticmethod
    def get_manhattan_distance(node_a: Vector2D, node_b: Vector2D) -> float:
        """
        calculates the manhattan distance of the 2d vector
        [see: https://lyfat.wordpress.com/2012/05/22/euclidean-vs-chebyshev-vs-manhattan-distance/]

        Parameters
        ----------
        node_a
            tuple with coordinate (x,y) or 2d vector
        node_b
            tuple with coordinate (x,y) or 2d vector

        Returns
        -------
        float
            Mahnhattan distance
        """
        delta = (Vec2dOperations.subtract(node_b, node_a))
        return np.abs(delta[0]) + np.abs(delta[1])

    @staticmethod
    def get_chebyshev_distance(node_a: Vector2D, node_b: Vector2D) -> float:
        """
        calculates the chebyshev norm of the 2d vector
        [see: https://lyfat.wordpress.com/2012/05/22/euclidean-vs-chebyshev-vs-manhattan-distance/]

        Parameters
        ----------
        node_a
            tuple with coordinate (x,y) or 2d vector
        node_b
            tuple with coordinate (x,y) or 2d vector

        Returns
        -------
        float
            the chebyshev distance
        """
        delta = (Vec2dOperations.subtract(node_b, node_a))
        return max(np.abs(delta[0]), np.abs(delta[1]))

    @staticmethod
    def normalize(node: Vector2D) -> Tuple[float, float]:
        """
        normalize the 2d vector = `v/|v|`

        :param node: tuple with coordinate (x,y) or 2d vector

        :return: tuple with coordinate (x,y) or 2d vector
        """
        n = Vec2dOperations.get_norm(node)
        if n > 0.0:
            n = 1 / n
        return Vec2dOperations.scale(node, n)

    @staticmethod
    def scale(node: Vector2D, scale: float) -> Vector2D:
        """
         scales the 2d vector = node * scale

         :param node: tuple with coordinate (x,y) or 2d vector
         :param scale: scalar to scale

         :return: tuple with coordinate (x,y) or 2d vector
         """
        return node[0] * scale, node[1] * scale

    @staticmethod
    def round(node: Vector2D) -> IntVector2D:
        """
         rounds the x and y coordinate and convert them to an integer values

         :param node: tuple with coordinate (x,y) or 2d vector

         :return: tuple with coordinate (x,y) or 2d vector
         """
        return int(np.round(node[0])), int(np.round(node[1]))

    @staticmethod
    def ceil(node: Vector2D) -> IntVector2D:
        """
         ceiling the x and y coordinate and convert them to an integer values

         :param node: tuple with coordinate (x,y) or 2d vector

         :return:
            tuple with coordinate (x,y) or 2d vector
         """
        return int(np.ceil(node[0])), int(np.ceil(node[1]))

    @staticmethod
    def floor(node: Vector2D) -> IntVector2D:
        """
         floor the x and y coordinate and convert them to an integer values

         :param node: tuple with coordinate (x,y) or 2d vector

         :return:
            tuple with coordinate (x,y) or 2d vector
         """
        return int(np.floor(node[0])), int(np.floor(node[1]))

    @staticmethod
    def bound(node: Vector2D, min_value: float, max_value: float) -> Vector2D:
        """
         force the values x and y to be between min_value and max_value

         :param node: tuple with coordinate (x,y) or 2d vector
         :param min_value: scalar value
         :param max_value: scalar value

         :return:
            tuple with coordinate (x,y) or 2d vector
         """
        return max(min_value, min(max_value, node[0])), max(min_value, min(max_value, node[1]))

    @staticmethod
    def rotate(node: Vector2D, rot_in_degree: float) -> Vector2D:
        """
         rotate the 2d vector with given angle in degree

         :param node: tuple with coordinate (x,y) or 2d vector
         :param rot_in_degree:  angle in degree

         :return:
            tuple with coordinate (x,y) or 2d vector
         """
        alpha = rot_in_degree / 180.0 * np.pi
        x0 = node[0]
        y0 = node[1]
        x1 = x0 * np.cos(alpha) - y0 * np.sin(alpha)
        y1 = x0 * np.sin(alpha) + y0 * np.cos(alpha)
        return x1, y1


def position_to_coordinate(depth: int, positions: List[int]):
    """Converts coordinates to positions::

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

    Parameters
    ----------
    depth : int
    positions : List[Tuple[int,int]]
    """
    coords = ()
    for p in positions:
        coords = coords + ((int(p) % depth, int(p) // depth),)  # changed x_dim to y_dim
    return coords


def coordinate_to_position(depth, coords):
    """
    Converts positions to coordinates::

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
        # Set None type coordinates off the grid
        if np.isnan(t[0]):
            position[idx] = -1
        else:
            position[idx] = int(t[1] * depth + t[0])
        idx += 1
    return position


def distance_on_rail(pos1, pos2, metric="Euclidean"):
    if metric == "Euclidean":
        return np.sqrt(np.power(pos1[0] - pos2[0], 2) + np.power(pos1[1] - pos2[1], 2))
    if metric == "Manhattan":
        return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])
