import numpy as np

from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d


def test_vec2d_is_equal():
    node_a = (1, 2)
    node_b = (2, 4)
    node_c = (1, 2)
    res_1 = Vec2d.is_equal(node_a, node_b)
    res_2 = Vec2d.is_equal(node_a, node_c)

    assert not res_1
    assert res_2


def test_vec2d_subtract():
    node_a = (1, 2)
    node_b = (2, 4)
    res_1 = Vec2d.subtract(node_a, node_b)
    res_2 = Vec2d.subtract(node_b, node_a)
    assert res_1 != res_2
    assert res_1 == (-1, -2)
    assert res_2 == (1, 2)


def test_vec2d_add():
    node_a = (1, 2)
    node_b = (2, 3)
    res_1 = Vec2d.add(node_a, node_b)
    res_2 = Vec2d.add(node_b, node_a)
    assert res_1 == res_2
    assert res_1 == (3, 5)


def test_vec2d_make_orthogonal():
    node_a = (1, 2)
    res_1 = Vec2d.make_orthogonal(node_a)
    assert res_1 == (2, -1)


def test_vec2d_euclidean_distance():
    node_a = (3, -7)
    node_0 = (0, 0)
    assert Vec2d.get_euclidean_distance(node_a, node_0) == Vec2d.get_norm(node_a)


def test_vec2d_manhattan_distance():
    node_a = (3, -7)
    node_0 = (0, 0)
    assert Vec2d.get_manhattan_distance(node_a, node_0) == 3 + 7


def test_vec2d_chebyshev_distance():
    node_a = (3, -7)
    node_0 = (0, 0)
    assert Vec2d.get_chebyshev_distance(node_a, node_0) == 7
    node_b = (-3, 7)
    node_0 = (0, 0)
    assert Vec2d.get_chebyshev_distance(node_b, node_0) == 7
    node_c = (3, 7)
    node_0 = (0, 0)
    assert Vec2d.get_chebyshev_distance(node_c, node_0) == 7


def test_vec2d_norm():
    node_a = (1, 2)
    node_b = (1, -2)
    res_1 = Vec2d.get_norm(node_a)
    res_2 = Vec2d.get_norm(node_b)
    assert np.sqrt(1 * 1 + 2 * 2) == res_1
    assert np.sqrt(1 * 1 + (-2) * (-2)) == res_2


def test_vec2d_normalize():
    node_a = (1, 2)
    node_b = (1, -2)
    res_1 = Vec2d.normalize(node_a)
    res_2 = Vec2d.normalize(node_b)
    assert np.isclose(1.0, Vec2d.get_norm(res_1))
    assert np.isclose(1.0, Vec2d.get_norm(res_2))


def test_vec2d_scale():
    node_a = (1, 2)
    node_b = (1, -2)
    res_1 = Vec2d.scale(node_a, 2)
    res_2 = Vec2d.scale(node_b, -2.5)
    assert res_1 == (2, 4)
    assert res_2 == (-2.5, 5)


def test_vec2d_round():
    node_a = (-1.95, -2.2)
    node_b = (1.95, 2.2)
    res_1 = Vec2d.round(node_a)
    res_2 = Vec2d.round(node_b)
    assert res_1 == (-2, -2)
    assert res_2 == (2, 2)


def test_vec2d_ceil():
    node_a = (-1.95, -2.2)
    node_b = (1.95, 2.2)
    res_1 = Vec2d.ceil(node_a)
    res_2 = Vec2d.ceil(node_b)
    assert res_1 == (-1, -2)
    assert res_2 == (2, 3)


def test_vec2d_floor():
    node_a = (-1.95, -2.2)
    node_b = (1.95, 2.2)
    res_1 = Vec2d.floor(node_a)
    res_2 = Vec2d.floor(node_b)
    assert res_1 == (-2, -3)
    assert res_2 == (1, 2)


def test_vec2d_bound():
    node_a = (-1.95, -2.2)
    node_b = (1.95, 2.2)
    res_1 = Vec2d.bound(node_a, -1, 0)
    res_2 = Vec2d.bound(node_b, 2, 2.2)
    assert res_1 == (-1, -1)
    assert res_2 == (2, 2.2)


def test_vec2d_rotate():
    node_a = (-1.95, -2.2)
    res_1 = Vec2d.rotate(node_a, -90.0)
    res_2 = Vec2d.rotate(node_a, 0.0)
    res_3 = Vec2d.rotate(node_a, 90.0)
    res_4 = Vec2d.rotate(node_a, 180.0)
    res_5 = Vec2d.rotate(node_a, 270.0)
    res_6 = Vec2d.rotate(node_a, 30.0)

    res_1 = (Vec2d.get_norm(Vec2d.subtract(res_1, (-2.2, 1.95))))
    res_2 = (Vec2d.get_norm(Vec2d.subtract(res_2, (-1.95, -2.2))))
    res_3 = (Vec2d.get_norm(Vec2d.subtract(res_3, (2.2, -1.95))))
    res_4 = (Vec2d.get_norm(Vec2d.subtract(res_4, (1.95, 2.2))))
    res_5 = (Vec2d.get_norm(Vec2d.subtract(res_5, (-2.2, 1.95))))
    res_6 = (Vec2d.get_norm(Vec2d.subtract(res_6, (-0.5887495373796556, -2.880255888325765))))

    assert np.isclose(0, res_1)
    assert np.isclose(0, res_2)
    assert np.isclose(0, res_3)
    assert np.isclose(0, res_4)
    assert np.isclose(0, res_5)
    assert np.isclose(0, res_6)
