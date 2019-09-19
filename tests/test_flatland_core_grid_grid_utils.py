from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d


def test_vec2d_add():
    node_a = (1, 2)
    node_b = (2, 3)
    res_1 = Vec2d.add(node_a, node_b)
    res_2 = Vec2d.add(node_b, node_a)
    assert res_1 == res_2
    assert res_1 == (3, 5)


def test_vec2d_subtract():
    node_a = (1, 2)
    node_b = (2, 4)
    res_1 = Vec2d.subtract(node_a, node_b)
    res_2 = Vec2d.subtract(node_b, node_a)
    assert res_1 != res_2
    assert res_1 == (-1, -2)
    assert res_2 == (1, 2)


def test_vec2d_make_orthogonal():
    node_a = (1, 2)
    res_1 = Vec2d.make_orthogonal(node_a)
    assert res_1 == (2, -1)


def test_vec2d_subtract():
    node_a = (1, 2)
    node_b = (2, 4)
    node_c = (1, 2)
    res_1 = Vec2d.is_equal(node_a, node_b)
    res_2 = Vec2d.is_equal(node_a, node_c)

    assert not res_1
    assert res_2


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


def test_vec2d_normalize():
    node_a = (1, 2)
    node_b = (4, 12)
    res_1 = Vec2d.normalize(node_a)
    res_2 = Vec2d.normalize(node_b)
    eps = 0.000000000001
    assert 1.0 - Vec2d.get_norm(res_1) < eps
    assert 1.0 - Vec2d.get_norm(res_2) < eps
