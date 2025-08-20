from flatland.evaluators.service import version_check


def test_version_check():
    assert version_check("4.1.4", ["4.1.4"])
    assert not version_check("4.1.4", ["4.1"])
    assert version_check("4.1.4", "", ">=4.1.4")
    assert version_check("4.1.4", "", ">=4.1.4,<4.1.5")
    assert not version_check("4.1.4", "", ">=4.1.4,<4.1.3")
    assert not version_check("4.1.4", "", "")
