from flatland.envs.agent_chains import MotionCheck


def test_add_agent_waiting():
    mc = MotionCheck()

    # two agents desiring to enter the same cell from off the map at the same time step
    mc.addAgent(0, None, (0, 0))
    mc.addAgent(1, None, (0, 0))

    mc.find_conflicts()

    assert mc.check_motion(0, None)
    assert not mc.check_motion(1, None)


def test_add_agent_waiting_blocked():
    mc = MotionCheck()

    # first agent already on the cell
    mc.addAgent(0, (0, 0), (0, 0))

    # second agent from off the map desiring to enter the same cell
    mc.addAgent(1, None, (0, 0))

    mc.find_conflicts()

    assert not mc.check_motion(0, (0, 0))  # current behaviour is motion check false if staying
    assert not mc.check_motion(1, None)
