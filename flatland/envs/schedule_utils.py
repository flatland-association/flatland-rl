import collections

Schedule = collections.namedtuple('Schedule',   'agent_positions '
                                                'agent_directions '
                                                'agent_targets '
                                                'agent_speeds '
                                                'agent_malfunction_rates '
                                                'max_episode_steps')
