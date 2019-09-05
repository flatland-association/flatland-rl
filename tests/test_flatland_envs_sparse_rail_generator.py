import numpy as np

from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool


def test_sparse_rail_generator():
    env = RailEnv(width=50,
                  height=50,
                  rail_generator=sparse_rail_generator(num_cities=10,  # Number of cities in map
                                                       num_intersections=10,  # Number of interesections in map
                                                       num_trainstations=50,  # Number of possible start/targets on map
                                                       min_node_dist=6,  # Minimal distance of nodes
                                                       node_radius=3,  # Proximity of stations to city center
                                                       num_neighb=3,  # Number of connections to other cities
                                                       seed=5,  # Random seed
                                                       grid_mode=False  # Ordered distribution of nodes
                                                       ),
                  schedule_generator=sparse_schedule_generator(),
                  number_of_agents=10,
                  obs_builder_object=GlobalObsForRailEnv())
    # reset to initialize agents_static
    env_renderer = RenderTool(env, gl="PILSVG", )
    env_renderer.render_env(show=True, show_observations=True, show_predictions=False)
    env_renderer.gl.save_image("./sparse_generator_false.png")
    # TODO test assertions!


def test_rail_env_action_required_info():
    np.random.seed(0)
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train
    env_always_action = RailEnv(width=50,
                                height=50,
                                rail_generator=sparse_rail_generator(num_cities=10,  # Number of cities in map
                                                                     num_intersections=10,
                                                                     # Number of interesections in map
                                                                     num_trainstations=50,
                                                                     # Number of possible start/targets on map
                                                                     min_node_dist=6,  # Minimal distance of nodes
                                                                     node_radius=3,
                                                                     # Proximity of stations to city center
                                                                     num_neighb=3,
                                                                     # Number of connections to other cities
                                                                     seed=5,  # Random seed
                                                                     grid_mode=False  # Ordered distribution of nodes
                                                                     ),
                                schedule_generator=sparse_schedule_generator(speed_ration_map),
                                number_of_agents=10,
                                obs_builder_object=GlobalObsForRailEnv())
    np.random.seed(0)
    env_only_if_action_required = RailEnv(width=50,
                                          height=50,
                                          rail_generator=sparse_rail_generator(num_cities=10,  # Number of cities in map
                                                                               num_intersections=10,
                                                                               # Number of interesections in map
                                                                               num_trainstations=50,
                                                                               # Number of possible start/targets on map
                                                                               min_node_dist=6,
                                                                               # Minimal distance of nodes
                                                                               node_radius=3,
                                                                               # Proximity of stations to city center
                                                                               num_neighb=3,
                                                                               # Number of connections to other cities
                                                                               seed=5,  # Random seed
                                                                               grid_mode=False
                                                                               # Ordered distribution of nodes
                                                                               ),
                                          schedule_generator=sparse_schedule_generator(speed_ration_map),
                                          number_of_agents=10,
                                          obs_builder_object=GlobalObsForRailEnv())
    env_renderer = RenderTool(env_always_action, gl="PILSVG", )

    for step in range(100):
        print("step {}".format(step))

        action_dict_always_action = dict()
        action_dict_only_if_action_required = dict()
        # Chose an action for each agent in the environment
        for a in range(env_always_action.get_num_agents()):
            action = np.random.choice(np.arange(4))
            action_dict_always_action.update({a: action})
            if step == 0 or info_only_if_action_required['action_required'][a]:
                action_dict_only_if_action_required.update({a: action})
            else:
                print("[{}] not action_required {}, speed_data={}".format(step, a,
                                                                          env_always_action.agents[a].speed_data))

        obs_always_action, rewards_always_action, done_always_action, info_always_action = env_always_action.step(
            action_dict_always_action)
        obs_only_if_action_required, rewards_only_if_action_required, done_only_if_action_required, info_only_if_action_required = env_only_if_action_required.step(
            action_dict_only_if_action_required)

        for a in range(env_always_action.get_num_agents()):
            assert len(obs_always_action[a]) == len(obs_only_if_action_required[a])
            for i in range(len(obs_always_action[a])):
                assert np.array_equal(obs_always_action[a][i], obs_only_if_action_required[a][i])
            assert np.array_equal(rewards_always_action[a], rewards_only_if_action_required[a])
            assert np.array_equal(done_always_action[a], done_only_if_action_required[a])
            assert info_always_action['action_required'][a] == info_only_if_action_required['action_required'][a]

        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

        if done_always_action['__all__']:
            break


def test_rail_env_malfunction_speed_info():
    np.random.seed(0)
    stochastic_data = {'prop_malfunction': 0.5,  # Percentage of defective agents
                       'malfunction_rate': 30,  # Rate of malfunction occurence
                       'min_duration': 3,  # Minimal duration of malfunction
                       'max_duration': 10  # Max duration of malfunction
                       }
    env = RailEnv(width=50,
                  height=50,
                  rail_generator=sparse_rail_generator(num_cities=10,  # Number of cities in map
                                                       num_intersections=10,
                                                       # Number of interesections in map
                                                       num_trainstations=50,
                                                       # Number of possible start/targets on map
                                                       min_node_dist=6,  # Minimal distance of nodes
                                                       node_radius=3,
                                                       # Proximity of stations to city center
                                                       num_neighb=3,
                                                       # Number of connections to other cities
                                                       seed=5,  # Random seed
                                                       grid_mode=False  # Ordered distribution of nodes
                                                       ),
                  schedule_generator=sparse_schedule_generator(),
                  number_of_agents=10,
                  obs_builder_object=GlobalObsForRailEnv(),
                  stochastic_data=stochastic_data)

    env_renderer = RenderTool(env, gl="PILSVG", )
    for step in range(100):
        action_dict = dict()
        # Chose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            action = np.random.choice(np.arange(4))
            action_dict.update({a: action})

        obs, rewards, done, info = env.step(
            action_dict)

        assert 'malfunction' in info
        for a in range(env.get_num_agents()):
            assert info['malfunction'][a] >= 0
            assert info['speed'][a] >= 0 and info['speed'][a] <= 1
            assert info['speed'][a] == env.agents[a].speed_data['speed']

        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

        if done['__all__']:
            break


def test_sparse_generator_with_too_man_cities_does_not_break_down():
    np.random.seed(0)

    RailEnv(width=50,
            height=50,
            rail_generator=sparse_rail_generator(
                num_cities=100,  # Number of cities in map
                num_intersections=10,  # Number of interesections in map
                num_trainstations=50,  # Number of possible start/targets on map
                min_node_dist=6,  # Minimal distance of nodes
                node_radius=3,  # Proximity of stations to city center
                num_neighb=3,  # Number of connections to other cities
                seed=5,  # Random seed
                grid_mode=False  # Ordered distribution of nodes
            ),
            schedule_generator=sparse_schedule_generator(),
            number_of_agents=10,
            obs_builder_object=GlobalObsForRailEnv())
