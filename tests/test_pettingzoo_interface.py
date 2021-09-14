import pytest

@pytest.mark.skip(reason="Only for testing pettingzoo interface and wrappers")
def test_petting_zoo_interface_env():
    import numpy as np
    import os
    import PIL
    import shutil

    from flatland.contrib.interface import flatland_env
    from flatland.contrib.utils import env_generators

    from flatland.envs.observations import TreeObsForRailEnv
    from flatland.envs.predictions import ShortestPathPredictorForRailEnv


    # First of all we import the Flatland rail environment
    from flatland.utils.rendertools import RenderTool, AgentRenderVariant

    from flatland.contrib.wrappers.flatland_wrappers import SkipNoChoiceCellsWrapper
    from flatland.contrib.wrappers.flatland_wrappers import ShortestPathActionWrapper  # noqa

    # Custom observation builder without predictor
    # observation_builder = GlobalObsForRailEnv()

    # Custom observation builder with predictor
    observation_builder = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(30))
    seed = 11
    save = True
    np.random.seed(seed)
    experiment_name = "flatland_pettingzoo"
    total_episodes = 2

    if save:
        try:
            if os.path.isdir(experiment_name):
                shutil.rmtree(experiment_name)
            os.mkdir(experiment_name)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    rail_env = env_generators.sparse_env_small(seed, observation_builder)
    rail_env = env_generators.small_v0(seed, observation_builder)

    rail_env.reset(random_seed=seed)

    # For Shortest Path Action Wrapper, change action to 1
    # rail_env = ShortestPathActionWrapper(rail_env)  
    rail_env = SkipNoChoiceCellsWrapper(rail_env, accumulate_skipped_rewards=False, discounting=0.0)

    dones = {}
    dones['__all__'] = False

    step = 0
    ep_no = 0
    frame_list = []
    all_actions_env = []
    all_actions_pettingzoo_env = []
    # while not dones['__all__']:
    while ep_no < total_episodes:
        action_dict = {}
        # Chose an action for each agent
        for a in range(rail_env.get_num_agents()):
            # action = env_generators.get_shortest_path_action(rail_env, a)
            action = 2
            all_actions_env.append(action)
            action_dict.update({a: action})
            step += 1
            # Do the environment step

        observations, rewards, dones, information = rail_env.step(action_dict)
        frame_list.append(PIL.Image.fromarray(rail_env.render(mode="rgb_array")))

        if dones['__all__']:
            completion = env_generators.perc_completion(rail_env)
            print("Final Agents Completed:", completion)
            ep_no += 1
            if save:
                frame_list[0].save(f"{experiment_name}{os.sep}out_{ep_no}.gif", save_all=True, 
                                   append_images=frame_list[1:], duration=3, loop=0)       
            frame_list = []
            rail_env.reset(random_seed=seed+ep_no)

    
#  __sphinx_doc_begin__
    env = flatland_env.env(environment=rail_env)
    seed = 11
    env.reset(random_seed=seed)
    step = 0
    ep_no = 0
    frame_list = []
    while ep_no < total_episodes:
        for agent in env.agent_iter():
            obs, reward, done, info = env.last()
            # act = env_generators.get_shortest_path_action(env.environment, get_agent_handle(agent))
            act = 2
            all_actions_pettingzoo_env.append(act)
            env.step(act)
            frame_list.append(PIL.Image.fromarray(env.render(mode='rgb_array')))
            step += 1
# __sphinx_doc_end__
        completion = env_generators.perc_completion(env)
        print("Final Agents Completed:", completion)
        ep_no += 1
        if save:
            frame_list[0].save(f"{experiment_name}{os.sep}pettyzoo_out_{ep_no}.gif", save_all=True, 
                               append_images=frame_list[1:], duration=3, loop=0)
        frame_list = []
        env.close()
        env.reset(random_seed=seed+ep_no)
        min_len = min(len(all_actions_pettingzoo_env), len(all_actions_env))
        assert all_actions_pettingzoo_env[:min_len] == all_actions_env[:min_len], "actions do not match"


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-sv", __file__]))
