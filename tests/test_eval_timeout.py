from flatland.evaluators.client import FlatlandRemoteClient
from flatland.core.env_observation_builder import DummyObservationBuilder
#from my_observation_builder import CustomObservationBuilder
import numpy as np
import time



from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder


class CustomObservationBuilder(ObservationBuilder):

    def __init__(self):
        super(CustomObservationBuilder, self).__init__()

    def set_env(self, env: Environment):
        super().set_env(env)
        # Note :
        # The instantiations which depend on parameters of the Env object should be 
        # done here, as it is only here that the updated self.env instance is available
        self.rail_obs = np.zeros((self.env.height, self.env.width))

    def reset(self):
        """
        Called internally on every env.reset() call, 
        to reset any observation specific variables that are being used
        """
        self.rail_obs[:] = 0        
        for _x in range(self.env.width):
            for _y in range(self.env.height):
                # Get the transition map value at location _x, _y
                transition_value = self.env.rail.get_full_transitions(_y, _x)
                self.rail_obs[_y, _x] = transition_value

    def get(self, handle: int = 0):

        agent = self.env.agents[handle]


        status = agent.status
        position = agent.position
        direction = agent.direction
        initial_position = agent.initial_position
        target = agent.target


        return self.rail_obs, (status, position, direction, initial_position, target)



def my_controller(obs, number_of_agents):
    _action = {}
    for _idx in range(number_of_agents):
        _action[_idx] = np.random.randint(0, 5)
    return _action


def __disabled__test_random_timeouts():
    remote_client = FlatlandRemoteClient(verbose=False)

    my_observation_builder = CustomObservationBuilder()

    evaluation_number = 0
    n_evalations = 10

    step_delay_rate = 0.001
    step_delay = 6

    reset_delay_rate = 0.2
    reset_delay = 10

    while evaluation_number < n_evalations:

        evaluation_number += 1
        # Switch to a new evaluation environemnt
        # 
        # a remote_client.env_create is similar to instantiating a 
        # RailEnv and then doing a env.reset()
        # hence it returns the first observation from the 
        # env.reset()
        # 
        # You can also pass your custom observation_builder object
        # to allow you to have as much control as you wish 
        # over the observation of your choice.
        time_start = time.time()
        observation, info = remote_client.env_create(
                        obs_builder_object=my_observation_builder
                    )
        env_creation_time = time.time() - time_start
        if not observation:
            #
            # If the remote_client returns False on a `env_create` call,
            # then it basically means that your agent has already been 
            # evaluated on all the required evaluation environments,
            # and hence its safe to break out of the main evaluation loop
            break
        
        print("Evaluation Number : {}".format(evaluation_number))

        if np.random.uniform() < reset_delay_rate:
            print(f"eval {evaluation_number} sleeping for {reset_delay} seconds")
            time.sleep(reset_delay)

        local_env = remote_client.env
        number_of_agents = len(local_env.agents)

        time_taken_by_controller = []
        time_taken_per_step = []
        steps = 0



        while True:
            time_start = time.time()
            action = my_controller(observation, number_of_agents)
            time_taken = time.time() - time_start
            time_taken_by_controller.append(time_taken)

            time_start = time.time()

            try:
                observation, all_rewards, done, info = remote_client.env_step(action)
            except StopAsyncIteration as err:
                print("timeout error ", err)
                break

            steps += 1
            time_taken = time.time() - time_start
            time_taken_per_step.append(time_taken)

            if np.random.uniform() < step_delay_rate:
                print(f"step {steps} sleeping for {step_delay} seconds")
                time.sleep(step_delay)

            if done['__all__']:
                print("Reward : ", sum(list(all_rewards.values())))
                break
        
        np_time_taken_by_controller = np.array(time_taken_by_controller)
        np_time_taken_per_step = np.array(time_taken_per_step)
        print("="*100)
        print("="*100)
        print("Evaluation Number : ", evaluation_number)
        print("Current Env Path : ", remote_client.current_env_path)
        print("Env Creation Time : ", env_creation_time)
        print("Number of Steps : ", steps)
        print("Mean/Std of Time taken by Controller : ", np_time_taken_by_controller.mean(), np_time_taken_by_controller.std())
        print("Mean/Std of Time per Step : ", np_time_taken_per_step.mean(), np_time_taken_per_step.std())
        print("="*100)

    print("Evaluation of all environments complete...")
    ########################################################################
    # Submit your Results
    # 
    # Please do not forget to include this call, as this triggers the 
    # final computation of the score statistics, video generation, etc
    # and is necesaary to have your submission marked as successfully evaluated
    ########################################################################
    print(remote_client.submit())

if __name__ == "__main__":
    test_random_timeouts()