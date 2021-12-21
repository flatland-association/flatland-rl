from typing import List
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.step_utils.states import TrainState
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.step_utils import env_utils

class Deadlock_Checker:
  def __init__(self, env):
    self.env = env
    self.deadlocked_agents = []
    self.immediate_deadlocked = []
    

  def reset(self) -> None:
    self.deadlocked_agents = []
    self.immediate_deadlocked = []

        
  # an immediate deadlock consists of two trains "trying to pass through each other".
  # An agent may have a free possible transition, but took a bad action and "ran into another train". This is now a deadlock, and the other free
  # direction can not be chosen anymore!
  def check_immediate_deadlocks(self, action_dict) -> List[EnvAgent]:
    """
      output: list of agents who are in immediate deadlocks
    """
    env = self.env
    newly_deadlocked_agents = []
 
    # TODO: check restrictions to relevant agents (status ACTIVE, etc.)
    relevant_agents = [agent for agent in env.agents if agent.state != TrainState.DONE and agent.position is not None]
    for agent in relevant_agents:
      other_agents = [other_agent for other_agent in env.agents if other_agent != agent] # check if this is a good test for inequality. Maybe use handles...

      # get the transitions the agent can take from his current position and orientation
      # an indicator array of the form e.g. (0,1,1,0) meaning that he can only go to east and south, not to north and west.
      possible_transitions = env.rail.get_transitions(*agent.position, agent.direction)
      #print(f"possible transitions: {possible_transitions}")

      # the directions are: 0(north), 1(east), 2(south) and 3(west)
      #possible_directions = [direction for direction, flag in enumerate(possible_transitions) if flag == 1]
      #print(f"possible directions: {possible_directions}")


      ################### only consider direction for actually chosen action ###############################
      new_position, new_direction = env_utils.apply_action_independent(action=action_dict[agent.handle], rail=env.rail, position=agent.position, direction=agent.direction)
      
      #assert new_direction in possible_directions, "Error, action leads to impossible direction"
      assert new_position == get_new_position(agent.position, new_direction), "Error, something is wrong with new position"


      opposed_agent_id = env.agent_positions[new_position] # TODO: check that agent_positions now works correctly in flatland V3 (i.e. gets correctly updated...)
      # agent_positions[cell] is an agent_id if an agent is there, otherwise -1.
      if opposed_agent_id != -1:
        opposed_agent = env.agents[opposed_agent_id]

        # other agent with opposing direction is in the way --> deadlock
        # an opposing direction means having a different direction than our agent would have if he moved to the new cell. (180 degrees or 90 degrees to our agent)
        if opposed_agent.direction != new_direction:
          if agent not in newly_deadlocked_agents: # to avoid duplicates
            newly_deadlocked_agents.append(agent)
          if opposed_agent not in newly_deadlocked_agents: # to avoid duplicates
            newly_deadlocked_agents.append(opposed_agent)
     
    self.immediate_deadlocked = newly_deadlocked_agents

    return newly_deadlocked_agents


  # main method to check for all deadlocks
  def check_deadlocks(self, action_dict) -> List[EnvAgent]:
    env = self.env

    relevant_agents = [agent for agent in env.agents if agent.state != TrainState.DONE and agent.position is not None]
 
    immediate_deadlocked = self.check_immediate_deadlocks(action_dict)
    self.immediate_deadlocked = immediate_deadlocked

   
    deadlocked = immediate_deadlocked[:] 

    # now we have to "close": each train which is blocked by another deadlocked train becomes deadlocked itself.
    still_changing = True
    while still_changing:

      still_changing = False # will be overwritten below if a change did occur

      # check if for any agent, there is a new deadlock found
      for agent in relevant_agents:
        #possible_transitions = env.rail.get_transitions(*agent.position, agent.direction)
        #print(f"possible transitions: {possible_transitions}")

        # the directions are: 0 (north), 1(east), 2(south) and 3(west)
        #possible_directions = [direction for direction, flag in enumerate(possible_transitions) if flag == 1]
        #print(f"possible directions: {possible_directions}")

        new_position, new_direction = env_utils.apply_action_independent(action=action_dict[agent.handle], rail=env.rail, position=agent.position, direction=agent.direction)
        #assert new_direction in possible_directions, "Error, action leads to impossible direction"
        assert new_position == get_new_position(agent.position, new_direction), "Error, something is wrong with new position"

        opposed_agent_id = env.agent_positions[new_position]
      
        if opposed_agent_id != -1: # there is an opposed agent there
          opposed_agent = env.agents[opposed_agent_id]

          if opposed_agent in deadlocked:
            if agent not in deadlocked: # to avoid duplicates
              deadlocked.append(agent)
              still_changing = True

    self.deadlocked_agents = deadlocked

    return deadlocked