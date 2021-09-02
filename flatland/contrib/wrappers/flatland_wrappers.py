import numpy as np
import os
import PIL
import shutil
# MICHEL: my own imports
import unittest
import typing
from collections import defaultdict
from typing import Dict, Any, Optional, Set, List, Tuple


from flatland.envs.observations import TreeObsForRailEnv,GlobalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.core.grid.grid4_utils import get_new_position

# First of all we import the Flatland rail environment
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

from flatland.envs.agent_utils import EnvAgent, RailAgentStatus
from flatland.envs.rail_env import RailEnv, RailEnvActions


def possible_actions_sorted_by_distance(env: RailEnv, handle: int):
    agent = env.agents[handle]
    

    if agent.status == RailAgentStatus.READY_TO_DEPART:
        agent_virtual_position = agent.initial_position
    elif agent.status == RailAgentStatus.ACTIVE:
        agent_virtual_position = agent.position
    elif agent.status == RailAgentStatus.DONE:
        agent_virtual_position = agent.target
    else:
        print("no action possible!")
        if agent.status == RailAgentStatus.DONE_REMOVED:
          print(f"agent status: DONE_REMOVED for agent {agent.handle}")
          print("to solve this problem, do not input actions for removed agents!")
          return [(RailEnvActions.DO_NOTHING, 0)] * 2
        print("agent status:")
        print(RailAgentStatus(agent.status))
        #return None
        # NEW: if agent is at target, DO_NOTHING, and distance is zero.
        # NEW: (needs to be tested...)
        return [(RailEnvActions.DO_NOTHING, 0)] * 2

    possible_transitions = env.rail.get_transitions(*agent_virtual_position, agent.direction)
    print(f"possible transitions: {possible_transitions}")
    distance_map = env.distance_map.get()[handle]
    possible_steps = []
    for movement in list(range(4)):
      # MICHEL: TODO: discuss with author of this code how it works, and why it breaks down in my test!
      # should be much better commented or structured to be readable!
        if possible_transitions[movement]:
            if movement == agent.direction:
                action = RailEnvActions.MOVE_FORWARD
            elif movement == (agent.direction + 1) % 4:
                action = RailEnvActions.MOVE_RIGHT
            elif movement == (agent.direction - 1) % 4:
                action = RailEnvActions.MOVE_LEFT
            else:
                # MICHEL: prints for debugging
                print(f"An error occured. movement is: {movement}, agent direction is: {agent.direction}")
                if movement == (agent.direction + 2) % 4 or (movement == agent.direction - 2) % 4:
                    print("it seems that we are turning by 180 degrees. Turning in a dead end?")

                # MICHEL: can this happen when we turn 180 degrees in a dead end?
                # i.e. can we then have movement == agent.direction + 2 % 4 (resp. ... == - 2 % 4)?

                # TRY OUT: ASSIGN MOVE_FORWARD HERE...
                action = RailEnvActions.MOVE_FORWARD
                print("Here we would have a ValueError...")
                #raise ValueError("Wtf, debug this shit.")
               
            distance = distance_map[get_new_position(agent_virtual_position, movement) + (movement,)]
            possible_steps.append((action, distance))
    possible_steps = sorted(possible_steps, key=lambda step: step[1])


    # MICHEL: what is this doing?
    # if there is only one path to target, this is both the shortest one and the second shortest path.
    if len(possible_steps) == 1:
        return possible_steps * 2
    else:
        return possible_steps


class RailEnvWrapper:
  def __init__(self, env:RailEnv):
    self.env = env

    assert self.env is not None
    assert self.env.rail is not None, "Reset original environment first!"
    assert self.env.agents is not None, "Reset original environment first!"
    assert len(self.env.agents) > 0, "Reset original environment first!"

    # rail can be seen as part of the interface to RailEnv.
    # is used by several wrappers, to e.g. access rail.get_valid_transitions(...)
    #self.rail = self.env.rail
    # same for env.agents
    # MICHEL: DOES THIS HERE CAUSE A PROBLEM with agent status not being updated?
    #self.agents = self.env.agents
    #assert self.env.agents == self.agents
    #print(f"agents of RailEnvWrapper are: {self.agents}")
    #self.width = self.rail.width
    #self.height = self.rail.height


  # TODO: maybe do this in a generic way, like "for each method of self.env, ..."
  # maybe using dir(self.env) (gives list of names of members)

  # MICHEL: this seems to be needed after each env.reset(..) call
  # otherwise, these attribute names refer to the wrong object and are out of sync...
  # probably due to the reassignment of new objects to these variables by RailEnv, and how Python treats that.

  # simple example: a = [1,2,3] b=a. But then a=[0]. Now we still have b==[1,2,3].

  # it's better tou use properties here!

  # @property
  # def number_of_agents(self):
  #   return self.env.number_of_agents
  
  # @property
  # def agents(self):
  #   return self.env.agents

  # @property
  # def _seed(self):
  #   return self.env._seed

  # @property
  # def obs_builder(self):
  #   return self.env.obs_builder

  def __getattr__(self, name):
    try:
      return super().__getattr__(self,name)
    except:
      """Expose any other attributes of the underlying environment."""
      return getattr(self.env, name)


  @property
  def rail(self):
    return self.env.rail
  
  @property
  def width(self):
    return self.env.width
  
  @property
  def height(self):
    return self.env.height

  @property
  def agent_positions(self):
    return self.env.agent_positions

  def get_num_agents(self):
    return self.env.get_num_agents()

  def get_agent_handles(self):
    return self.env.get_agent_handles()

  def step(self, action_dict: Dict[int, RailEnvActions]):
    #self.agents = self.env.agents
    # ERROR. something is wrong with the references for self.agents...
    #assert self.env.agents == self.agents
    return self.env.step(action_dict)

  def reset(self, **kwargs):
    # MICHEL: I suspect that env.reset() does not simply change values of variables, but assigns new objects
    # that might cause some attributes not be properly updated here, because of how Python treats assignments differently from modification..
    #assert self.env.agents == self.agents
    obs, info = self.env.reset(**kwargs)
    #assert self.env.agents == self.agents, "after resetting internal env, self.agents names wrong object..."
    #self.reset_attributes()
    #print(f"calling RailEnvWrapper.reset()")
    #print(f"obs: {obs}, info:{info}")
    return obs, info


class ShortestPathActionWrapper(RailEnvWrapper):

    def __init__(self, env:RailEnv):
        super().__init__(env)
        #self.action_space = gym.spaces.Discrete(n=3)  # 0:stop, 1:shortest path, 2:other direction

    # MICHEL: we have to make sure that not agents with agent.status == DONE_REMOVED are in the action dict.
    # otherwise, possible_actions_sorted_by_distance(self.env, agent_id)[action - 1][0] will crash.
    def step(self, action_dict: Dict[int, RailEnvActions]) -> Tuple[Dict, Dict, Dict, Dict]:
      ########## MICHEL: NEW (just for debugging) ########
      for agent_id, action in action_dict.items():
        agent = self.agents[agent_id]
        # assert agent.status != RailAgentStatus.DONE_REMOVED # this comes with agent.position == None...
        # assert agent.status != RailAgentStatus.DONE # not sure about this one...
        print(f"agent: {agent} with status: {agent.status}")
      ######################################################

      # input: action dict with actions in [0, 1, 2].
      transformed_action_dict = {}
      for agent_id, action in action_dict.items():
          if action == 0:
              transformed_action_dict[agent_id] = action
          else:
              assert action in [1, 2]
              # MICHEL: how exactly do the indices work here?
              #transformed_action_dict[agent_id] = possible_actions_sorted_by_distance(self.rail_env, agent_id)[action - 1][0]
              #print(f"possible actions sorted by distance(...) is: {possible_actions_sorted_by_distance(self.env, agent_id)}")
              #assert agent.status != RailAgentStatus.DONE_REMOVED
              # MICHEL: THIS LINE CRASHES WITH A "NoneType is not subscriptable" error...
              assert possible_actions_sorted_by_distance(self.env, agent_id) is not None
              assert possible_actions_sorted_by_distance(self.env, agent_id)[action - 1] is not None
              transformed_action_dict[agent_id] = possible_actions_sorted_by_distance(self.env, agent_id)[action - 1][0]
      obs, rewards, dones, info = self.env.step(transformed_action_dict)
      return obs, rewards, dones, info

    #def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        #return self.rail_env.reset(random_seed)

    # MICHEL: should not be needed, as we inherit that from RailEnvWrapper...
    #def reset(self, **kwargs) -> Tuple[Dict, Dict]:
    #  obs, info = self.env.reset(**kwargs)
    #  return obs, info


def find_all_cells_where_agent_can_choose(env: RailEnv):
    """
    input: a RailEnv (or something which behaves similarly, e.g. a wrapped RailEnv),
    WHICH HAS BEEN RESET ALREADY!
    (o.w., we call env.rail, which is None before reset(), and crash.)
    """
    switches = []
    switches_neighbors = []
    directions = list(range(4))
    for h in range(env.height):
        for w in range(env.width):

            # MICHEL: THIS SEEMS TO BE A BUG. WRONG ODER OF COORDINATES.
            # will not show up in quadratic environments.
            # should be pos = (h, w)
            #pos = (w, h)

            # MICHEL: changed this
            pos = (h, w)

            is_switch = False
            # Check for switch: if there is more than one outgoing transition
            for orientation in directions:
                #print(f"env is: {env}")
                #print(f"env.rail is: {env.rail}")
                possible_transitions = env.rail.get_transitions(*pos, orientation)
                num_transitions = np.count_nonzero(possible_transitions)
                if num_transitions > 1:
                    switches.append(pos)
                    is_switch = True
                    break
            if is_switch:
                # Add all neighbouring rails, if pos is a switch
                for orientation in directions:
                    possible_transitions = env.rail.get_transitions(*pos, orientation)
                    for movement in directions:
                        if possible_transitions[movement]:
                            switches_neighbors.append(get_new_position(pos, movement))

    decision_cells = switches + switches_neighbors
    return tuple(map(set, (switches, switches_neighbors, decision_cells)))


class NoChoiceCellsSkipper:
    def __init__(self, env:RailEnv, accumulate_skipped_rewards: bool, discounting: float) -> None:
      self.env = env
      self.switches = None
      self.switches_neighbors = None
      self.decision_cells = None
      self.accumulate_skipped_rewards = accumulate_skipped_rewards
      self.discounting = discounting
      self.skipped_rewards = defaultdict(list)

      # env.reset() can change the rail grid layout, so the switches, etc. will change! --> need to do this in reset() as well.
      #self.switches, self.switches_neighbors, self.decision_cells = find_all_cells_where_agent_can_choose(self.env)

      # compute and initialize value for switches, switches_neighbors, and decision_cells.
      self.reset_cells()

    # MICHEL: maybe these three methods should be part of RailEnv?
    def on_decision_cell(self, agent: EnvAgent) -> bool:
        """
        print(f"agent {agent.handle} is on decision cell")
        if agent.position is None:
          print("because agent.position is None (has not been activated yet)")
        if agent.position == agent.initial_position:
          print("because agent is at initial position, activated but not departed")
        if agent.position in self.decision_cells:
          print("because agent.position is in self.decision_cells.")
        """
        return agent.position is None or agent.position == agent.initial_position or agent.position in self.decision_cells

    def on_switch(self, agent: EnvAgent) -> bool:
        return agent.position in self.switches

    def next_to_switch(self, agent: EnvAgent) -> bool:
        return agent.position in self.switches_neighbors

    # MICHEL: maybe just call this step()...
    def no_choice_skip_step(self, action_dict: Dict[int, RailEnvActions]) -> Tuple[Dict, Dict, Dict, Dict]:
        o, r, d, i = {}, {}, {}, {}
      
        # MICHEL: NEED TO INITIALIZE i["..."]
        # as we will access i["..."][agent_id]
        i["action_required"] = dict()
        i["malfunction"] = dict()
        i["speed"] = dict()
        i["status"] = dict()

        while len(o) == 0:
            #print(f"len(o)==0. stepping the rail environment...")
            obs, reward, done, info = self.env.step(action_dict)

            for agent_id, agent_obs in obs.items():

                ######  MICHEL: prints for debugging  ###########
                if not self.on_decision_cell(self.env.agents[agent_id]):
                      print(f"agent {agent_id} is NOT on a decision cell.")
                #################################################


                if done[agent_id] or self.on_decision_cell(self.env.agents[agent_id]):
                    ######  MICHEL: prints for debugging  ######################
                    if done[agent_id]:
                      print(f"agent {agent_id} is done.")
                    #if self.on_decision_cell(self.env.agents[agent_id]):
                      #print(f"agent {agent_id} is on decision cell.")
                      #cell = self.env.agents[agent_id].position
                      #print(f"cell is: {cell}")
                      #print(f"the decision cells are: {self.decision_cells}")
                    
                    ############################################################

                    o[agent_id] = agent_obs
                    r[agent_id] = reward[agent_id]
                    d[agent_id] = done[agent_id]

                    # MICHEL: HAVE TO MODIFY THIS HERE
                    # because we are not using StepOutputs, the return values of step() have a different structure.
                    #i[agent_id] = info[agent_id]
                    i["action_required"][agent_id] = info["action_required"][agent_id] 
                    i["malfunction"][agent_id] = info["malfunction"][agent_id]
                    i["speed"][agent_id] = info["speed"][agent_id]
                    i["status"][agent_id] = info["status"][agent_id]
                                                                  
                    if self.accumulate_skipped_rewards:
                        discounted_skipped_reward = r[agent_id]
                        for skipped_reward in reversed(self.skipped_rewards[agent_id]):
                            discounted_skipped_reward = self.discounting * discounted_skipped_reward + skipped_reward
                        r[agent_id] = discounted_skipped_reward
                        self.skipped_rewards[agent_id] = []

                elif self.accumulate_skipped_rewards:
                    self.skipped_rewards[agent_id].append(reward[agent_id])
                # end of for-loop

            d['__all__'] = done['__all__']
            action_dict = {}
            # end of while-loop

        return o, r, d, i

    # MICHEL: maybe just call this reset()...
    def reset_cells(self) -> None:
        self.switches, self.switches_neighbors, self.decision_cells = find_all_cells_where_agent_can_choose(self.env)


# IMPORTANT: rail env should be reset() / initialized before put into this one!
# IDEA: MAYBE EACH RAILENV INSTANCE SHOULD AUTOMATICALLY BE reset() / initialized upon creation!
class SkipNoChoiceCellsWrapper(RailEnvWrapper):
  
    # env can be a real RailEnv, or anything that shares the same interface
    # e.g. obs, rewards, dones, info = env.step(action_dict) and obs, info = env.reset(), and so on.
    def __init__(self, env:RailEnv, accumulate_skipped_rewards: bool, discounting: float) -> None:
        super().__init__(env)
        # save these so they can be inspected easier.
        self.accumulate_skipped_rewards = accumulate_skipped_rewards
        self.discounting = discounting
        self.skipper = NoChoiceCellsSkipper(env=self.env, accumulate_skipped_rewards=self.accumulate_skipped_rewards, discounting=self.discounting)

        self.skipper.reset_cells()

        # TODO: this is clunky..
        # for easier access / checking
        self.switches = self.skipper.switches
        self.switches_neighbors = self.skipper.switches_neighbors
        self.decision_cells = self.skipper.decision_cells
        self.skipped_rewards = self.skipper.skipped_rewards

  
    # MICHEL: trying to isolate the core part and put it into a separate method.
    def step(self, action_dict: Dict[int, RailEnvActions]) -> Tuple[Dict, Dict, Dict, Dict]:
        obs, rewards, dones, info = self.skipper.no_choice_skip_step(action_dict=action_dict)
        return obs, rewards, dones, info
        

    # MICHEL: TODO: maybe add parameters like regenerate_rail, regenerate_schedule, etc.
    # arguments from RailEnv.reset() are: self, regenerate_rail: bool = True, regenerate_schedule: bool = True, activate_agents: bool = False, random_seed: bool = None
    # TODO: check the type of random_seed. Is it bool or int?
    # MICHEL: changed return type from Dict[int, Any] to Tuple[Dict, Dict].
    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        obs, info = self.env.reset(**kwargs)
        # resets decision cells, switches, etc. These can change with an env.reset(...)!
        # needs to be done after env.reset().
        self.skipper.reset_cells()
        return obs, info