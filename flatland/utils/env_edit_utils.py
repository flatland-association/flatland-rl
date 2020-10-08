

import flatland.envs.schedule_generators as sg
import flatland.envs.rail_generators as rg
import flatland.envs.observations as obs
from flatland.utils import editor
from flatland.envs.rail_env import RailEnv


# Start and end all agents at the same place
class SchedGen2(sg.BaseSchedGen):
    def __init__(self, rcStart, rcEnd, iDir):
        self.rcStart = rcStart
        self.rcEnd = rcEnd
        self.iDir = iDir
        
    def generate(self, rail, num_agents, hints=None, num_resets=None, np_random=None) -> sg.Schedule:
        return sg.Schedule(agent_positions = [self.rcStart] * num_agents, 
                           agent_directions= [self.iDir] * num_agents,
                           agent_targets = [self.rcEnd] * num_agents,
                           agent_speeds = [1.0] * num_agents,
                           agent_malfunction_rates = None,
                           max_episode_steps=100)



# cycle through lists of start, end and direction
class SchedGen3(sg.BaseSchedGen):
    def __init__(self, lrcStarts, lrcTargs, liDirs):
        self.lrcStarts = lrcStarts
        self.lrcTargs = lrcTargs
        self.liDirs = liDirs
        
    def generate(self, rail, num_agents, hints=None, num_resets=None, np_random=None) -> sg.Schedule:
        return sg.Schedule(agent_positions = [ self.lrcStarts[i % len(self.lrcStarts)] for i in range(num_agents) ],
                           agent_directions= [ self.liDirs[i % len(self.liDirs)] for i in range(num_agents) ],
                           agent_targets = [ self.lrcTargs[i % len(self.lrcTargs)] for i in range(num_agents) ],
                           agent_speeds = [1.0] * num_agents,
                           agent_malfunction_rates = None,
                           max_episode_steps=100)


def makeEnv(nAg=2, width=20, height=10, oSG=None):
    env = RailEnv(width=width, height=height, rail_generator=rg.empty_rail_generator(),
                number_of_agents=nAg,
                schedule_generator=oSG,
                obs_builder_object=obs.TreeObsForRailEnv(max_depth=1))

    envModel = editor.EditorModel(env)
    env.reset()
    return env, envModel


def makeEnv2(nAg=2, shape=(20,10), llrcPaths=[], lrcStarts=[], lrcTargs=[], liDirs=[], bUCF=True):
    oSG = SchedGen3(lrcStarts, lrcTargs, liDirs)

    env = RailEnv(width=shape[0], height=shape[1], 
                rail_generator=rg.empty_rail_generator(),
                number_of_agents=nAg,
                schedule_generator=oSG,
                obs_builder_object=obs.TreeObsForRailEnv(max_depth=1),
                close_following=bUCF,
                record_steps=True)

    envModel = editor.EditorModel(env)
    env.reset()

    for lrcPath in llrcPaths:
        envModel.mod_rail_cell_seq(envModel.interpolate_path(lrcPath))

    return env, envModel


ddEnvSpecs = {
        # opposing stations with single alternative path
        "single_alternative":{
            "llrcPaths":  [
                [(1,0), (1,15)],  # across the top
                [(1,4), (1,6), (3,6), (3, 12), (1,12), (1,14)], # alternative loop below
                ],
            "lrcStarts": [ (1,3), (1,14) ],
            "lrcTargs" : [(1,14), (1,3)],
            "liDirs" : [1,3]
            },

        # single spur so one agent needs to wait
        "single_spur": {
            "llrcPaths" : [
                [(1,0), (1,15)],
                [(4,0), (4,6), (1,6), (1, 8)]],
            "lrcStarts": [(1,3), (1,14) ],
            "lrcTargs" : [(1,14), (4,2)],
            "liDirs" : [1,3]
            },
        
        # single spur so one agent needs to wait
        "merging_spurs": {
            "llrcPaths" : [
                [(1,0), (1,15), (7, 15), (7,0)],
                [(4,0), (4,6), (1,6), (1, 8)],
                #[((1,14), (1,16), (7,16), )]
                ],
            "lrcStarts": [(1,2), (4,2) ],
            "lrcTargs" : [(7,3)],
            "liDirs" : [1]
            },

        # Concentric Loops
        "concentric_loops": {
            "llrcPaths": [
                [(1,1), (1,5), (8, 5), (8,1), (1,1), (1,3)],
                [(1,3), (1,10), (8,10), (8,3)]
                ],
            
            "lrcStarts": [(1,3)],
            "lrcTargs": [(2,1)],
            "liDirs":  [1]
            },

        # two loops
        "loop_with_loops": {
            "llrcPaths": [
                # big outer loop Row 1, 8; Col 1, 15
                [(1,1), (1,15), (8, 15), (8,1), (1,1), (1,3)],
                # alternative 1
                [(1,3), (1,5), (3,5), (3,10), (1, 10), (1, 12)],
                # alternative 2
                [(8,3), (8,5), (6,5), (6,10), (8, 10), (8, 12)],
                
                ],
            
            # list of row,col of agent start cells
            "lrcStarts": [(1,3), (8, 3)],
            # list of row,col of targets
            "lrcTargs": [(8,2), (1,2)],
            # list of initial directions
            "liDirs":  [1, 1], 
            }

        }
    

def makeTestEnv(sName="single_alternative", nAg=2, bUCF=True):
    global ddEnvSpecs
    
    dSpec = ddEnvSpecs[sName]

    return makeEnv2(nAg=nAg, bUCF=bUCF, **dSpec)

def getAgentState(env):
    dAgState={}
    for iAg, ag in enumerate(env.agents):
        dAgState[iAg] = (*ag.position, ag.direction)
    return dAgState