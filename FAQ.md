# Frequently Asked Questions (FAQs)

## Questions about the Flatland Challenge:
These are the most common questions regarding the [Flatland Challenge](https://www.aicrowd.com/challenges/flatland-challenge).
If your questions are not answered please check the [Forum](https://discourse.aicrowd.com/c/flatland-challenge?_ga=2.33753761.1627822449.1571622829-1432296534.1549103074) and post your question there.

### How can I win prizes in this challenge?
You can win prizes in different categories.

Best Solution Prize: Won by the participants with the best performing submission on our test set. Only your rankings from the Round 1 and Round 2 are taken into account. Check the leader board on this site regularly for the latest information on your ranking.

The top three submissions in this category will be awarded the following cash prizes (in Swiss Francs):

CHF 7’500.- for first prize

CHF 5’000.- for second prize

CHF 2’500.- for third prize

Community Contributions Prize: Awarded to the person/group who makes the biggest contribution to the community - done through generating new observations and sharing them with the community.

The top submission in this category will be awarded the following cash prize (in Swiss Francs): CHF 5’000.-

In addition, we will hand-pick and award up to five (5) travel grants to the Applied Machine Learning Days 2019 in Lausanne, Switzerland. Participants with promising solutions may be invited to present their solutions at SBB in Bern, Switzerland.

To check your eligibility please read the prizes section in the [rules](https://www.aicrowd.com/challenges/flatland-challenge/challenge_rules/68).

### What are the deadlines for the flatland challenge?
- The beta round starts on the 1st of July 2019 and ends on the 30th of July 2019
- Round 1 closed on Sunday, 13th of October 2019, 12 PM. UTC +1
- Round 2 closes on Sunday, 5th of January 2020, 12 PM. UTC +1

### How is the score of a submission computed?
The scores of your submission are computed as follows:

1. Mean number of agents done, in other words how many agents reached their target in time.
2. Mean reward is just the mean of the cummulated reward.
3. If multiple participants have the same number of done agents we compute a "nomralized" reward as follows:
```
normalized_reward =cumulative_reward / (self.env._max_episode_steps +self.env.get_num_agents()
```
The mean number of agents done is the primary score value, only when it is tied to we use the "normalized" reward to determine the position on the leaderboard.

### How do I submit to the Flatland Challenge?
Follow the instructions in the [starter kit](https://github.com/AIcrowd/flatland-challenge-starter-kit) to get your first submission.

### Can I use env variables with my controller?
Yes you can. You can access all environment variables as you please. We recommend you use a custom observation builder to do so as explained [here](http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/03_tutorials.html#custom-observations-and-custom-predictors-tutorial).

### What are the time limits for my submission?
If there is no action on the server for 10 minutes the submission will be cancelled and a time-out error wil be produced.

If the submissions in total takes longer than 8 hours a time-out will occur.

### What are the parameters for the environments for the submission scoring?
The environments vary in size and number of agents as well as malfunction parameters. The upper limit of these variables for submissions are:
- `(x_dim, y_dim) <= (150, 150)`
- `n_agents <= 250` (this might be updated)
- `malfunction rates` this is currently being refactored

## Questions about the Flatland Repository:
This section provides you with information about the most common questions around the Flatland repository. If your question is still not answered either reach out to the contacts listed on the repository directly or open an issue by following these [guidlines](http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/06_contributing.html).
### How can I get started with Flatland?
Install Flatland by running `pip install -U flatland-rl` or directly from source by cloning the flatland repository and running `python setup.py --install` in the repository directory.

These [Tutorials](http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/03_tutorials.html) help you get a basic understanding of the flatland environment.
### How do I train agents on Flatland?
Once you have installed Flatland, head over to the [baselines repository](https://gitlab.aicrowd.com/flatland/baselines) to see how you can train your own reinforcement learning agent on Flatland.

Check out this [tutorial](https://gitlab.aicrowd.com/flatland/baselines/blob/master/torch_training/Getting_Started_Training.md?_ga=2.193077805.1627822449.1571622829-1432296534.1549103074) to get a sense of how it works.

### What is a observation builder and which should I use?
Observation builders give you the possibility to generate custom observations for your controller (reinfocement learning agent, optimization algorithm,...). The observation builder has access to all environment data and can perform any operations on them as long as they are not changed.
This [tutorial](http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/03_tutorials.html#custom-observations-and-custom-predictors-tutorial) will give you a sense on how to use them.
### What is a predictor and which one should I use?
Because railway traffic is limited to rails, many decisions that you have to take need to consider future situations and detect upcoming conflicts ahead of time. Therefore, flatland provides the possibility of predictors that predict where agents will be in the future. We provide a stock predictor that assumes each agent just travels along its shortest path.
You can build more elaborate predictors and use them as part of your observation builder. You find more information [here](http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/03_tutorials.html#custom-observations-and-custom-predictors-tutorial).
### What information is available about each agent?
Each agent is an object and contains the following information:

- `initial_position = attrib(type=Tuple[int, int])`: The initial position of an agent. This is where the agent will enter the environment. It is the start of the agent journey.
- `position = attrib(default=None, type=Optional[Tuple[int, int]])`: This is the actual position of the agent. It is updated every step of the environment. Before the agent has entered the environment and after it leaves the environment it is set to `None`
- `direction = attrib(type=Grid4TransitionsEnum)`: This is the direction an agent is facing. The values for directions are `North:0`, `East:1`, `South:2` and `West:3`.
- `target = attrib(type=Tuple[int, int])`: This is the target position the agent has to find and reach. Once the agent reaches this position its taks is done.
- `moving = attrib(default=False, type=bool)`: Because agents can have malfunctions or be stopped because their path is blocked we store the current state of an agent. If `agent.moving == True` the agent is currently advancing. If it is `False` the agent is either blocked or broken.
- `speed_data = attrib(default=Factory(lambda: dict({'position_fraction': 0.0, 'speed': 1.0, 'transition_action_on_cellexit': 0})))`: This contains all the relevant information about the speed of an agent:
    - The attribute `'position_fraction'` indicates how far the agent has advanced within the cell. As soon as this value becomes larger than `1` the agent advances to the next cell as defined by `'transition_action_on_cellexit'`.
    - The attribute `'speed''` defines the travel speed of an agent. It can be any fraction smaller than 1.
    - The attribute `'transition_action_on_cellexit'` contains the information about the action that will be performed at the exit of the cell. Due to speeds smaller than 1. agents have to take several steps within a cell. We however only allow an action to be chosen at cell entry.
- `malfunction_data = attrib(default=Factory(lambda: dict({'malfunction': 0, 'malfunction_rate': 0, 'next_malfunction': 0, 'nr_malfunctions': 0,'moving_before_malfunction': False})))`: Contains all information relevant for agent malfunctions:
    - The attribute `'malfunction` indicates if the agent is currently broken. If the value is larger than `0` the agent is broken. The integer value represents the number of `env.step()` calls the agent will still be broken.
    - The attribute `'next_malfunction'` will be REMOVED as it serves no purpose anymore, malfunctions are now generated by a poisson process.
    - The attribute `'nr_malfunctions'` is a counter that keeps track of the number of malfunctions a specific agent has had.
    - The attribute `'moving_before_malfunction'` is an internal parameter used to restart agents that were moving automatically after the malfunction is fixed.
- `status = attrib(default=RailAgentStatus.READY_TO_DEPART, type=RailAgentStatus)`: The status of the agent explains what the agent is currently doing. It can be in either one of these states:
    - `READY_TO_DEPART` not in grid yet (position is None) 
    - `ACTIVE` in grid (position is not None), not done
    - `DONE` in grid (position is not None), but done
    - `DONE_REMOVED` removed from grid (position is None)

### Can I use my own reward function?
Yes you can do reward shaping as you please. All information can be accessed directly in the env.
### What are rail and schedule generators?
To generate environments for Flatland you need to provide a railway infrastructure (rail) and a set of tasks for each agent to complete (schedule).
### What is the max number of timesteps per episode?
The maximum number of timesteps is `max_time_steps = 4 * 2 * (env.width + env.height + 20)`
### What are malfunctions and what can i do to resolve them?
Malfunctions occur according to a Poisson process. The hinder an agent from performing its actions and update its position. While an agent is malfunctioning it is blocking the paths for other agents. There is nothing you can do to fix an agent, it will get fixed automatically as soon as `agent.malfunction_data['malfunction'] == 0` .
You can however adjust the other agent actions to avoid delay propagation within the railway network and keeping traffic as smooth as possible.

### Can agents communication with each other?
There is no communitcation layer built into Flatland directly. You can however build a communication layer outside of the Flatland environment if necessary.

## Questions about bugs
### Why are my trains drawn outside of the rails?
If you render your environment and the agents appear to be off the rail it is usually due to changes in the railway infrastructure. Make sure that you reset your renderer anytime the infrastructure changes by calling `env_renderer.reset().
`
