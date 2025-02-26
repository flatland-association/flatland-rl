Trajectories
============


TODO move to `flatland-book`?


Data Model
----------

```mermaid
classDiagram
    class Trajectory
    Trajectory: +Path data_dir
    Trajectory: +UUID ep_id
    Trajectory: +run(int from_step, int to_step=-1)
    Trajectory: +verify()
    Trajectory: +from_submission(Policy policy, ObservationBuilder obs_builder, int snapshot_interval)$ Trajectory

    class EnvSnapshot
    EnvSnapshot: +Path data_dir
    Trajectory: +UUID ep_id

    class EnvConfiguration
    EnvConfiguration: +int max_episode_steps
    EnvConfiguration: +int height
    EnvConfiguration: +int width
    EnvConfiguration: +Rewards reward_function
    EnvConfiguration: +MalGen
    EnvConfiguration: +RailGen etc. reset

    class EnvState
    EnvState: +Grid rail

    namespace RailEnv {
        class EnvConfiguration

        class EnvState
    }

    class EnvActions

    class EnvRewards

    EnvSnapshot --> "1" EnvConfiguration
    EnvSnapshot --> "1" EnvState
    Trajectory --> "1" EnvConfiguration
    Trajectory --> "1..*" EnvState
    Trajectory --> "1..*" EnvActions
    Trajectory --> "1..*" EnvRewards

    class Policy
    Policy: act(int handle, Observation observation)

    class ObservationBuilder
    ObservationBuilder: get()
    ObservationBuilder: get_many()

    class Submission
    Submission --> "1" Policy
    Submission --> ObservationBuilder
```

Remarks:

* Trajectory needs not start at step 0
* Trajectory needs not contain state for every step - however, when starting the trajectory from an intermediate step, the snapshot must exist.

Flow Trajectory Run
-------------------

```mermaid
flowchart TD
    subgraph Trajectory.run
        start((start)) -->|data_dir| D0
        D0(RailEnvPersister.load_new) -->|env| E{env done?}
        E -->|no:\nobservations| G{Agent loop:\n more agents?}
        G --->|observation| G1(policy.act)
        G1 -->|action| G
        G -->|no:\n actions| F3(env.step)
        F3 -->|observations,rewards,info| E
        E -->|yes:\n rewards| H(((end)))
    end

    style Policy fill: #ffe, stroke: #333, stroke-width: 1px
    style G1 fill: #ffe, stroke: #333, stroke-width: 1px
    style Env fill: #fcc, stroke: #333, stroke-width: 1px, color: #300
    style F3 fill: #fcc, stroke: #333, stroke-width: 1px, color: #300
    subgraph legend
        Env(Environment)
        Policy(Policy)
        Trajectory(Trajectory)
    end

```
