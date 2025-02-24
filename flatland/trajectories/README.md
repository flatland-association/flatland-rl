Data Model
==========

TODO move to flatland-book?

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
