# Flatland 2.0 Introduction (Beta)

Curious to see whats coming in *Flat*land 2.0? Have a look at the current development and report bugs and give us feedback on the environment.

*WARNING*: Flatlnadn 2.0 Beta is under current development and not stable nor final. We would however like you to play with the code and help us get the best possible environment for multi-agent control problems.

## Whats new

In this version of *Flat*land we are moving closer to realistic and more complex railway problems. Earlier versions of *Flat*land which introduced you to the concept of restricted transitions was still to simplified to give us feasible solutions for daily operations. Thus the following changes are coming in the next version to be closer to real railway network challenges:

- *New Level Generator* with less connections between different nodes in the network and thus much higher agent densities on rails.
- *Stochastic Events* that cause agents to stop and get stuck for different number of time steps.
- *Different Speed Classes* allow agents to move at different speeds and thus enhance complexity in the search for optimal solutions.

Below we explain these changes in more detail and how you can play with their parametrization. We appreciate *your feedback* on the performance and the difficulty on these levels to help us shape the best possible *Flat*land 2.0 environment.

## Get the new level generators
Since this is currently still in *beta* phase you can only install this version of *Flat*land through the gitlab repository. Once you have downloaded the [Flatland Repository](https://gitlab.aicrowd.com/flatland/flatland) you have to switch to the [147_new_level_generator](https://gitlab.aicrowd.com/flatland/flatland/tree/147_new_level_generator) branch to be able access the latest changes in *Flat*land.

Once you have switched to this branch install *Flat*land by running `python setup.py install`.

## Generate levels

We are currently working on different new level generators and you can expect that the levels in the submission testing will not all come from just one but rather different level generators to be sure that the controllers can handle any railway specific challenge.

For this early *beta* testing we suggest you have a look at the `sparse_rail_generator` and `realistic_rail_generator`.

### Sparse Rail Generator
![Example_Sparse](https://i.imgur.com/DP8sIyx.png)

## Add Stochasticity

## Add different speed profiles