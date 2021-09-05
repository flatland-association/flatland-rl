
PettingZoo
==========

..

   PettingZoo (https://www.pettingzoo.ml/) is a collection of multi-agent environments for reinforcement learning. We build a pettingzoo interface for flatland.


Background
----------

PettingZoo is a popular multi-agent environment library (https://arxiv.org/abs/2009.14471) that aims to be the gym standard for Multi-Agent Reinforcement Learning. We list the below advantages that make it suitable for use with flatland


* Works with both rllib (https://docs.ray.io/en/latest/rllib.html) and stable baselines 3 (https://stable-baselines3.readthedocs.io/) using wrappers from Super Suit.
* Clean API (https://www.pettingzoo.ml/api) with additional facilities/api for parallel, saving observation, recording using gym monitor, processing, normalising observations
* Scikit-learn inspired api
  e.g.

.. code-block:: python

   act = model.predict(obs, deterministic=True)[0]


* Parallel learning using literally 2 lines of code to use with stable baselines 3

.. code-block:: python

   env = ss.pettingzoo_env_to_vec_env_v0(env)
   env = ss.concat_vec_envs_v0(env, 8, num_cpus=4, base_class=’stable_baselines3’)


* Tested and supports various multi-agent environments with many agents comparable to flatland. e.g. https://www.pettingzoo.ml/magent
* Clean interface means we can custom add an experimenting tool like wandb and have full flexibility to save information we want
