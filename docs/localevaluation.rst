=====
Local Evaluation
=====

This document explains you how to locally evaluate your submissions before making 
an official submission to the competition.

Requirements
--------------

* **flatland-rl** : We expect that you have `flatland-rl` installed by following the instructions in  :doc:`installation`.

* **redis** : Additionally you will also need to have  `redis installed <https://redis.io/topics/quickstart>`_ and **should have it running in the background.**

Test Data
--------------

* **test env data** : You can `download and untar the test-env-data <https://www.aicrowd.com/challenges/flatland-challenge/dataset_files>`_, 
at a location of your choice, lets say `/path/to/test-env-data/`. After untarring the folder, the folder structure should look something like : 


.. code-block:: console

    .
    └── test-env-data
        ├── Test_0
        │   ├── Level_0.pkl
        │   └── Level_1.pkl
        ├── Test_1
        │   ├── Level_0.pkl
        │   └── Level_1.pkl
        ├..................
        ├..................
        ├── Test_8
        │   ├── Level_0.pkl
        │   └── Level_1.pkl
        └── Test_9
            ├── Level_0.pkl
            └── Level_1.pkl

Evaluation Service
------------------

* **start evaluation service** : Then you can start the evaluator by running : 

.. code-block:: console

    flatland-evaluator --tests /path/to/test-env-data/

RemoteClient
------------------

* **run client** : Some `sample submission code can be found in the starter-kit <https://github.com/AIcrowd/flatland-challenge-starter-kit/>`_, 
but before you can run your code locally using `FlatlandRemoteClient`, you will have to set the `AICROWD_TESTS_FOLDER` environment variable to the location where you 
previous untarred the folder with `the test-env-data`:

.. code-block:: console

    export AICROWD_TESTS_FOLDER="/path/to/test-env-data/"

    # or on Windows :
    # 
    # set AICROWD_TESTS_FOLDER "\path\to\test-env-data\"

    # and then finally run your code
    python run.py
