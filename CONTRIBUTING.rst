.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://gitlab.aicrowd.com/flatland/flatland/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the Repository Issue Tracker for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the Repository Issue Tracker for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

flatland could always use more documentation, whether as part of the
official flatland docs, in docstrings, or even on the web in blog posts,
articles, and such. A quick reference for writing good docstrings is available at : https://docs.python-guide.org/writing/documentation/#writing-docstrings

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://gitlab.aicrowd.com/flatland/flatland/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `flatland` for local development.

1. Fork the `flatland` repo on https://gitlab.aicrowd.com/flatland/flatland .
2. Clone your fork locally::

    $ git clone git@gitlab.aicrowd.com:flatland/flatland.git

3. Install the software dependencies via Anaconda-3 or Miniconda-3. (This assumes you have Anaconda installed by following the instructions `here <https://www.anaconda.com/distribution>`_)

    $ conda install -c conda-forge tox-conda
    $ conda install tox
    $ tox -v --recreate

    This will create a virtual env you can then use.

    These steps are performed if you run

    $ getting_started/getting_started.bat/.sh

    from Anaconda prompt.


4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::

    $ flake8 flatland tests examples benchmarks
    $ python setup.py test or py.test
    $ tox

   To get flake8 and tox, just pip install them into your virtualenv.

6. Commit your changes and push your branch to Gitlab::

    $ git add .
    $ git commit -m "Addresses #<issue-number> Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a merge request through the Gitlab repository website.

Merge Request Guidelines
-------------------------

Before you submit a merge request, check that it meets these guidelines:

1. The merge request should include tests.
2. The code must be formatted (PyCharm)
3. If the merge request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
4. The merge request should work for Python 3.6, 3.7 and for PyPy. Check
   https://gitlab.aicrowd.com/flatland/flatland/pipelines
   and make sure that the tests pass for all supported Python versions.
   We force pipelines to be run successfully for merge requests to be merged.
5. Although we cannot enforce it technically, we ask for merge requests to be reviewed by at least one core member
   in order to ensure that the Technical Guidelines below are respected and that the code is well tested:

5.1.  The remarks from the review should be resolved/implemented and communicated using the 'discussions resolved':

.. image:: images/DiscussionsResolved.png

5.2.  When a merge request is merged, source branches should be deleted and commits squashed:

.. image:: images/SourceBranchSquash.png

Tips
----

To run a subset of tests::

$ py.test tests.test_flatland


Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed .
Then run::

$ bumpversion patch # possible: major / minor / patch
$ git push
$ git push --tags

TODO: Travis will then deploy to PyPI if tests pass. (To be configured properly by Mohanty)


Local Evaluation
----------------

This document explains you how to locally evaluate your submissions before making
an official submission to the competition.

Requirements
~~~~~~~~~~~~

* **flatland-rl** : We expect that you have `flatland-rl` installed by following the instructions in  [README.md](README.md).

* **redis** : Additionally you will also need to have  `redis installed <https://redis.io/topics/quickstart>`_ and **should have it running in the background.**

Test Data
~~~~~~~~~

* **test env data** : You can `download and untar the test-env-data <https://www.aicrowd.com/challenges/flatland-challenge/dataset_files>`, at a location of your choice, lets say `/path/to/test-env-data/`. After untarring the folder, the folder structure should look something like:


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
~~~~~~~~~~~~~~~~~~

* **start evaluation service** : Then you can start the evaluator by running :

.. code-block:: console

    flatland-evaluator --tests /path/to/test-env-data/

RemoteClient
~~~~~~~~~~~~

* **run client** : Some `sample submission code can be found in the starter-kit <https://github.com/AIcrowd/flatland-challenge-starter-kit/>`_, but before you can run your code locally using `FlatlandRemoteClient`, you will have to set the `AICROWD_TESTS_FOLDER` environment variable to the location where you previous untarred the folder with `the test-env-data`:


.. code-block:: console

    export AICROWD_TESTS_FOLDER="/path/to/test-env-data/"

    # or on Windows :
    #
    # set AICROWD_TESTS_FOLDER "\path\to\test-env-data\"

    # and then finally run your code
    python run.py


Technical Guidelines
--------------------

Clean Code
~~~~~~~~~~
Please adhere to the general `Clean Code <https://www.planetgeek.ch/wp-content/uploads/2014/11/Clean-Code-V2.4.pdf>`_ principles,
for instance we write short and concise functions and use appropriate naming to ensure readability.

Naming Conventions
~~~~~~~~~~~~~~~~~~

We use the pylint naming conventions:

`module_name`, `package_name`, `ClassName`, `method_name`, `ExceptionName`, `function_name`, `GLOBAL_CONSTANT_NAME`, `global_var_name`, `instance_var_name`, `function_parameter_name`, `local_var_name`.


numpydoc
~~~~~~~~

Docstrings should be formatted using numpydoc_.


.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html


Acessing resources
~~~~~~~~~~~~~~~~~~

We use `importlib-resources <https://importlib-resources.readthedocs.io/en/latest/>`_ to read from local files.
    Sample usages:

    .. code-block:: python

        from importlib_resources import path

        with path(package, resource) as file_in:
            new_grid = np.load(file_in)

    And:

    .. code-block:: python

        from importlib_resources import read_binary

        load_data = read_binary(package, resource)
        self.set_full_state_msg(load_data)



    Renders the scene into a image (screenshot)

    .. code-block:: python

        renderer.gl.save_image("filename.bmp")

Type Hints
~~~~~~~~~~

We use Type Hints (`PEP 484 <https://www.python.org/dev/peps/pep-0484/>`_) for better readability and better IDE support.

    .. code-block:: python
        # This is how you declare the type of a variable type in Python 3.6
        age: int = 1

        # In Python 3.5 and earlier you can use a type comment instead
        # (equivalent to the previous definition)
        age = 1  # type: int

        # You don't need to initialize a variable to annotate it
        a: int  # Ok (no value at runtime until assigned)

        # The latter is useful in conditional branches
        child: bool
        if age < 18:
            child = True
        else:
            child = False

Have a look at the `Type Hints Cheat Sheet <https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html>`_ to get started with Type Hints.

Caveat: We discourage the usage of Type Aliases for structured data since its members remain unnamed (see `Issue #284 <https://gitlab.aicrowd.com/flatland/flatland/issues/284/>`_).

    .. code-block:: python
        # Discouraged: Type Alias with unnamed members
        Tuple[int, int]

        # Better: use NamedTuple
        from typing import NamedTuple

        Position = NamedTuple('Position',
            [
                ('r', int),
                ('c', int)
            ]



NamedTuple
~~~~~~~~~~
For structured data containers for which we do not write additional methods, we use
`NamedTuple` instead of plain `Dict` to ensure better readability by

    .. code-block:: python
        from typing import NamedTuple

        RailEnvNextAction = NamedTuple('RailEnvNextAction',
            [
                ('action', RailEnvActions),
                ('next_position', RailEnvGridPos),
                ('next_direction', Grid4TransitionsEnum)
            ])

Members of NamedTuple can then be accessed through `.<member>` instead of `['<key>']`.

If we have to ensure some (class) invariant over multiple members
(for instance, `o.A` always changes at the same time as `o.B`),
then we should uses classes instead, see the next section.

Class Attributes
~~~~~~~~~~~~~~~~

We use classes for data structures if we need to write methods that ensure (class) invariants over multiple members,
for instance, `o.A` always changes at the same time as `o.B`.
We use the attrs_ class decorator and a way to declaratively define the attributes on that class:

    .. code-block:: python
        @attrs
        class Replay(object):
            position = attrib(type=Tuple[int, int])

.. _attrs: https://github.com/python-attrs/attrs


Abstract Base Classes
~~~~~~~~~~~~~~~~~~~~~
We use the abc_ class decorator and a way to declaratively define the attributes on that class:

    .. code-block:: python
        # abc_base.py

        import abc


        class PluginBase(metaclass=abc.ABCMeta):

            @abc.abstractmethod
            def load(self, input):
                """Retrieve data from the input source
                and return an object.
                """

            @abc.abstractmethod
            def save(self, output, data):
                """Save the data object to the output."""




And then

    .. code-block:: python

        # abc_subclass.py

        import abc
        from abc_base import PluginBase


        class SubclassImplementation(PluginBase):

            def load(self, input):
                return input.read()

            def save(self, output, data):
                return output.write(data)


        if __name__ == '__main__':
            print('Subclass:', issubclass(SubclassImplementation,
                                          PluginBase))
            print('Instance:', isinstance(SubclassImplementation(),
                                          PluginBase))

.. _abc: https://pymotw.com/3/abc/



Currying
~~~~~~~~
We discourage currying to encapsulate state since we often want the stateful object to have multiple methods
(but the curried function has only its signature and abusing params to switch behaviour is not very readable).

Thus, we should refactor our generators and use classes instead (see `Issue #283 <https://gitlab.aicrowd.com/flatland/flatland/issues/283>`_).

    .. code-block:: python
        # Type Alias
        RailGeneratorProduct = Tuple[GridTransitionMap, Optional[Dict]]
        RailGenerator = Callable[[int, int, int, int], RailGeneratorProduct]

        # Currying: a function that returns a confectioned function with internal state
        def complex_rail_generator(nr_start_goal=1,
                                   nr_extra=100,
                                   min_dist=20,
                                   max_dist=99999,
                                   seed=1) -> RailGenerator:


