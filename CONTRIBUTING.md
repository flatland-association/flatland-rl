# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs using our [issue tracker](https://github.com/flatland-association/flatland-rl/issues).

If you are reporting a bug, please make sure to fill out the appropriate issue template.

### Fix Bugs

Look through the Repository Issue Tracker for bugs. Anything tagged with "bug" and "help wanted" is open to whoever
wants to implement it.

### Implement Features

Look through the Repository Issue Tracker for features. Anything tagged with "enhancement" and "help wanted" is open to
whoever wants to implement it.

### Write Documentation

flatland could always use more documentation, whether as part of the
official flatland docs, in docstrings, or even on the web in blog posts,
articles, and such. A quick reference for writing good docstrings is available
at [writing-docstrings](https://docs.python-guide.org/writing/documentation/#writing-docstrings).

### Submit Feedback

The best way to send feedback is to [file an issue](https://github.com/flatland-association/flatland-rl/issues).

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `flatland` for local development.

1. Clone the `flatland` repo:

    ```shell
    git clone git@github.com:flatland-association/flatland-rl.git
    ```

2. Setup a virtual environment using your preferred method (e.g. venv) and activate it. Make sure python 3.7, 3.8 and
   3.9 interpreters are available. Note that if you are using an Apple Macbook with an M1 or M2 processor, you need to
   use python 3.8 or 3.9.

3. Install the software dependencies using pip:

    ```shell
    pip install -r requirements_dev.txt
    ```

4. Create a branch for local development:

    ```shell
    git checkout -b name-of-your-bugfix-or-feature
    ```

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the tests. Use tox to run them as it will automatically
   test on all supported python versions:

    ```shell
    tox
    ```

6. Commit your changes and push your branch to Github:

    ```shell
    git add .
    git commit -m "Addresses #<issue-number> Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature
    ```

7. Open a pull request on Github targeting the `main` branch.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. The code must be formatted (using an IDE like PyCharm can do this for you).
3. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a
   docstring, and add the feature to the list in README.rst.
4. The pull request should work for Python 3.7, 3.8, 3.9. We force pipelines to be run successfully for pull requests to
   be merged.
5. Pull requests must be approved by at least one member of the core team. This is to ensure that the Technical
   Guidelines below are respected and that the code is well tested.

## Technical Guidelines

### Clean Code

Please adhere to the general [Clean Code](https://www.planetgeek.ch/wp-content/uploads/2014/11/Clean-Code-V2.4.pdf)
principles, for instance, we write short and concise functions and use appropriate naming to ensure readability.

### Naming Conventions

We use the pylint naming conventions:

- `module_name`
- `package_name`
- `ClassName`
- `method_name`
- `ExceptionName`
- `function_name`
- `GLOBAL_CONSTANT_NAME`
- `global_var_name`
- `instance_var_name`
- `function_parameter_name`
- `local_var_name`

### numpydoc

Docstrings should be formatted using [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html).

### Accessing resources

We use [importlib-resources](https://importlib-resources.readthedocs.io/en/latest/) to read from local files.

Sample usages:

```python
from importlib_resources import path

with path(package, resource) as file_in:
    new_grid = np.load(file_in)
```

and:

```python
from importlib_resources import read_binary

load_data = read_binary(package, resource)
self.set_full_state_msg(load_data)
```

Renders the scene into a image (screenshot):

```python
renderer.gl.save_image("filename.bmp")
```

## Type Hints

We use Type Hints ([PEP 484](https://www.python.org/dev/peps/pep-0484/)) for better readability and better IDE support:

```python
# This is how you declare the type of a variable type in Python 3.6
age: int = 1

# You don't need to initialize a variable to annotate it
a: int  # Ok (no value at runtime until assigned)

# The latter is useful in conditional branches
child: bool
if age < 18:
    child = True
else:
    child = False
```

To get started with Type Hints, you can have a look at
the [Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html).

Caveat: We discourage the usage of Type Aliases for structured data since its members remain unnamed. Instead, consider
using NamedTuple:

```python
# Discouraged: Type Alias with unnamed members
Tuple[int, int]

# Better: use NamedTuple
from typing import NamedTuple

Position = NamedTuple('Position',
                      [
                          ('r', int),
                          ('c', int)
                      ])
```

## NamedTuple

For structured data containers for which we do not write additional methods, we use `NamedTuple` instead of plain `Dict`
to ensure better readability:

```python
from typing import NamedTuple

RailEnvNextAction = NamedTuple('RailEnvNextAction',
                               [
                                   ('action', RailEnvActions),
                                   ('next_position', RailEnvGridPos),
                                   ('next_direction', Grid4TransitionsEnum)
                               ])
```

Members of NamedTuple can then be accessed through `.<member>` instead of `['<key>']`.

## Class Attributes

We use classes for data structures if we need to write methods that ensure (class) invariants over multiple members, for
instance, `o.A` always changes at the same time as `o.B`. We use the [attrs](https://github.com/python-attrs/attrs)
class decorator and a way to declaratively define the attributes on that class:

```python
@attrs
class Replay(object):
    position = attrib(type=Tuple[int, int])
```

## Abstract Base Classes

We use the [abc](https://pymotw.com/3/abc/) class decorator and a way to declaratively define the attributes on that
class:

```python
# abc_base.py

import abc


class PluginBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def load(self, input):
        """Retrieve data from the input source and return an object."""

    @abc.abstractmethod
    def save(self, output, data):
        """Save the data object to the output."""
```

And then:

```python
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
```

## Currying

We discourage currying to encapsulate state since we often want the stateful object to have multiple methods (but the
curried function has only its signature, and abusing params to switch behavior is not very readable). Thus, we should
refactor our generators and use classes instead.

```python
# Type Alias
RailGeneratorProduct = Tuple[GridTransitionMap, Optional[Dict]]
RailGenerator = Callable[[int, int, int, int], RailGeneratorProduct]


# Currying: a function that returns a confectioned function with internal state
def complex_rail_generator(nr_start_goal=1,
                           nr_extra=100,
                           min_dist=20,
                           max_dist=99999,
                           seed=1) -> RailGenerator:
```
