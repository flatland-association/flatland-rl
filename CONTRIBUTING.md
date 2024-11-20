# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

## Types of Contributions

### Report Bugs

Report bugs using our [issue tracker](https://github.com/flatland-association/flatland-rl/issues).

If you are reporting a bug, please make sure to fill out the appropriate issue template.

### Fix Bugs

Look through the repository issue tracker for bugs. Anything tagged with "bug" and "help wanted" is open to whoever
wants to implement it.

### Implement Features

Look through the repository issue tracker for features. Anything tagged with "enhancement" and "help wanted" is open to
whoever wants to implement it.

### Write Documentation

flatland could always use more documentation, whether as part of the official flatland docs, in docstrings, or even on
the web in blog posts, articles, and such. A quick reference for writing good docstrings is available
at [writing-docstrings](https://docs.python-guide.org/writing/documentation/#writing-docstrings).

### Submit Feedback

The best way to send feedback is to [file an issue](https://github.com/flatland-association/flatland-rl/issues).

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `flatland` for local development.

1. Make sure all supported Python interpreters (3.10, 3.11, 3.12) are available.
   This is important because you want to be able to run the test with all supported versions.
   We recommend [pyenv](https://github.com/pyenv/pyenv) to manage Python versions.
   See their docs for installation instructions.

2. Open up a terminal, clone the `flatland` repository and open it:

    ```shell
    git clone git@github.com:flatland-association/flatland-rl.git
    cd flatland-rl
    ```
3. When using `pyenv`, set the local Python versions:

   ```shell
   pyenv local 3.10 3.11 3.12
   ```

4. Set up a virtual environment using your preferred method (we suggest the built-in `venv`) and activate it.
   You can use your IDE to do this or by using the command line:

    ```shell
    python -m venv .venv
    source .venv/bin/activate
    ```

5. Install dependencies required for development using pip:

    ```shell
    python -m pip install -r requirements-dev.txt
    python -m pip install -r requirements-ml.txt
    ```

6. Create a branch for local development:

    ```shell
    git checkout -b name-of-your-bugfix-or-feature
    ```

7. Make all the changes you want to make!

8. When you're done making changes, check that your changes pass the tests.
   Use `tox` to run them as it will automatically test on all supported Python versions:

    ```shell
    tox
    ```
9. If you used a Python package not yet used before in the project, add it to the `dependencies` section of
   the `pyproject.toml` file.
   Make sure to re-generate the requirement files:

    ```shell
    tox -e requirements
    ```

10. Whenever you feel like you completed an iteration of your changes, commit and push them to GitHub:

    ```shell
    git add .
    git commit
    # Your favorite editor opens, allowing you to enter a message that describes your changes. The first line is the
    # subject line. Use sentence capitalisation (but don't end with a period) and limit it to 50 characters. It's good
    # practice to use imperative mood, e.g. "Add new feature that does X". If you need more space to describe your
    # changes (focus on the what and why, less on the how), add an empty line and then continue with the body. Try to
    # limit every line in body to 72 characters.
    git push --set-upstream origin name-of-your-bugfix-or-feature
    # After the first time, a simple `git push` will do the trick.
    ```

11. Open a pull request on GitHub targeting the `main` branch.
    Make sure to fill out the template.
    A review from a core team member is automatically requested.
    At least one approval is required to merge.
12. Once successfully reviewed, squash-merge the PR.
    This collapses all the commits into one and merges it into the `main` branch.
    Please adjust the subject line and body of the commit to accurately reflect your changes.

## Technical Guidelines

### Clean Code

Please adhere to the general [Clean Code](https://www.planetgeek.ch/wp-content/uploads/2014/11/Clean-Code-V2.4.pdf)
principles. For instance, we write short and concise functions and use appropriate naming to ensure readability.

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

### Accessing Resources

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

### Type Hints

We use type hints ([PEP 484](https://www.python.org/dev/peps/pep-0484/)) for better readability and better IDE support:

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

To get started with type hints, you can have a look at
the [Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html).

Caveat: We discourage the usage of type aliases for structured data since its members remain unnamed. Instead, consider
using `NamedTuple`:

```python
# Discouraged: Type Alias with unnamed members
Tuple[int, int]

# Better: use NamedTuple
from typing import NamedTuple

Position = NamedTuple('Position', [
    ('r', int),
    ('c', int)
])
```

### NamedTuple

For structured data containers for which we do not write additional methods, we use `NamedTuple` instead of plain `Dict`
to ensure better readability:

```python
from typing import NamedTuple

RailEnvNextAction = NamedTuple('RailEnvNextAction', [
    ('action', RailEnvActions),
    ('next_position', RailEnvGridPos),
    ('next_direction', Grid4TransitionsEnum)
])
```

Members of NamedTuple can then be accessed through `.<member>` instead of `['<key>']`.

### Class Attributes

We use classes for data structures if we need to write methods that ensure (class) invariants over multiple members, for
instance, `o.A` always changes at the same time as `o.B`. We use the [attrs](https://github.com/python-attrs/attrs)
class decorator and a way to declaratively define the attributes on that class:

```python
@attrs
class Replay(object):
    position = attrib(type=Tuple[int, int])
```

### Abstract Base Classes

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

### Currying

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

## Publishing

To publish a new version of the package:

1. Pick an appropriate version number (this project follows semantic versioning, hence chose wisely) and
   update `CHANGELOG.md` accordingly.
2. Create a branch using the naming convention `release/<version-number>`, e.g. `release/4.0.0`.
3. Commit your changes to `CHANGELOG.md`.
4. Push the branch and open a PR targeting `main` and get it approved by a core team member.
5. Merge the PR, a GitHub action will pick it up and release the new version.

