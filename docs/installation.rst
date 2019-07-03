.. highlight:: shell

============
Installation
============

Software Runtime & Dependencies
-------------------------------

This is the recommended way of installation and running flatland's dependencies.

* Install `Anaconda <https://www.anaconda.com/distribution/>`_ by following the instructions `here <https://www.anaconda.com/distribution/>`_
* Create a new conda environment 

.. code-block:: console

    $ conda create python=3.6 --name flatland-rl
    $ conda activate flatland-rl

* Install the necessary dependencies

.. code-block:: console

    $ conda install -c conda-forge cairosvg pycairo
    $ conda install -c anaconda tk  


Stable release
--------------

To install flatland, run this command in your terminal:

.. code-block:: console

    $ pip install flatland-rl

This is the preferred method to install flatland, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for flatland can be downloaded from the `Gitlab repo`_.

You can clone the public repository:

.. code-block:: console

    $ git clone git@gitlab.aicrowd.com:flatland/flatland.git

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Gitlab repo: https://gitlab.aicrowd.com/flatland/flatland


Jupyter Canvas Widget
---------------------
If you work with jupyter notebook you need to install the Jupyer Canvas Widget. To install the Jupyter Canvas Widget read also
https://github.com/Who8MyLunch/Jupyter_Canvas_Widget#installation
