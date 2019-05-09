========
Flatland
========



.. image:: https://gitlab.aicrowd.com/flatland/flatland/badges/master/pipeline.svg
     :target: https://gitlab.aicrowd.com/flatland/flatland/pipelines
     :alt: Test Running
.. image:: https://gitlab.aicrowd.com/flatland/flatland/badges/master/coverage.svg
     :target: https://gitlab.aicrowd.com/flatland/flatland/pipelines
     :alt: Test Coverage



Multi Agent Reinforcement Learning on Trains

Getting Started
===============

Online Docs
------------

The documentation for the latest code on the master branch is found at  `http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/ <http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/>`_ 



Generate Docs
--------------

The docs have a lot more details about how to interact with this codebase.  

**TODO**: Mohanty to add atleast a neat outline herefor the contents to the docs here ::

    git clone git@gitlab.aicrowd.com:flatland/flatland.git
    cd flatland
    pip install -r requirements_dev.txt

* On, Linux and macOS ::

    make docs


* On, Windows ::

    python setup.py develop (or)
    python setup.py install
    python make_docs.py


Features
--------

TODO


Installation
============

Stable Release
--------------

To install flatland, run this command in your terminal ::

    pip install flatland-rl

This is the preferred method to install flatland, as it will always install the most recent stable release.

If you don’t have `pip <https://pip.pypa.io/en/stable/>`_ installed, this `Python installation guide <https://docs.python-guide.org/starting/installation/>`_ can guide you through the process.


From Sources
------------
The sources for flatland can be downloaded from the `Gitlab repo <https://gitlab.aicrowd.com/flatland/flatland>`_.

You can clone the public repository ::

    $ git clone git@gitlab.aicrowd.com:flatland/flatland.git

Once you have a copy of the source, you can install it with ::

    $ python setup.py install
    
    
Usage
=====
To use flatland in a project ::
    
    import flatland
    
flatland
========
TODO: explain the interface here


Authors
--------
* Sharada Mohanty <mohanty@aicrowd.com>
* Giacomo Spigler <giacomo.spigler@gmail.com>
* Mattias Ljungström
* Jeremy Watson
* Erik Nygren <erik.nygren@sbb.ch>
* Adrian Egli <adrian.egli@sbb.ch>
* Vaibhav Agrawal <theinfamouswayne@gmail.com>


<please fill yourself in>
