Multi-Agent Interface
=======

.. include:: interface/pettingzoo.rst
.. include:: interface/wrappers.rst

Multi-Agent Pettingzoo Usage
=======

We can use the PettingZoo interface by proving the rail env to the petting zoo wrapper as shown below in the example.

.. literalinclude:: ../tests/test_pettingzoo_interface.py
   :language: python
   :start-after: __sphinx_doc_begin__
   :end-before: __sphinx_doc_end__


Multi-Agent Interface Stable Baseline 3 Training
=======

.. literalinclude:: ../flatland/contrib/training/flatland_pettingzoo_stable_baselines.py
   :language: python
   :start-after: __sphinx_doc_begin__
   :end-before: __sphinx_doc_end__


Multi-Agent Interface Rllib Training
=======

.. literalinclude:: ../flatland/contrib/training/flatland_pettingzoo_rllib.py
   :language: python
   :start-after: __sphinx_doc_begin__
   :end-before: __sphinx_doc_end__