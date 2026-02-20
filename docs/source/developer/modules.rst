API Reference
=============

The API reference provides a module-level map of the Res-IRF4 implementation.

Module guide
------------

.. list-table::
   :header-rows: 1
   :widths: 22 45 33

   * - Module
     - Use this when you need to...
     - Typical entrypoints
   * - ``project.model``
     - run full scenario pipelines
     - ``res_irf``, ``prepare_config``
   * - ``project.building``
     - inspect agent-level building and renovation decisions
     - class and helper methods in ``building``
   * - ``project.thermal``
     - inspect thermal / EPC calculations
     - thermal calculation utilities
   * - ``project.read_input``
     - load config, stock, and policy inputs
     - input and policy parsing functions
   * - ``project.dynamic``
     - inspect exogenous drivers and stock dynamics
     - dynamic update helpers
   * - ``project.coupling``
     - couple Res-IRF outputs with external workflows
     - coupling utilities
   * - ``project.write_output``
     - generate result files and figures
     - plotting and report generation functions
   * - ``project.runs``
     - execute many configs in batch
     - CLI entrypoint for directory-based runs
   * - ``project.utils``
     - use shared data, formatting, and utility helpers
     - generic helper functions

CLI entrypoints
---------------

Run one config:

.. code-block:: bash

   python -m project.main -c project/config/config.json

Run many configs:

.. code-block:: bash

   python -m project.runs -d project/config/policies/realistic

.. toctree::
   :maxdepth: 2

   api/model
   api/building
   api/thermal
   api/read_input
   api/write_output
   api/utils
   api/dynamic
   api/coupling
   api/runs
