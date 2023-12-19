# Res-IRF

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6553021.svg)](https://doi.org/10.5281/zenodo.6553021)

## Disclaimer

**_The contents of this repository are all in-progress and should not be expected to be free of errors or to perform any
specific functions. Use only with care and caution._**

## Overview

> The Res-IRF model is a tool for simulating energy consumption and energy efficiency improvements in the French
> residential building sector. It currently focuses on space heating as the main usage. The rationale for its development
> is to integrate a detailed description of the energy performance of the dwelling stock with a rich description of
> household behaviour. Res-IRF has been developed to improve the behavioural realism that is typically lacking in
> integrated models of energy demand.

## Installation

**Step 1**: Git **clone Res-IRF folder** in your computer.

- Use your terminal and go to a location where you want to store the Res-IRF project.
- `https://github.com/CIRED/Res-IRF4.git`
- `git clone https://github.com/CIRED/Res-IRF4.git`

**Step 2**: **Create a conda environment** from the environment.yml file:

- A conda environment contains the required version of Python and packages. It is an easy way to install all the
  requirements.
- The `environment.yml` file is in the Res-IRF folder.
- Use the **terminal** and go to the Res-IRF folder stored on your computer.
- Type: `conda env create -f environment.yml`

**Step 3**: **Activate the new environment**.

- The first line of the yml file sets the new environment's name.
- Type: `conda activate envResIRF` (in the terminal)

**Step 4**: **Launch Res-IRF**

- Launch from Res-IRF root folder (not from `/project`):
- `python -m project.main -c project/config/config.json`
- `project/config/config.json` is the path to the configuration file
- It is possible that some new packages are not integrated in the yml file. In this case you have to install these
  additional packages manually in the environment.

## Getting started

Project includes libraries, scripts and notebooks.  
`/project` is the folder containing scripts, notebooks, inputs and outputs.

The standard way to run Res-IRF:

**Launch Res-IRF main script.**

### Inputs

Res-IRF come with a set of input data. Input data are stored in the `project/input` folder, and are divided into
sub-folders. 
A configuration file must be declared to run the model, this file selects the input data to be used.

#### Configuration file

The configuration file contains the settings of the scenario.
The details of how the configuration file works are not yet available. Nevertheless, the file has been designed to be as
user-friendly as possible.
A configuration file contain one or multiple headers and one or several scenarios (which makes it possible to get
automatic comparisons between these
scenarios). Each scenario setting is described in a json dictionary.

Among the possible parameters used to describe a scenario:
- `name` : name of the scenario,
- `end` : end year of the simulation,
- `file` : use another calibration file as a starting point to run the simulation. This feature allows to run a scenario
with the same calibration but with different settings (e.g. different energy prices). It also facilitates the modification
of one parameter in the Reference scenario without changing all configuration files.
- `policies`: policies can be included directly in the configuration file or by making reference to an independent policy file (
see `input/policies/policies_ref.json`). This feature ensures that different scenarios have the same policies setting.

An example of configuration file is in the `config` folder under the name of `config.json`.

Headers are a way to easily create variants of scenario while ensuring to keep the same main settings for all scenarios:
- `scenarios` : basic variant of the Reference scenario. An example of such header can be found in `test/config_validation.json`.
Among the possible variants that can be created automatically:
  - "Current Policies" keeps the same public policies used during calibration throughout the simulation.
  - "No policy" deletes all public policies in the year after calibration.
  - "Constant Prices" retains the same energy prices as used during calibration.
- `sensitivity` : modify one or multiple parameters of the Reference scenario,
- `uncertainty` : run all possible permutation of modified parameters, 
- `policies_scenarios` : simplified output for large number of scenarios,
- `assessment_test` : run assessment indicators for one policy.

##### Run large number of policies scenarios

A notebook stored in `project/input/policies/proposals/variant_current_policies` allows to create a configuration file with
different variants of the current policies.
By default, all permutation of the policies are created in `policies_scenarios.json`. 
It is then recommended to run this file with the `policies_scenarios` header.

#### Code performance, memory and parallelization

The Res-IRF script use Multiprocessing tool to launch multiple scenarios in the same time.

The model requires a significant amount of RAM (up to 2GB per scenario).
The RAM requirement depends on the level of details of the scenario, and some specific settings allow to perform
simplified simulations reducing this requirement.
Nevertheless, it is recommended not to run more than 3 scenarios in parallel on a laptop.

A feature allows to simplify some functionalities of the model. This is the "simple" part of the configuration file.
Among the possible parameters, some reduce the number of combination and thus allow to realize simulations with less
computing time and especially less memory:

- the configuration parameter "quintiles" allows to pass quintiles or deciles,
- the configuration parameter "heating_system" allows to restrict the number of heating systems.

Other parameters, allow to launch simulations by keeping the public policies of the calibration or on the contrary to
remove them to measure their effect.

### Outputs

The model creates results in a folder in `project/output`.  
Folder name is by default `ddmmyyyy_hhmm` (launching date and hour).
By default, only a selection of the most important results are available and graphs.

In the `output/ddmmyyyy_hhmm` folder:

- A summary file `summary_run.pdf` is automatically created in the output folder aggregating the most important results.
- One folder for each scenario declared in the configuration file with detailed outputs:
    - `output.csv` detailed output readable directly with an Excel-like tool
    - `/img`folder
- A folder `/img` at the root of the output folder contains graphs comparing scenarios launch in the same config file.

## API

It is also possible to get data and Python object directly (useful to create its own scripts).  
`config = get_config()` allows to get the Reference configuration file.  
`inputs = get_inputs(building_stock=path)` allow to get data.
`inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config)`: create Python objects from
raw data.  
Finally:  
`buildings, energy_prices, taxes, post_inputs, cost_heater, ms_heater, cost_insulation, ms_intensive, renovation_rate_ini, policies_heater, policies_insulation, flow_built = initialize(inputs, stock, year, policies_heater, policies_insulation, taxes, config, path)`
parse and create Python objects used by Res-IRF.  
Moreover, the user can use all the methods of AgentBuildings object `buildings` defined in buildings.py.

## About the authors

The development of the Res-IRF model was initiated at CIRED in 2008. Coordinated by Louis-Gaëtan Giraudet, it involved
over the years, in alphabetic order, Louise Asselin, Cyril Bourgeois, Frédéric Branger, François Chabrol, David Glotin,
Céline Guivarch, Philippe Quirion, and Lucas Vivier.

The last version of the code was rewritten entirely and developed by Lucas Vivier.

## Meta

If you find `Res-IRF` useful, please kindly cite our last paper:

```
@article{
  author  = {Giraudet, Louis-Gaëtan and Bourgeois, Cyril and Quirion, Philippe},
  title   = {Policies for low-carbon and affordable home heating: A French outlook},
  journal = {Energy Policy},
  year    = {2021},
  volume  = {151},
  url     = {https://www.sciencedirect.com/science/article/pii/S0301421521000094}
}
```

Lucas Vivier – [@VivierLucas](https://twitter.com/VivierLucas) – lucas.vivier@enpc.fr

Distributed under the GNU GENERAL PUBLIC LICENSE. See ``LICENSE`` for more information.
