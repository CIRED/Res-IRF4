
# Res-IRF

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6553021.svg)](https://doi.org/10.5281/zenodo.6553021)

## Disclaimer

**_The contents of this repository are all in-progress and should not be expected to be free of errors or to perform any
specific functions. Use only with care and caution._**

## Overview

> The Res-IRF model is a tool for simulating energy consumption and energy efficiency improvements in the French residential building sector. It currently focuses on space heating as the main usage. The rationale for its development is to integrate a detailed description of the energy performance of the dwelling stock with a rich description of household behaviour. Res-IRF has been developed to improve the behavioural realism that is typically lacking in integrated models of energy demand.

## Installation

**Step 1**: Git **clone Res-IRF folder** in your computer.
   - Use your terminal and go to a location where you want to store the Res-IRF project.
   - `https://github.com/CIRED/Res-IRF4.git`
   - If the code is not released in open-source. [Create a token](https://docs.github.com/en/enterprise-server@3.4/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token). 
   - `git clone https://oauth-key-goes-here@/github.com/CIRED/Res-IRF4.git`


**Step 2**: **Create a conda environment** from the environment.yml file:
   - A conda environment contains the required version of Python and packages. It is an easy way to install all the requirements.
   - The `environment.yml` file is in the Res-IRF folder.
   - Use the **terminal** and go to the Res-IRF folder stored on your computer.
   - Type: `conda env create -f environment.yml`

**Step 3**: **Activate the new environment**.
   - The first line of the yml file sets the new environment's name.
   - Type: `conda activate envResIRF` (in the terminal)

**Step 4**: **Launch Res-IRF**
   - Launch from Res-IRF root folder (not from `/project`):
   - `python -m project.main -c project/config/test/config.json`
   - `project/config/test/config.json` is the path to the configuration file

## Getting started

Project includes libraries, scripts and notebooks.  
`/project` is the folder containing scripts, notebooks, inputs and outputs.  

The standard way to run Res-IRF:  

**Launch Res-IRF main script.**  
The model creates results in a folder in project/output.  
Folder name is by default `ddmmyyyy_hhmm` (launching date and hour).
By default, only a  selection of the most important results are available and graphs.

A configuration file must be declared.
An example of configuration file is in the `input` folder under the name of `config.json`.
The Res-IRF script use Multiprocessing tool to launch multiple scenarios in the same time. 

In the `output/ddmmyyyy_hhmm` folder:
- One folder for each scenario declared in the configuration file with detailed outputs:
    - `output.csv` detailed output readable directly with an Excel-like tool
- `.png` graphs comparing scenarios launch in the same config file.

## API

It is also possible to get data and Python object directly (useful to create its own scripts).  
`config = get_config()` allows to get the Reference configuration file.  
`inputs = get_inputs(building_stock=path)` allow to get data.
`inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config)`: create Python objects from raw data.  
Finally:  
`buildings, energy_prices, taxes, post_inputs, cost_heater, ms_heater, cost_insulation, ms_intensive, renovation_rate_ini, policies_heater, policies_insulation, flow_built = initialize(inputs, stock, year, policies_heater, policies_insulation, taxes, config, path)`
parse and create Python objects used by Res-IRF.  
Moreover, the user can use all the methods of AgentBuildings object `buildings` defined in buildings.py.  


## About the authors

The development of the Res-IRF model was initiated at CIRED in 2008. Coordinated by Louis-Gaëtan Giraudet, it involved
over the years, in alphabetic order, Louise Asselin, Cyril Bourgeois, Frédéric Branger, François Chabrol, David Glotin, 
Céline Guivarch, Philippe Quirion, and Lucas Vivier.

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


Lucas Vivier – [@VivierLucas](https://twitter.com/VivierLucas) – vivier@centre-cired.fr

Distributed under the GNU GENERAL PUBLIC LICENSE. See ``LICENSE`` for more information.
