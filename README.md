# Purpose
A data analysis project for dblp  dataset found at: https://dblp.org/.

Download the `.xml.gz` and `.dtd` files add them to the data/raw folder.

# Current stage
The project is almost completely finished, with only minor changes needed, like adding documentation and some small changes to the `Dockerfile`.

# Setup
- `make up` - creates the docker container
- `make init-db` - creates the duckdb database and saves it in /data/db folder, requires dblp.xml.gz and dblp.dtd

at `localhost:8888` you're going to have access to the jupyter notebook, the token by default is set to "nokia". 


# Presentation
The data analysis itself is performed in the `notebooks/EDA.ipynb` there you can view:
- analysing trends in the scientific papers like:
    + number of papers over time
    + most active authors by year
    + Average size of the team
    + etc.
- most commonly covered topics over time
- collaboration graphs of the authors
- a recommendation system demo

# File structure

- `/src/data_loading`   - files in this folder are responsible for converting an xml into the duckdb database. The scripts are run during the `make init-db` command
- `/src/trend_analysis` - contains a module for figuring out the topics covered over time in the articles
- `/src/utils/logger.py`- a simple logger
- `/src/constants.py`   - file with defined constants, like folder locations, tags, etc.
- `/notebooks/EDA.ipynb`- Main presentable notebook with data analysis proper
- `/notebooks/initial_data_profiling.ipynb` - notebook which was used to scour out the structure of the initial xml itself

# TBA
Documentation