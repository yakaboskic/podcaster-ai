# Quick Start
First, clone the podcaster ai repository to a directory of your choosing using a command such as:
```bash
git clone git@github.com:yakaboskic/podcaster-ai.git
```
### Download Data Files
There are a number of necessary data files required to run Podcaster AI. They are various model files as well as data files that I prefer not to package in the github repo for faster cloning speeds. These data files are hosted at Zenodo at the following [link](https://zenodo.org/records/10460039). 

Once you navigate to the link download the data files to a folder in labeled `data` that lines in the root directory of your cloned podcaster-ai directory. 

### Configure Poetry
Podcaster AI uses [poetry](https://python-poetry.org/) as it's dependency management framework as many of our dependencies are also in development. Please install poetry on your system using the instructions found [here](https://python-poetry.org/docs/#installation).

Once, poetry is installed, you can then quickly install all necessary dependencies using the following poetry command:

```bash
poetry install
```

This command will create a new virtual environment on your machine, and install all required packages into that virtual environment.
