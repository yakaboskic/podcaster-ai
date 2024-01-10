# Quick Start
This section will walk you through how to setup Podcaster AI to run in a virtual environment. First, clone the podcaster ai repository to a directory of your choosing using a command such as:
```bash
git clone git@github.com:yakaboskic/podcaster-ai.git
```
### Requirements
Ensure that at least Python 3.11 is installed and if you are using Ubuntu or another Linux distro ensure that all the dev libraries are installed with Python 3.11 as well. You may wish to run a command such as:

```bash
sudo apt install python3.11 python3.11-dev
```
If you want to install a different version of Python than that which is default on your system, this [link](https://www.howtogeek.com/install-latest-python-version-on-ubuntu/) may be helpful.

### Setting Up Your Virtual Environment
We recommend installing podcaster into a virtual environment of your choosing. To follow are instructions for creating a virtual environment using Python's built-in venv. 

In the podcaster-ai repo folder that we just clone, create a virtual environment using the following command:
```bash
python3.11 -m venv podcaster-venv
```
Next, activate your virtual with the following command:
```bash
. podcaster-venv/bin/activate
```
Now we can continue installing podcaster-ai into this virtual.

### Download Data Files
There are a number of necessary data files required to run Podcaster AI. They are various model files as well as data files that I prefer not to package in the github repo for faster cloning speeds. These data files are hosted at Zenodo at the following [link](https://zenodo.org/records/10460039). 

Once you navigate to the link download the data files to a folder in labeled `data` that lines in the root directory of your cloned podcaster-ai directory. 

### Install the Project
We are leveraging a pyproject.toml and a requirements.txt file to manage project dependencies, so you can just use the following command to install the project once your virtual environment is activated:
```bash
pip install .
```
More documentation to come. 
