from setuptools import setup, find_packages

# Read the content of requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    packages=find_packages(),
    install_requires=requirements,
)