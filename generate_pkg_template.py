#!/usr/bin/python
# -- coding: utf-8 --

import os
import sys
print(f"number of parameters: ",len(sys.argv))
for i in range(len(sys.argv)):
    print(f"param{i}:{sys.argv[i]}")

package_name = sys.argv[1]
#package_name = "pkg"

#create folders
folder_names = ['api', 'notebooks', package_name, 'scripts', 'tests']
for folder in folder_names: 
    os.system(f"mkdir -p {folder}")

#create files
file_names = [f'{package_name}/__init__.py',
                'README.md',
                '.gitignore',
                'requirements.txt',
                'Makefile',
                'setup.py']

for file in file_names:
    os.system(f"touch {file}")

#prepare requirements.txt
packages = ["pandas", "numpy", "requests", "setuptools"]
f = open('requirements.txt', "w")
for package in packages:

    f.write(f"{package}\n")
f.close()


#prepare .gitignore
ignore_files = [".gitignore", 
                f"{package_name}/__pycache__/*",
                "notebooks/.ipynb_checkpoints",
                "notebooks/.ipynb_checkpoints/*"
                ]

f = open('.gitignore', "w")
for file in ignore_files:
    f.write(f"{file}\n")
f.close()

#create template README.md
f = open('README.md', "w")
f.write(f"""# *NAME OF THE PROJECT/PROGRAM* 

## *What is does*

PROGRAM is a python/c++/c/html/ipython notebook which makes/solve/consists
what is a goal
input
output 

## General Prerequisites (for running and building)

* [Anaconda](https://www.anaconda.com/products/individual)
* [GitHub](https://github.com)
* library-n.version 5
* some other library

## To build

NAME PROGRAM can be build on win/mac/linux
Instruction how to build

## To run (for development or testing)

```
# Clone this repository 
git clone http://github.com/rolandina
# Go into the directory
cd dirname
# python run.py
```
## Contributing

This project was built on another project/[framework](framework link)
If you have any questions, please reach me at 
[License](www.lisense.com)""")
f.close()

f = open('setup.py', "w")

f.write(f"""
from setuptools import setup

setup(
    name="{package_name}",
    version='0.0.1',
    packages=["{package_name}"],
    install_requires={packages},
    #scripts = ['scripts/generate_plots']
    )
""")
f.close()



