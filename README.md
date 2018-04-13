[![build status](
https://gitlab.spaceapplications.com/InFuse/CDFF_dev/badges/master/build.svg)](
https://gitlab.spaceapplications.com/InFuse/CDFF_dev)
[![coverage report](
https://gitlab.spaceapplications.com/InFuse/CDFF_dev/badges/master/coverage.svg)](
https://gitlab.spaceapplications.com/InFuse/CDFF_dev)

# CDFF-Dev

The CDFF-Dev provides the tools to develop, test, visualize, and perform
analysis on data fusion products. The CDFF-Dev includes tools such as data
log replay, visualization tools, and data analysis tools. No component of
the CDFF-Dev is deployed on the target system.

CDFF-Dev was initiated and is currently developed by the InFuse consortium:
* [Robotics Innovation Center](http://robotik.dfki-bremen.de/en/startpage.html)
  of the [German Research Center for Artificial Intelligence (DFKI)](http://www.dfki.de)
  in Bremen
* ...

![DFKI RIC](https://www.dfki.de/web/presse/bildmaterial/dfki-logo-e-schrift.jpg)

## Dependencies of CDFF-Dev

The CDFF's Dev component, available in this `CDFF_dev` repository, depends on the CDFF's Core and Support components, available in the [`CDFF`](https://gitlab.spaceapplications.com/InFuse/CDFF_dev) repository. It also requires the Python 3 interpreter, the Python 3 headers, and the following Python packages:

* pip
* PyYAML
* Jinja2
* Cython
* NumPy
* msgpack-python
* Graphviz
* pydot

On Ubuntu 16.04 you can install them as follow:

```
# Python interpreter, headers, and package manager
$ sudo apt-get install python3 python3-dev python3-pip

# Install Python packages and newer version of package manager
# in /usr/local/{bin,lib/python3.X/dist-packages} for all users
$ sudo pip3 install --upgrade pip
$ sudo pip3 install -r requirements.txt

# Or install Python packages and newer version of package manager
# in $HOME/.local/{bin,lib/python3.X/site-packages} for the current user
$ pip3 install --user --upgrade pip
$ pip3 install --user -r requirements.txt
```

Unit testing CDFF-Dev require additional Python packages, see further.

## Compiling and installing CDFF-Dev

After you have built (and optionally installed) the CDFF's Core and Support components, you can compile and install the CDFF's Dev component as a Python package:

```
# Install CDFF-Dev in /usr/local/{bin,lib/python3.X/dist-packages} for all users
$ sudo CDFFPATH=/path/to/CDFF pip3 install --editable /path/to/CDFF_dev

# Or install it in $HOME/.local/{bin,lib/python3.X/site-packages} for the current user
$ CDFFPATH=/path/to/CDFF pip3 install --user --editable /path/to/CDFF_dev
```

## Documentation about CDFF-Dev

The overall concept and conventions are described in these documents:

* [CDFF-Support](https://drive.google.com/open?id=1BzKnNrRw6yIFllrITiEGZXD8awtsmvNslqRuB4j29mw)
* [CDFF-Dev](https://drive.google.com/open?id=1yz_w7Eut6Rtg0d4I6R4mze2G8Oip4agyqrTDlKVgC6g)
* [Guide for creating a DFN](https://drive.google.com/open?id=1hFTRKgJNN3n_brT3aajMA03AR_jQ2eCo-ZM33ggY5cE)

## Using CDFF-Dev's DFN code generator

Once CDFF-Dev is compiled and installed, you can run the DFN code generator as follow:

```
$ dfn_template_generator my_node_desc.yaml my_node_output_dir --cdffpath path/to/CDFF/
```

where:

* `my_node_desc.yaml` is the node description file
* `my_node_output_dir` is the directory where the generated C++ files and the generated Python bindings will be written (it will be created if it doesn't exist)
* `path/to/CDFF/` is the directory which contains your local clone of the `CDFF` repository

## Testing CDFF-Dev

Unit testing CDFF-Dev requires the following additional Python packages:

* nose
* nose2

You can install them as follow:

```
# Install nose and nose2 in /usr/local/{bin,lib/python3.X/dist-packages} for all users
$ sudo pip3 install nose nose2

# Or install them in $HOME/.local/{bin,lib/python3.X/site-packages} for the current user
$ pip3 install --user nose nose2
```

Then to run the unit tests:

```
# Either
nosetests -sv

# Or
make test
```

## Contributing to CDFF-Dev

It is not allowed to push directly to the `master` branch. To develop a new
feature, please make a feature branch and open a merge request.

## Status

The following features work:

1. Generating a DFN template in C++
2. Generating Python bindings for a DFN
3. Generating a DFPC template in C++

## License

There is no license at the moment.

## Copyright

Copyright 2017, DFKI GmbH / Robotics Innovation Center, ...
