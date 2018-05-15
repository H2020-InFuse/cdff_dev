[![build status](
https://gitlab.spaceapplications.com/InFuse/CDFF_dev/badges/master/build.svg)](
https://gitlab.spaceapplications.com/InFuse/CDFF_dev)
[![coverage report](
https://gitlab.spaceapplications.com/InFuse/CDFF_dev/badges/master/coverage.svg)](
https://gitlab.spaceapplications.com/InFuse/CDFF_dev)

## CDFF-Dev

CDFF-Dev provides tools to develop, test, visualize, and perform analysis on data fusion products. It includes tools such as data log replay, visualization tools, and data analysis tools. None of it is deployed on the target system.

CDFF-Dev was initiated and is currently developed by the InFuse consortium:
* [Robotics Innovation Center](http://robotik.dfki-bremen.de/en/startpage.html)
  of the [German Research Center for Artificial Intelligence (DFKI)](http://www.dfki.de)
  in Bremen
* [Space Mechatronics Systems Technology Laboratory](https://www.strath.ac.uk/engineering/designmanufactureengineeringmanagement/thespacemechatronicsystemstechnologylaboratory/) of the [University of Strathclyde](https://www.strath.ac.uk/) in Glasgow
* ...

<img src="https://www.dfki.de/web/presse/bildmaterial/dfki-logo-e-schrift.jpg" alt="DFKI RIC" height="75px" />
<img src="https://www.strath.ac.uk/media/1newwebsite/webteam/logos/xUoS_Logo_Tab.png.pagespeed.ic.LkNMQldh_5.png" alt="Strathclyde SMeSTech" height="75px" />

### Dependencies of CDFF-Dev

The CDFF's Dev component, available in this `CDFF_dev` repository, depends on the CDFF's Core and Support components, available in the [`CDFF`](https://gitlab.spaceapplications.com/InFuse/CDFF) repository. It also requires the Python 3 interpreter, the Python 3 headers, the Python 3 package manager, Graphviz, and the following Python packages:

* PyYAML
* Jinja2
* Cython
* NumPy
* msgpack-python
* pydot

On Ubuntu 16.04 you can install those requirements as follow:

```
# Python interpreter, headers, package manager, and Graphviz
$ sudo apt-get install python3 python3-dev python3-pip graphviz

# Install Python packages and a newer version of package manager
# in /usr/local/{bin,lib/python3.X/dist-packages} for all users
$ sudo pip3 install --upgrade pip
$ sudo pip3 install -r requirements.txt

# Or install Python packages and a newer version of package manager
# in $HOME/.local/{bin,lib/python3.X/site-packages} for the current user
$ pip3 install --user --upgrade pip
$ pip3 install --user -r requirements.txt
```

Or you can not install anything and instead use CDFF-Dev inside a Docker container started from the InFuse Docker image. It is the same image as for CDFF-Core and CDFF-Support, and comes with all the necessary dependencies. You only need to provide your local clones of the `CDFF` and `CDFF_dev` repositories. Have a look at [this section](https://gitlab.spaceapplications.com/InFuse/CDFF/blob/master/External/Readme.md#usage) in the documentation on the `CDFF` repository. With an adequately-defined alias, you can start a container as follow:

```
user@computer:~$ docker cdff-dev
Found CDFF-core and CDFF-support: /opt/cdff/
Found compiled ASN.1 data types:  /opt/cdff/Common/Types/C/
Found CDFF-dev: /opt/cdff-dev/
Setting up CDFF-dev... 
Obtaining file:///opt/cdff-dev
Installing collected packages: cdff-dev
  Running setup.py develop for cdff-dev
Successfully installed cdff-dev
Setting up CDFF-dev: done
user@cdff-dev:/$ 
```

Unit testing CDFF-Dev requires additional Python packages, see further.

### Compiling and installing CDFF-Dev

After you have built (and optionally installed) the CDFF's Core and Support components, you can compile and install the CDFF's Dev component as a Python package:

```
# Install CDFF-Dev in /usr/local/{bin,lib/python3.X/dist-packages} for all users
$ sudo CDFFPATH=/path/to/CDFF pip3 install --editable /path/to/CDFF_dev

# Or install it in $HOME/.local/{bin,lib/python3.X/site-packages} for the current user
$ CDFFPATH=/path/to/CDFF pip3 install --user --editable /path/to/CDFF_dev
```

When using the InFuse Docker image, this is performed automatically at container startup, and CDFF-Dev is of course installed inside the container, not on your hard disk.

### Documentation about CDFF-Dev

The overall concept and conventions are described in these documents:

* [CDFF-Support](https://drive.google.com/open?id=1BzKnNrRw6yIFllrITiEGZXD8awtsmvNslqRuB4j29mw)
* [CDFF-Dev](https://drive.google.com/open?id=1yz_w7Eut6Rtg0d4I6R4mze2G8Oip4agyqrTDlKVgC6g)
* [Guide for creating a DFN](https://drive.google.com/open?id=1hFTRKgJNN3n_brT3aajMA03AR_jQ2eCo-ZM33ggY5cE)

### Using CDFF-Dev's DFN code generator

Once CDFF-Dev is compiled and installed, you can run the DFN code generator as follow:

```
# Provide the path to the CDFF on the command line
$ dfn_template_generator --cdffpath=/path/to/CDFF/ dfn_desc.yaml [output_dir]

# Or through an environment variable
$ export CDFFPATH=/path/to/CDFF/
$ dfn_template_generator dfn_desc.yaml [output_dir]

# Or through a hint file in the current directory
$ echo -n /path/to/CDFF/ > cdffpath
$ dfn_template_generator dfn_desc.yaml [output_dir]

# If you don't do either of those things, you will be asked for the path
# and it will be written to a hint file in the current directory
$ dfn_template_generator dfn_desc.yaml
Please enter the path to CDFF:
```

In these commands:

* `dfn_desc.yaml` is the DFN description file.

* `output_dir` is the directory where the generated C++ files and the generated Python bindings are written. It will be created if it doesn't exist, and the default value is the current directory.

    Existing DFN implementation files (`dfnname.{hpp,cpp}`) are not overwritten. Existing DFN interface files (`dfnnameInterface.{hpp,cpp}`) and existing Python bindings (`dfnname.{pxd,pyx}` and `_dfnname.pxd`) are overwritten.

* `/path/to/CDFF/` is the directory that contains your local clone of the `CDFF` repository (CDFF-Core and CDFF-Support). It is looked for on the command line, in the `CDFFPATH` environment variable, and in the hint file `cdffpath` in the current directory, in this order of precedence.

### Testing CDFF-Dev

Unit testing CDFF-Dev requires the following additional Python packages:

* nose
* nose2

You can install them as follow (already installed in the InFuse Docker image):

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

### Contributing to CDFF-Dev

It is not allowed to push to the `master` branch. To contribute a new feature, please develop it in a feature branch, push the feature branch, and open a merge request.

### Status

The following features work:

1. Generating a DFN template in C++
2. Generating Python bindings for a DFN
3. Generating a DFPC template in C++

### License

There is no license at the moment.

### Copyright

Copyright 2017, DFKI GmbH / Robotics Innovation Center, ...
