[![build status](
https://gitlab.spaceapplications.com/InFuse/CDFF_dev/badges/master/build.svg)](
https://gitlab.spaceapplications.com/InFuse/CDFF_dev)
[![coverage report](
https://gitlab.spaceapplications.com/InFuse/CDFF_dev/badges/master/coverage.svg)](
https://gitlab.spaceapplications.com/InFuse/CDFF_dev)

## CDFF-Dev

CDFF-Dev provides tools to develop, test, visualize, and perform
analysis on data fusion products. It includes tools such as data
log replay, visualization tools, and data analysis tools. None of
it is deployed on the target system.

CDFF-Dev was initiated and is currently developed by the
[InFuse](https://www.h2020-infuse.eu/) consortium. The following
institutes contributed to the software:
* [Robotics Innovation Center](http://robotik.dfki-bremen.de/en/startpage.html)
  of the [German Research Center for Artificial Intelligence (DFKI)](http://www.dfki.de)
  in Bremen
* [Space Mechatronics Systems Technology Laboratory](https://www.strath.ac.uk/engineering/designmanufactureengineeringmanagement/thespacemechatronicsystemstechnologylaboratory/)
  of the [University of Strathclyde](https://www.strath.ac.uk/) in
  Glasgow

<img src="https://www.dfki.de/web/presse/bildmaterial/dfki-logo-e-schrift.jpg" alt="DFKI RIC" height="50px" />&emsp;&emsp;
<img src="doc/static/strathclyde.jpg" alt="Strathclyde SMeSTech" height="70px" />&emsp;&emsp;

### Dependencies of CDFF-Dev

The recommended way of installing all dependencies of CDFF-Dev is autoproj.
See [autoproj installation instructions](https://gitlab.spaceapplications.com/InFuse/cdff-buildconf/tree/cdff_dev#infuse-framework-install-instructions)
for details.

The CDFF's Dev component, available in this `CDFF_dev` repository, depends
on the CDFF's Core and Support components, available in the
[`CDFF`](https://gitlab.spaceapplications.com/InFuse/CDFF) repository.
In addition to the EnviRe components that are required in CDFF, CDFF-Dev
requires the simplified EnviRe visualizer interface.
It also requires the Python 3 interpreter, the Python 3 headers, the Python
3 package manager, Graphviz, and the following Python packages:

* PyYAML
* Jinja2
* Cython
* NumPy
* msgpack-python
* pydot
* PyQt 4

On Ubuntu 16.04 you can install those requirements as follow:

```
# Python interpreter, headers, package manager, Qt, and Graphviz
$ sudo apt-get install python3 python3-dev python3-pip python3-pyqt4 graphviz

# Install Python packages and a newer version of the package manager
# in /usr/local/{bin,lib/python3.X/dist-packages} for all users
$ sudo -H pip3 install --upgrade pip
$ sudo -H pip3 install -r requirements.txt

# Or install Python packages and a newer version of the package manager
# in $HOME/.local/{bin,lib/python3.X/site-packages} for the current user
$ pip3 install --user --upgrade pip
$ pip3 install --user -r requirements.txt
```

Or you can do without installing anything and instead use CDFF-Dev inside a Docker
container started from the InFuse Docker image. It is the same image as for
CDFF-Core and CDFF-Support, and comes with all the necessary dependencies.
You only need to provide your local clones of the `CDFF` and `CDFF_dev`
repositories. Have a look at [this section](https://gitlab.spaceapplications.com/InFuse/CDFF/blob/master/External/Readme.md#usage)
in the documentation on the `CDFF` repository. With an adequately-defined
alias, you can start a container as follow:

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
$ sudo -H CDFFPATH=/path/to/CDFF pip3 install --editable /path/to/CDFF_dev

# Or install it in $HOME/.local/{bin,lib/python3.X/site-packages} for the current user
$ CDFFPATH=/path/to/CDFF pip3 install --user --editable /path/to/CDFF_dev
```

When using the InFuse Docker image, compilation as `root` is performed automatically at container startup, and CDFF-Dev is of course installed inside the container, not on your hard disk.

Notes:

* Compiling CDFF-Dev produces a number of build files in `/path/to/CDFF_dev`:

    ```
    build/
    cdff_dev/__pycache__/
    cdff_dev.egg-info/
    cdff_types.cpp
    cdff_types*.so
    ```

    They can all be removed except `cdff_types*.so` and `cdff_envire*.so`.

* Compiling CDFF-Dev as `root` means that the aforementioned files and directories are owned by `root`. If they are not removed or chowned, subsequently compiling as a normal user fails because the build process isn't allowed to overwrite them.

* About the `-H` flag (`--set-home`): the Python package manager caches data in `$XDG_CACHE_HOME/pip`, where `$XDG_CACHE_HOME` is `$HOME/.cache` by default, where `$HOME` is the home directory of the superuser only if `sudo -H` is used, since `sudo` does not change the home directory by default.

    Running `pip3` through `sudo` without the `-H` flag disables this caching to avoid writing `root`-owned files to the cache directory of the current user. Run `pip3` through `sudo -H` to write those files to the cache directory of the superuser instead.

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
$ sudo -H pip3 install nose nose2

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

Notes:

* Running the unit tests produces a number of test-related files in `/path/to/CDFF_dev`:

    ```
    build/temp.*/test/test_output/
    cdff_dev/__pycache__/
    cdff_dev/test/__pycache__/
    test/__pycache__/
    <testname>*.so
    ```

    If deleted they will be generated again at the next unit test run.

* If CDFF-dev was compiled as `root`, the directory `build/temp.*/` is owned by `root` and non world-writable, so the subdirectory `test/` cannot be created by a normal user, consequently tests that write to `build/temp.*/test/test_output/` fail.

    Work around this issue by adding the writable mode bit for "others" to `build/temp.*/`, or chowning it to your user.

### Contributing to CDFF-Dev

It is not allowed to push to the `master` branch. To contribute a new
feature, please develop it in a feature branch, push the feature branch,
and open a merge request.

### Status

The following features work:

1. Generating a DFN template in C++ with Python bindings
1. Generating a DFPC template in C++ with Python bindings
1. Replaying logfiles with EnviRe visualizer

### License

There is no license at the moment.

### Copyright

Copyright 2017-2018, DFKI GmbH / Robotics Innovation Center,
University of Strathclyde / Space Mechatronics Systems Technology
Laboratory
