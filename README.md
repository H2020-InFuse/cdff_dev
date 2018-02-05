[![build status](
https://gitlab.spaceapplications.com/InFuse/CDFF_dev/badges/master/build.svg)](
https://gitlab.spaceapplications.com/InFuse/CDFF_dev)
[![coverage report](
https://gitlab.spaceapplications.com/InFuse/CDFF_dev/badges/master/coverage.svg)](
https://gitlab.spaceapplications.com/InFuse/CDFF_dev)

# CDFF-Dev

The CDFF-Dev provides the tools to develop, test, visualize, and perform
analysis on data fusion products. The CDFF-Dev include tools such as data
log replay, visualization tools, and data analysis tools. None of the
components of the CDFF-Dev are deployed on the target system.

CDFF-Dev was initiated and is currently developed by the InFUSE constortium:
* [Robotics Innovation Center](http://robotik.dfki-bremen.de/en/startpage.html)
  of the [German Research Center for Artificial Intelligence (DFKI)](http://www.dfki.de)
  in Bremen
* ...

![DFKI RIC](https://www.dfki.de/web/presse/bildmaterial/dfki-logo-e-schrift.jpg)

# Dependencies

CDFF-Dev depends on CDFF. In addition, the following packages are required:

* Python 3 + headers
* pip
* Python YAML
* Jinja 2
* Cython

On Ubuntu 16.04 you can install them with

    sudo apt-get install python3 python3-dev python3-pip
    sudo pip3 install -r requirements.txt

## Installation

As CDFF-Dev is under development, the most convenient way is to install a
symbolic link to the source code in your system:

    sudo pip3 install -e .

## Documentation

The overall concept and conventions are described in the documentation
of CDFF-Support [here](https://docs.google.com/document/d/1BzKnNrRw6yIFllrITiEGZXD8awtsmvNslqRuB4j29mw/edit#heading=h.lsr1bgv0ntf5).
The link to the CDFF-dev Document [here](https://docs.google.com/document/d/1yz_w7Eut6Rtg0d4I6R4mze2G8Oip4agyqrTDlKVgC6g/edit#heading=h.1xul7efma9uy)

### DFN Code Generator

You can run the DFN code generator with

    dfn_template_generator my_node_desc.yaml my_node_output_folder

## Testing

You can run the tests with nosetests:

    make test

Nosetests can be installed with pip:

    sudo pip3 install nose

## Contributing

It is not allowed to push directly to the master branch. To develop a new
feature, please make a feature branch and open a merge request.

## Current State

Only a rough scaffold is available at the moment.

## License

There is no license at the moment.

## Copyright

Copyright 2017, DFKI GmbH / Robotics Innovation Center, ...
