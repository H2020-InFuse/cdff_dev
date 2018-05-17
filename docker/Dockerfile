FROM nexus.spaceapplications.com/repository/infuse/cdff-ci:1.7.1
# Python interpreter, headers, and package manager
RUN apt-get update && apt-get -y install python3 python3-dev python3-pip graphviz curl

# Install Python packages and newer version of package manager
# in /usr/local/{bin,lib/python3.X/dist-packages}
RUN pip3 install --upgrade pip
RUN pip3 install pyyaml jinja2 cython numpy msgpack-python pydot nose nose2

LABEL maintainer="Alexander.Fabisch@dfki.de"
LABEL version="1.7.3"
