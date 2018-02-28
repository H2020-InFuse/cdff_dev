FROM nexus.spaceapplications.com/repository/infuse/cdff-ci:1.6.5.2

RUN apt-get update && apt-get -y install python3 python3-dev python3-pip python3-yaml curl

RUN pip3 install --upgrade pip

RUN pip3 install jinja2 cython nose nose2 numpy msgpack

LABEL maintainer="Alexander.Fabisch@dfki.de"

LABEL version="1.6.5.2"
