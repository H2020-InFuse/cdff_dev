FROM nexus.spaceapplications.com/repository/infuse/infuse:latest

LABEL maintainer="Alexander.Fabisch@dfki.de"

RUN apt-get update && apt-get -y install python3 python3-dev python3-pip python3-yaml
RUN pip3 install jinja2 cython nose
