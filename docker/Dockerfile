FROM nexus.spaceapplications.com/repository/infuse/cdff-ci:1.6.5.2

RUN apt-get update && apt-get -y install python3 python3-dev python3-pip python3-yaml curl

RUN pip3 install --upgrade pip

RUN pip3 install jinja2 cython nose nose2 numpy msgpack-python

# clone CDFF
RUN mkdir -p infuse/build_tools
WORKDIR /infuse
RUN git clone --depth 1 https://gitlab-ci-token:pVUF6xEhoz2kgWAUyyCr@gitlab.spaceapplications.com:InFuse/Infuse/CDFF.git
# fetch artifacts
COPY get_cdff_artifacts.sh /infuse/build_tools
WORKDIR /infuse/build_tools
RUN /bin/bash get_cdff_artifacts.sh

RUN cp -rf /CDFF/External/install /infuse/CDFF/External/install
WORKDIR /infuse/CDFF/Tools/ASNtoC
RUN ./FetcherScript.sh

LABEL maintainer="Alexander.Fabisch@dfki.de"

LABEL version="1.6.5.4"
