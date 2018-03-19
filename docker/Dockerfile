FROM nexus.spaceapplications.com/repository/infuse/cdff-ci:1.6.5.2

RUN apt-get update && apt-get -y install python3 python3-dev python3-pip python3-yaml curl

RUN pip3 install --upgrade pip

RUN pip3 install jinja2 cython nose nose2 numpy msgpack-python

# clone CDFF
RUN mkdir -p /builds/InFuse/build_tools
WORKDIR /builds/InFuse
RUN git clone --depth 1 https://gitlab-ci-token:pVUF6xEhoz2kgWAUyyCr@gitlab.spaceapplications.com:InFuse/Infuse/CDFF.git
# fetch artifacts
COPY get_cdff_artifacts.sh /builds/InFuse/build_tools
WORKDIR /builds/InFuse/build_tools
RUN /bin/bash get_cdff_artifacts.sh

RUN ln -sf /CDFF/External/install /builds/InFuse/CDFF/External/install
#WORKDIR /builds/InFuse/CDFF/Tools/ASNtoC
#RUN ./FetcherScript.sh
WORKDIR /builds/InFuse/CDFF/build/
RUN cmake -DCMAKE_INSTALL_PREFIX=./ ..
WORKDIR /builds/InFuse

LABEL maintainer="Alexander.Fabisch@dfki.de"
LABEL version="1.6.5.5"
