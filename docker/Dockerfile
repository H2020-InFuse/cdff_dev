FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
  build-essential \
  apt-utils \
  cmake \
  sudo \
  git \
  mono-devel \
  ruby \
  ruby-dev \
  wget \
  curl \
  unzip

RUN git config --global user.email infuse@example.com
RUN git config --global user.name InFuse

RUN apt-get update && apt-get install -y \
  python3 \
  python3-dev \
  python3-pyqt4 \
  python3-pip \
  autoconf \
  automake \
  freeglut3-dev \
  graphviz \
  graphviz-dev \
  libgoogle-glog-dev \
  libjsoncpp-dev \
  libopenscenegraph-dev \
  libpoco-dev \
  libproj-dev \
  libqt4-dev \
  libqt4-opengl-dev \
  libtinyxml-dev \
  qt4-qmake
RUN pip3 install pyyaml jinja2 cython numpy msgpack-python pydot nose nose2

RUN mkdir -p -m 700 /root/.ssh/
RUN ssh-keyscan gitlab.spaceapplications.com >> /root/.ssh/known_hosts

# Prepare non-interactive autoproj setup with CI user
RUN mkdir -p /tmp/infuse/.autoproj
WORKDIR /tmp/infuse
ADD config.yml /tmp/infuse/.autoproj
RUN wget http://www.rock-robotics.org/autoproj_bootstrap
ENV AUTOPROJ_BOOTSTRAP_IGNORE_NONEMPTY_DIR 1
RUN ruby autoproj_bootstrap git https://cdff-dev:cdff-dev@gitlab.spaceapplications.com/InFuse/cdff-buildconf.git branch=cdff_dev
ADD 00_ci_user.yml /tmp/infuse/autoproj/overrides.d/
ADD manifest /tmp/infuse/autoproj/

# Workaround for error in ESROCOS' package set
RUN wget https://raw.githubusercontent.com/ESROCOS/buildconf/orbital/esrocos/esrocos_autoproj.rb
RUN mkdir -p autoproj/esrocos
RUN mv esrocos_autoproj.rb autoproj/esrocos
RUN echo "Autobuild.env_push_path 'RUBYLIB', File.join(Autoproj.root_dir, 'autoproj', 'esrocos')" >> autoproj/init.rb

RUN bash -c "source env.sh && autoproj update"
RUN bash -c "source env.sh && autoproj build"

RUN rm -rf /tmp/infuse/cdff/external/opencv/build
RUN rm -rf /tmp/infuse/cdff/external/boost/libs
RUN rm -rf /tmp/infuse/cdff/external/pcl/build
RUN rm -rf /tmp/infuse/cdff/external/vtk/build
RUN rm -rf /tmp/infuse/envire/envire_core/build
RUN rm -rf /tmp/infuse/envire/envire_visualizer/build
RUN rm -rf /tmp/infuse/envire/envire_smurf/build
RUN rm -rf /tmp/infuse/envire/envire_urdf/build

LABEL maintainer="Alexander.Fabisch@dfki.de"
LABEL version="1.8"
