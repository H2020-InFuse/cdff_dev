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

RUN mkdir -p /opt/infuse/.autoproj
RUN mkdir -p /root/.ssh/
WORKDIR /opt/infuse
ADD config.yml /opt/infuse/.autoproj
RUN wget http://www.rock-robotics.org/autoproj_bootstrap
ENV AUTOPROJ_BOOTSTRAP_IGNORE_NONEMPTY_DIR 1
ARG ssh_prv_key
RUN ssh-keyscan gitlab.spaceapplications.com >> /root/.ssh/known_hosts \
  && echo "$ssh_prv_key" > /root/.ssh/id_rsa && chmod 600 /root/.ssh/* && eval `ssh-agent -s` && ssh-add /root/.ssh/id_rsa \
  && ruby autoproj_bootstrap git git@gitlab.spaceapplications.com:InFuse/cdff-buildconf branch=cdff_dev \
  && rm -f /root/.ssh/*
RUN apt-get update && apt-get install -y \
  python3 \
  python3-dev \
  python3-pyqt4 \
  python3-pip \
  autoconf \
  automake \
  freeglut3-dev \
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

# Workaround for error in ESROCOS' package set
RUN wget https://raw.githubusercontent.com/ESROCOS/buildconf/orbital/esrocos/esrocos_autoproj.rb
RUN mkdir -p autoproj/esrocos
RUN mv esrocos_autoproj.rb autoproj/esrocos
RUN echo "Autobuild.env_push_path 'RUBYLIB', File.join(Autoproj.root_dir, 'autoproj', 'esrocos')" >> autoproj/init.rb

RUN ssh-keyscan gitlab.spaceapplications.com >> /root/.ssh/known_hosts \
  && echo "$ssh_prv_key" > /root/.ssh/id_rsa && chmod 600 /root/.ssh/* && eval `ssh-agent -s` && ssh-add /root/.ssh/id_rsa \
  && bash -c "source env.sh && autoproj update" \
  && rm -f /root/.ssh/*
RUN bash -c "source env.sh && autoproj build"
#RUN apt-get update && apt-get install -y ipython3
