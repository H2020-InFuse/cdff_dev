FROM nexus.spaceapplications.com/repository/infuse/cdff-ci:1.8.1

RUN apt-get update && apt-get install -y \
  python3 \
  python3-dev \
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
  qt4-qmake \
  pkg-config \
  ruby \
  ruby-dev \
  sudo \
  python3-pyqt4
RUN pip3 install --upgrade pip
RUN pip3 install pyyaml jinja2 cython numpy msgpack-python pydot nose nose2

RUN mkdir -p -m 700 /root/.ssh/ && \
  ssh-keyscan gitlab.spaceapplications.com >> /root/.ssh/known_hosts

# Prepare non-interactive autoproj setup with CI user
WORKDIR /tmp/infuse
COPY config.yml /tmp/infuse/.autoproj/
RUN wget http://www.rock-robotics.org/autoproj_bootstrap --directory-prefix=/tmp/infuse/
RUN AUTOPROJ_BOOTSTRAP_IGNORE_NONEMPTY_DIR=1 ruby autoproj_bootstrap git https://cdff-dev:cdff-dev@gitlab.spaceapplications.com/InFuse/cdff-buildconf.git branch=cdff_dev
COPY 00_ci_user.yml /tmp/infuse/autoproj/overrides.d/
COPY manifest /tmp/infuse/autoproj/

# Temporary workaround for error in ESROCOS' package set.
#RUN mkdir -p /tmp/infuse/autoproj/esrocos
#RUN wget https://raw.githubusercontent.com/ESROCOS/buildconf/orbital/esrocos/esrocos_autoproj.rb --directory-prefix=/tmp/infuse/autoproj/esrocos/
#RUN echo "Autobuild.env_push_path 'RUBYLIB', File.join(Autoproj.root_dir, 'autoproj', 'esrocos')" >> autoproj/init.rb

# Autoproj needs these to be defined (will be removed later):
RUN git config --global user.email infuse@example.com \
  && git config --global user.name InFuse

SHELL ["/bin/bash", "-c"]

RUN source env.sh && autoproj update

# We use CDFF's preinstalled dependencies.
RUN echo "Autobuild::Package['cdff/CDFF'].define 'USE_BUNDLED_DEPENDENCIES', 'OFF'" >> autoproj/overrides.rb

# Fix missing pkg-config files
COPY pcl_registration-1.8.pc pcl_features-1.8.pc pcl_2d-1.8.pc /tmp/infuse/install/lib/pkgconfig/

# Workaround for Eigen include directory
COPY setup.py /tmp/infuse/cdff/CDFF_dev/

# Install direct dependencies of CDFF_dev
RUN source env.sh && autoproj build \
  && rm -rf /tmp/infuse/cdff/CDFF_dev

# Install EnviRe visualizer because it is not yet in the master branch
#RUN source env.sh && autoproj build envire/envire_visualizer \
#  && rm -rf /tmp/infuse/envire/envire_core/build \
#  && rm -rf /tmp/infuse/envire/envire_visualizer/build \
#  && rm -rf /tmp/infuse/envire/envire_smurf/build \
#  && rm -rf /tmp/infuse/envire/envire_urdf/build \
#  && rm -rf /tmp/infuse/external/ogdf/build \
#  && rm -rf /tmp/infuse/gui/vizkit3d/build \
#  && rm -rf /tmp/infuse/gui/osgviz/build \
#  && rm -rf /tmp/infuse/base/types/build

# Reset git configuration
#RUN rm /root/.gitconfig

#WORKDIR /

LABEL maintainer="Alexander.Fabisch@dfki.de"
#LABEL version="1.8.6"
#LABEL description="Image providing the dependencies of CDFF-dev for continuous integration"
