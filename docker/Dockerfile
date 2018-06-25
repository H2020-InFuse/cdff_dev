FROM nexus.spaceapplications.com/repository/infuse/cdff-ci:1.8

RUN apt-get update && apt-get install -y \
  python3 \
  python3-dev \
  python3-pip \
  python3-pyqt4 \
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
  sudo

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

RUN mkdir -p /root/.ssh/

# Workaround for error in ESROCOS' package set
RUN wget https://raw.githubusercontent.com/ESROCOS/buildconf/orbital/esrocos/esrocos_autoproj.rb
RUN mkdir -p autoproj/esrocos
RUN mv esrocos_autoproj.rb autoproj/esrocos
RUN echo "Autobuild.env_push_path 'RUBYLIB', File.join(Autoproj.root_dir, 'autoproj', 'esrocos')" >> autoproj/init.rb

# Autoproj needs these to be defined (will be removed later):
RUN git config --global user.email infuse@example.com
RUN git config --global user.name InFuse

RUN bash -c "source env.sh && autoproj update && echo done"

RUN rm /root/.gitconfig

RUN pip3 install --upgrade pip
RUN pip3 install pyyaml jinja2 cython numpy msgpack-python pydot nose nose2
RUN echo "Autobuild::Package['cdff/CDFF'].define 'USE_BUNDLED_DEPENDENCIES', 'OFF'" >> autoproj/overrides.rb

RUN bash -c "source env.sh && autoproj build && rm -rf /tmp/infuse/envire/envire_core/build && rm -rf /tmp/infuse/envire/envire_visualizer/build && rm -rf /tmp/infuse/envire/envire_smurf/build && rm -rf /tmp/infuse/envire/envire_urdf/build && rm -rf /tmp/infuse/external/ogdf/build && rm -rf /tmp/infuse/gui/vizkit3d/build && rm -rf /tmp/infuse/gui/osgviz/build && rm -rf /tmp/infuse/base/types/build"

# TODO remove CDFF(_dev) and clone them in .gitlab-ci.yml
