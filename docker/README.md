# CDFF-Dev Dockerfile instructions for CDFF_Dev

## Install Docker

For Ubuntu:

    sudo apt-get install docker.io

Installation instructions for other systems are available at the
[official documentation](https://docs.docker.com/engine/installation/).

## Use Image from Docker Registry

* Log in SpaceApps docker server.
  The login step are explained for security reason in this
  [document](https://owncloud.spaceapplications.com/owncloud/index.php/apps/files/ajax/download.php?dir=%2F%2BInFuse-SHARED%2F%2BWP9%20-%20CDFF%20Software%20Infrastructure%20and%20Supporting%20Tools&files=docker-server-logging.txt).
  If you don't have access to the Owncloud of Space Applications, contact
  someone from Space Applications to get access.
* Select the docker image that you want to use from the registry, for example,
  `nexus.spaceapplications.com/repository/infuse/cdff_dev:1.6.4`. You can
  browse available images at [this website](https://nexus.spaceapplications.com)
  if you login with the same account.
* Create a container from the image. See Section 'Create Container' for details.

## Build Image

Note: The average user does not have to build an image. Usually the image will
be distributed to the users through a docker registry.

* Clone the cdff_dev if you haven't yet

    git clone git@gitlab.spaceapplications.com:InFuse/CDFF_dev.git

* Go to the directory `docker`.
* Log in SpaceApps docker server.
  The login step are explained for security reason in this
  [document](https://owncloud.spaceapplications.com/owncloud/index.php/apps/files/ajax/download.php?dir=%2F%2BInFuse-SHARED%2F%2BWP9%20-%20CDFF%20Software%20Infrastructure%20and%20Supporting%20Tools&files=docker-server-logging.txt).
  If you don't have access to the Owncloud of Space Applications, contact
  someone from Space Applications to get access.

* Build base image with dependencies (cdff-dev-ci):

    docker build -t cdff-dev-ci:latest .

* Build version for users (cdff-dev):

    docker build -t cdff-dev:latest -f Dockerfile_user .

Sometimes it is necessary to clean the docker cache if you want to rebuild the
image. You just have to add `--no-cache` in this case.

## Create Container

You can create a container (runtime environment) from the image.

Without GUI:

    docker run -it <docker image>

With GUI:

    xhost local:root
    docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix --privileged <docker image>

If you have a custom DNS server you can set it for the docker container with

    --dns <ip>

Additionally, you can mount external directories in the container. This can
be useful because you might want to destroy and recreate containers from time
to time. It can be done with the option

    -v <host-directory>:<mount-point>

You can give containers names that can be used like their IDs:

    --name <name>

Full example:

    xhost local:root
    export HOSTWORKDIR=...
    docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOSTWORKDIR:/external --privileged --name dev <docker image> bash

## Setup GPU

You usually have to set up your GPU if you use the EnviRe visualizer. For
example, to make an Nvidia GPU available in the docker container, the following
steps have to be taken:

On the host, check your driver version:

    # you might have to install this package:
    $ sudo apt-get install mesa-utils
    $ glxinfo | grep "OpenGL version"
    OpenGL version string: 4.5.0 NVIDIA 375.39

If you only want to know the version number you can run

    glxinfo | grep -o "OpenGL version .* NVIDIA .*" | cut -d' ' -f 6

Alternatively you can look it up in your driver settings gui, where you also can activate the proprietary nvidia driver.

In this example, the driver version is "375.39". Now, you have install exactly
the same driver in the container:

    export VERSION=375.39
    wget http://us.download.nvidia.com/XFree86/Linux-x86_64/$VERSION/NVIDIA-Linux-x86_64-$VERSION.run
    chmod +x NVIDIA-Linux-x86_64-$VERSION.run
    apt-get install -y module-init-tools
    ./NVIDIA-Linux-x86_64-$VERSION.run -a -N --ui=none --no-kernel-module

## Don't forget to set up git

    git config --global user.name "<name>"
    git config --global user.email <email>

## Working with Docker

Overview of containers:

    docker ps -a

Start container:

    docker start <id>

Connect to container:

    docker attach <id>

Typing the first 2-3 characters of the container ID is usually sufficient.

## Troubleshooting

### Error: Cannot connect to the Docker daemon

Add the docker group if it doesn't already exist:

    sudo groupadd docker

Add the connected user "${USER}" to the docker group:

    sudo gpasswd -a ${USER} docker

Restart the Docker daemon:

    sudo service docker restart

Either do a newgrp docker or log out/in to activate the changes to groups.

### Error: Can't connect to X11 window server

Do this at your host pc

    sudo xhost +
