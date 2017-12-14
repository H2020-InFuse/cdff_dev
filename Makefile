all: CDFF build_cdff


CDFF:
	git clone https://gitlab-ci-token:pVUF6xEhoz2kgWAUyyCr@gitlab.spaceapplications.com:InFuse/Infuse/CDFF.git

build_cdff: autogeneration dependencies mkdir_build
	cd CDFF/build; cmake -DCMAKE_INSTALL_PREFIX=./.. ..
	cd CDFF/build; make

autogeneration:
	cd CDFF/Tools/ASNtoC; ./GeneratorScript.sh

dependencies: CDFF/External/opencv3/README.md
	echo "Installed dependencies."

mkdir_build:
	cd CDFF; mkdir build

CDFF/External/opencv3/README.md:
	cd CDFF/; git submodule init; git submodule update
	cd CDFF/External/opencv3/; mkdir build
	cd CDFF/External/opencv3/build; cmake -DBUILD_DOCS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_PROTOBUF=OFF -DBUILD_TESTS=OFF -DWITH_CUDA=OFF -DBUILD_opencv_dnn=OFF -DCMAKE_INSTALL_PREFIX=./.. ..
	cd CDFF/External/opencv3/build; make install

External/tinyxml2/readme.md:
	cd CDFF/; git submodule init; git submodule update
	cd CDFF/External/opencv3/; mkdir build
	cd CDFF/External/opencv3/build; cmake -DCMAKE_INSTALL_PREFIX=./.. ..
	cd CDFF/External/opencv3/build; make install

test:
	nosetests3 -sv

.PHONY: test
