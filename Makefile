all: CDFF build_cdff


dependencies: CDFF CDFF/External/opencv3/README.md External/tinyxml2/readme.md
	echo "Installed dependencies."

CDFF:
	git clone --depth 1 git@gitlab.spaceapplications.com:InFuse/CDFF.git

build_cdff: autogeneration
	cp -rf /CDFF/External/install CDFF/External/install
	mkdir -p CDFF/build
	cd CDFF/build; cmake -DCMAKE_INSTALL_PREFIX=./.. .. ; make install

autogeneration:
	cd CDFF/Tools/ASNtoC; ./GeneratorScript.sh

CDFF/External/opencv3/README.md:
	cd CDFF/; git submodule init; git submodule update
	mkdir -p CDFF/External/opencv3/build
	cd CDFF/External/opencv3/build; cmake -DBUILD_DOCS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_PROTOBUF=OFF -DBUILD_TESTS=OFF -DWITH_CUDA=OFF -DBUILD_opencv_dnn=OFF -DCMAKE_INSTALL_PREFIX=./.. ..
	cd CDFF/External/opencv3/build; make install

External/tinyxml2/readme.md:
	cd CDFF/; git submodule init; git submodule update
	mkdir -p CDFF/External/opencv3/build
	cd CDFF/External/opencv3/build; cmake -DCMAKE_INSTALL_PREFIX=./.. ..
	cd CDFF/External/opencv3/build; make install

test:
	nosetests -sv

.PHONY: test
