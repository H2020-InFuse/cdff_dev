all: CDFF build_cdff


dependencies: CDFF build_cdff
	echo "Installed dependencies."

CDFF:
	git clone --depth 1 git@gitlab.spaceapplications.com:InFuse/CDFF.git

build_cdff: autogeneration
	cd build_tools; ./install_externals.sh
	cd build_tools; ./get_cdff_artifacts.sh

autogeneration:
	cd CDFF/Tools/ASNtoC; ./GeneratorScript.sh || ./FetcherScript.sh

test:
	nosetests -sv
	nosetests -sv cdff_dev/cdff_types

.PHONY: test
