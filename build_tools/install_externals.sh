#!/bin/bash
#alexander.fabisch@dfki.de
if [ -d "$DIRECTORY" ]; then
    cp -rf /CDFF/External/install ../CDFF/External/install
else
    cd ../CDFF/External/
    ./fetch_compile_install_dependencies.sh
fi

