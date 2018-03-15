#!/bin/bash
#xma@spaceapplications.com
#alexander.fabisch@dfki.de
#This file fetches the latest generated files directly from the build server

#exit immediately if a simple command exits with a nonzero exit value.
set -e

function get_source_function(){
    #Get working directory and script containing directory
    SOURCE="${BASH_SOURCE[0]}"
    while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
        DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
        SOURCE="$(readlink "$SOURCE")"
        [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
    done
    DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

    #output directory where the files will be uncompressed
    OUTPUT_DIR=$DIR"/../../Common/Types/C"

    if [ $# -eq 0 ]
    then
        echo "No arguments supplied, output directory will be :$OUTPUT_DIR"
    else
        OUTPUT_DIR=$1
    fi
}

function show_error_exit {
    echo "Could not retreive correct Artifacts for you."
    echo "Please run GeneratorScript.sh instead, or switch to a branch having successfully been build on the server (eg master)."
    exit -1
}

function download_artifact_function(){
    if [[ `wget -S --spider $1  2>&1 | grep 'HTTP/1.1 200 OK'` ]]; then
        echo "Fetching latest Artifacts for branch $branch_name."
        curl -o generatedFiles.gz -LOk -X GET --header "PRIVATE-TOKEN: pVUF6xEhoz2kgWAUyyCr" $1
    else
        show_error_exit
    fi
}

function unzip_function(){
    echo "Unzipping to $OUTPUT_DIR"
    unzip -joq generatedFiles.gz -d $OUTPUT_DIR
    rm generatedFiles.gz
    echo "Done."
}

get_source_function ../CDFF/build
branch_name=master
download_artifact_function https://gitlab.spaceapplications.com/InFuse/CDFF/-/jobs/artifacts/$branch_name/download?job=build
unzip_function

