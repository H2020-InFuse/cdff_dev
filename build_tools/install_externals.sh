if [ -d "$DIRECTORY" ]; then
    cp -rf /CDFF/External/install ../CDFF/External/install
else
    echo "Currently there is no way to install dependencies automatically."
    echo "Please follow the instructions on the readme of CDFF:"
    echo ""
    echo "    https://gitlab.spaceapplications.com/InFuse/CDFF/tree/master#get-all-dependencies-first"
    echo ""
fi

