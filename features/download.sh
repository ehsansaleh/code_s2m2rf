#!/bin/bash

# Usage: ./download.sh
# note:  This file downloads the features from the google drive.
#        It only downloads the novel and val set features by default.
#        If you need the base features, you need to generate them yourself.

SCRIPTDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPTDIR

# importing the gdluntar helper function from the utils directory
source ../utils/bashfuncs.sh

FILEID="1Z-oO1hvcZkwsCZo9n7R3UGwHvFLi6Arf"
FILENAME="s2m2r_features.tar"
GDRIVEURL="https://drive.google.com/file/d/1Z-oO1hvcZkwsCZo9n7R3UGwHvFLi6Arf/view?usp=sharing"
PTHMD5FILE="s2m2r_features.md5"
REMOVETARAFTERDL="1"
ISSMALL="0"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL} ${ISSMALL}


