#!/bin/bash

# Usage: ./download.sh
# note:  This file downloads the filelist templates from the google drive.

SCRIPTDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPTDIR

# importing the gdluntar helper function from the utils directory
source ../utils/bashfuncs.sh

FILEID="1AnakHzG3tf-ijT8udNDaqx_nbJRKflw6"
FILENAME="filelists.tar"
GDRIVEURL="https://drive.google.com/file/d/1AnakHzG3tf-ijT8udNDaqx_nbJRKflw6/view?usp=sharing"
PTHMD5FILE="filelists.md5"
REMOVETARAFTERDL="1"
ISSMALL="1"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL} ${ISSMALL}

echo "Now you need to run python json_maker.py to generate your filelists from the templates..."
