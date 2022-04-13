#!/bin/bash

# Usage: ./download.sh
# note:  This file downloads the pre-trained backbone cehckpoint from 
#        the google drive. These were provided by the S2M2 authors at 
#        https://drive.google.com/open?id=1S-t56H8YWzMn3sjemBcwMtGuuUxZnvb_
#        This script only automates the downloading, and the placement process.

SCRIPTDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPTDIR

# importing the gdluntar helper function from the utils directory
source ../utils/bashfuncs.sh

FILEID="1uqE614-xsTd8nBrfYAi5iHz02DpPwm4U"
FILENAME="cifar_model.tar.gz"
GDRIVEURL="https://drive.google.com/file/d/1uqE614-xsTd8nBrfYAi5iHz02DpPwm4U/view?usp=sharing"
PTHMD5FILE="cifar.md5"
REMOVETARAFTERDL="1"
ISSMALL="0"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL} ${ISSMALL}

FILEID="1xaB9tN6jvM-FfrMzOK_yq849L8XhSZ2b"
FILENAME="CUB_model.tar.gz"
GDRIVEURL="https://drive.google.com/file/d/1xaB9tN6jvM-FfrMzOK_yq849L8XhSZ2b/view?usp=sharing"
PTHMD5FILE="CUB.md5"
REMOVETARAFTERDL="1"
ISSMALL="0"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL} ${ISSMALL}

FILEID="15x4Z1y2KlEbFA8kM427sAEfWw40akFNf"
FILENAME="miniImagenet_model.tar.gz"
GDRIVEURL="https://drive.google.com/file/d/15x4Z1y2KlEbFA8kM427sAEfWw40akFNf/view?usp=sharing"
PTHMD5FILE="miniImagenet.md5"
REMOVETARAFTERDL="1"
ISSMALL="0"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL} ${ISSMALL}

INNERDIR=${SCRIPTDIR}"/tieredImagenet/WideResNet28_10_S2M2_R"
mkdir -p ${INNERDIR}
FILEID="1Sojs77vt_ziQqq2IgA34d9bDO-OlKNfq"
FILENAME="${INNERDIR}/tiered_s2m2_199.tar.pth"
GDRIVEURL="https://drive.google.com/file/d/1Sojs77vt_ziQqq2IgA34d9bDO-OlKNfq/view?usp=sharing"
PTHMD5FILE="tieredImagenet.md5"
REMOVETARAFTERDL="0"
ISSMALL="0"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL} ${ISSMALL}
[[ -f ${FILENAME} ]] && mv ${FILENAME} ${INNERDIR}/199.tar


