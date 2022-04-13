#!/bin/bash

# Usage: ./download.sh [tiered]
# note:  This file downloads the datasets from the google drive.
#        It only downloads CUB, cifar and miniImagenet by default.
#        If you need the tieredImagenet data, you should know that 
#        it is 82GB, and you need to add the tiered option to the command. 
#        The raw links are also included, so feel free to
#        download and untar the files yourself.

DATASETSDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $DATASETSDIR

# importing the gdluntar helper function from the utils directory
source ../utils/bashfuncs.sh

echo "Working on miniImagenet" && cd ${DATASETSDIR}/miniImagenet
FILEID="1FzafdVYo3hhT8qd7NCtimfcnVNu9DYgj"
FILENAME="miniImagenet_Data.tar.gz"
GDRIVEURL="https://drive.google.com/file/d/1FzafdVYo3hhT8qd7NCtimfcnVNu9DYgj/view?usp=sharing"
PTHMD5FILE="miniImagenet_Data.md5"
REMOVETARAFTERDL="1"
ISSMALL="0"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL} ${ISSMALL}
echo "----------------------------------------"

echo "Working on CUB" &&  cd ${DATASETSDIR}/CUB
FILEID="17VP9Gk55ZMAeW8HnlN-ipATt79HGu1YG"
FILENAME="CUB_Data.tar.gz"
GDRIVEURL="https://drive.google.com/file/d/17VP9Gk55ZMAeW8HnlN-ipATt79HGu1YG/view?usp=sharing"
PTHMD5FILE="CUB_Data.md5"
REMOVETARAFTERDL="1"
ISSMALL="0"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL} ${ISSMALL}
echo "----------------------------------------"

echo "Working on cifar" && cd ${DATASETSDIR}/cifar
FILEID="1E2Dq_0CL78iyOqyZlSxHI0XpxrqJr9bK"
FILENAME="cifar_Data.tar.gz"
GDRIVEURL="https://drive.google.com/file/d/1E2Dq_0CL78iyOqyZlSxHI0XpxrqJr9bK/view?usp=sharing"
PTHMD5FILE="cifar_Data.md5"
REMOVETARAFTERDL="1"
ISSMALL="0"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL} ${ISSMALL}
echo "----------------------------------------"

[[ $1 == "tiered" ]] && [[ -d  ${DATASETSDIR}/tieredImagenet/Data ]] && \
   echo "The ${DATASETSDIR}/tieredImagenet/Data directory already exists. " && \
   echo "Remove it if you want to re-download and extract the tieredImagenet data from scratch."

if [[ $1 == "tiered" ]] && [[ ! -d  ${DATASETSDIR}/tieredImagenet/Data ]]; then
  cd ${DATASETSDIR}/tieredImagenet
  # First we download the parts one-by-one without any untarring attempts.
  # This will take a while. 
  REMOVETARAFTERDL="0"
  
  echo "Checking (or downloading) the tieredImagenet tar ball parts..."
  
  FILEID="1hYVpu66LkrwFPrN66FM5q0YiQYcWvx28"
  FILENAME="tieredImagenet_Data.tar.gz_part_aa"
  GDRIVEURL="https://drive.google.com/file/d/1hYVpu66LkrwFPrN66FM5q0YiQYcWvx28/view?usp=sharing"
  PTHMD5FILE="tieredImagenet_aa.md5"
  ISSMALL="0"
  gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL} ${ISSMALL}

  FILEID="1-PV5-5VcSLKehpO356aLq9rXlI_0TShQ"
  FILENAME="tieredImagenet_Data.tar.gz_part_ab"
  GDRIVEURL="https://drive.google.com/file/d/1-PV5-5VcSLKehpO356aLq9rXlI_0TShQ/view?usp=sharing"
  PTHMD5FILE="tieredImagenet_ab.md5"
  ISSMALL="0"
  gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL} ${ISSMALL}

  FILEID="1rM3RZ899pMqkjhp3cjBPWqW_6AZY6T5X"
  FILENAME="tieredImagenet_Data.tar.gz_part_ac"
  GDRIVEURL="https://drive.google.com/file/d/1rM3RZ899pMqkjhp3cjBPWqW_6AZY6T5X/view?usp=sharing"
  PTHMD5FILE="tieredImagenet_ac.md5"
  gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL} ${ISSMALL}

  FILEID="14FV_B_q4BXVxvr9AQNEGPQsnUzfzDCbK"
  FILENAME="tieredImagenet_Data.tar.gz_part_ad"
  GDRIVEURL="https://drive.google.com/file/d/14FV_B_q4BXVxvr9AQNEGPQsnUzfzDCbK/view?usp=sharing"
  PTHMD5FILE="tieredImagenet_ad.md5"
  ISSMALL="0"
  gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL} ${ISSMALL}

  FILEID="1P-HLia25XdLIazMePw50xffem2FVs-3O"
  FILENAME="tieredImagenet_Data.tar.gz_part_ae"
  GDRIVEURL="https://drive.google.com/file/d/1P-HLia25XdLIazMePw50xffem2FVs-3O/view?usp=sharing"
  PTHMD5FILE="tieredImagenet_ae.md5"
  ISSMALL="0"
  gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL} ${ISSMALL}

  FILEID="1lP1LF-WhH2zly_S-QhhUag_ILXuXKHDg"
  FILENAME="tieredImagenet_Data.tar.gz_part_af"
  GDRIVEURL="https://drive.google.com/file/d/1lP1LF-WhH2zly_S-QhhUag_ILXuXKHDg/view?usp=sharing"
  PTHMD5FILE="tieredImagenet_af.md5"
  ISSMALL="0"
  gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL} ${ISSMALL}
  
  echo "Untarring the tieredImagenet tar ball parts..."
  echo '  + cat tieredImagenet_Data.tar.gz_part_* | tar -xzvf -'
  cat tieredImagenet_Data.tar.gz_part_aa \
      tieredImagenet_Data.tar.gz_part_ab \
      tieredImagenet_Data.tar.gz_part_ac \
      tieredImagenet_Data.tar.gz_part_ad \
      tieredImagenet_Data.tar.gz_part_ae \
      tieredImagenet_Data.tar.gz_part_af | tar -xzvf -
  
  echo '--> You can remove the downloaded tieredImagenet_Data.tar.gz_part_a* tar ball files '
  echo '    in the tieredImagenet directory to save some space!'
  echo '    Here is the template command if you are interested:'
  echo '    '
  echo "    rm ${DATASETSDIR}"'/tieredImagenet/tieredImagenet_Data.tar.gz_part_*'
  cd ..
fi

