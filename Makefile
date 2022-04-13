figures: plot_dacc_vs_nways
.PHONY: summary venv figures tables filelists

SHELL := /bin/bash
PROJBASE := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
MPIPRECMD := $(shell command -v mpirun >/dev/null 2>&1 && echo "mpirun -n 10")

##########################################################
####################      VENV     #######################
##########################################################
venv:
	python -m venv venv
	source venv/bin/activate && python -m pip install --upgrade pip
	source venv/bin/activate && python -m pip install torch==1.8.2+cu102 \
		torchvision==0.9.2+cu102 \
		-f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
	source venv/bin/activate && python -m pip install -r requirements.txt

##########################################################
###########      Summarizing CSV Results     #############
##########################################################
summary:
	source .env.sh && cd utils && ${MPIPRECMD} python csv2summ.py

tables:
	source .env.sh && cd utils && python summ2tbls.py

##########################################################
####################   Filelists   #######################
##########################################################
filelists:
	source .env.sh && cd filelists &&  python json_maker.py

##########################################################
####################     Figures     #####################
##########################################################
plot_dacc_vs_nways:
	source .env.sh && cd utils && python plot_dacc_vs_nways.py

##########################################################
#####################     Clean     ######################
##########################################################
fix_crlf:
	find ${PROJBASE} -maxdepth 3 -type f -name "*.md5" \
	  -o -name "*.py" -o -name "*.sh" -o -name "*.json" | xargs dos2unix

clean:
	@echo Nothing to clean
