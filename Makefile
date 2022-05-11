.PHONY : env
env : 
		mamba env create -f environment.yml
		conda activate hw7

.PHONY : clean
clean: 
		rm -f figures/*.png


.PHONY : all
all: 
		jupyter execute income.ipynb
		jupyter execute eda_gender_explore.ipynb
		jupyter execute main.ipynb