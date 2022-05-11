import matplotlib.pyplot as plt
import pandas as pd

"""
plotting function for all the gender exploration plots
plots sex by income, education, marital status, and workclass
"""
def plot_gender_eda(df):
	"""
	input: df = dataframe 
	output: 4 plots that also save as figures into figures/ folder
	columns should include: label, education, marital-status, workclass, and sex
	"""
	if 'label' not in df.columns:
		return 'missing column'
	if 'marital-status' not in df.columns:
		return 'missing column'
	if 'education' not in df.columns:
		return 'missing column'
	if 'workclass' not in df.columns:
		return 'missing column'
	
	plot1 = df.groupby(['label'])['sex'].value_counts().unstack(fill_value=0).plot(kind = 'bar', title = "Income by Gender")
	fig1 = plot1.get_figure()
	fig1.savefig("figures/gender_plot1.png")

	plot2 = df.groupby(['education'])['sex'].value_counts().unstack(fill_value=0).plot(kind = 'bar', title = "Education by Gender")
	fig2 = plot1.get_figure()
	fig2.savefig("figures/gender_plot2.png")
	
	plot3 = df.groupby(['marital-status'])['sex'].value_counts().unstack(fill_value=0).plot(kind = 'bar', title = "Marital Status by Gender")
	fig3 = plot3.get_figure()
	fig3.savefig("figures/gender_plot3.png")
	
	plot4 = df.groupby(['workclass'])['sex'].value_counts().unstack(fill_value=0).plot(kind = 'bar', title = "Working Class by Gender")
	fig4 = plot4.get_figure()
	fig4.savefig("figures/gender_plot4.png")
