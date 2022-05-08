import matplotlib.pyplot as plt
import pandas as pd


def plot_gender_eda(df):
	plot1 = df.groupby(['label'])['sex'].value_counts().unstack(fill_value=0).plot.bar()
	fig1 = plot1.get_figure()
	fig1.savefig("figures/gender_plot1.png")

	plot2 = df.groupby(['education'])['sex'].value_counts().unstack(fill_value=0).plot.bar()
	fig2 = plot1.get_figure()
	fig2.savefig("figures/gender_plot2.png")
	
	plot3 = df.groupby(['marital-status'])['sex'].value_counts().unstack(fill_value=0).plot.bar()
	fig3 = plot3.get_figure()
	fig3.savefig("figures/gender_plot3.png")
	
	plot4 = df.groupby(['workclass'])['sex'].value_counts().unstack(fill_value=0).plot.bar()
	fig4 = plot4.get_figure()
	fig4.savefig("figures/gender_plot4.png")
