from gender_functions import plot_gender_eda
import pandas as pd

def test():
	df = pd.DataFrame(data = [0,1,0,1,0,1], columns = ['test'])
	answer = plot_gender_eda(df)
	assert answer == 'missing column'