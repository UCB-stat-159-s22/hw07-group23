from gender_functions import process_df

import numpy as np
import pandas as pd



def test_df():
	df = pd.DataFrame(data = [0,1,0,1,0,1], columns = ['test'])
	answer = process_df(df, 'test')
	assert answer == "incorrect format"
