import pandas as pd


def process_df(df, target_column):
	'''
	this function cleans and preps the df for the decision tree function
	variables must be categorical
	
	Input: df = dataframe with target column and categorical variable columns
	output:new datafame with one-hot encoded columns, target variable array
	'''
	if len(df.columns) < 2:
		return "incorrect format"
	newdf = df.drop(columns = ['age', 'fnlwgt','education-num','relationship', 'race', 'hours-per-week','capital-gain','capital-loss','native country'])
	target = newdf[target_column]
	newdf = newdf.drop(columns = [target_column])
	X = pd.get_dummies(newdf)

	return X, target