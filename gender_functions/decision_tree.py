from sklearn import tree
import matplotlib.pyplot as plt


def plot_tree(clf, target):
	fig = plt.figure(figsize=(25,20))
	tree_df = tree.plot_tree(clf, 
                   feature_names=clf.feature_names,  
                   class_names=target,
                   filled=True,fontsize=14)
	fig.savefig('figures/gender_decisiontree.png')
	