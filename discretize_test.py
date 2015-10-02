import math
import numpy as np
from sklearn.cluster.bicluster import SpectralCoclustering
from matplotlib import pyplot as plt

N= 10
input_list = []
agent1_dict = {}

def biclustering(inputl,num_clusters):
	global agent1_dict
	data = np.matrix(inputl)
	model = SpectralCoclustering(n_clusters=num_clusters,random_state=0) 
	#model = SpectralBiclustering(n_clusters=num_clusters)
	model.fit(data)
	#create agent 1 dictionary
	for c in range(num_clusters): 	
		agent1_dict[c] = model.get_indices(c)[0].tolist() #0 row indices, 1 column indices
	fit_data = data[np.argsort(model.row_labels_)]
	fit_data = fit_data[:, np.argsort(model.column_labels_)]
	plot(fit_data)
	return agent1_dict

		
def plot(data_to_plot):
	fig = plt.figure()
	plt.imshow(data_to_plot, aspect='auto', cmap = plt.cm.Blues)
	plt.show()

def read_input():
	global input_list
	for line in file('default_features_1059_tracks.txt'):
		templist = []
		arr = line.strip().split(',')
		for i in range(len(arr)-2):
			templist.append(float(arr[i]))
		#print templist
		low = min(templist)
		high = max(templist)
		data = np.array(templist)
		bins = np.linspace(low,high+1,10)
		#print bins
		digitized = np.digitize(data,bins)
		#print digitized
		#bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]				
		#print bin_means
		templist = list(digitized)
		input_list.append(templist)
	return input_list
		
	
def main():
	input_list = read_input()
	print biclustering(input_list,10)

if __name__ == "__main__":
	main()

