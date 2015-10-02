import string
import sys
import numpy as np
import math
from scipy import spatial,cluster
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn import metrics
from sklearn import neighbors
from matplotlib import pyplot as plt
import copy
from sklearn.datasets import make_biclusters

N = 10 #Number of clusters
m = 3 #Number of nearest neighbors
w = 0.25 #weight factor

input_list_disc = []
input_list = []
input_list2 = [] 
agent1_dict = {}
agent2_dict = {}
track_rewards1 = {} 
track_rewards2 = {}
track_knn1 = {} 
track_knn2 = {}

def epsilon(basetable): #current measure: average of all track distances
	dist_matrix = spatial.distance.pdist(basetable,'euclidean')

	dist_matrix = spatial.distance.squareform(dist_matrix)
	return np.average(dist_matrix)	

def build_track_rewards():
	global input_list,input_list2,agent1_dict,track_rewards1,track_rewards2,agent2_dict,track_knn1,track_knn2
	cluster_table1,cluster_table2 = [],[]
	e1 = epsilon(input_list)
	e2 = epsilon(input_list2) 
	for cnum,tracklist in agent1_dict.iteritems():
		cluster_table1 = [input_list[t] for t in tracklist]
		fillknn(cluster_table1,tracklist,1)#build knns--'update' works for non overlapping keys
		for t in tracklist:
			temp_table1 = copy.copy(cluster_table1)
			temp_table1.remove(input_list[t])
			track_rewards1[t] = calc_trackReward(input_list[t],temp_table1,e1)
	for cnum,tracklist in agent2_dict.iteritems():
		cluster_table2 = [input_list2[t] for t in tracklist]
		fillknn(cluster_table2,tracklist,2)#build knns
		for t in tracklist:
			temp_table2 = copy.copy(cluster_table2)
			temp_table2.remove(input_list2[t])
			track_rewards2[t] = calc_trackReward(input_list2[t],temp_table2,e2)


def fillknn(cluster,tracklist,agent):
	global track_knn1,track_knn2
	#print cluster
	tree = spatial.KDTree(cluster)
	for i in range(len(cluster)):
		pos = tree.query(cluster[i],k=m+1,p=2)[1] #p=2 rep Euclidean; m set to m+1 as knn list includes the query track
		temp = list(pos) #unnecessarily redundant 'temp'
		#if i in temp:
		#	temp.remove(i)
		if(agent==1):           
			track_knn1[tracklist[i]] = [tracklist[p] for p in temp]
		else:
			track_knn2[tracklist[i]] = [tracklist[p] for p in temp]
		
def calc_trackReward(v,sub_list,e): #e is not the same for agent1 and agent2
	score = 0
	for record in sub_list:
		dist = spatial.distance.euclidean(v,record)
		if dist<=e: #epsilon
				score = score+1
	return score+1 #k+1


def totalPairReward():
	a1a2reward_dict = create_empty_rewardpair_dict()
	a2a1reward_dict = create_empty_rewardpair_dict()
	for i in range(N):
		for j in range(N):
			a1a2reward_dict[i][j] = calculate_pair_reward(i,j,1)
			a2a1reward_dict[i][j] = calculate_pair_reward(i,j,2)
	return a1a2reward_dict,a2a1reward_dict

def create_empty_rewardpair_dict():
	x = {}
	for i in range(N):
		x[i] = {}
	for k,v in x.iteritems():
		for i in range(N):
			v[i] = 0
	return x


#modification
def calculate_pair_reward(i,j,agent):
	global agent1_dict, agent2_dict,track_rewards1,track_rewards2,track_knn1,track_knn2 
	cluster_reward_list = []
	value = 0
	intersum = 0
	cluster_knnset = set([])
	not_in_knnset_sum = 0
	if(agent==1):
		for t in agent1_dict[i]:
			m_set = track_knn1[t]
			inter = set(m_set).intersection(set(agent2_dict[j]))
			cluster_knnset = inter.union(cluster_knnset) #knn list for the cluster
			if inter!=set([]):
				for i in inter:
					intersum += track_rewards2[i]
		not_in_knnset = set(agent2_dict[j]).difference(cluster_knnset)
		not_in_knnset_sum = sum(track_rewards2[i] for i in not_in_knnset)
		value = intersum - (w*not_in_knnset_sum)
	else:
		for t in agent2_dict[i]:
			m_set = track_knn2[t]
			inter = set(m_set).intersection(set(agent1_dict[j]))
			cluster_knnset = inter.union(cluster_knnset) #knn list for the cluster
			if inter!=set([]):
				for i in inter:
					intersum += track_rewards1[i]
		not_in_knnset = set(agent1_dict[j]).difference(cluster_knnset)
		not_in_knnset_sum = sum(track_rewards1[i] for i in not_in_knnset)
		value = intersum - (w*not_in_knnset_sum)
		
	return value

def biclustering(input,num_clusters):
	global agent1_dict
	data = np.asmatrix(input)
	model = SpectralCoclustering(n_clusters=num_clusters,random_state=0) 
	#model = SpectralBiclustering(n_clusters=num_clusters)
	model.fit(data)
	#create agent 1 dictionary
	agent1_dict = {}
	for c in range(num_clusters): 	
		agent1_dict[c] = model.get_indices(c)[0].tolist() #0 row indices, 1 column indices
	#fit_data = data[np.argsort(model.row_labels_)]
	#fit_data = fit_data[:, np.argsort(model.column_labels_)]
	#plot(fit_data)
	return agent1_dict

def hierarchical(input_list2,num_clusters):
	global agent2_dict	
	condensed_dist = cluster.hierarchy.distance.pdist(input_list2)
	z = cluster.hierarchy.linkage(condensed_dist)
	labels = cluster.hierarchy.fcluster(z,num_clusters,'maxclust')
	#create agent 2 dictionary
	agent2_dict = {}
	for l in range(len(labels)):
		agent2_dict.setdefault(labels[l]-1,[]).append(l)
	dendro = cluster.hierarchy.dendrogram(z,N,'level')
	#plt.show()
	return agent2_dict
		
def plot(data_to_plot):
	fig = plt.figure()
	plt.imshow(data_to_plot, aspect='auto', cmap = plt.cm.Blues)
	plt.show()

def read_input():
	global input_list,input_list2
	for line in file('default_features_1059_tracks.txt'):
		templist = []
		templist2 = []
		arr = line.strip().split(',')
		length = len(arr)
		for i in range(length-2):
			templist.append(float(arr[i]))
		templist2.append(float(arr[length-1]))
		templist2.append(float(arr[length-2]))
		low = min(templist)
		high = max(templist)
		data = np.array(templist)
		bins = np.linspace(low,high+1,10)
		digitized = np.digitize(data,bins)
		templist = list(digitized)
		input_list.append(templist)
		input_list2.append(templist2)
	return input_list,input_list2
		
def getMax(d):
	maxx = max(d.values())
	return [x for x,y in d.items() if y==maxx]

'''def display():
	for cnum,tracklist in agent1_dict.iteritems():
		for t in tracklist:
			print t,'\t',input_list[t],'\t',track_rewards1[t],'\t',track_knn1[t],'\t'
		print '\n' '''

def main():
	global input_list_disc,input_list,input_list2
	rewards = {}
	#np.set_printoptions(threshold='nan') #to print the entire numpy array
	#threshold = float(sys.argv[1])
	input_list,input_list2 = read_input()
	agent1_dict = biclustering(input_list,N)
	agent2_dict = hierarchical(input_list2,N)
	build_track_rewards()
	rewards = totalPairReward()
	print rewards
	
	
if __name__ == "__main__":
	main()


























	

