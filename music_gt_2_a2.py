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

N = 10 #Number of clusters
m = 3 #Number of nearest neighbors
w = 0.25 #weight factor
input_list_disc = []
input_list = []
input_list2 = [] 
agent1_dict = {}
agent2_dict = {}
#track_rewards1 = {} #track_rewards2 = {}
track_rewards2 = {}
#track_knn1 = {} 
track_knn2 = {}

def epsilon(basetable): #current measure: average of all track distances
	dist_matrix = spatial.distance.pdist(basetable,'euclidean')
	dist_matrix = spatial.distance.squareform(dist_matrix)
	return np.average(dist_matrix)	

def build_track_rewards():
	global input_list,input_list2,agent1_dict, track_rewards2,agent2_dict
	cluster_table1,cluster_table2 = [],[]
	e1 = epsilon(input_list)
	e2 = epsilon(input_list2)
	for cnum,tracklist in agent2_dict.iteritems():
		cluster_table2 = [input_list2[t] for t in tracklist]
		fillknn(cluster_table2,tracklist)#build knns
		for t in tracklist:
			cluster_table2.remove(input_list2[t])
			track_rewards2[t] = calc_trackReward(input_list2[t],cluster_table2,e2)

def fillknn(cluster,tracklist):
	global track_knn2
	#print cluster
	tree = spatial.KDTree(cluster)
	for i in range(len(cluster)):
		print cluster[i],' ',
		pos = tree.query(cluster[i],k=m+1,p=2)[1] #p=2 rep Euclidean; m set to m+1 as knn list includes the query track
		print i,' ',pos
		temp = list(pos) #unnecessarily redundant 'temp'
		#temp.remove(i)           
		track_knn2[tracklist[i]] = [tracklist[p] for p in temp]

def calc_trackReward(v,sub_list,e): #e is not the same for agent1 and agent2
	score = 0
	for record in sub_list:
		dist = spatial.distance.euclidean(v,record)
		if dist<=e: #epsilon
				score = score+1
	return score+1 #k+1


def totalPairReward():
	a2a1reward_dict = create_empty_rewardpair_dict()
	for i in range(N):
		for j in range(N):
			a2a1reward_dict[i][j] = calculate_pair_reward(i,j)
	return a2a1reward_dict

def create_empty_rewardpair_dict():
	x = {}
	for i in range(N):
		x[i] = {}
	for k,v in x.iteritems():
		for i in range(N):
			v[i] = 0
	return x


#modification
def calculate_pair_reward(i,j):
	global agent1_dict, agent2_dict 
	cluster_reward_list = []
	value = 0
	intersum = 0
	cluster_knnset = set([])
	not_in_knnset_sum = 0
	for t in agent2_dict[i]:
		m_set = track_knn2[t]
		inter = set(m_set).intersection(set(agent1_dict[j]))
		cluster_knnset = inter.union(cluster_knnset) #knn list for the cluster
		if inter!=set([]):
			for i in inter:
				intersum += track_rewards2[i]#accumulated m-set merit sum in Agent2 cluster
	not_in_knnset = set(agent1_dict[j]).difference(cluster_knnset)
	not_in_knnset_sum = sum(track_rewards2[i] for i in not_in_knnset)
	value = intersum - (w*not_in_knnset_sum)
	return value


def discretize(l):#current measure: 75th percentile for the purposes of biclustering agent1
	val = math.ceil(np.percentile(l,75))
	for i in range(len(l)):
		if l[i] < val:
			l[i] = 0
		else:
			l[i] = 1
	return l

def biclustering(input,num_clusters):
	global agent1_dict
	data = np.matrix(input)
	model = SpectralCoclustering(n_clusters=num_clusters,random_state=0) 
	model.fit(data)
	#create agent 1 dictionary
	agent1_dict = {}
	for c in range(num_clusters): 	
		agent1_dict[c] = model.get_indices(c)[0].tolist() #0 row indices, 1 column indices
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
	dendro = cluster.hierarchy.dendrogram(z)
	return agent2_dict
		
def plot(data_to_plot):
	fig = plt.figure()
	plt.imshow(data_to_plot, aspect='auto', cmap = plt.cm.Blues)
	plt.show()
	
def read_input():
	for line in file('default_features_1059_tracks.txt'):
		arr = line.split(',')
		templist = []
		templist2 = []
		for i in range(len(arr)-2):
			templist.append(float(arr[i]))
			templist_normal = list(templist)
			templist_disc = discretize(templist)
		templist2.append(float(arr[len(arr)-2]))
		templist2.append(float(arr[len(arr)-1]))		
		input_list.append(templist_normal)
		input_list_disc.append(templist_disc)
		input_list2.append(templist2)
	return input_list_disc,input_list2,input_list

def getMax(d):
	maxx = max(d.values())
	return [x for x,y in d.items() if y==maxx]

def main():
	global input_list_disc,input_list,input_list2
	input_list_disc,input_list2,input_list = read_input()
	agent1_dict = biclustering(input_list_disc,N)
	agent2_dict = hierarchical(input_list2,N)
	build_track_rewards()
	print totalPairReward()
	 
	
if __name__ == "__main__":
	main()

