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
track_rewards1 = {} #track_rewards2 = {}
track_knn1 = {} #track_knn2 = {}

def epsilon(basetable): #current measure: average of all track distances
	dist_matrix = spatial.distance.pdist(basetable,'euclidean')
	dist_matrix = spatial.distance.squareform(dist_matrix)
	return np.average(dist_matrix)	

def build_track_rewards():
	global input_list,input_list2,agent1_dict, track_rewards1#,agent2_dict
	cluster_table1,cluster_table2 = [],[]
	e1 = epsilon(input_list)
	e2 = epsilon(input_list2) 
	for cnum,tracklist in agent1_dict.iteritems():
		cluster_table1 = [input_list[t] for t in tracklist]
		fillknn(cluster_table1,tracklist)#build knns
		for t in tracklist:
			cluster_table1.remove(input_list[t])
			track_rewards1[t] = calc_trackReward(input_list[t],cluster_table1,e1)

def fillknn(cluster,tracklist):
	global track_knn1
	#print cluster
	tree = spatial.KDTree(cluster)
	for i in range(len(cluster)):
		pos = tree.query(cluster[i],k=m+1,p=2)[1] #p=2 rep Euclidean; m set to m+1 as knn list includes the query track
		temp = list(pos) #unnecessarily redundant 'temp'
		temp.remove(i)           
		track_knn1[tracklist[i]] = [tracklist[p] for p in temp]

##################STOPPED HERE######################################################### sub_list not a dict
def calc_trackReward(v,sub_list,e): #e is not the same for agent1 and agent2
	score = 0
	for record in sub_list:
		dist = spatial.distance.euclidean(v,record)
		if dist<=e: #epsilon
				score = score+1
	return score+1 #k+1


def totalPairReward():
	a1a2reward_dict = create_empty_rewardpair_dict()
	for i in range(10):
		for j in range(1,11):
			a1a2reward_dict[i][j] = calculate_pair_reward(i,j)
	return a1a2reward_dict

def create_empty_rewardpair_dict():
	x = {}
	for i in range(10):
		x[i] = {}
	for k,v in x.iteritems():
		for i in range(1,11):
			v[i] = []
	return x


def calculate_pair_reward(i,j):
	global agent1_dict, agent2_dict 
	cluster_reward_list = []
	for t in agent1_dict[i]:
		value = 0
		intersum = 0
		remsum = 0
		m_set = track_knn1[t]
		inter = set(m_set).intersection(set(agent2_dict[j]))
		if inter!=set([]):
			for i in inter:
				intersum += track_rewards1[i]
			for rem in set(agent2_dict).difference(inter):
				remsum += track_rewards1[rem]
			value = intersum - (w*remsum)
		cluster_reward_list.append(value)
	return cluster_reward_list	


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
	fit_data = data[np.argsort(model.row_labels_)]
	fit_data = fit_data[:, np.argsort(model.column_labels_)]
	plot(fit_data)
	return agent1_dict
	
def hierarchical(input_list2,num_clusters):
	global agent2_dict	
	condensed_dist = cluster.hierarchy.distance.pdist(input_list2)
	z = cluster.hierarchy.linkage(condensed_dist)
	labels = cluster.hierarchy.fcluster(z,num_clusters,'maxclust')
	#create agent 2 dictionary
	agent2_dict = {}
	for l in range(len(labels)):
		agent2_dict.setdefault(labels[l],[]).append(l)
	dendro = cluster.hierarchy.dendrogram(z)
	return agent2_dict
		
def plot(data_to_plot):
	fig = plt.figure()
	plt.imshow(data_to_plot, aspect='auto', cmap = plt.cm.Blues)
	plt.show()
	
def read_input():
	for line in file('default_features_1059_tracks.txt'):
		arr = line.split(',')
		#print arr,'\n'
		templist = []
		templist2 = []
		for i in range(len(arr)-2):
			#print float(arr[i])
			templist.append(float(arr[i]))
			templist_normal = list(templist)
			templist_disc = discretize(templist) ###LOOKINTO!!!
		#templist_disc = discretize(templist)### Error when statement is present here
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
	#np.set_printoptions(threshold='nan') #to print the entire numpy array
	#threshold = float(sys.argv[1])
	input_list_disc,input_list2,input_list = read_input()
	agent1_dict = biclustering(input_list_disc,N)
	agent2_dict = hierarchical(input_list2,N)
	build_track_rewards()
	#print agent2_dict
	print totalPairReward()
	 
	
if __name__ == "__main__":
	main()

