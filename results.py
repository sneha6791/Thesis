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
from copy import deepcopy
from sklearn.datasets import make_biclusters
from pylab import *

N = 25 #Number of clusters
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
track_a2reward = {}
track_a1reward = {}
a1a2reward_dict = {}
a2a1reward_dict = {}

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
			temp_table1 = deepcopy(cluster_table1)
			temp_table1.remove(input_list[t])
			track_rewards1[t] = calc_trackReward(input_list[t],temp_table1,e1)
	for cnum,tracklist in agent2_dict.iteritems():
		cluster_table2 = [input_list2[t] for t in tracklist]
		fillknn(cluster_table2,tracklist,2)#build knns
		for t in tracklist:
			temp_table2 = deepcopy(cluster_table2)
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
	global a1a2reward_dict,a2a1reward_dict,N
	a1a2reward_dict = create_empty_rewardpair_dict()
	a2a1reward_dict = create_empty_rewardpair_dict()
	for i in range(N):
		for j in range(N):
			a1a2reward_dict[i][j] = calculate_pair_reward(i,j,1)
			a2a1reward_dict[i][j] = calculate_pair_reward(i,j,2)
	return a1a2reward_dict, a2a1reward_dict

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
	global agent1_dict, agent2_dict,track_rewards1,track_rewards2,track_knn1,track_knn2,track_a2reward,track_a1reward
	value = 0
	intersum = 0
	cluster_knnset = set([])
	not_in_knnset_sum = 0
	clusterreward = 0
	if(agent==1):
		print i,'\t',j,
		for t in agent1_dict[i]:
			a2rewardsum = 0
			clusterreward += track_rewards1[t]
			m_set = track_knn1[t]
			inter = set(m_set).intersection(set(agent2_dict[j]))
			cluster_knnset = cluster_knnset.union(m_set)	#inter?		
			if inter!=set([]):
				for x in inter:
					a2rewardsum += track_rewards2[x]
				intersum += a2rewardsum
			track_a2reward[t] = a2rewardsum
			#print t,'\t',track_rewards1[t],'\t',track_knn1[t],'\t',inter,'\t',a2rewardsum,'\t'
		
		not_in_knnset = set(agent2_dict[j]).difference(cluster_knnset)
		not_in_knnset_sum = sum(track_rewards2[y] for y in not_in_knnset)
		value = intersum - (w*not_in_knnset_sum)
		print '\t',clusterreward,'\t',value
		#print clusterreward,'\t',value
	else:
		print i,'\t',j,
		for t in agent2_dict[i]:
			a1rewardsum = 0
			clusterreward += track_rewards2[t]
			m_set = track_knn2[t]
			inter = set(m_set).intersection(set(agent1_dict[j]))
			cluster_knnset = cluster_knnset.union(m_set)	#inter?			
			if inter!=set([]):
				for x in inter:
					a1rewardsum += track_rewards1[x]
				intersum += a1rewardsum
			track_a1reward[t] = a1rewardsum
			#print t,'\t',track_rewards2[t],'\t',track_knn2[t],'\t',inter,'\t',a1rewardsum,'\t'
		not_in_knnset = set(agent1_dict[j]).difference(cluster_knnset)
		not_in_knnset_sum = sum(track_rewards1[y] for y in not_in_knnset)
		value = intersum - (w*not_in_knnset_sum)
		print '\t',clusterreward,'\t',value
		#print i,'\t',j,'\t',clusterreward,'\t',value
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

def display(agent):
	if agent==1:
		for k,v in agent1_dict.iteritems():
				for x,y in agent2_dict.iteritems():
					for j in v:
						for i in y:
							print k,'\t',j,'\t',track_rewards1[j],'\t',track_knn1[j],'\t',x,'\t',i,'\t',track_rewards2[i],'\t',track_a2reward[j],'\t',a1a2reward_dict[k][x]
	else:
		for k,v in agent2_dict.iteritems():
				for x,y in agent1_dict.iteritems():
					for j in v:
						for i in y:
							print k,'\t',j,'\t',track_rewards2[j],'\t',track_knn2[j],'\t',x,'\t',i,'\t',track_rewards1[i],'\t',track_a1reward[j],'\t',a2a1reward_dict[k][x]

def displaya2(n):
	global input_list2
	a2 = biclustering(input_list2,n)
	for k,v in a2.iteritems():
		x = []
		y = []
		print k
		for i in v:
			x.append(input_list2[i][0])
			y.append(input_list2[i][1])
		print x
		print y
		print '\n'

def plot_a2_clusters():
	#plotting clustered data
	cmap = plt.get_cmap('gnuplot')
	#colors = [cmap(i) for i in np.linspace(0, 1, N)]
	colors = ['bo','ro','go','bs','rs','gs','b^','r^','g^','co','mo','ko','c^','m^','k^','cs','ks','ms','ys','y^','r+','b+','g+','c+','k+','r*','b*','g*','c*','k*']
	fig, ax = plt.subplots()
	grid(True)
	i = 0
	for key,value in agent2_dict.iteritems():
		p,q = [],[]
		for v in value:
			p.append(input_list2[v][0])
			q.append(input_list2[v][1])
		plt.plot(p, q, colors[i])
		for X, Y in zip(p,q):
			ax.annotate('{}'.format(key), fontsize = 10, fontstyle = 'oblique', xy=(X,Y),xytext =(-8,8),textcoords='offset points')
		i = i+1
	plt.show()
	

def main():
	global input_list_disc,input_list,input_list2,a1a2reward_dict,a2a1reward_dict
	#np.set_printoptions(threshold='nan') #to print the entire numpy array
	#threshold = float(sys.argv[1])
	input_list,input_list2 = read_input()
	agent1_dict = biclustering(input_list,N)
	agent2_dict = hierarchical(input_list2,N)
	#displaya2(10)
	build_track_rewards()
	a1a2reward_dict,a2a1reward_dict = totalPairReward()
	#plot_a2_clusters()
	
		
if __name__ == "__main__":
	main()
