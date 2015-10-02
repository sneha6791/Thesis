import string
import sys
import numpy as np
import math
from scipy import spatial,cluster
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn import metrics
from matplotlib import pyplot as plt
import copy



input_list = []
input_list2 = [] 
agent1_dict = {}
agent2_dict = {}


def discretize(l):#current measure: 75th percentile for the purposes of biclustering agent1
	val = math.ceil(np.percentile(l,75))
	for i in range(len(l)):
		if l[i] < val:
			l[i] = 0
		else:
			l[i] = 1
	return l


def biclustering(input_list,num_clusters):
	global agent1_dict
	#clustering agent 1
	data = np.matrix(input_list)
	#plot(data)#original data
	
	#model = SpectralBiclustering(n_clusters=num_clusters) #Biclustering refer http://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_biclustering.html#example-bicluster-plot-spectral-biclustering-py

	model = SpectralCoclustering(n_clusters=num_clusters,random_state=0) #Coclustering refer http://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_coclustering.html

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
	#clustering agent 2
	condensed_dist = cluster.hierarchy.distance.pdist(input_list2)
	z = cluster.hierarchy.linkage(condensed_dist)
	labels = cluster.hierarchy.fcluster(z,num_clusters,'maxclust')
	#create agent 2 dictionary
	agent2_dict = {}
	for l in range(len(labels)):
		agent2_dict.setdefault(labels[l],[]).append(l)
	dendro = cluster.hierarchy.dendrogram(z)
	plt.show() #clustered plot
	return agent2_dict
	
	
def plot(data_to_plot):
	fig = plt.figure()
	plt.imshow(data_to_plot, aspect='auto', cmap = plt.cm.Blues)
	plt.show()
	

def exchange(dict1,dict2,threshold):
	mappings = {}
	#print "agent2_key", "score", "cluster1", "cluster2", "common", "total"
        for key1,value1 in dict1.iteritems():
		#print key1
		#agent2keys = []
		max_score = -1
		max_key = -1
		for key2,value2 in dict2.iteritems():
			score = jaccard(value1,value2)  
			#print key2, score, len(value1),len(value2),len(set(value1).intersection(value2)),len(set(value1).union(value2))
			#find the max_jaccard score for each of the agent2 clusters
			if score>max_score:
				max_score = score
				max_key = key2
		#filter out those clusters that satisfy a score > threshold
		if max_score > threshold:
			#print key1,"---->",max_key," ",dict1[key1],"---->",dict2[max_key]
			mappings[key1] = max_key
	return mappings
	
def jaccard(cluster1,cluster2):
        inter = len(set(cluster1).intersection(cluster2))
        union = len(set(cluster1).union(cluster2))
	return round(float(inter)/union,4)
	

def calculate_reward(mapping_list):
	global input_list,input_list2,agent1_dict,agent2_dict
	agent1_rewards,agent2_rewards, agent1_uncommon, agent2_uncommon = {},{},{},{}
	#calcualting avg distance for the base input tables
	e2 = epsilon(input_list2) 
	e1 = epsilon(input_list)
	print 'avg dists agent1: ',e1,' agent2 ',e2
	for lhs,rhs in mapping_list.iteritems():
		sub_inputlist,sub_inputlist2 = {},{}#sub matrix dicts
		for x in agent2_dict[rhs]:
			sub_inputlist2[x] = input_list2[x] # sub-matrix of agent2's base table pertaining to agent1 cluster
		for y in agent1_dict[lhs]:
			sub_inputlist[y] = input_list[x]# sub-matrix of agent1's base table pertaining to agent2 cluster
		common = set(agent1_dict[lhs]).intersection(agent2_dict[rhs])
		temp1 = []
		temp2 = []
		l1,l2= [],[]
		
		for track1 in agent1_dict[lhs]:
			if track1 in common:
				temp1.append(trackReward(track1,input_list2[track1],sub_inputlist2,e2))
			else:
				temp1.append(0)
				l1.append(track1)
		agent1_rewards[lhs] = sum(temp1) #cluster1 rewards
		agent1_uncommon[lhs] = l1
		
		
		
		for track2 in agent2_dict[rhs]:
			if track2 in common:
				temp2.append(trackReward(track2,input_list[track2],sub_inputlist,e1))
			else:
				temp2.append(0)
				l2.append(track2)
		agent2_rewards[rhs] = sum(temp2) #cluster2 rewards
		agent2_uncommon[rhs] = l2
		print lhs,'-->',rhs,'  ',agent1_rewards[lhs],'-->',agent2_rewards[rhs]
	return agent1_rewards,agent2_rewards,agent1_uncommon,agent2_uncommon
	
def epsilon(basetable): #current measure: average of all track distances
	dist_matrix = spatial.distance.pdist(basetable,'euclidean')
	dist_matrix = spatial.distance.squareform(dist_matrix)
	return np.average(dist_matrix)	
	
def trackReward(t,v,sub_list,e):
	score = 0
	for key,val in sub_list.iteritems():
		if key!=t:
			dist = spatial.distance.euclidean(v,val)
			if dist<=e: #epsilon
				score = score+1
	return score+1 #k+1
	
	
def hypothesis1(uncommon,mappings,agent,cluster_reward,other_agent_reward):
        global agent1_dict,agent2_dict
		
	sub_inputlist,sub_inputlist2 = {},{}
	maxdictlist = []
	for lhs,rhs in mappings.iteritems():
		revised_reward = {}
		maxx = []
		maxdict = {}
		
		for x in agent2_dict[rhs]:
			sub_inputlist2[x] = input_list2[x] # sub-matrix of agent2's base table pertaining to agent1 cluster
		for y in agent1_dict[lhs]:
			sub_inputlist[y] = input_list[x]# sub-matrix of agent1's base table pertaining to agent2 cluster
		
		if agent == 1:
			for i in uncommon[lhs]:
				#print i
				revised_reward[i] = cluster_reward[lhs] + trackReward(i,input_list2[i],sub_inputlist2,epsilon(input_list2))+other_agent_reward[rhs]
			
		else:
			for i in uncommon[rhs]:
				#print i
				revised_reward[i] = cluster_reward[rhs] + trackReward(i,input_list[i],sub_inputlist,epsilon(input_list))+other_agent_reward[lhs]
		#print '\n',lhs,'-->',rhs,'\t',revised_reward
		maxx = getMax(revised_reward)
		for key in maxx:
			maxdict[key] = revised_reward[key]
		maxdictlist.append(maxdict)
	return dict((k,v) for d in maxdictlist for (k,v) in d.items())		
			 
def hypothesis2(uncommon,mappings,agent,cluster_reward,other_agent_reward):
        global agent1_dict,agent2_dict
		
	sub_inputlist,sub_inputlist2 = {},{}
	maxdictlist = []
	for lhs,rhs in mappings.iteritems():
		revised_reward = {}
		maxx = []
		maxdict = {}
		
		for x in agent2_dict[rhs]:
			sub_inputlist2[x] = input_list2[x] # sub-matrix of agent2's base table pertaining to agent1 cluster
		for y in agent1_dict[lhs]:
			sub_inputlist[y] = input_list[x]# sub-matrix of agent1's base table pertaining to agent2 cluster
		
		if agent == 1:
			for i in uncommon[lhs]:
				#print i
				revised_reward[i] = cluster_reward[lhs] - trackReward(i,input_list2[i],sub_inputlist2,epsilon(input_list2))+other_agent_reward[rhs]
			
		else:
			for i in uncommon[rhs]:
				#print i
				revised_reward[i] = cluster_reward[rhs] - trackReward(i,input_list[i],sub_inputlist,epsilon(input_list))+other_agent_reward[lhs]
		#print '\n',lhs,'-->',rhs,'\t',revised_reward
		maxx = getMax(revised_reward)
		for key in maxx:
			maxdict[key] = revised_reward[key]
		maxdictlist.append(maxdict)
	return dict((k,v) for d in maxdictlist for (k,v) in d.items())		
				
		


def getMax(d):
	maxx = max(d.values())
	return [x for x,y in d.items() if y==maxx]

def read_input():
	for line in file('default_features_1059_tracks.txt'):
		arr = line.split(',')
		templist = []
		templist2 = []
		for i in range(len(arr)-2):
			templist.append(float(arr[i]))
			templist = discretize(templist)
		templist2.append(float(arr[len(arr)-2]))
		templist2.append(float(arr[len(arr)-1]))
		input_list.append(templist)
		input_list2.append(templist2)
	return input_list,input_list2


def main():
	np.set_printoptions(threshold='nan') #to print the entire numpy array
	threshold = float(sys.argv[1])
	input_list,input_list2 = read_input()
	agent1_dict = biclustering(input_list,10) #numclusters
	agent2_dict = hierarchical(input_list2,10)
	'''mappings = exchange(agent1_dict,agent2_dict,threshold)#jaccard score threshold 
	agent1_rewards,agent2_rewards,agent1_uncommon,agent2_uncommon = calculate_reward(mappings)#euclidean dist epsilon value 
	
	##### NOTE: add reward(agent1 cluster) + reward(agent2 cluster) after hypothesis calculation #######

	#print 'initial agent1 agent2 rewards\n',agent1_rewards[0],'\t',agent2_rewards[25]
	print 'agent1 hypothesis1\n'
	agent1h1max = hypothesis1(agent1_uncommon,mappings,1,agent1_rewards,agent2_rewards)
	for k in sorted(agent1h1max, key=agent1h1max.get, reverse=True):
		print k,agent1h1max[k]
	print 'agent2 hypothesis1\n'
	agent2h1max = hypothesis1(agent2_uncommon,mappings,2,agent2_rewards,agent1_rewards)
	for k in sorted(agent2h1max, key=agent2h1max.get, reverse=True):
		print k,agent2h1max[k]
	##update maxdicts with R1 + R2 total rewards##
	print 'agent1 hypothesis2\n'
	agent1h2max = hypothesis2(agent1_uncommon,mappings,1,agent1_rewards,agent2_rewards)
	for k in sorted(agent1h2max, key=agent1h2max.get, reverse=True):
		print k,agent1h2max[k]
	print 'agent2 hypothesis2\n'
	agent2h2max = hypothesis2(agent2_uncommon,mappings,2,agent2_rewards,agent1_rewards)
	for k in sorted(agent2h2max, key=agent2h2max.get, reverse=True):
		print k,agent2h2max[k]'''
	
	
	

if __name__ == "__main__":
	main()

