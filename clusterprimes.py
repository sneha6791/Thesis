#import string
#import sys

import numpy as np
#import math
from scipy import spatial,cluster
#from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.cluster.bicluster import SpectralCoclustering
#from sklearn import metrics
#from sklearn import neighbors
from matplotlib import pyplot as plt
from copy import deepcopy
#from sklearn.datasets import make_biclusters
#from pylab import *
import heapq

N = 15 #Number of clusters 10,15,20,25
m = 3 #Number of nearest neighbors
w = 0.25 #weight factor --modified to equal to 'm'

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
track_pairreward1 = {}
track_pairreward2 = {}
bestmatch_agent1 = {}
bestmatch_agent2 = {}

agent1_rejects = {}
agent2_rejects = {}
agent1_primes = {}
agent2_primes = {}

bm1 ={}
bm2 = {}
def epsilon(basetable): #current measure: average of all track distances
	dist_matrix = spatial.distance.pdist(basetable,'euclidean')
	dist_matrix = spatial.distance.squareform(dist_matrix)
	return np.average(dist_matrix)

def build_track_rewards():
	global input_list,input_list2,agent1_dict,track_rewards1,track_rewards2,agent2_dict,track_knn1,track_knn2
	cluster_table1,cluster_table2 = [],[]
	for cnum,tracklist in agent1_dict.iteritems():
		cluster_table1 = [input_list[t] for t in tracklist]
		fillknn(cluster_table1,tracklist,1)#build knns--'update' works for non overlapping keys
		e1 = epsilon(cluster_table1) #modified epsilon
		for t in tracklist:
			temp_table1 = deepcopy(cluster_table1)
			temp_table1.remove(input_list[t])
			track_rewards1[t] = calc_trackReward(input_list[t],temp_table1,e1)
	for cnum,tracklist in agent2_dict.iteritems():
		cluster_table2 = [input_list2[t] for t in tracklist]
		fillknn(cluster_table2,tracklist,2)#build knns
		e2 = epsilon(cluster_table2) #modified epsilon
		for t in tracklist:
			temp_table2 = deepcopy(cluster_table2)
			temp_table2.remove(input_list2[t])
			track_rewards2[t] = calc_trackReward(input_list2[t],temp_table2,e2)


def fillknn(cluster,tracklist,agent):
	global track_knn1,track_knn2
	tree = spatial.KDTree(cluster)
	for i in range(len(cluster)):
		pos = tree.query(cluster[i],k=m+1,p=2)[1] #p=2 rep Euclidean; m set to m+1 as knn list includes the query track
		if(agent==1):
			track_knn1[tracklist[i]] = [tracklist[p] for p in list(pos)]
		else:
			track_knn2[tracklist[i]] = [tracklist[p] for p in list(pos)]

def calc_trackReward(v,sub_list,e): #e is not the same for agent1 and agent2
	global m
	score = 0
	for record in sub_list:
		dist = spatial.distance.euclidean(v,record)
		if dist<=e: #epsilon
				score = score+1
	return score+1 #m+1

def scale_track_rewards():
    global track_rewards1,track_rewards2
    for cnum,tracklist in agent1_dict.iteritems():
            meritlist = [track_rewards1[t] for t in tracklist]
            maxval1 = max(meritlist)
            for t in tracklist:
                track_rewards1[t] = float(track_rewards1[t])/maxval1
    for cnum,tracklist in agent2_dict.iteritems():
            meritlist = [track_rewards2[t] for t in tracklist]
            maxval2 = max(meritlist)
            for t in tracklist:
                track_rewards2[t] = float(track_rewards2[t])/maxval2



def totalPairReward():
	global a1a2reward_dict,a2a1reward_dict,N,bestmatch_agent1,bestmatch_agent2,track_pairreward1,track_pairreward2
	a1a2reward_dict = create_empty_rewardpair_dict(N,N)
	a2a1reward_dict = create_empty_rewardpair_dict(N,N)
	track_pairreward1 = create_track_rewardpair_dict()#number of tracks
	track_pairreward2 = create_track_rewardpair_dict()#number of tracks
	for i in range(N):
		maxreward1 = 0
		maxreward2 = 0
		for j in range(N):
			a1a2reward_dict[i][j] = calculate_pair_reward(i,j,1)
			a2a1reward_dict[i][j] = calculate_pair_reward(i,j,2)
			if maxreward1 < a1a2reward_dict[i][j]:
				maxreward1 = a1a2reward_dict[i][j]
				maxcluster1 = j
			if maxreward2 < a2a1reward_dict[i][j]:
				maxreward2 = a2a1reward_dict[i][j]
				maxcluster2 = j
		bestmatch_agent1[i] = maxcluster1
		bestmatch_agent2[i] = maxcluster2
	return a1a2reward_dict,a2a1reward_dict,bestmatch_agent1,bestmatch_agent2


def calculate_pair_reward(i,j,agent):
    global track_a2reward,track_a1reward,track_pairreward2,track_pairreward2
    pair_reward = -1
    intersum = 0
    cluster_knnset = set([])
    #not_in_knnset_sum = 0
    clusterreward = 0
    if agent==1:
        for t in agent1_dict[i]:
            a2rewardsum = 0
            clusterreward += track_rewards1[t]
            m_set = track_knn1[t]
            inter = set(m_set).intersection(set(agent2_dict[j]))
            cluster_knnset = cluster_knnset.union(m_set)
            if inter!=set([]):
                for x in inter:
                    a2rewardsum += track_rewards2[x]
                intersum += a2rewardsum
            track_pairreward1[i][j][t] = a2rewardsum
    elif agent==2:
        for t in agent2_dict[i]:
            a1rewardsum = 0
            clusterreward += track_rewards2[t]
            m_set = track_knn2[t]
            inter = set(m_set).intersection(set(agent1_dict[j]))
            cluster_knnset = cluster_knnset.union(m_set)
            if inter!=set([]):
                for x in inter:
                   a1rewardsum += track_rewards1[x]
                intersum += a1rewardsum
            track_pairreward2[i][j][t] = a1rewardsum
    pair_reward = intersum
    #print track_rewards1,track_rewards2
    return pair_reward





def create_empty_rewardpair_dict(n,m):
	x = {}
	for i in range(n):
		x[i] = {}
	for k,v in x.iteritems():
		for i in range(m):
			v[i] = 0
	return x

def create_track_rewardpair_dict():
	x = {}
	for i in range(N):
		x[i] = {}
	for k,v in x.iteritems():
		for j in range(N):
			v[j] = {}
	return x


def biclustering(input,num_clusters):
	global agent1_dict
	data = np.asmatrix(input)
	model = SpectralCoclustering(n_clusters=num_clusters,random_state=0)
	'''model = SpectralBiclustering(n_clusters=num_clusters)'''
	model.fit(data)
	'''create agent 1 dictionary'''
	agent1_dict = {}
	for c in range(num_clusters):
		agent1_dict[c] = model.get_indices(c)[0].tolist() #0 row indices, 1 column indices
	'''fit_data = data[np.argsort(model.row_labels_)]
	fit_data = fit_data[:, np.argsort(model.column_labels_)]
	plot(fit_data)'''
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
	#dendro = cluster.hierarchy.dendrogram(z,N,'level')
	'''plt.show()'''
	return agent2_dict

def plot(data_to_plot):
	#fig = plt.figure()
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

def printresults(bestmatchpair,track_rewards_in_agent,track_rewards_in_other):
	for agent1_cluster,agent2_cluster in bestmatchpair.iteritems():
		for t,r in track_rewards_in_other[agent1_cluster][agent2_cluster].iteritems():
			print agent1_cluster,'\t',agent2_cluster,'\t',t,'\t',track_rewards_in_agent[t],'\t',r

def knnsum(agent):
	knnsum = {}
	if agent==1:
		for t,nlist in track_knn1.iteritems():
			knnsum[t] = sum(track_rewards1[n] for n in nlist)
	elif agent==2:
		for t,nlist in track_knn2.iteritems():
			knnsum[t] = sum(track_rewards2[n] for n in nlist)
	return knnsum

def print_best_pairs(bm):
	for agent1,agent2 in bm.iteritems():
		print agent1,'-',agent2

def print_detail_a1(c1,c2):
	for t in agent1_dict[c1]:
		print c1,'\t',t,'\t',track_pairreward1[c1][c2][t],'\t',c2

'''def meanstddev(a1,a2,track_pairreward):
	mean = np.mean([x for x in track_pairreward[a1][a2].values() if x!=0])
	stddev = np.std(track_pairreward[a1][a2].values())
	print a1,' ',a2,' ',mean,' ',stddev
	for k,v in track_pairreward[a1][a2].iteritems():
		print k,' ',v,
		if v < (mean - stddev):
			print ' ','x'
		else: print '' '''

def meanstddev(a1,a2,track_pairreward,agent):
	global agent1_rejects,agent2_rejects
	l = []
	mean = np.mean([x for x in track_pairreward[a1][a2].values() if x!=0])
	stddev = np.std([x for x in track_pairreward[a1][a2].values() if x!=0])
	#stddev = np.std(track_pairreward[a1][a2].values())
	#print a1,' ',a2,' ',mean,' ',stddev
	for k,v in track_pairreward[a1][a2].iteritems():
		if v < (mean - stddev):
			l.append(k)
	if agent == 1:	agent1_rejects[a1] = l
	else:	agent2_rejects[a1] = l

def clusterprimes(agent):
	global agent1_primes,agent2_primes
	if agent == 1:
		for cnum,tracklist in agent1_dict.iteritems():
			primes = [x for x in tracklist if x not in agent1_rejects[cnum]]
			agent1_primes[cnum] = primes
	elif agent==2:
		for cnum,tracklist in agent2_dict.iteritems():
			primes = [x for x in tracklist if x not in agent2_rejects[cnum]]
			agent2_primes[cnum] = primes

def pairreward_upd(cfrom,cto,afrom,ato):
    if afrom==1 and ato==2:
        agentf_dict = agent1_primes
        track_knn = track_knn1
        agentt_dict = agent2_dict
        track_rewards = track_rewards2
    elif afrom==2 and ato==1:
        agentf_dict = agent1_primes
        track_knn = track_knn1
        agentt_dict = agent2_dict
        track_rewards = track_rewards2
    else: return -1
    pair_reward = 0
    for t in agentf_dict[cfrom]:
        kset = track_knn[t]
        inter = set(kset).intersection(agentt_dict[cto])
        if inter!=set([]):
            for x in inter:
                pair_reward += track_rewards[x]
    return pair_reward

def bland_altman_plot():
    #c = 10 #cluster number
    for c in range(N):
        bm = bm1[c] #best match agent 2 cluster
        meritlist = [track_rewards1[t] for t in agent1_dict[c]]
        rewardlist = [float(track_pairreward1[c][bm][t])/(m+1) for t in agent1_dict[c]]
        x = np.matrix(rewardlist)
        y =np.matrix(meritlist)
        plt.plot(x+y,x-y,'ro')
        plt.ylabel('Reward-Merit')
        plt.xlabel('Reward+Merit')
        plt.title(c)
        plt.show()



def main():
    global input_list_disc,input_list,input_list2,a1a2reward_dict,a2a1reward_dict,agent1_primes,agent2_primes,agent1_dict,agent2_dict,bm1,bm2
    global track_rewards1,track_rewards2
    knnsum_1,knnsum_2 = {},{}
    input_list,input_list2 = read_input()
    agent1_dict = biclustering(input_list,N)
    agent2_dict = hierarchical(input_list2,N)
    build_track_rewards()
    scale_track_rewards()
    a1a2reward_dict,a2a1reward_dict,bm1,bm2 = totalPairReward()
    bland_altman_plot()
    #knnsum1 = knnsum(1)
    #knnsum2 = knnsum(2)
    #printresults(bm1,knnsum1,track_pairreward1)
    #printresults(bm2,knnsum2,track_pairreward2)
    #print_best_pairs(bm1)
    #print_best_pairs(bm2)
    '''
    for k,v in bm1.iteritems():
        meanstddev(k,v,track_pairreward1,1)
    for k,v in bm2.iteritems():
        meanstddev(k,v,track_pairreward2,2)
    #print agent1_rejects,'\n',agent2_rejects
    clusterprimes(1)
    clusterprimes(2)
    '''
    '''
    for cnum,tracklist in agent1_dict.iteritems():
        bm = bm1[cnum]
        for t in tracklist:
            print cnum,'\t',t,'\t',track_rewards1[t],'\t',track_pairreward1[cnum][bm][t],'\t',bm
    '''
    '''
    for c1,c2 in bm1.iteritems():
        print c1,'\t',c2

    for cnum,tracklist in agent1_dict.iteritems():
        for t in tracklist:
            print cnum,'\t',t,'\t',track_rewards1[t]
    '''

    '''for cnum,tr in agent1_rejects.iteritems():
		for t in tr:
			b = bm1[cnum]
			old_rewd = pairreward_upd(cnum,b,1,2)
			agent1_primes[cnum].append(t)
			new_rewd = pairreward_upd(cnum,b,1,2)
			print cnum,'\t',t,'\t',new_rewd,'\t',old_rewd
			if old_rewd < new_rewd: print 'Increase','\t',new_rewd-old_rewd
			elif old_rewd==new_rewd:print 'No change'
			else: print 'Decrease','\t',old_rewd-new_rewd
			agent1_primes[cnum].remove(t)

		print '\n' '''

    '''for cnum in range(N):
            for t in agent1_rejects[cnum]:
                    kset = track_knn1[t]
                    inter = set(kset).intersection(agent2_dict[bm1[cnum]])
                    print cnum,'\t',bm1[cnum],'\t',t,'\t',inter
    for cnum in range(N):
            for t in agent1_primes[cnum]:
                    kset = track_knn1[t]
                    inter = set(kset).intersection(agent2_dict[bm1[cnum]])
                    print cnum,'\t',bm1[cnum],'\t',t,'\t',inter,'\t',track_rewards1[t],'\t',track_pairreward1[cnum][bm1[cnum]][t]'''


if __name__ == "__main__":
	main()

