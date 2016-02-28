# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 20:52:10 2016

@author: dinesh
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 14:45:48 2016

@author: sneha
"""

import numpy as np
import scipy as sp
from scipy import spatial,cluster
#from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.cluster.bicluster import SpectralCoclustering
from matplotlib import pyplot as plt
from copy import deepcopy
from sklearn.decomposition import RandomizedPCA

N = 33 #Number of clusters 10,15,20,25
m = 4 #Number of nearest neighbors 3,4,5,7

inputlist1=[]
input_list_disc = []
input_list = []
input_list2 = []
agent1_dict = {}
agent2_dict = {}
track_merits1 = {}
track_merits2 = {}
track_knn1 = {}
track_knn2 = {}
a1a2reward_dict = {}
a2a1reward_dict = {}
track_pairreward1 = {}
track_pairreward2 = {}
bestmatch_agent1 = {}
bestmatch_agent2 = {}
bm1 = {}
bm2 = {}
agent1_dict_temp,agent2_dict_temp = {},{}
avgdist_agent1_clusters = {}
avgdist_agent2_clusters = {}
it  =0
X = []
e1,e2 = -1,-1
ctr = 0


def getMax(d):
	maxx = max(d.values())
	return [x for x,y in d.items() if y==maxx]

def bland_altman_plot():
    for c in range(N):
        bm = bm1[c]
        meritlist = [track_merits1[t] for t in agent1_dict[c]]
        rewardlist = [float(track_pairreward1[c][bm][t])/(m+1) for t in agent1_dict[c]]
        print c,'\t',meritlist
        print c,'\t',rewardlist
        x = np.matrix(rewardlist)
        y =np.matrix(meritlist)
        plt.plot(x+y/2,x-y/2,'ro')
        plt.ylabel('Reward-Merit')
        plt.xlabel('Reward+Merit')
        plt.title(c)
        #plt.show()


def observe_cluster(cnum):
    global it,X
    fig, ax = plt.subplots()
    tracklist = agent2_dict_temp[cnum]
    x = [X[t,0] for t in tracklist]
    y = [X[t,1] for t in tracklist]
    ax.scatter(x, y)
    for i, txt in enumerate(tracklist):
        ax.annotate(txt, (x[i],y[i]))
    fig.savefig('/home/dinesh/Documents/sneha/thesis/results/'+'a2'+str(cnum)+'_rmsum'+str(it)+'.png')



def update_clusters(RMsum_1,RMsum_2,rmsum1,rmsum2):
    global agent1_dict_temp,agent2_dict_temp
    '''
    for cnum,tracklist in agent1_dict_temp.iteritems():
        if tracklist == []:
            rmsum1.__delitem__(cnum)
    for cnum,tracklist in agent2_dict_temp.iteritems():
        if tracklist == []:
            rmsum2.__delitem__(cnum)
    '''
    min_rmsum_cluster1 = min(rmsum1, key=rmsum1.get)
    min_rmsum_cluster2 = min(rmsum2, key=rmsum2.get)
    '''
    min_rmsum_cluster1 = min(rmsum1.iterkeys(),key=lambda k:rmsum1[k])
    min_rmsum_cluster2 = min(rmsum2.iterkeys(),key=lambda k:rmsum2[k])
    '''
    track_remove_1 = min(track_merits1.viewkeys() & agent1_dict_temp[min_rmsum_cluster1], key=track_merits1.get)
    track_remove_2 = min(track_merits2.viewkeys() & agent2_dict_temp[min_rmsum_cluster2], key=track_merits2.get)

    print it,'\t',min_rmsum_cluster1,'\t',rmsum1[min_rmsum_cluster1],'\t',min_rmsum_cluster2,'\t',rmsum2[min_rmsum_cluster2]


    #code to remove track
    if agent1_dict_temp[min_rmsum_cluster1] != []:
        for cnum,tracklist in agent1_dict_temp.iteritems():
            if track_remove_1 in tracklist:
                tracklist.remove(track_remove_1)
                RMsum_1.__delitem__(track_remove_1)
                agent1_dict_temp[cnum] = tracklist
                break
    if agent2_dict_temp[min_rmsum_cluster2] != []:
        for cnum,tracklist in agent2_dict_temp.iteritems():
            if track_remove_2 in tracklist:
                tracklist.remove(track_remove_2)
                RMsum_2.__delitem__(track_remove_2)
                agent2_dict_temp[cnum] = tracklist
                break
    RMsum_1,RMsum_2 = {},{}

def sum_mer_rewd():
    global agent1_dict_temp,agent2_dict_temp
    RMsum_1,RMsum_2 = {},{}
    rmsum1,rmsum2 = {},{}
    rmsumval1,rmsumval2 = 0,0
    #print agent1_dict_temp.viewkeys()
    for cnum,tracklist in agent1_dict_temp.iteritems():
        if tracklist != []:
            #print it,'\t',cnum,'\t',tracklist
            for t in tracklist:
                val = track_merits1[t] + track_pairreward1[cnum][bm1[cnum]][t]/(m+1)
                rmsumval1 = rmsumval1 + val
                RMsum_1[t] = val
            rmsum1[cnum] = rmsumval1/len(tracklist)
            #print cnum,'\t',tracklist,'\t',rmsumval1
    for cnum,tracklist in agent2_dict_temp.iteritems():
        if tracklist != []:
            for t in tracklist:
                val = track_merits2[t] + track_pairreward2[cnum][bm2[cnum]][t]/(m+1)
                rmsumval2 = rmsumval2 + val
                RMsum_2[t] = val
            rmsum2[cnum] = rmsumval2/len(tracklist)
            #print cnum,'\t',tracklist,'\t',rmsumval2

    return RMsum_1,RMsum_2,rmsum1,rmsum2

def calculate_pair_reward(i,j,agent):
    global track_pairreward1,track_pairreward2
    pair_reward = -1
    intersum = 0
    if agent==1:
        for t in agent1_dict_temp[i]:
            a2rewardsum = 0
            m_set = track_knn1[t]
            inter = set(m_set).intersection(set(agent2_dict_temp[j]))
            if inter!=set([]):
                for x in inter:
                    a2rewardsum += track_merits2[x]
                intersum += a2rewardsum
            track_pairreward1[i][j][t] = a2rewardsum
            pair_reward = intersum
    elif agent==2:
        for t in agent2_dict[i]:
            a1rewardsum = 0
            m_set = track_knn2[t]
            inter = set(m_set).intersection(set(agent1_dict_temp[j]))
            if inter!=set([]):
                for x in inter:
                   a1rewardsum += track_merits1[x]
                intersum += a1rewardsum
            track_pairreward2[i][j][t] = a1rewardsum
            pair_reward = intersum
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

def totalPairReward():
    global a1a2reward_dict,a2a1reward_dict,N,bestmatch_agent1,bestmatch_agent2,track_pairreward1,track_pairreward2
    a1a2reward_dict = create_empty_rewardpair_dict(N,N)
    a2a1reward_dict = create_empty_rewardpair_dict(N,N)
    track_pairreward1 = create_track_rewardpair_dict()#number of tracks
    track_pairreward2 = create_track_rewardpair_dict()#number of tracks
    maxcluster1,maxcluster2 = -1,-1
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


def scale_track_rewards():
    global track_merits1,track_merits2
    for cnum,tracklist in agent1_dict_temp.iteritems():
            meritlist = [track_merits1[t] for t in tracklist]
            maxval1 = max(meritlist or [-1])
            for t in tracklist:
                track_merits1[t] = float(track_merits1[t])/maxval1
    for cnum,tracklist in agent2_dict_temp.iteritems():
            meritlist = [track_merits2[t] for t in tracklist]
            maxval2 = max(meritlist or [-1])
            for t in tracklist:
                track_merits2[t] = float(track_merits2[t])/maxval2

def calc_trackReward(v,sub_list,e):
    #e is not the same for agent1 and agent2
    global m
    score = 0
    for record in sub_list:
        dist = spatial.distance.euclidean(v,record)
        if dist<=e: #epsilon
            score = score+1
    return score+1

def epsilon(basetable):
    #current measure: average of all track distances
    dist_matrix = spatial.distance.pdist(basetable,'euclidean')
    dist_matrix = spatial.distance.squareform(dist_matrix)
    return np.average(dist_matrix)

def fillknn(cluster,tracklist,agent):
    global track_knn1,track_knn2
    tree = spatial.KDTree(cluster)
    for i in range(len(cluster)):
        pos = tree.query(cluster[i],k=min(m+1,len(cluster)),p=2)[1] #p=2 rep Euclidean; m set to m+1 as knn list includes the query track
        if len(cluster)<=1:
            if(agent==1):
                track_knn1[tracklist[i]] = [tracklist[i]]
            else:
                track_knn2[tracklist[i]] = [tracklist[i]]
        else:
            if(agent==1):
                track_knn1[tracklist[i]] = [tracklist[p] for p in list(pos)]
            else:
                    #print cluster
                    #print tracklist
                    #if pos != 0:
                track_knn2[tracklist[i]] = [tracklist[p] for p in list(pos)]
            #print tracklist[i],'\t',list(pos),'\t',i,'\t',track_knn2[tracklist[i]]

def build_track_rewards():
    global input_list,input_list2,track_merits1,track_merits2,track_knn1,track_knn2
    global e1,e2
    global track_scores_1,track_scores_2
    cluster_table1,cluster_table2 = [],[]
    for cnum,tracklist in agent1_dict_temp.iteritems():
        if(tracklist!=[]):
            cluster_table1 = [input_list[t] for t in tracklist]
            #print 'ITERATION ',it,'\t','CLUSTER# ',cnum
            fillknn(cluster_table1,tracklist,1)
            e1 = epsilon(cluster_table1)
            avgdist_agent1_clusters.setdefault(cnum,[]).append(e1)
            for t in tracklist:
                temp_table1 = deepcopy(cluster_table1)
                temp_table1.remove(input_list[t])
                track_merits1[t] = calc_trackReward(input_list[t],temp_table1,e1)
    for cnum,tracklist in agent2_dict_temp.iteritems():
        if(tracklist!=[]):
            #print 'ITERATION ',it,'\t','CLUSTER# ',cnum
            cluster_table2 = [input_list2[t] for t in tracklist]
            fillknn(cluster_table2,tracklist,2)
            e2 = epsilon(cluster_table2)
            avgdist_agent2_clusters.setdefault(cnum,[]).append(e2)
            for t in tracklist:
                temp_table2 = deepcopy(cluster_table2)
                temp_table2.remove(input_list2[t])
                track_merits2[t] = calc_trackReward(input_list2[t],temp_table2,e2)

def reduced_dimensions():
    X = np.array(input_list) #only for agent1
    pca = RandomizedPCA(n_components = 2)
    X_rpca = pca.fit_transform(X)
    return X_rpca

def plot(data_to_plot):
	#fig = plt.figure()
	plt.imshow(data_to_plot, aspect='auto', cmap = plt.cm.Blues)
	plt.show()

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
	agent2_dict = {}
	for l in range(len(labels)):
		agent2_dict.setdefault(labels[l]-1,[]).append(l)
	#dendro = cluster.hierarchy.dendrogram(z,1059,'level', leaf_font_size=14)
	#plt.show()
	return agent2_dict

def read_input():
    global input_list,input_list2,inputlist1
    for line in file('default_features_1059_tracks.txt'):
        templist = []
        templist2 = []
        arr = line.strip().split(',')
        length = len(arr)
        for i in range(length-2):
            templist.append(float(arr[i]))
        temp = deepcopy(templist)
        inputlist1.append(temp)
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


def main():

    global input_list_disc,input_list,input_list2,a1a2reward_dict,a2a1reward_dict,agent1_primes,agent2_primes,agent1_dict,agent2_dict,bm1,bm2
    global track_merits1,track_merits2
    global zscore_agent1,zscore_agent2
    global agent1_dict_temp,agent2_dict_temp
    global X,it

    RMsum_1,RMsum_2 = {},{}
    input_list,input_list2 = read_input()
    agent1_dict = biclustering(input_list,N)
    agent2_dict = hierarchical(input_list2,N)
    agent1_dict_temp = deepcopy(agent1_dict)
    agent2_dict_temp = deepcopy(agent2_dict)
    '''
    for cnum,tracklist in agent1_dict.iteritems():
        for t in tracklist:
            print cnum,'\t',t,'\t',input_list[t],'\t',input_list2[t]
    '''
    #X = reduced_dimensions()
    for it in range(0,150):
        build_track_rewards()
        scale_track_rewards()
        a1a2reward_dict,a2a1reward_dict,bm1,bm2 = totalPairReward()
        RMsum_1,RMsum_2,rmsum1,rmsum2 = sum_mer_rewd()
        update_clusters(RMsum_1,RMsum_2,rmsum1,rmsum2)
        it = it+1
    plt.close()


if __name__ == "__main__":
	main()

