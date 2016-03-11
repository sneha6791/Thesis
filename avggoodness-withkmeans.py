# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:55:32 2016

@author: dinesh
"""
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
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from copy import deepcopy
from sklearn.decomposition import RandomizedPCA

N = 15 #Number of clusters 10,15,20,25
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
ctr1,ctr2 = 0,0
prev1,curr1 = -1,-1
prev2,curr2 = -1,-1
flag1,flag2 = False,False
a1rejects = {}
a2rejects = {}
a1bucket,a2bucket =[],[]
window_counter1,window_counter2 = 0,0
L1,L2,AL1,AL2 = [],[],[],[]
window_size1,window_size2 = 4,4
window_stop1,window_stop2 = 8,8

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


def update_clusters(RMcluster,agent):
    global agent1_dict_temp,agent2_dict_temp
    global prev1,curr1,prev2,curr2,ctr1,flag1,ctr2,flag2
    global a1rejects,a2rejects,a1bucket,a2bucket
    global window_counter1,window_counter2
    global L1,L2,AL1,AL2
    global window_size1,window_size2,window_stop1,window_stop2

    if agent==1:
        min_rmsum_cluster1 = min(RMcluster, key=RMcluster.get)
        track_remove_1 = min(track_merits1.viewkeys() & agent1_dict_temp[min_rmsum_cluster1], key=track_merits1.get)
        #code for stopping criterion
        print 'a1','\t',it,'\t',RMcluster[min_rmsum_cluster1]
        L1.append(RMcluster[min_rmsum_cluster1])
        if len(L1) == window_size1:
            #calculate avg
            AL1.append(np.mean(L1))
            #print 'a1',it,'\t',AL1[-1]
            L1.remove(L1[0])
            if len(AL1) == window_stop1:
                #calculate stopping criterion
                if AL1[0] >= AL1[-1]:
                    flag1 = True
                    L1 = []
                AL1 = []

        try:
            a1bucket.append(track_remove_1)
        except KeyError:
            a1bucket = [track_remove_1]
            #code to remove track
        for cnum,tracklist in agent1_dict_temp.iteritems():
            if track_remove_1 in tracklist:
                tracklist.remove(track_remove_1)
                agent1_dict_temp[cnum] = tracklist
                break

        #print 'a1','\t',it,'\t',L1,'\t',AL1,'\t',flag1
    elif agent==2:
        #code
        min_rmsum_cluster2 = min(RMcluster, key=RMcluster.get)
        track_remove_2 = min(track_merits2.viewkeys() & agent2_dict_temp[min_rmsum_cluster2], key=track_merits2.get)

        print 'a2','\t',it,'\t',RMcluster[min_rmsum_cluster2]
        #code for stopping criterion
        L2.append(RMcluster[min_rmsum_cluster2])
        if len(L2) == window_size1:
            #calculate avg
            AL2.append(np.mean(L2))
            #print 'a2',it,'\t',AL2[-1]
            L2.remove(L2[0])
            if len(AL2) == window_stop2:
                #calculate stopping criterion
                if AL2[0] >= AL2[-1]:
                    flag2 = True
                    L2 = []
                AL2 = []

        try:
            a2bucket.append(track_remove_2)
        except KeyError:
            a2bucket = [track_remove_2]

        #code to remove track
        for cnum,tracklist in agent2_dict_temp.iteritems():
            if track_remove_2 in tracklist:
                tracklist.remove(track_remove_2)
                agent2_dict_temp[cnum] = tracklist
                break
        #print 'a2','\t',it,'\t',L2,'\t',AL2,'\t',flag2
    else:
        return None


def sum_mer_rewd(agent):
    global agent1_dict_temp,agent2_dict_temp
    if agent==1:
        RMsum_1 = {}
        rmsum1 = {}
        rmsumval1 = 0
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
        return RMsum_1,rmsum1
    elif agent==2:
        RMsum_2 = {}
        rmsum2 = {}
        rmsumval2 = 0
        for cnum,tracklist in agent2_dict_temp.iteritems():
            if tracklist != []:
                for t in tracklist:
                    val = track_merits2[t] + track_pairreward2[cnum][bm2[cnum]][t]/(m+1)
                    rmsumval2 = rmsumval2 + val
                    RMsum_2[t] = val
                rmsum2[cnum] = rmsumval2/len(tracklist)
                #print cnum,'\t',tracklist,'\t',rmsumval2
        #print it
        #print rmsum2
        return RMsum_2,rmsum2
    else: return None

def calculate_pair_reward(i,j,agent):
    global track_pairreward1,track_pairreward2
    pair_reward = 0
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
        for t in agent2_dict_temp[i]:
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

def create_empty_rewardpair_dict(agent):
    if agent ==1:
        x = {}
        for i in agent1_dict_temp.viewkeys():
            x[i] = {}
        for k,v in x.iteritems():
            for i in agent2_dict_temp.viewkeys():
                v[i] = 0
        return x
    elif agent ==2:
        x = {}
        for i in agent2_dict_temp.viewkeys():
            x[i] = {}
        for k,v in x.iteritems():
            for i in agent1_dict_temp.viewkeys():
                v[i] = 0
        return x




def create_track_rewardpair_dict(agent):
    if agent==1:
        x = {}
        for i in agent1_dict_temp.viewkeys():
            x[i] = {}
        for k,v in x.iteritems():
            for j in agent2_dict_temp.viewkeys():
                v[j] = {}
        return x
    elif agent==2:
        x = {}
        for i in agent2_dict_temp.viewkeys():
            x[i] = {}
        for k,v in x.iteritems():
            for j in agent1_dict_temp.viewkeys():
                v[j] = {}
        return x




def totalPairReward(agent):
    global a1a2reward_dict,a2a1reward_dict,N,bestmatch_agent1,bestmatch_agent2,track_pairreward1,track_pairreward2
    if agent==1:
        a1a2reward_dict = create_empty_rewardpair_dict(1)
        track_pairreward1 = create_track_rewardpair_dict(1)#number of tracks
        #maxcluster1 = -1
        for i in agent1_dict_temp.viewkeys():
            maxreward1 = 0
            for j in agent2_dict_temp.viewkeys():
                a1a2reward_dict[i][j] = calculate_pair_reward(i,j,1)
                if maxreward1 <= a1a2reward_dict[i][j]:#included = condition to eliminate bm to be -1
                    maxreward1 = a1a2reward_dict[i][j]
                    maxcluster1 = j
            bestmatch_agent1[i] = maxcluster1
        return a1a2reward_dict,bestmatch_agent1
    elif agent==2:
        a2a1reward_dict = create_empty_rewardpair_dict(2)
        track_pairreward2 = create_track_rewardpair_dict(2)#number of tracks
        #maxcluster2 = -1
        for i in agent2_dict_temp.viewkeys():
            maxreward2 = 0
            for j in agent1_dict_temp.viewkeys():
                a2a1reward_dict[i][j] = calculate_pair_reward(i,j,2)
                if maxreward2 <= a2a1reward_dict[i][j]:#included = condition to eliminate bm to be -1
                    maxreward2 = a2a1reward_dict[i][j]
                    maxcluster2 = j
            bestmatch_agent2[i] = maxcluster2
        return a2a1reward_dict,bestmatch_agent2
    else: return None



def scale_track_rewards(agent):
    global track_merits1,track_merits2
    if agent == 1:
        for cnum,tracklist in agent1_dict_temp.iteritems():
                meritlist = [track_merits1[t] for t in tracklist]
                maxval1 = max(meritlist or [-1])
                for t in tracklist:
                    track_merits1[t] = float(track_merits1[t])/maxval1
    elif agent==2:
        for cnum,tracklist in agent2_dict_temp.iteritems():
                meritlist = [track_merits2[t] for t in tracklist]
                maxval2 = max(meritlist or [-1])
                for t in tracklist:
                    track_merits2[t] = float(track_merits2[t])/maxval2
    else:
        return None

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

def build_track_rewards(agent):
    global input_list,input_list2,track_merits1,track_merits2,track_knn1,track_knn2
    global e1,e2
    global track_scores_1,track_scores_2
    cluster_table1,cluster_table2 = [],[]
    if agent==1:
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
    elif agent==2:
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
    else: return None

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


def kmeansclustering(features,tracks,ncenters):
    clusters = {}
    model = KMeans(n_clusters = ncenters)
    data = [features[x] for x in tracks]
    model.fit(data)
    labels = model.labels_
    for i in range(len(tracks)):
        try:
            clusters[labels[i]].append(tracks[i])
        except KeyError:
            clusters[labels[i]] = [tracks[i]]
    return clusters

def main():

    global input_list_disc,input_list,input_list2,a1a2reward_dict,a2a1reward_dict,agent1_primes,agent2_primes,agent1_dict,agent2_dict,bm1,bm2
    global track_merits1,track_merits2
    global zscore_agent1,zscore_agent2
    global agent1_dict_temp,agent2_dict_temp
    global X,it
    global a1bucket,a2bucket
    global window_counter1,window_counter2,flag1,flag2


    a1final,a2final= False,False
    stop = False
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
    #for it in range(0,50):
    while True:
        build_track_rewards(1)
        build_track_rewards(2)
        scale_track_rewards(1)
        scale_track_rewards(2)
        a1a2reward_dict,bm1 = totalPairReward(1)
        a2a1reward_dict,bm2 = totalPairReward(2)
        RMsum_1,rmsum1 = sum_mer_rewd(1)
        RMsum_2,rmsum2 = sum_mer_rewd(2)
        update_clusters(rmsum1,1)
        update_clusters(rmsum2,2)
        if flag1==True:
            a1final = True
            break
        if flag2==True:
            a2final = True
            break
        it = it+1

    if a1final==True and stop==False:
        '''
        for cnum,tracklist in a1rejects.iteritems():
            for t in tracklist:
                agent1_dict_temp[cnum].append(t)
        '''
        while a2final==False:
            build_track_rewards(2)
            scale_track_rewards(2)
            a2a1reward_dict,bm2 = totalPairReward(2)
            RMsum_2,rmsum2 = sum_mer_rewd(2)
            update_clusters(rmsum2,2)
            it= it+1
            if flag2==True:
                a2final = True
                stop = True

    if a2final==True and stop==False:
        '''
        for cnum,tracklist in a2rejects.iteritems():
            for t in tracklist:
                agent2_dict_temp[cnum].append(t)
        '''
        while a1final==False:
            build_track_rewards(1)
            scale_track_rewards(1)
            a2a1reward_dict,bm2 = totalPairReward(1)
            RMsum_1,rmsum1 = sum_mer_rewd(1)
            update_clusters(rmsum1,1)
            it= it+1
            if flag1==True:
                a1final = True
                stop = True


    print a1bucket,'\t',a2bucket
    kmeanscenters = 10
    c = N
    if len(a1bucket) > kmeanscenters:
        a1kmeans = kmeansclustering(input_list,a1bucket,10)
        print a1kmeans
        for val in a1kmeans.viewvalues():
            agent1_dict_temp.update({c:val})
            c =c+1

    else:
        agent1_dict_temp.update({c:a1bucket})

    c = N
    if len(a2bucket) > kmeanscenters:
        a2kmeans = kmeansclustering(input_list2,a2bucket,10)
        print a2kmeans
        for val in a2kmeans.viewvalues():
            agent2_dict_temp.update({c:val})
            c= c+1
    else:
        agent2_dict_temp.update({c:a2bucket})

    print agent1_dict_temp
    print agent2_dict_temp
#repeat
    window_counter1 = 0
    window_counter2 = 0
    flag1,flag2 = False,False
    a1final,a2final= False,False
    stop = False
    print 'REPEAT WITH KMEANS CLUSTERS'

    while True:
        build_track_rewards(1)
        build_track_rewards(2)
        scale_track_rewards(1)
        scale_track_rewards(2)
        a1a2reward_dict,bm1 = totalPairReward(1)
        a2a1reward_dict,bm2 = totalPairReward(2)
        RMsum_1,rmsum1 = sum_mer_rewd(1)
        RMsum_2,rmsum2 = sum_mer_rewd(2)
        update_clusters(rmsum1,1)
        update_clusters(rmsum2,2)
        if flag1==True:
            a1final = True
            break
        if flag2==True:
            a2final = True
            break
        it = it+1

    if a1final==True and stop==False:
        '''
        for cnum,tracklist in a1rejects.iteritems():
            for t in tracklist:
                agent1_dict_temp[cnum].append(t)
        '''
        while a2final==False:
            build_track_rewards(2)
            scale_track_rewards(2)
            a2a1reward_dict,bm2 = totalPairReward(2)
            RMsum_2,rmsum2 = sum_mer_rewd(2)
            update_clusters(rmsum2,2)
            it= it+1
            if flag2==True:
                a2final = True
                stop = True

    if a2final==True and stop==False:
        '''
        for cnum,tracklist in a2rejects.iteritems():
            for t in tracklist:
                agent2_dict_temp[cnum].append(t)
        '''
        while a1final==False:
            build_track_rewards(1)
            scale_track_rewards(1)
            a2a1reward_dict,bm2 = totalPairReward(1)
            RMsum_1,rmsum1 = sum_mer_rewd(1)
            update_clusters(rmsum1,1)
            it= it+1
            if flag1==True:
                a1final = True
                stop = True
    plt.close()


if __name__ == "__main__":
	main()

