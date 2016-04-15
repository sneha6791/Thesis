import numpy as np
import scipy as sp
from scipy import spatial,cluster
#from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from copy import deepcopy
from sklearn.decomposition import RandomizedPCA
import math
from sklearn import preprocessing

N = 25 #Number of clusters 10,15,20,25,30,35,40
m = 4 #Number of nearest neighbors 3,4,5,7

inputlist1=[]
input_list_disc = []
input_list = []
input_list2 = []
agent1_dict = {}
agent2_dict = {}
track_merits1 = {}
track_merits2 = {}
track_rewards1 = {}
track_rewards2 = {}
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
it =0
X = []
e1,e2 = -1,-1
flag1,flag2 = False,False
a1rejects = {}
a2rejects = {}
a1bucket,a2bucket =[],[]
max1,max2 = 0,0
rm_tracks1,rm_cluster1,rm_tracks2,rm_cluster2 = {},{},{},{}

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

def addclustersh(listoftracks,agent,numclusters):
    if agent==1:
        dict_to_update = agent1_dict_temp
        featurelist = [input_list[x] for x in listoftracks]
    elif agent==2:
        dict_to_update = agent2_dict_temp
        featurelist = [input_list2[x] for x in listoftracks]
    else:
        return None
    c = dict_to_update.keys()[-1] + 1
    clusters = hierarchical(listoftracks,featurelist,numclusters)
    print 'Rejected track clusters ',agent,' ',clusters
    for val in clusters.viewvalues():
        dict_to_update.update({c:val})
        c = c+1
    return dict_to_update


def addclustersk(listoftracks,agent,kmeanscenters):
    if agent==1:
        dict_to_update = agent1_dict_temp
    elif agent==2:
        dict_to_update = agent2_dict_temp
    else:
        return None
    c = dict_to_update.keys()[-1] + 1
    if len(listoftracks) > kmeanscenters:
        clusters = kmeansclustering(input_list,listoftracks,10)
        print 'kmeans ',clusters
        for val in clusters.viewvalues():
            dict_to_update.update({c:val})
            c = c+1
    else:
        dict_to_update.update({c:listoftracks})
    return dict_to_update

def update_clusters(agent):
    global agent1_dict_temp,agent2_dict_temp
    global flag1,flag2
    global a1bucket,a2bucket
    global max1,max2
    global rm_tracks1,rm_tracks2
    global rm_cluster1,rm_cluster2

    if agent==1:
        #min_rmsum_cluster1 = min(rm_cluster1, key=rm_cluster1.get)
        minval,min_rmsum_cluster1 = min((v,k) for k,v in rm_cluster1.items())
        track_remove_1 = min(track_merits1.viewkeys() & agent1_dict_temp[min_rmsum_cluster1], key=track_merits1.get)
        val1 = rm_cluster1[min_rmsum_cluster1]
        max1 = max(max1,val1)
        #bm = bm1[min_rmsum_cluster1]
        print 'a1','\t',it,'\t',val1,'\t',max1,'\t',track_merits1[track_remove_1],'\t',track_rewards1[track_remove_1],'\t',min_rmsum_cluster1,'\t',track_remove_1,'\t',len(agent1_dict_temp[min_rmsum_cluster1])
        if (max1-val1)/max1 > 0.15:
            flag1 = True
            print 'a1','\t','REMOVED TRACKS','\t',it,'REMAINING TRACKS','\t',1059-it
        a1bucket.append(track_remove_1)
        return track_remove_1,min_rmsum_cluster1

    elif agent==2:
        #min_rmsum_cluster2 = min(rm_cluster2, key=rm_cluster2.get)
        minval,min_rmsum_cluster2 = min((v,k) for k,v in rm_cluster2.items())
        track_remove_2 = min(track_merits2.viewkeys() & agent2_dict_temp[min_rmsum_cluster2], key=track_merits2.get)
        val2 = rm_cluster2[min_rmsum_cluster2]
        max2 = max(max2,val2)
        #bm = bm2[min_rmsum_cluster2]
        print 'a2','\t',it,'\t',val2,'\t',max2,'\t',track_merits2[track_remove_2],'\t',track_rewards2[track_remove_2],'\t',min_rmsum_cluster2,'\t',track_remove_2,'\t',len(agent2_dict_temp[min_rmsum_cluster2])
        if (max2-val2)/max2 > 0.15:
            flag2 = True
            print 'a2','\t','REMOVED TRACKS','\t',it,'REMAINING TRACKS','\t',1059-it
        a2bucket.append(track_remove_2)
        return track_remove_2,min_rmsum_cluster2
    else:
        return None


def sum_mer_rewd(clusterID,agent):
    rmtrack,rmcluster = {},{}
    if agent==1:
        agentdict = agent1_dict_temp
        trackmerits = track_merits1
        trackrewards = track_rewards1
        #trackrewards = track_pairreward1
        #bm = bm1[clusterID]
    elif agent==2:
        agentdict = agent2_dict_temp
        trackmerits = track_merits2
        trackrewards = track_rewards2
        #trackrewards = track_pairreward2
        #bm = bm2[clusterID]
    else: return None
    tracklist = agentdict[clusterID]
    rmsumval = 0
    if tracklist != []:
        for t in tracklist:
            val = trackmerits[t] + trackrewards[t]
            rmsumval = rmsumval + val
            rmtrack[t] = val
        rmcluster[clusterID] = rmsumval/len(tracklist)
    return rmtrack,rmcluster

def build_track_rewards(clusterID,agent):
    temp_track_rewards = {}
    if agent == 1:
        agentdict = agent1_dict_temp
        track_pairreward = track_pairreward1
        bm = bm1[clusterID]
    elif agent == 2:
        agentdict = agent2_dict_temp
        track_pairreward = track_pairreward2
        bm = bm2[clusterID]
    for t in agentdict[clusterID]:
        temp_track_rewards[t] = track_pairreward[clusterID][bm][t]/(m+1) if bm!=None else 0
    return temp_track_rewards

def best_match(clusterID,agent):
    global a1a2reward_dict,a2a1reward_dict
    if agent == 1:
        pairrewards = a1a2reward_dict
    elif agent == 2:
        pairrewards = a2a1reward_dict
    bestmatchID = max(pairrewards[clusterID], key=pairrewards[clusterID].get)
    if pairrewards[clusterID][bestmatchID] == 0:
        bestmatchID = None
    return bestmatchID


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


def scale_track_merits(dict_of_values):
    if any(dict_of_values) == True:
        m = max(v for v in dict_of_values.viewvalues())
        for k,v in dict_of_values.iteritems():
            dict_of_values.update({k:float(v)/m})
    return dict_of_values

def calc_track_merit(v,sub_list,e):
    score = 0
    for record in sub_list:
        dist = spatial.distance.euclidean(v,record)
        if dist<=e: #average distance treshold
            score = score+1
    return score+1

def epsilon(basetable):
    dist_matrix = spatial.distance.pdist(basetable,'euclidean')
    dist_matrix = spatial.distance.squareform(dist_matrix)#display only upper traingle
    return np.average(dist_matrix)#current measure: average of track pairwise distances within cluster

def fillknn(clusterID,agent):
    track_knn = {}
    if agent==1:
        input_data = input_list
        agentdict = agent1_dict_temp
    elif agent==2:
        input_data = input_list2
        agentdict = agent2_dict_temp
    tracklist = agentdict[clusterID]
    if(tracklist!=[]):
        features_table = [input_data[t] for t in tracklist]
        tree = spatial.KDTree(features_table)
        for i in range(len(features_table)):
            pos = tree.query(features_table[i],k=min(m+1,len(features_table)),p=2)[1] #p=2 rep Euclidean; m set to m+1 as knn list includes the query track
            if len(features_table)<=1:
                track_knn[tracklist[i]] = [tracklist[i]]
            else:
                track_knn[tracklist[i]] = [tracklist[p] for p in list(pos)]
    return track_knn

def build_track_merits(clusterID,agent):
    trackmerits = {}
    if agent==1:
        input_data = input_list
        agentdict = agent1_dict_temp
    elif agent==2:
        input_data = input_list2
        agentdict = agent2_dict_temp
    else: return None
    tracklist = agentdict[clusterID]
    if tracklist!=[]:
        features_table = [input_data[t] for t in tracklist]
        e = epsilon(features_table)
        for t in tracklist:
            temp_table = deepcopy(features_table)
            temp_table.remove(input_data[t])
            trackmerits[t] = calc_track_merit(input_data[t],temp_table,e)
    return trackmerits

def plot(data_to_plot):
	#fig = plt.figure()
	plt.imshow(data_to_plot, aspect='auto', cmap = plt.cm.Blues)
	plt.show()

def biclustering(data,num_clusters):
	clusters = {}
	data = np.asmatrix(data)
	model = SpectralCoclustering(n_clusters=num_clusters,random_state=0)
	#model = SpectralBiclustering(n_clusters=num_clusters)
	model.fit(data)
	for c in range(num_clusters):
		clusters[c] = model.get_indices(c)[0].tolist() #0 row indices, 1 column indices
	#fit_data = data[np.argsort(model.row_labels_)]
	#fit_data = fit_data[:, np.argsort(model.column_labels_)]
	#plot(fit_data)
	return clusters

def hierarchical(tracklist,data,num_clusters):
	clusters = {}
	condensed_dist = cluster.hierarchy.distance.pdist(data)
	z = cluster.hierarchy.linkage(condensed_dist,'average')
	labels = cluster.hierarchy.fcluster(z,num_clusters,'maxclust')
	for l in range(len(labels)):
		clusters.setdefault(labels[l]-1,[]).append(tracklist[l])
	#dendro = cluster.hierarchy.dendrogram(z,1059,'level', leaf_font_size=14)
	#plt.show()
	return clusters

def binning(datalist,num_intervals):
    low,high = min(datalist),max(datalist)#min and max values for binning interval calculation
    data = np.array(datalist)
    bins = np.linspace(low,high,num_intervals)#generates 10 samples spaced equally between low and high
    digitized = np.digitize(data,bins)#labels data points corresponding to the interval under which they belong
    return list(digitized)

def read_input(datafile):
    true_input_a1,true_input_a2,discretized_input_a1= [],[],[]
    for line in file(datafile):
        templist = []
        templist2 = []
        arr = line.strip().split(',')
        length = len(arr)
        templist = [float(x) for x in arr[:length-2]]#last 2 features belong to agent2
        templist2 = [float(x) for x in arr[length-2:]]
        temp = deepcopy(templist)#retain true value
        true_input_a1.append(temp)#true audio values
        templist = binning(templist,10)#discretized a1 values
        discretized_input_a1.append(templist)
        true_input_a2.append(templist2)#true lat,long value
    return discretized_input_a1,true_input_a2

def initialize(clusterID,agent):
    global track_merits1,track_knn1
    global track_merits2,track_knn2
    if agent == 1:
        if agent1_dict_temp[clusterID] != []:
            temp_trackmerits = build_track_merits(clusterID,1)
            scaled_trackmerits = scale_track_merits(temp_trackmerits)
            track_merits1.update(scaled_trackmerits)#track merits minclusterid2
            temp_knn = fillknn(clusterID,1)
            track_knn1.update(temp_knn)#track knn minclusterid1
    elif agent == 2:
        if agent2_dict_temp[clusterID] != []:
            temp_trackmerits = build_track_merits(clusterID,2)
            scaled_trackmerits = scale_track_merits(temp_trackmerits)
            track_merits2.update(scaled_trackmerits)#track merits minclusterid2
            temp_knn = fillknn(clusterID,2)
            track_knn2.update(temp_knn)#track knn minclusterid2
    else: return None

def delete_from_track_dicts(mintrackid,minclusterid,agent):
    global track_merits1,track_knn1,rm_tracks1,track_pairreward1,agent1_dict_temp,track_rewards1
    global track_merits2,track_knn2,rm_tracks2,track_pairreward2,agent2_dict_temp,track_rewards2
    if agent==1:
        track_merits1.pop(mintrackid,None)
        track_knn1.pop(mintrackid,None)
        track_rewards1.pop(mintrackid,None)
        for key,subdict in track_pairreward1[minclusterid].iteritems():
           subdict.pop(mintrackid,None)
        rm_tracks1.pop(mintrackid,None)
        agent1_dict_temp[minclusterid].remove(mintrackid)
        return track_merits1,track_knn1,track_rewards1,rm_tracks1,agent1_dict_temp,track_pairreward1
    elif agent==2:
        track_merits2.pop(mintrackid,None)
        track_knn2.pop(mintrackid,None)
        track_rewards2.pop(mintrackid,None)
        for key,subdict in track_pairreward2[minclusterid].iteritems():
            subdict.pop(mintrackid,None)
        rm_tracks2.pop(mintrackid,None)
        agent2_dict_temp[minclusterid].remove(mintrackid)
        return track_merits2,track_knn2,track_rewards2,rm_tracks2,agent2_dict_temp,track_pairreward2

def initial_setting():
    global agent1_dict_temp,track_merits1,track_knn1,rm_tracks1,rm_cluster1,a1a2reward_dict,track_pairreward1,bm1,track_rewards1
    global agent2_dict_temp,track_merits2,track_knn2,rm_tracks2,rm_cluster2,a2a1reward_dict,track_pairreward2,bm2,track_rewards2
    for cnum,tracklist in agent1_dict_temp.iteritems():
        initialize(cnum,1)
    for cnum,tracklist in agent2_dict_temp.iteritems():
        initialize(cnum,2)

    for i in agent1_dict_temp.keys():
        track_pairreward1[i] = {}
        a1a2reward_dict[i] = {}
        for j in agent2_dict_temp.keys():
            track_pairreward1[i][j] = {}
            a1a2reward_dict[i][j] = calculate_pair_reward(i,j,1)
    for i in agent2_dict_temp.keys():
        a2a1reward_dict[i] = {}
        track_pairreward2[i] = {}
        for j in agent1_dict_temp.keys():
            track_pairreward2[i][j] = {}
            a2a1reward_dict[i][j] = calculate_pair_reward(i,j,2)

    for cnum in agent1_dict_temp.keys():
        bm1[cnum] = best_match(cnum,1)
        temp_trackrewards = build_track_rewards(cnum,1)
        track_rewards1.update(temp_trackrewards)
        temp_rmtrack,temp_rmcluster = sum_mer_rewd(cnum,1)
        rm_tracks1.update(temp_rmtrack)
        rm_cluster1.update(temp_rmcluster)
    for cnum in agent2_dict_temp.keys():
        bm2[cnum] = best_match(cnum,2)
        temp_trackrewards = build_track_rewards(cnum,2)
        track_rewards2.update(temp_trackrewards)
        temp_rmtrack,temp_rmcluster = sum_mer_rewd(cnum,2)
        rm_tracks2.update(temp_rmtrack)
        rm_cluster2.update(temp_rmcluster)

def display_all():
    print 'a1 clusters ',agent1_dict_temp
    print 'a2 clusters ',agent2_dict_temp
    print 'a1 merits ',track_merits1
    print 'a2 merits ',track_merits2
    print 'a1 rewards ',track_rewards1
    print 'a2 rewards ',track_rewards2
    print 'a1 knn ',track_knn1
    print 'a2 knn ',track_knn2
    #print 'a1 pairrewards ',track_pairreward1
    #print 'a2 pairrewards ',track_pairreward2
    print 'a1 bm ',bm1
    print 'a2 bm ',bm2
    print 'a1 rmcluster ',rm_cluster1
    print 'a1 rmtrack ',rm_tracks1
    print 'a2 rmcluster ',rm_cluster2
    print 'a2 rmtrack ',rm_tracks2
    for cnum,tracklist in agent1_dict_temp.iteritems():
        bm = bm1[cnum]
        print 'a1','\t',cnum
        for t in tracklist:
            print t,'\t',track_merits1[t],'\t',input_list[t]
        print 'bestmatch','\t',bm
        for t in agent2_dict_temp[bm]:
            print t,'\t',input_list2[t]
    for cnum,tracklist in agent2_dict_temp.iteritems():
        bm = bm2[cnum]
        print 'a2','\t',cnum
        for t in tracklist:
            print t,'\t',input_list2[t]
        print 'bestmatch','\t',bm
        for t in agent1_dict_temp[bm]:
            print t,'\t',input_list[t]

def main():
    global input_list,input_list2,agent1_dict,agent2_dict
    global track_knn1,track_knn2
    global a1a2reward_dict,a2a1reward_dict,bm1,bm2,track_pairreward1,track_pairreward2
    global track_merits1,track_merits2
    global agent1_dict_temp,agent2_dict_temp
    global it
    global a1bucket,a2bucket
    global flag1,flag2,max1,max2
    global rm_tracks1,rm_cluster1,rm_tracks2,rm_cluster2
    global bm1,bm2
    global track_rewards1,track_rewards2

    input_list,input_list2 = read_input('default_features_1059_tracks.txt')
    tracklist = [x for x in range(len(input_list))]
    agent1_dict = biclustering(input_list,N)#*use tracklist
    agent2_dict = hierarchical(tracklist,input_list2,N)
    agent1_dict_temp = deepcopy(agent1_dict)
    agent2_dict_temp = deepcopy(agent2_dict)
    epochcounter = 1
    min_capacity = 10
    total_epochs = 50

    while epochcounter <= total_epochs+1:
        initial_setting()
        #print 'Rewards ',track_rewards1
        #print 'Rewards ',track_rewards2
        print 'Average Goodness of all clusters (including rejected tracks): '
        print 'aga','\t','a1', '\t', np.average(rm_cluster1.values()),'\t','a2','\t',np.average(rm_cluster2.values())
        while True:
            it = it+1
            mintrackid1,minclusterid1 = update_clusters(1)
            mintrackid2,minclusterid2 = update_clusters(2)
            track_merits1,track_knn1,track_rewards1,rm_tracks1,agent1_dict_temp,track_pairreward1= delete_from_track_dicts(mintrackid1,minclusterid1,1)
            track_merits2,track_knn2,track_rewards2,rm_tracks2,agent2_dict_temp,track_pairreward2= delete_from_track_dicts(mintrackid2,minclusterid2,2)
            #if agent1_dict_temp[minclusterid1] == []:
            #    print '\nA1 EMPTY CLUSTER',minclusterid1
            #    print '\nempty cluster metrics before'
            #    print '\nclusters ',agent1_dict_temp
            #    print '\na1a2rewards ',a1a2reward_dict
            #    print '\nbestmatch ',bm1
            #    print '\bestmatch a2 ',bm2
            #    print '\nrm_cluster ',rm_cluster1
            #if agent2_dict_temp[minclusterid2] == []:
                #print '\nA2 EMPTY CLUSTER',minclusterid2
            initialize(minclusterid1,1)
            initialize(minclusterid2,2)
            #update mincluster rewards and other rewards wrt mincluster
            if agent1_dict_temp[minclusterid1] != []:
                for j in agent2_dict_temp.keys():
                    a1a2reward_dict[minclusterid1][j] = calculate_pair_reward(minclusterid1,j,1)
                    a2a1reward_dict[j][minclusterid1] = calculate_pair_reward(j,minclusterid1,2)
            else:
                for j in agent2_dict_temp.keys():
                    a1a2reward_dict[minclusterid1][j] = 0
                    a2a1reward_dict[j][minclusterid1] = 0

            if agent2_dict_temp[minclusterid2] != []:
                for j in agent1_dict_temp.keys():
                    a2a1reward_dict[minclusterid2][j] = calculate_pair_reward(minclusterid2,j,2)
                    a1a2reward_dict[j][minclusterid2] = calculate_pair_reward(j,minclusterid2,1)
            else:
                for j in agent1_dict_temp.keys():
                    a2a1reward_dict[minclusterid2][j] = 0
                    a1a2reward_dict[j][minclusterid2] = 0
            #update best matches,rmagent1_dict_temp.pop(minclusterid1,None)
            if agent1_dict_temp[minclusterid1] != []:
                for cnum in agent1_dict_temp.keys():
                    bm1[cnum] = best_match(cnum,1)
                    temp_trackrewards = build_track_rewards(cnum,1)
                    track_rewards1.update(temp_trackrewards)
                    temp_rmtrack,temp_rmcluster = sum_mer_rewd(cnum,1)
                    rm_tracks1.update(temp_rmtrack)
                    rm_cluster1.update(temp_rmcluster)
            else:
                bm1.pop(minclusterid1,None)
                #re-caluclate dependencies
                for c,x in bm2.iteritems():
                    if x==minclusterid1:
                        bm2[c] = best_match(c,2)
                rm_cluster1.pop(minclusterid1,None)
                agent1_dict_temp.pop(minclusterid1,None)
                #print '\nA1 EMPTY CLUSTER',minclusterid1
                #print '\nempty cluster metrics after'
                #print '\nclusters ',agent1_dict_temp
                #print '\na1a2rewards ',a1a2reward_dict
                #print '\nbestmatch ',bm1
                #print '\bestmatch a2 ',bm2
                #print '\nrm_cluster ',rm_cluster1
                #epochcounter = 99
                #break

            if agent2_dict_temp[minclusterid2] != []:
                for cnum in agent2_dict_temp.keys():
                    bm2[cnum] = best_match(cnum,2)
                    temp_trackrewards = build_track_rewards(cnum,2)
                    track_rewards2.update(temp_trackrewards)
                    temp_rmtrack,temp_rmcluster = sum_mer_rewd(cnum,2)
                    rm_tracks2.update(temp_rmtrack)
                    rm_cluster2.update(temp_rmcluster)
            else:
                bm2.pop(minclusterid2,None)
                #re-caluclate dependencies
                for c,x in bm1.iteritems():
                    if x==minclusterid2:
                        bm1[c] = best_match(c,1)
                rm_cluster2.pop(minclusterid2,None)
                agent2_dict_temp.pop(minclusterid2,None)

            if flag1==True and flag2==False:
                while flag2==False:
                    it= it+1
                    mintrackid2,minclusterid2 = update_clusters(2)
                    track_merits2,track_knn2,track_rewards2,rm_tracks2,agent2_dict_temp,track_pairreward2 = delete_from_track_dicts(mintrackid2,minclusterid2,2)
                    #if agent2_dict_temp[minclusterid2] == []:
                    #    print '\nA2 EMPTY CLUSTER',minclusterid2
                    initialize(minclusterid2,2)
                    if agent2_dict_temp[minclusterid2] != []:
                        for j in agent1_dict_temp.keys():
                            a2a1reward_dict[minclusterid2][j] = calculate_pair_reward(minclusterid2,j,2)
                            a1a2reward_dict[j][minclusterid2] = calculate_pair_reward(j,minclusterid2,1)
                    else:
                        for j in agent1_dict_temp.keys():
                            a2a1reward_dict[minclusterid2][j] = 0
                            a1a2reward_dict[j][minclusterid2] = 0

                    if agent2_dict_temp[minclusterid2] != []:
                        for cnum in agent2_dict_temp.keys():
                            bm2[cnum] = best_match(cnum,2)
                            temp_trackrewards = build_track_rewards(cnum,2)
                            track_rewards2.update(temp_trackrewards)
                            temp_rmtrack,temp_rmcluster = sum_mer_rewd(cnum,2)
                            rm_tracks2.update(temp_rmtrack)
                            rm_cluster2.update(temp_rmcluster)
                    else:
                        bm2.pop(minclusterid2,None)
                        #re-caluclate dependencies
                        for c,x in bm1.iteritems():
                            if x==minclusterid2:
                                bm1[c] = best_match(c,1)
                        rm_cluster2.pop(minclusterid2,None)
                        agent2_dict_temp.pop(minclusterid2,None)
            if flag1==False and flag2==True:
                while flag1==False:
                    it= it+1
                    mintrackid1,minclusterid1 = update_clusters(1)
                    track_merits1,track_knn1,track_rewards1,rm_tracks1,agent1_dict_temp,track_pairreward1 = delete_from_track_dicts(mintrackid1,minclusterid1,1)
                    #if agent1_dict_temp[minclusterid1] == []:
                    #    print '\nA1 EMPTY CLUSTER',minclusterid1
                    #    print '\nA1 EMPTY CLUSTER',minclusterid1
                    #    print '\nempty cluster metrics before'
                    #    print '\nclusters ',agent1_dict_temp
                    #    print '\na1a2rewards ',a1a2reward_dict
                    #    print '\nbestmatch ',bm1
                    #    print '\bestmatch a2 ',bm2
                    #    print '\nrm_cluster ',rm_cluster1
                    initialize(minclusterid1,1)
                    if agent1_dict_temp[minclusterid1] != []:
                        for j in agent2_dict_temp.keys():
                            a1a2reward_dict[minclusterid1][j] = calculate_pair_reward(minclusterid1,j,1)
                            a2a1reward_dict[j][minclusterid1] = calculate_pair_reward(j,minclusterid1,2)
                    else:
                        for j in agent2_dict_temp.keys():
                            a1a2reward_dict[minclusterid1][j] = 0
                            a2a1reward_dict[j][minclusterid1] = 0

                    if agent1_dict_temp[minclusterid1] != []:
                        for cnum in agent1_dict_temp.keys():
                            bm1[cnum] = best_match(cnum,1)
                            temp_trackrewards = build_track_rewards(cnum,1)
                            track_rewards1.update(temp_trackrewards)
                            temp_rmtrack,temp_rmcluster = sum_mer_rewd(cnum,1)
                            rm_tracks1.update(temp_rmtrack)
                            rm_cluster1.update(temp_rmcluster)
                    else:
                        bm1.pop(minclusterid1,None)
                        #re-caluclate dependencies
                        for c,x in bm2.iteritems():
                            if x==minclusterid1:
                                bm2[c] = best_match(c,2)
                        rm_cluster1.pop(minclusterid1,None)
                        agent1_dict_temp.pop(minclusterid1,None)
                        #print '\nA1 EMPTY CLUSTER',minclusterid1
                        #print '\nempty cluster metrics after'
                        #print '\nclusters ',agent1_dict_temp
                        #print '\na1a2rewards ',a1a2reward_dict
                        #print '\nbestmatch ',bm1
                        #print '\bestmatch a2 ',bm2
                        #print '\nrm_cluster ',rm_cluster1
                        #epochcounter = 99
                        #break
            if flag1==True and flag2==True:
                print 'EPOCH ', epochcounter
                print 'Average Goodness of all clusters (remaining tracks): '
                print 'agr','\t','a1', '\t', np.average(rm_cluster1.values()),'\t','a2','\t',np.average(rm_cluster2.values())
                print 'Remaining clusters'
                print 'a1 ',agent1_dict_temp
                print 'a2 ',agent2_dict_temp
                #restrict the number of clusters being added to avoid unnecessarily large number of clusters
                num_extra_clusters1 =  math.floor(len(a1bucket)/min_capacity) #constant
                num_extra_clusters2 =  math.floor(len(a2bucket)/min_capacity) #constant
                print 'Clustering Rejected Tracks '
                agent1_dict_temp = addclustersh(a1bucket,1,num_extra_clusters1)
                agent2_dict_temp = addclustersh(a2bucket,2,num_extra_clusters2)
                a1bucket = []
                a2bucket = []
                track_merits1,track_merits2 = {},{}
                a1a2reward_dict,a2a1reward_dict = {},{}
                track_pairreward1,track_pairreward2 = {},{}
                track_knn1,track_knn2 = {},{}
                bm1,bm2 = {},{}
                rm_cluster1,rm_cluster2 = {},{}
                rm_tracks1,rm_tracks2 = {},{}
                track_rewards1,track_rewards2 = {},{}

                max1,max2 = 0,0
                flag1,flag2 = False,False
                it = 0
                break
        epochcounter = epochcounter + 1
    print 'The End'
if __name__ == "__main__":
	main()

