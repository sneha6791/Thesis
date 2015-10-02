import string
import sys
from scipy import spatial,cluster
import numpy as np



def main():
	
	input_list = []
	input_list2 = []
	for line in file('default_features_1059_tracks.txt'):
		arr = line.split(',')
		templist = []
		templist2 = []
		for i in range(len(arr)-2):
			templist.append(float(arr[i]))
		templist2.append(float(arr[len(arr)-2]))
		templist2.append(float(arr[len(arr)-1]))
		input_list.append(templist)
		input_list2.append(templist2)
	
	print 'Agent 1 \n'
	agent1 = dict(enumerate(input_list))
	print '\n\nAgent 2 \n'
	agent2 = dict(enumerate(input_list2))

	agent2_dist = []
	maxdist = 0.0
	for key1,val1 in agent2.iteritems():
		temp = []
		for key2,val2 in agent2.iteritems():
			if key1 != key2:
				dist = spatial.distance.euclidean(val1,val2)
			else:
				dist = 0.0
			temp.append(dist)
		agent2_dist.append(temp)
	

	'''for i in range(10):
		for j in range(10):
			print agent2_dist[i][j],' ',
		print '\n' 
        '''
	test_list = []
	#cluster.hierarchy.linkage(agent2_matrix,method='single',metric='euclidean')
	#agent2_matrix = np.matrix(agent2_dist)
	#print "maximum euclidean distance ",agent2_matrix.max()
	print test_list
	test_matrix = np.matrix(test_list)
	#Z = cluster.hierarchy.linkage(agent2_matrix)
	Z = cluster.hierarchy.linkage(test_matrix)	
	print cluster.hierarchy.dendrogram(Z)

if __name__ == "__main__":
	main()

 
