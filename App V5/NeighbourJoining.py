#!/usr/bin/python

#
# Use the neighbour joining algorithm to build a tree
#


fileCount = 1; 
clusterCount = 1;
matrixCount = 0;
originalMatrix = [];
temp = [];
mergedAlready = "";

import sys
import numpy as np
import math
import time
import random
import scipy as scipy
import itertools
import sys
import os
import filegen;
import ast;


from ete3 import Tree
from skbio import DistanceMatrix
from skbio.tree import nj

max = sys.maxsize



class Node:
    '''
    Simple binary tree node with name + height
    also left_height and right_height functions to return height - child height
    if you want getters and setters write them your own damn self
    '''
        
    def __init__(self, name, height, left=None, right=None):
        self.name = name
        self.height = height
        self.left = left
        self.right = right

    def left_height(self):
        """
        Returns edge length btwn self and left child
        """
        
        if self.left is not None:
            return self.height - self.left.height
        else:
            return 0
    
    def right_height(self):
        """
        Returns edge length btwn self and right child
        """
        
        if self.right is not None:
            return self.height - self.right.height
        else:
            return 0
    
    def __str__(self):
        """
        Recursively prints tree structure
        """
        
        if self.left is not None:
            # Has children
            return ("({ln}:{lh}{lr})({rn}:{rh}{rr})".format(

                    ln=self.left.name,
                    lh=round(self.left_height(),1),
                    lr=self.left.__str__(),
                    rn=self.right.name,
                    rh=round(self.right_height(),1),
                    rr=self.right.__str__()
                    ))
        else:
            return ""
        
        
    def traverse(self):
        """
        Unused, lol
        """
        if self.left is not None:
            self.left.traverse()
        if self.right is not None:
            self.right.traverse()
        print(self.__str__())


def matrixFileOutput(Matrix):

    global fileCount;

    res = (Matrix.tolist());

    # Write matrix to file
    print("Outputfile " + str(fileCount) +":");
    out1 = "Matrix.o" + str(fileCount);
    print(out1);
    print("");

    if not os.path.isfile(out1):
        open(out1, 'w').close();

    out = open(out1, 'w');
    out.write(str(res));



def round_down_1_dp(num):
    """ 
    Does some rounding-down chicanery because the numbers weren't quite right
    """
    return math.floor(num*10) / 10


def printSaveTree():
    pass;


def dist(seq1, seq2):
    # Helper method for finding distances
    
    distance  = 0
    for x, y in zip(seq1, seq2):
        if x != y: 
            distance += 1
    return distance


def distance_matx(sequences):
    # Do the comparisons and create distance matrix
    # Takes a list of sequences
    # Returns an integer distance matrix
    
    # Empty matrix
    distance_matrix = np.zeros((len(sequences), len(sequences)), dtype=int)
    
    # Do the comparisons
    x, y = 0, 0
    for seqA in sequences:
        for seqB in sequences:
            score = dist(seqA, seqB)
            distance_matrix[x][y] = score
            y += 1
        x += 1
        y = 0

    return distance_matrix

def tril_matx(dist_matx):
    # Make it lower triangular by replacing upper tri w/ inf
    # removes duplicates and diagonal 0s
    
    dist_matx = dist_matx.astype(float)
    dist_matx[np.triu_indices(dist_matx.shape[0], 0)] = float("inf")
    
    return dist_matx

def min_cell(matx):
    # Takes a matrix
    # Return a list of tuples, with all indicies containing min value.
    
    mins = []
    # Use np to find a min
    min_dist = matx[np.unravel_index(np.argmin(matx, axis=None), matx.shape)]
    
    # Check if there's any other minima
    for i in range(len(matx)):
        for j in range(len(matx)):
            if matx[i][j] == min_dist:
                mins.append((i,j)) 
    
    return mins



def dist(seq1, seq2):
    # Helper method for finding distances
    
    distance  = 0
    for x, y in zip(seq1, seq2):
        if x != y: 
            distance += 1
    return distance

def distance_matx(sequences):
    # Do the comparisons and create distance matrix
    # Takes a list of sequences
    # Returns an integer distance matrix
    
    # Empty matrix
    distance_matrix = np.zeros((len(sequences), len(sequences)), dtype=int)
    
    # Do the comparisons
    x, y = 0, 0
    for seqA in sequences:
        for seqB in sequences:
            score = dist(seqA, seqB)
            distance_matrix[x][y] = score
            y += 1
        x += 1
        y = 0

    return distance_matrix

def tril_matx(dist_matx):
    # Make it lower triangular by replacing upper tri w/ inf
    # removes duplicates and diagonal 0s
    
    dist_matx = dist_matx.astype(float)
    dist_matx[np.triu_indices(dist_matx.shape[0], 0)] = float("inf")
    
    return dist_matx

def min_cell(matx):
    # Takes a matrix
    # Return a list of tuples, with all indicies containing min value.
    
    mins = []
    # Use np to find a min
    min_dist = matx[np.unravel_index(np.argmin(matx, axis=None), matx.shape)]
    
    # Check if there's any other minima
    for i in range(len(matx)):
        for j in range(len(matx)):
            if matx[i][j] == min_dist:
                mins.append((i,j)) 
    
    return mins

def calculateQ(d):
    r = d.shape[0]
    q = np.zeros((r,r))
    for i in range(r):
        for j in range(r):
            if i == j:
                q[i][j] = 0
            else:
                sumI = 0
                sumJ = 0
                for k in range(r):
                    sumI += d[i][k]
                    sumJ += d[j][k]
                q[i][j] = (r-2) * d[i][j] - sumI - sumJ

    return q

def findLowestPair(q):
    r = q.shape[0]
    minVal = max
    for i in range(0,r):
        for j in range(i,r):
            if (q[i][j] < minVal):
                minVal = q[i][j]
                minIndex = (i,j)

    print("Min Val: " + str(minVal));

    return minIndex


def doDistOfPairMembersToNewNode(i,j,d):
    r = d.shape[0]
    sumI = 0
    sumJ = 0


    global fileCount;
    global matrixCount;
    global clusterCount;


    if((2.0 * (r - 2.0)) != 0):

        print("Matrix #" + str(fileCount) + "\n",d)


        return (-1,-1);

    # End Condition, Save files of information
    else:



        # Write matrix count to a file
        matrixCount = fileCount-1;
        print("There are: " + str(matrixCount) + " matrices");

        print("Outputfile matrixCount:");
        out1 = "matrixCount.o";
        print(out1);
        print("");

        if not os.path.isfile(out1):
            open(out1, 'w').close();

        out = open(out1, 'w');
        out.write(str(matrixCount));


        # Write matrix merge count to a file
        clusterCount = fileCount-1;
        print("There are: " + str(clusterCount) + " clusters");

        print("Outputfile mergeCount:");
        out1 = "mergeCount.o";
        print(out1);
        print("");

        if not os.path.isfile(out1):
            open(out1, 'w').close();

        out = open(out1, 'w');
        out.write(str(clusterCount));






def calculateNewDistanceMatrix(f,g,d):
    print (d)
    r = d.shape[0]
    nd = np.zeros((r-1,r-1))

    # Copy over the old data to this matrix
    ii = jj = 1
    for i in range(0,r):
        if i == f or i == g:
            continue
        for j in range(0,r):
            if j == f or j == g:
                continue
            nd[ii][jj] = d[i][j]
            jj += 1
        ii += 1
        jj = 1
            
    # Calculate the first row and column
    ii = 1
    for i in range (0,r):
        if i == f or i == g:
            continue
        nd[0][ii] = (d[f][i] + d[g][i] - d[f][g]) / 2.0
        nd[ii][0] = (d[f][i] + d[g][i] - d[f][g]) / 2.0
        ii += 1

    return nd


def doNeighbourJoining(d, seq_labels):

    print("this is the distance Matrix: d\n",d);
    


    global fileCount;
    global clusterCount;
    global mergedAlready;


    filegen.makeXML_blank("./trees/tree0.xml")
    filegen.makePNG("./trees/tree0.xml",
                    "./trees/tree0")

    leaf_nodes = [] 

    for sequence in seq_labels:
        leaf_nodes.append(Node(sequence, 0))

    matx = np.array(d);
    matx = matx.astype(float)


    # Save this matrix, d, into an output file e.g "neighborMatrix.o1" or "neighborMatrix.o2" etc.
    print("Outputfile" + str(fileCount) + ":");
    out1 = "Matrix.o" + str(fileCount);
    print(out1);
    print("");
    if not os.path.isfile(out1):
        open(out1,'w').close();

    out = open(out1, 'w');
    out.write(str(d.tolist()));

    fileCount = fileCount + 1;


    #ITERATE
    while len(seq_labels) > 2:

        # Calculate q score matrix
        q = calculateQ(matx)
        print("this is q:\n",q)

        # Save this matrix, q, into an output file e.g "qMatrix.o1" or "neighbq.o2" etc.
        print("Outputfile" + str(fileCount-1) + ":");
        out1 = "qMatrix.o" + str(fileCount-1);
        print(out1);
        print("");
        if not os.path.isfile(out1):
            open(out1,'w').close();

        out = open(out1, 'w');
        out.write(str(q.tolist()));        



        ### this code is finding the first lowest pair ###
        lowestPair = findLowestPair(q)
        print ("Joining:")
        print (lowestPair[0])
        print (lowestPair[1])
        i = lowestPair[0]
        j = lowestPair[1]
        seq_i_idx = i;
        seq_j_idx = j;

        if seq_i_idx > seq_j_idx: # Put them in ascending order
            seq_i_idx, seq_j_idx = seq_j_idx, seq_i_idx
            
        # Get dij
        dij = matx[seq_i_idx][seq_j_idx]



        # Honestly this code is not even used anywhere. Use it to output files.
        pairDist = doDistOfPairMembersToNewNode(i,j,matx);

        # Calculate Averages
        d = calculateNewDistanceMatrix(i,j,matx);


        # Reassign matrix for next iter
        matx = d;
        matx = matx.astype(float)


        # Define label for new cluster k
        k_label = "S" + seq_labels[seq_i_idx].strip("S") + seq_labels[seq_j_idx].strip("S")


        # Write merged clusters to file
        clusterRes = str(seq_labels[seq_i_idx].strip("S")) + " " + str(seq_labels[seq_j_idx].strip("S"));
        print("Outputfile " + str(clusterCount) +":");
        out1 = "Cluster.o" + str(clusterCount);
        print(out1);
        print("");

        if not os.path.isfile(out1):
            open(out1, 'w').close();

        out = open(out1, 'w');
        out.write(str(clusterRes));


        # Remove merged sequence labels from list and add the merged one
        del seq_labels[seq_j_idx]; del seq_labels[seq_i_idx]
        seq_labels.insert(0, k_label)

        # Create node, assign height and children
        new_node = Node(k_label, round(round_down_1_dp(dij / 2),1), leaf_nodes[seq_i_idx], leaf_nodes[seq_j_idx])
        del leaf_nodes[seq_j_idx]; del leaf_nodes[seq_i_idx]
        leaf_nodes.insert(0, new_node)

        filegen.makeXML(new_node, "./trees/tree{}.xml".format(str(clusterCount)))
        filegen.makePNG("./trees/tree{}.xml".format(str(clusterCount)),
                        "./trees/tree{}".format(str(clusterCount)))


        # New Matrix Created
        matrixFileOutput(matx);



        # Honestly this code is not even used anywhere. Use it to output files.
        pairDist = doDistOfPairMembersToNewNode(i,j,matx)
        


        # Increment Everything
        fileCount = fileCount + 1;
        clusterCount = clusterCount + 1;


    # TERMINATE
    
    # Only 2 clusters remain
    # Define label for root cluster k
    # Get distance, create final node

    k_label = "S" + seq_labels[0].strip("S") + seq_labels[1].strip("S")


    # Write merged clusters to file
    clusterRes = str(seq_labels[0].strip("S")) + " " + str(seq_labels[1].strip("S"));
    print("Outputfile " + str(clusterCount) +":");
    out1 = "Cluster.o" + str(clusterCount);
    print(out1);
    print("");

    if not os.path.isfile(out1):
        open(out1, 'w').close();

    out = open(out1, 'w');
    out.write(str(clusterRes));



    dij = matx[0][1]
    root = Node(k_label, (dij / 2), leaf_nodes[0], leaf_nodes[1])

    filegen.makeXML(root, "./trees/tree{}.xml".format(str(clusterCount)))
    filegen.makePNG("./trees/tree{}.xml".format(str(clusterCount)),
                    "./trees/tree{}".format(str(clusterCount)))


    # Output last qmatrix
    q = calculateQ(matx)
    print("this is q:\n",q)

    # Save this matrix, q, into an output file e.g "qMatrix.o1" or "neighbq.o2" etc.
    print("Outputfile" + str(fileCount-1) + ":");
    out1 = "qMatrix.o" + str(fileCount-1);
    print(out1);
    print("");
    if not os.path.isfile(out1):
        open(out1,'w').close();

    out = open(out1, 'w');
    out.write(str(q.tolist()));  

    # Honestly this code is not even used anywhere. Use it to output files.
    pairDist = doDistOfPairMembersToNewNode(i,j,matx)
      


    return root
    


 ### this is the main method where we start running of the neighbour joining methods ##   
def run(distMatrix, seq_labels):
    doNeighbourJoining(distMatrix, seq_labels)



def runNeighbour(filename):

    ### trying to read the file from here ###
    try:
        file = open(filename)
        print("this file exist")

        # Write filename to a file
        print("Outputfile " + str("3.in") +":");
        out1 = "inputFileName.o";
        print(out1);
        print("");

        if not os.path.isfile(out1):
            open(out1, 'w').close();

        out = open(out1, 'w');
        out.write(str("3.in"));

    except:
        print("Error reading input file")


    in_sequences = []
    in_seq_labels = []

    # Assumes format of header line followed by sequence line
    while True:
        seq_name = file.readline().strip(">\n") # Read header, strip stuff
        sequence = file.readline().strip() # Read sequence, strip newline
        if not sequence: break
        # sequences[seq_name] = sequence # Add to dict
        in_sequences.append(sequence)
        in_seq_labels.append(seq_name)

    print("Sequences read: ")
    print(in_sequences)
    print(in_seq_labels)


    int_dist_matx = distance_matx(in_sequences)
    print("Distance Matrix:");
    print(int_dist_matx)




    # Start
    matx_copy = int_dist_matx.copy();
    labels_copy = in_seq_labels.copy();

    run(matx_copy, labels_copy);


    # Testing
    # data = int_dist_matx;
    # ids = ["S1", "S2", "S3", "S4", "S5"];

    # dm = DistanceMatrix(data, ids);

    # tree = nj(dm);
    # print(tree.ascii_art());




# runNeighbour("3.in");




# http://scikit-bio.org/docs/0.2.1/generated/skbio.tree.nj.html
