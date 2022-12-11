'''Merge Sort'''

def mergesort(arr):
    n = len(arr)

    if n <= 1:
        return arr
    else:
        mid = n // 2
        left, right = arr[:mid], arr[mid:]

        mergesort(left);mergesort(right)

        l = r = k = 0

        while l < len(left) and r < len(right):
            if left[l] < right[r]:
                arr[k] = left[l]
                l += 1
            else:
                arr[k] = right[r]
                r += 1
            k += 1

        while l < len(left):
            arr[k] = left[l]
            l += 1
            k += 1

        while r < len(right):
            arr[k] = right[r]
            r += 1
            k += 1

myl = [7,0,9,9,1,3,3]
mergesort(myl)
print(myl)


'''Quick Sort'''

def quick_sort(arr) :
    if len(arr) <= 1:
        return arr
    else :
        pivot = arr.pop()
    greater_list = []
    lesser_list = []

    for ele in arr :
        if ele > pivot :
            greater_list.append(ele)
        else :
            lesser_list.append(ele)
    return quick_sort(lesser_list) + [pivot] + quick_sort(greater_list)

print(quick_sort([12,34,1,2,63,98,43]))


'''Warshall prolem'''

from math import inf
g = [[0,3,inf,5],[2,0,inf,4],[inf,1,0,inf],[inf,inf,2,0]]

for k in range(len(g)) :
    for i in range(len(g)) :
        for j in range(len(g)) :
            g[i][k] = min( g[i][k] , g[i][j] + g[j][k] )

for i in g :
    print(' '.join([str(j) for j in i]))


'''Knapsack Problem'''

def knapSack(W, wt, val, n):

    if n == 0 or W == 0:
        return 0
 
    if (wt[n-1] > W):
        return knapSack(W, wt, val, n-1)
 
    else:
        return max( val[n-1] + knapSack( W-wt[n-1], wt, val, n-1), knapSack(W, wt, val, n-1))

val = [60, 100, 120]
wt = [10, 20, 30]
W = 50
n = len(val)
print (knapSack(W, wt, val, n))


'''Dijkstra algorithm'''

import sys
class Graph():

	def __init__(self, vertices):
		self.V = vertices

	def printSolution(self, dist):
		print("Vertex \tDistance from Source")
		for node in range(self.V):
			print(node, "\t", dist[node])

	def minDistance(self, dist, sptSet):

		min = sys.maxsize
		for u in range(self.V):
			if dist[u] < min and sptSet[u] == False:
				min = dist[u]
				min_index = u

		return min_index

	def dijkstra(self, src):

		dist = [sys.maxsize] * self.V
		dist[src] = 0
		sptSet = [False] * self.V

		for cout in range(self.V):

			x = self.minDistance(dist, sptSet)
			sptSet[x] = True

			for y in range(self.V):
				if self.graph[x][y] > 0 and sptSet[y] == False and dist[y] > dist[x] + self.graph[x][y] :
					dist[y] = dist[x] + self.graph[x][y]

		self.printSolution(dist)

g = Graph(9)
g.graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
            [4, 0, 8, 0, 0, 0, 0, 11, 0],
            [0, 8, 0, 7, 0, 4, 0, 0, 2],
            [0, 0, 7, 0, 9, 14, 0, 0, 0],
            [0, 0, 0, 9, 0, 10, 0, 0, 0],
            [0, 0, 4, 14, 10, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 1, 6],
            [8, 11, 0, 0, 0, 0, 1, 0, 7],
            [0, 0, 2, 0, 0, 0, 6, 7, 0]
        ]

g.dijkstra(0)


''' Travelling Salesman Problem '''

from sys import maxsize
from itertools import permutations
V = 4

def travellingSalesmanProblem(graph, s):
 
    vertex = []
    for i in range(V):
        if i != s:
            vertex.append(i)

    min_path = maxsize
    next_permutation = permutations(vertex)
    for i in next_permutation:

        current_pathweight = 0
        k = s
        for j in i:
            current_pathweight += graph[k][j]
            k = j
        current_pathweight += graph[k][s]
 
        min_path = min(min_path, current_pathweight)

    return min_path

graph = [[0, 10, 15, 20], [10, 0, 35, 25],
        [15, 35, 0, 30], [20, 25, 30, 0]]
s = 0
print(travellingSalesmanProblem(graph, s))


''' Kruskal '''

class Graph:

	def __init__(self, vertices):
		self.V = vertices
		self.graph = []

	def addEdge(self, u, v, w):
		self.graph.append([u, v, w])

	def find(self, parent, i):
		if parent[i] != i:
	
			parent[i] = self.find(parent, parent[i])
		return parent[i]

	def union(self, parent, rank, x, y):
	
		if rank[x] < rank[y]:
			parent[x] = y
		elif rank[x] > rank[y]:
			parent[y] = x

		else:
			parent[y] = x
			rank[x] += 1

	def KruskalMST(self):

		result = []
		i = 0
		e = 0

		self.graph = sorted(self.graph, key=lambda item: item[2])

		parent = []
		rank = []

		for node in range(self.V):
			parent.append(node)
			rank.append(0)

		while e < self.V - 1:

			u, v, w = self.graph[i]
			i = i + 1
			x = self.find(parent, u)
			y = self.find(parent, v)

			if x != y:
				e = e + 1
				result.append([u, v, w])
				self.union(parent, rank, x, y)

		minimumCost = 0
		print("Edges in the constructed MST")
		for u, v, weight in result:
			minimumCost += weight
			print("%d -- %d == %d" % (u, v, weight))
		print("Minimum Spanning Tree", minimumCost)

g = Graph(4)
g.addEdge(0, 1, 10)
g.addEdge(0, 2, 6)
g.addEdge(0, 3, 5)
g.addEdge(1, 3, 15)
g.addEdge(2, 3, 4)

g.KruskalMST()

''' Prims '''

import sys
class Graph():

	def __init__(self, vertices):
		self.V = vertices

	def printMST(self, parent):
		print("Edge \tWeight")
		for i in range(1, self.V):
			print(parent[i], "-", i, "\t", self.graph[i][parent[i]])

	def minKey(self, key, mstSet):

		min = sys.maxsize
		for v in range(self.V):
			if key[v] < min and mstSet[v] == False:
				min = key[v]
				min_index = v

		return min_index

	def primMST(self):

		key = [sys.maxsize] * self.V
		parent = [None] * self.V
		key[0] = 0
		mstSet = [False] * self.V
		parent[0] = -1

		for cout in range(self.V):

			u = self.minKey(key, mstSet)
			mstSet[u] = True

			for v in range(self.V):

				if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
					key[v] = self.graph[u][v]
					parent[v] = u

		self.printMST(parent)

g = Graph(5)
g.graph = [[0, 2, 0, 6, 0],
        [2, 0, 3, 8, 5],
        [0, 3, 0, 0, 7],
        [6, 8, 0, 0, 9],
        [0, 5, 7, 9, 0]]

g.primMST()

