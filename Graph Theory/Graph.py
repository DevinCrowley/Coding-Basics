"""
About Graph.py:

This file is intended as a general structure for other programs 
to use in their implementations of algorithms in graph theory.

This file contains:
	- The Node class
	- The Graph class
	- Functions to print or return string representations of selections from a Graph object
	- Functions to generate particular Graph objects, primarily for testing purposes

A Node contains:
	- A unique key
	- An optional value
	- A parent attribute that may be assigned to another Node
	- A distance, i.e. the distance from some other Node
	- A discoveryTime, i.e. when the Node was first encountered in a search
	- A finishTime, i.e. when the Node and all Nodes accessible from it 
		have been explored in a search.

A Graph contains:
	- An adjacencyMap of the form 
		{key : {(adjacentKey1, edge weight), (adjacentKey2, edge weight)}}
	- A valueMap of the form {key : value}
	- A vertexMap of the form {key, Node}
"""

from collections import defaultdict
from math import inf as infinity
import numpy as np
import unittest

r"""
   ____    _         _                 _         
  / __ \  | |       (_)               | |        
 | |  | | | |__      _    ___    ___  | |_   ___ 
 | |  | | | '_ \    | |  / _ \  / __| | __| / __|
 | |__| | | |_) |   | | |  __/ | (__  | |_  \__ \
  \____/  |_.__/    | |  \___|  \___|  \__| |___/
                   _/ |                          
                  |__/                           
"""

class Node:
	"""key is a unique value to identify each Node. value is optional satellite data."""

	def __init__(self, key, value = None):
		self.key = key
		self.value = value

		self.parent = None
		self.searchStatus = "undiscovered"

		self.distance = infinity

		self.discoveryTime = infinity
		self.finishTime = infinity

	def __repr__(self):
		return f"key: {self.key}, value: {self.value}."


class Graph:
	"""
	Graph is a collection of Node objects and directional edges between them.

	keySet is the set of unique keys for the Nodes.
	adjacencyMap is a dictionary of edges of the form {key : {keys of all adjacent Nodes}}.

	Attributes:

	adjacencyMap --> dictionary of the form 
					adjacencyMap[key] = {(adjacentKey1, edge weight), (adjacentKey2, edge weight)}.
					If provided as {key : {adjacent keys}}, edge weights of 1 will be added.
	valueMap ------> defaultdict(lambda: None) of the form valueMap[key] = value.
	vertexMap -----> dictionary of the form vertexMap[key] = corresponding Node.

	IMPROVEMENT: remove keySet and make adjacencyMap a defaultdict.
				Update Test_Graph and provide Graph with a __repr__ method.
	"""

	def fixArguments(self, adjacencyMap, valueMap):
		"""
		Check that adjacencyMap and valueMap are dictionaries.

		Add default edge weights of 1 if not provided.

		Check each edge in adjacencyMap and ensure that there is an entry for the corresponding adjacent key.

		Check each key in valueMap and ensure it is represented in adjacencyMap.
		"""

		# Verify that adjacencyMap and valueMap are dictionaries; valueMap may be None.
		if not isinstance(adjacencyMap, dict):
			raise TypeError("adjacencyMap must be a dictionary.")
		if valueMap is not None and not isinstance(valueMap, dict):
			raise TypeError("valueMap must be a dictionary.")
		# Argument types verified.

		# Convert valueMap to a defaultdict with a default value of None, 
		# or set it as one if it was initialized as None.
		if valueMap is not None:
			valueMap = defaultdict(lambda: None, valueMap)
		else:
			valueMap = defaultdict(lambda: None)

		# Add default edge weights of 1 if not provided.
		for key, adjacencySet in adjacencyMap.items():
			newAdjacencySet = set()
			for edge in adjacencySet:
				if not isinstance(edge, tuple):
					edge = (edge, 1)
				newAdjacencySet.add(edge)
			adjacencyMap[key] = newAdjacencySet
		
		# Correct adjacencyMap to include an entry for each key encountered.
		staticAdjacencyMapValues = list(adjacencyMap.values())
		for adjacencySet in staticAdjacencyMapValues:
			for edge in adjacencySet:
				# For each adjacent key, ensure there is a corresponding entry in adjacencyMap.
				adjacentKey = edge[0]
				if adjacentKey not in adjacencyMap:
					adjacencyMap[adjacentKey] = set()

		# Incorporate valueMap's keys into adjacencyMap.
		for key in valueMap.keys():
			# Equivalent line: adjacencyMap.setdefault(key, set())
			if key not in adjacencyMap.keys():
				adjacencyMap[key] = set()

		return adjacencyMap, valueMap


	def __init__(self, adjacencyMap, valueMap = None):

		# If adjacencyMap or valueMap are invalid, this method raises an exception.
		# Else, it adjusts adjacencyMap to include an entry for all keys in both 
		# the edges it contains and the keys in valueMap, and converts valueMap to a 
		# defaultdict with default value None. 
		# Returns corrected (adjacencyMap, valueMap).

		self.adjacencyMap, self.valueMap = self.fixArguments(adjacencyMap, valueMap)

		# Generate and store Nodes
		self.vertexMap = {}
		for key in adjacencyMap.keys():
			# If self.valueMap does not contain key, self.valueMap[key] defaults to None.
			self.vertexMap[key] = Node(key, self.valueMap[key])


	def __repr__(self):
		"""
		Returns a string representing self.adjacencyMap and self.valueMap.
		Return value is of the form:
			"Key: {key1}, value: {value1}, parent: {parent1}, edges --> [(adjacentKey1, weight1), (adjacentKey2, weight2)...]
			"Key: {key2}, value: {value2}, parent: {parent2}, edges --> [(adjacentKey1, weight1), (adjacentKey2, weight2)...]
		"""

		reprString = ""
		# keyList used to enforce consistent order on key iteration.
		keyList = list(self.adjacencyMap.keys())
		keyList.sort()
		for key in keyList:
			sortedAdjacencySet = list(self.adjacencyMap[key])
			sortedAdjacencySet.sort()
			# f-string expressions insist on not having line breaks.
			reprString += 	f"key: {key : >3}, value: {str(self.valueMap[key]) : >4}, "\
							f"parent: {self.vertexMap[key].parent.key if self.vertexMap[key].parent is not None else str(None) : >4}, "\
							f"edges --> {sortedAdjacencySet}"
			# Each key is unique since it came from a set.
			if key != keyList[-1]:
				reprString += "\n"

		return reprString


	def make_undirected(self):
		"""
		For each edge A --> B, ensure that the opposite edge B --> A is present.
		Runtime is O(|vertices| + |edges|).
		"""

		for key, adjacencySet in self.adjacencyMap.items():
			for edge in adjacencySet:
				adjacentKey = edge[0]
				# Since the values in adjacencyMap are sets, we need not check for membership.
				self.adjacencyMap[adjacentKey].add((key, edge[1]))


	def reverse_edges(self, reverseKeySet = None):
		"""
		Replaces each edge in self of the form A --> B with B --> A, if A is in reverseKeySet. 
		Put another way, reverses all outgoing edges from each key in reverseKeySet.

		If reverseKeySet is not provided, all edges are reversed.
		"""

		if reverseKeySet is None:
			reverseKeySet = set(self.adjacencyMap.keys())

		# Check that reverseKeySet is a set.
		if not isinstance(reverseKeySet, set):
			raise KeyError("reverseKeySet must be a set.")
		# If there are any keys in reverseKeySet that are not in graph raise KeyError.
		if not reverseKeySet <= self.adjacencyMap.keys():
			raise KeyError("reverseKeySet contains keys not in self.adjacencyMap.")

		# Create blank adjacencyMap with an entry for each key in self.
		newAdjacencyMap = {key : set() for key in self.adjacencyMap.keys()}

		# Add reversed edges.
		for key in reverseKeySet:
			for edge in self.adjacencyMap[key]:
				adjacentKey = edge[0]
				newAdjacencyMap[adjacentKey].add((key, edge[1]))
		# Add non-reversed edges.
		for key in self.adjacencyMap.keys() - reverseKeySet:
			newAdjacencyMap[key] |= self.adjacencyMap[key]

		# Replace self.adjacencyMap.
		self.adjacencyMap = newAdjacencyMap


	def put(self, newAdjacencyMap, newValueMap = None):
		"""
		Checks arguments as with initialization, then adds new keys and values to self. 
		If a key is included that already exists in self, its new edges overlap with its old ones. 
		That is, the result state of a key's edges is the union of its old edges and new ones.
		Values are overridden.
		"""

		newAdjacencyMap, newValueMap = self.fixArguments(newAdjacencyMap, newValueMap)

		# Incorporate newAdjacencyMap into self.adjacencyMap.
		for key, edges in newAdjacencyMap.items():
			self.adjacencyMap.setdefault(key, set()).update(edges)

		# Incorporate newValueMap into self.valueMap.
		for key, value in newValueMap.items():
			self.valueMap[key] = value

		# Update self.vertexMap to include all new Nodes.
		for newKey in newAdjacencyMap.keys() - self.vertexMap.keys():
			# self.valueMap is a defaultdict with default value None.
			self.vertexMap[newKey] = Node(newKey, self.valueMap[newKey])

		# Update the appropriate Nodes' value attributes.
		for key in newValueMap.keys():
			self.vertexMap[key].value = self.valueMap[key]


	def reset(self):
		"""Resets the attribute values for each Node to their initialization values."""

		for node in self.vertexMap.values():
			node.parent = None
			node.searchStatus = "undiscovered"
			node.distance = infinity
			node.discoveryTime = infinity
			node.finishTime = infinity


	def adjacentKeys(self, key):
		"""Returns the set of keys to which there is an edge from key."""
		# self.adjacencyMap[key] --> {(adjacentKey1, weight1), adjacentKey2, weight2), ...}
		return set(map(lambda edge: edge[0], self.adjacencyMap[key]))

r"""
   _____                          _         _____    _                 _                       
  / ____|                        | |       |  __ \  (_)               | |                      
 | |  __   _ __    __ _   _ __   | |__     | |  | |  _   ___   _ __   | |   __ _   _   _   ___ 
 | | |_ | | '__|  / _` | | '_ \  | '_ \    | |  | | | | / __| | '_ \  | |  / _` | | | | | / __|
 | |__| | | |    | (_| | | |_) | | | | |   | |__| | | | \__ \ | |_) | | | | (_| | | |_| | \__ \
  \_____| |_|     \__,_| | .__/  |_| |_|   |_____/  |_| |___/ | .__/  |_|  \__,_|  \__, | |___/
                         | |                                  | |                   __/ |      
                         |_|                                  |_|                  |___/       
"""

def print_path(rootNode, targetNode):
    """Prints each successive parent of targetNode until rootNode or None is encountered."""

    if targetNode == rootNode:
        print(f"key: {targetNode.key}.");
    elif targetNode.parent == None:
        print(f"No path from Node {targetNode.key} to Node {rootNode.key}.")
    else:
        print_path(rootNode, targetNode.parent)
        print(targetNode.value)


def print_edges(graph):
    """For each vertex, print_edges prints each outgoing edge on its own line."""

    adjacencyMapList = list(graph.adjacencyMap.items())
    adjacencyMapList.sort()
    for key, edges in adjacencyMapList:
    	edges = list(edges)
    	edges.sort()
    	print(f"key: {key}, edges: {edges}.")


def print_path_to_root(graph, rootKey):
    """
    Follows the parent pointer of rootKey and its ancestors and prints each key 
    until a cycle is detected or None is encountered.
    """

    tracerNode = graph.vertexMap[rootKey]
    pathNodes = [tracerNode]
    while tracerNode != None:
        print(f"key: {tracerNode.key}.")
        tracerNode = tracerNode.parent
        if tracerNode not in pathNodes:
            pathNodes.append(tracerNode)
        else:
            raise Exception("Cycle detected.")


def path_to_string(graph, rootKey, targetKey, reverse = False):
	"""
	Returns a string representation of the path from targetKey to rootKey 
	through each Node's parent attribute, or None if there is no such path.

	Cycles are aborted and treated the same as if None is encountered, 
	i.e. if a Node along the path up from targetKey has parent value None.

	If reverse = True, then the path follows the parent pointers in reverse 
	from rootKey to targetKey.
	"""

	# Check rootKey and targetKey for membership in graph.
	if rootKey not in graph.adjacencyMap.keys():
		raise KeyError(f"key {rootKey} not found in graph.")
	if targetKey not in graph.adjacencyMap.keys():
		raise KeyError(f"key {targetKey} not found in graph.")

	rootNode = graph.vertexMap[rootKey]
	targetNode = graph.vertexMap[targetKey]

	pathNodes = [targetNode]
	while targetNode is not None and targetNode != rootNode:
		targetNode = targetNode.parent

		# Check for cycles.
		if targetNode in pathNodes:
			return None
		else:
			pathNodes.append(targetNode)

	# If targetNode is None, there was no path to rootKey.
	if targetNode is None:
		return None

	# Construct string representation.
	
	if reverse: pathNodes.reverse()

	pathString = f"[{pathNodes[0].key}"
	for node in pathNodes[1:]:
		pathString += f", {node.key}"
	pathString += "]"

	return pathString


def path_to_root_to_string(graph, startKey, reverse = False):
	"""
	Returns a string representation of the successive keys from startKey to 
	the nearest root by following each Node's parent pointers.

	If reverse == True, then the path returned traverses the parent pointers 
	backwards from the nearest root to startKey.

	If a cycle is detected, then None is returned.
	"""

	# Check startKey for membership in graph.
	if startKey not in graph.adjacencyMap.keys():
		raise KeyError(f"key {startKey} not found in graph.")

	tracerNode = graph.vertexMap[startKey]
	pathNodes = [tracerNode]

	while tracerNode.parent is not None:
		tracerNode = tracerNode.parent

		# Check for cycles
		if tracerNode in pathNodes:
			return None
		else:
			pathNodes.append(tracerNode)

	# Construct string representation.

	if reverse: pathNodes.reverse()

	pathString = f"[{pathNodes[0].key}"
	for node in pathNodes[1:]:
		pathString += f", {node.key}"
	pathString += "]"

	return pathString

r"""
   _____                               _             _____                          _           
  / ____|                             | |           / ____|                        | |          
 | (___     __ _   _ __ ___    _ __   | |   ___    | |  __   _ __    __ _   _ __   | |__    ___ 
  \___ \   / _` | | '_ ` _ \  | '_ \  | |  / _ \   | | |_ | | '__|  / _` | | '_ \  | '_ \  / __|
  ____) | | (_| | | | | | | | | |_) | | | |  __/   | |__| | | |    | (_| | | |_) | | | | | \__ \
 |_____/   \__,_| |_| |_| |_| | .__/  |_|  \___|    \_____| |_|     \__,_| | .__/  |_| |_| |___/
                              | |                                          | |                  
                              |_|                                          |_|                  
"""

def multiply_graph(graph, multiplicationFactor):
	"""
	Returns a graph consisting of multiplicationFactor copies of the input graph, 
	except with all their keys shifted by multiples of the range of key values in the input graph.
	The input graph's valueMap is shifted in the same manner.

	If graph is a tree, returns a forest with multiplicationFactor trees.
	"""

	# Verify multiplicationFactor.
	if not isinstance(multiplicationFactor, int) or multiplicationFactor < 1:
		raise ValueError("multiplicationFactor must be a positive integer.")

	# Find the extrema of graph's keys to determine the range of key values.
	smallestKey, largestKey = min(graph.adjacencyMap.keys()), max(graph.adjacencyMap.keys())
	keyRange = largestKey - smallestKey + 1

	# Make shifted copies of graph's adjacencyMap and valueMap.
	newAdjacencyMap = dict(graph.adjacencyMap)
	newValueMap = dict(graph.valueMap)
	for copyNumber in range(1, multiplicationFactor):
		for key in graph.adjacencyMap.keys():
			# Shift adjacent keys up by keyRange * copyNumber, leaving edge weights unchanged.
			newAdjacencySet = {(newKey, weight) for (newKey, weight) in \
				map(lambda edge: (edge[0] + keyRange * copyNumber, edge[1]), graph.adjacencyMap[key])}
			# Shift keys entries.
			newAdjacencyMap[key + keyRange * copyNumber] = newAdjacencySet
			# Shift valueMap.
			if graph.valueMap[key] is None:
				newValue = None
			else:
				newValue = graph.valueMap[key] + keyRange * copyNumber
			newValueMap[key + keyRange * copyNumber] = newValue

	# Create and return new Graph object.
	return Graph(newAdjacencyMap, newValueMap)


def sample_undirected_forest(shellDepth = 2, numTrees = 3, treeValueMap = None):
	"""Wraps multiply_graph around sample_undirected_graph."""
	return multiply_graph(sample_undirected_graph(shellDepth, treeValueMap), numTrees)


def sample_dag_forest(shellDepth = 2, numTrees = 3, treeValueMap = None):
	"""Wraps multiply_graph around sample_dag."""
	return multiply_graph(sample_dag(shellDepth, treeValueMap), numTrees)


def sample_undirected_graph(shellDepth = 3, valueMap = None):
	"""
	Returns a Graph object with some interesting structure. 
	Size scales nonlinearly with shellDepth.
	"""

	# To understand it, I recommend drawing the first half based on the initialization of adjacencyMap,
	# or using the print_edges(graph) function to show each vertex's outgoing edges.
	# This can make constructing the graph quite straightforward. 
	# Drawn as if it were a tree rooted at 0, this graph is rotationally symmetric.

	# For shellDepth = 3, 
	# numVertices = 1 + 3 + 3*2 + (3*2*2) + 3*2 + 3 + 1, 
	# and numCenterLine = 3*2*2.
	numCenterLine = 3 * 2**(shellDepth - 1)
	numVertices = (1 + sum([3 * 2**i for i in range(shellDepth - 1)])) * 2 + numCenterLine

	# First half of sample graph, built from the bottom up.
	adjacencyMap = {key : {2 * key + 1, 2 * key + 2, 2 * key + 3} 
					for key in range(0, int((numVertices - numCenterLine) / 2))}

	# Add intermediate vertices - redundant with Graph.fixArguments.
	for key in range(int((numVertices - numCenterLine) / 2), int((numVertices + numCenterLine) / 2)):
		adjacencyMap[key] = set()

	# Second half of sample map, built from the top down.
	for key in range(numVertices - 1, int((numVertices + numCenterLine) / 2) - 1, -1):
		reverseDistance = numVertices - 1 - 2 * (numVertices - 1 - key)
		adjacencyMap[key] = {reverseDistance - 1, reverseDistance - 2, reverseDistance - 3}

	# Both the top and bottom halves of adjacencyMap have been created, but there is no 
	# edge connecting them since only their inward directional edges exist.
	# To correct for this, we make all edges reciprocal with Graph.make_undirected().

	graph = Graph(adjacencyMap, valueMap)
	graph.make_undirected()

	return graph


def sample_dag(shellDepth = 3, valueMap = None):
	"""
	Returns a directed, acyclic Graph object with some interesting structure.
	Size scales nonlinearly with shellDepth.
	"""

	# For shellDepth = 3, 
	# numVertices = 1 + 3 + 3*2 + (3*2*2) + 3*2 + 3 + 1, 
	# and numCenterLine = 3*2*2.
	numCenterLine = 3 * 2**(shellDepth - 1)
	numVertices = (1 + sum([3 * 2**i for i in range(shellDepth - 1)])) * 2 + numCenterLine

	# First half of sample dag, built from the bottom up.
	adjacencyMap = {key : {2 * key + 1, 2 * key + 2, 2 * key + 3} 
					for key in range(0, int((numVertices - numCenterLine) / 2))}

	# Second half of sample map, built from the top down.
	for key in range(numVertices - 1, int((numVertices + numCenterLine) / 2) - 1, -1):
		reverseDistance = numVertices - 1 - 2 * (numVertices - 1 - key)
		adjacencyMap[key] = {reverseDistance - 1, reverseDistance - 2, reverseDistance - 3}

	# Both the top and bottom halves of adjacencyMap have been created, 
	# but the top half's edges are all backwards from what is intended.
	# Currently all edges direct inwards, forming two trees rooted at 0 and numVertices - 1.

	graph = Graph(adjacencyMap, valueMap)
	reverseKeySet = {key for key in range(int((numVertices + numCenterLine) / 2), numVertices)}
	graph.reverse_edges(reverseKeySet)

	return graph

r"""
  _    _           _   _       _______                _         
 | |  | |         (_) | |     |__   __|              | |        
 | |  | |  _ __    _  | |_       | |      ___   ___  | |_   ___ 
 | |  | | | '_ \  | | | __|      | |     / _ \ / __| | __| / __|
 | |__| | | | | | | | | |_       | |    |  __/ \__ \ | |_  \__ \
  \____/  |_| |_| |_|  \__|      |_|     \___| |___/  \__| |___/
                                                                
                                                                
"""

class Test_Graph(unittest.TestCase):

	def test_sample_undirected_graph(self):
		# Verify the state of the generated graph.
		shellDepth3SampleUndirectedGraph = \
"""key:   0, value:  100, parent: None, edges --> [(1, 1), (2, 1), (3, 1)]
key:   1, value: None, parent: None, edges --> [(0, 1), (3, 1), (4, 1), (5, 1)]
key:   2, value: None, parent: None, edges --> [(0, 1), (5, 1), (6, 1), (7, 1)]
key:   3, value:  103, parent: None, edges --> [(0, 1), (1, 1), (7, 1), (8, 1), (9, 1)]
key:   4, value: None, parent: None, edges --> [(1, 1), (9, 1), (10, 1), (11, 1)]
key:   5, value: None, parent: None, edges --> [(1, 1), (2, 1), (11, 1), (12, 1), (13, 1)]
key:   6, value:  106, parent: None, edges --> [(2, 1), (13, 1), (14, 1), (15, 1)]
key:   7, value: None, parent: None, edges --> [(2, 1), (3, 1), (15, 1), (16, 1), (17, 1)]
key:   8, value: None, parent: None, edges --> [(3, 1), (17, 1), (18, 1), (19, 1)]
key:   9, value:  109, parent: None, edges --> [(3, 1), (4, 1), (19, 1), (20, 1), (21, 1)]
key:  10, value: None, parent: None, edges --> [(4, 1), (22, 1)]
key:  11, value: None, parent: None, edges --> [(4, 1), (5, 1), (22, 1)]
key:  12, value:  112, parent: None, edges --> [(5, 1), (22, 1), (23, 1)]
key:  13, value: None, parent: None, edges --> [(5, 1), (6, 1), (23, 1)]
key:  14, value: None, parent: None, edges --> [(6, 1), (23, 1), (24, 1)]
key:  15, value:  115, parent: None, edges --> [(6, 1), (7, 1), (24, 1)]
key:  16, value: None, parent: None, edges --> [(7, 1), (24, 1), (25, 1)]
key:  17, value: None, parent: None, edges --> [(7, 1), (8, 1), (25, 1)]
key:  18, value:  118, parent: None, edges --> [(8, 1), (25, 1), (26, 1)]
key:  19, value: None, parent: None, edges --> [(8, 1), (9, 1), (26, 1)]
key:  20, value: None, parent: None, edges --> [(9, 1), (26, 1), (27, 1)]
key:  21, value:  121, parent: None, edges --> [(9, 1), (27, 1)]
key:  22, value: None, parent: None, edges --> [(10, 1), (11, 1), (12, 1), (27, 1), (28, 1)]
key:  23, value: None, parent: None, edges --> [(12, 1), (13, 1), (14, 1), (28, 1)]
key:  24, value:  124, parent: None, edges --> [(14, 1), (15, 1), (16, 1), (28, 1), (29, 1)]
key:  25, value: None, parent: None, edges --> [(16, 1), (17, 1), (18, 1), (29, 1)]
key:  26, value: None, parent: None, edges --> [(18, 1), (19, 1), (20, 1), (29, 1), (30, 1)]
key:  27, value:  127, parent: None, edges --> [(20, 1), (21, 1), (22, 1), (30, 1)]
key:  28, value: None, parent: None, edges --> [(22, 1), (23, 1), (24, 1), (30, 1), (31, 1)]
key:  29, value: None, parent: None, edges --> [(24, 1), (25, 1), (26, 1), (31, 1)]
key:  30, value:  130, parent: None, edges --> [(26, 1), (27, 1), (28, 1), (31, 1)]
key:  31, value: None, parent: None, edges --> [(28, 1), (29, 1), (30, 1)]"""
		# Values are assigned to every third key.
		valueMap = {i : i + 100 for i in range(0, 32, 3)}
		graph = sample_undirected_graph(3, valueMap)
		self.assertEqual(repr(graph), shellDepth3SampleUndirectedGraph)


	def test_sample_dag(self):
		# Verify the state of the generated graph.
		shellDepth3SampleDAG = \
"""key:   0, value:  100, parent: None, edges --> [(1, 1), (2, 1), (3, 1)]
key:   1, value: None, parent: None, edges --> [(3, 1), (4, 1), (5, 1)]
key:   2, value: None, parent: None, edges --> [(5, 1), (6, 1), (7, 1)]
key:   3, value:  103, parent: None, edges --> [(7, 1), (8, 1), (9, 1)]
key:   4, value: None, parent: None, edges --> [(9, 1), (10, 1), (11, 1)]
key:   5, value: None, parent: None, edges --> [(11, 1), (12, 1), (13, 1)]
key:   6, value:  106, parent: None, edges --> [(13, 1), (14, 1), (15, 1)]
key:   7, value: None, parent: None, edges --> [(15, 1), (16, 1), (17, 1)]
key:   8, value: None, parent: None, edges --> [(17, 1), (18, 1), (19, 1)]
key:   9, value:  109, parent: None, edges --> [(19, 1), (20, 1), (21, 1)]
key:  10, value: None, parent: None, edges --> [(22, 1)]
key:  11, value: None, parent: None, edges --> [(22, 1)]
key:  12, value:  112, parent: None, edges --> [(22, 1), (23, 1)]
key:  13, value: None, parent: None, edges --> [(23, 1)]
key:  14, value: None, parent: None, edges --> [(23, 1), (24, 1)]
key:  15, value:  115, parent: None, edges --> [(24, 1)]
key:  16, value: None, parent: None, edges --> [(24, 1), (25, 1)]
key:  17, value: None, parent: None, edges --> [(25, 1)]
key:  18, value:  118, parent: None, edges --> [(25, 1), (26, 1)]
key:  19, value: None, parent: None, edges --> [(26, 1)]
key:  20, value: None, parent: None, edges --> [(26, 1), (27, 1)]
key:  21, value:  121, parent: None, edges --> [(27, 1)]
key:  22, value: None, parent: None, edges --> [(27, 1), (28, 1)]
key:  23, value: None, parent: None, edges --> [(28, 1)]
key:  24, value:  124, parent: None, edges --> [(28, 1), (29, 1)]
key:  25, value: None, parent: None, edges --> [(29, 1)]
key:  26, value: None, parent: None, edges --> [(29, 1), (30, 1)]
key:  27, value:  127, parent: None, edges --> [(30, 1)]
key:  28, value: None, parent: None, edges --> [(30, 1), (31, 1)]
key:  29, value: None, parent: None, edges --> [(31, 1)]
key:  30, value:  130, parent: None, edges --> [(31, 1)]
key:  31, value: None, parent: None, edges --> []"""
		# Values are assigned to every third key.
		valueMap = {i : i + 100 for i in range(0, 32, 3)}
		graph = sample_dag(3, valueMap)
		self.assertEqual(repr(graph), shellDepth3SampleDAG)


	def test_put(self):

		modifiedSampleUndirectedGraph = \
"""key:   0, value:  100, parent: None, edges --> [(1, 1), (2, 1), (3, 1)]
key:   1, value: None, parent: None, edges --> [(0, 1), (3, 1), (4, 1), (5, 1)]
key:   2, value: None, parent: None, edges --> [(0, 1), (5, 1), (6, 1), (7, 1)]
key:   3, value:  103, parent: None, edges --> [(0, 1), (1, 1), (7, 1), (8, 1), (9, 1)]
key:   4, value: None, parent: None, edges --> [(1, 1), (9, 1), (10, 1), (11, 1)]
key:   5, value: None, parent: None, edges --> [(1, 1), (2, 1), (11, 1), (12, 1), (13, 1)]
key:   6, value:  106, parent: None, edges --> [(2, 1), (13, 1), (14, 1), (15, 1)]
key:   7, value: None, parent: None, edges --> [(2, 1), (3, 1), (15, 1), (16, 1), (17, 1)]
key:   8, value: None, parent: None, edges --> [(3, 1), (17, 1), (18, 1), (19, 1)]
key:   9, value:  109, parent: None, edges --> [(3, 1), (4, 1), (19, 1), (20, 1), (21, 1)]
key:  10, value: None, parent: None, edges --> [(4, 1), (22, 1)]
key:  11, value: None, parent: None, edges --> [(4, 1), (5, 1), (22, 1)]
key:  12, value:  112, parent: None, edges --> [(5, 1), (22, 1), (23, 1)]
key:  13, value: None, parent: None, edges --> [(5, 1), (6, 1), (23, 1)]
key:  14, value: None, parent: None, edges --> [(6, 1), (23, 1), (24, 1)]
key:  15, value:  115, parent: None, edges --> [(6, 1), (7, 1), (24, 1)]
key:  16, value: None, parent: None, edges --> [(7, 1), (24, 1), (25, 1)]
key:  17, value: None, parent: None, edges --> [(7, 1), (8, 1), (25, 1)]
key:  18, value:  118, parent: None, edges --> [(8, 1), (25, 1), (26, 1)]
key:  19, value: None, parent: None, edges --> [(8, 1), (9, 1), (26, 1)]
key:  20, value: None, parent: None, edges --> [(9, 1), (26, 1), (27, 1)]
key:  21, value:  121, parent: None, edges --> [(9, 1), (27, 1)]
key:  22, value: None, parent: None, edges --> [(10, 1), (11, 1), (12, 1), (27, 1), (28, 1)]
key:  23, value: None, parent: None, edges --> [(12, 1), (13, 1), (14, 1), (28, 1)]
key:  24, value:  124, parent: None, edges --> [(14, 1), (15, 1), (16, 1), (28, 1), (29, 1)]
key:  25, value: None, parent: None, edges --> [(16, 1), (17, 1), (18, 1), (29, 1)]
key:  26, value: None, parent: None, edges --> [(18, 1), (19, 1), (20, 1), (29, 1), (30, 1)]
key:  27, value:  127, parent: None, edges --> [(20, 1), (21, 1), (22, 1), (30, 1)]
key:  28, value: None, parent: None, edges --> [(22, 1), (23, 1), (24, 1), (30, 1), (31, 1)]
key:  29, value: None, parent: None, edges --> [(24, 1), (25, 1), (26, 1), (31, 1)]
key:  30, value: 3000, parent: None, edges --> [(26, 1), (27, 1), (28, 1), (31, 1)]
key:  31, value: 3100, parent: None, edges --> [(28, 1), (29, 1), (30, 1), (32, 1)]
key:  32, value: 3200, parent: None, edges --> [(33, 1), (34, 1), (35, 1)]
key:  33, value: None, parent: None, edges --> [(34, 1), (36, 1)]
key:  34, value: None, parent: None, edges --> []
key:  35, value: None, parent: None, edges --> []
key:  36, value: None, parent: None, edges --> []"""
		# Values are assigned to every third key.
		valueMap = {i : i + 100 for i in range(0, 32, 3)}
		graph = sample_undirected_graph(3, valueMap)
		newAdjacencyMap = {31: {32}, 32: {33, 34, 35}, 33: {34, 36}}
		newValueMap = {30: 3000, 31: 3100, 32: 3200}
		graph.put(newAdjacencyMap, newValueMap)
		self.assertEqual(repr(graph), modifiedSampleUndirectedGraph)


	def test_multiply_graph(self):
		multipliedShelLDepth2SampleDAG = \
"""key:   0, value:  100, parent: None, edges --> [(1, 1), (2, 1), (3, 1)]
key:   1, value: None, parent: None, edges --> [(3, 1), (4, 1), (5, 1)]
key:   2, value: None, parent: None, edges --> [(5, 1), (6, 1), (7, 1)]
key:   3, value:  103, parent: None, edges --> [(7, 1), (8, 1), (9, 1)]
key:   4, value: None, parent: None, edges --> [(10, 1)]
key:   5, value: None, parent: None, edges --> [(10, 1)]
key:   6, value:  106, parent: None, edges --> [(10, 1), (11, 1)]
key:   7, value: None, parent: None, edges --> [(11, 1)]
key:   8, value: None, parent: None, edges --> [(11, 1), (12, 1)]
key:   9, value:  109, parent: None, edges --> [(12, 1)]
key:  10, value: None, parent: None, edges --> [(12, 1), (13, 1)]
key:  11, value: None, parent: None, edges --> [(13, 1)]
key:  12, value:  112, parent: None, edges --> [(13, 1)]
key:  13, value: None, parent: None, edges --> []
key:  14, value:  114, parent: None, edges --> [(15, 1), (16, 1), (17, 1)]
key:  15, value: None, parent: None, edges --> [(17, 1), (18, 1), (19, 1)]
key:  16, value: None, parent: None, edges --> [(19, 1), (20, 1), (21, 1)]
key:  17, value:  117, parent: None, edges --> [(21, 1), (22, 1), (23, 1)]
key:  18, value: None, parent: None, edges --> [(24, 1)]
key:  19, value: None, parent: None, edges --> [(24, 1)]
key:  20, value:  120, parent: None, edges --> [(24, 1), (25, 1)]
key:  21, value: None, parent: None, edges --> [(25, 1)]
key:  22, value: None, parent: None, edges --> [(25, 1), (26, 1)]
key:  23, value:  123, parent: None, edges --> [(26, 1)]
key:  24, value: None, parent: None, edges --> [(26, 1), (27, 1)]
key:  25, value: None, parent: None, edges --> [(27, 1)]
key:  26, value:  126, parent: None, edges --> [(27, 1)]
key:  27, value: None, parent: None, edges --> []
key:  28, value:  128, parent: None, edges --> [(29, 1), (30, 1), (31, 1)]
key:  29, value: None, parent: None, edges --> [(31, 1), (32, 1), (33, 1)]
key:  30, value: None, parent: None, edges --> [(33, 1), (34, 1), (35, 1)]
key:  31, value:  131, parent: None, edges --> [(35, 1), (36, 1), (37, 1)]
key:  32, value: None, parent: None, edges --> [(38, 1)]
key:  33, value: None, parent: None, edges --> [(38, 1)]
key:  34, value:  134, parent: None, edges --> [(38, 1), (39, 1)]
key:  35, value: None, parent: None, edges --> [(39, 1)]
key:  36, value: None, parent: None, edges --> [(39, 1), (40, 1)]
key:  37, value:  137, parent: None, edges --> [(40, 1)]
key:  38, value: None, parent: None, edges --> [(40, 1), (41, 1)]
key:  39, value: None, parent: None, edges --> [(41, 1)]
key:  40, value:  140, parent: None, edges --> [(41, 1)]
key:  41, value: None, parent: None, edges --> []"""
		# Values are assigned to every third key.
		valueMap = {i : i + 100 for i in range(0, 14, 3)}
		graph = multiply_graph(sample_dag(2, valueMap), 3)
		self.assertEqual(repr(graph), multipliedShelLDepth2SampleDAG)
		sameGraph = sample_dag_forest(2, 3, valueMap)
		self.assertEqual(repr(sameGraph), multipliedShelLDepth2SampleDAG)


if __name__ == "__main__":
	unittest.main()
