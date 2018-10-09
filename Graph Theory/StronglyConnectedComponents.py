from Graph import *
from TopologicalSort import *
import unittest


class SCCGraph(Graph):
	"""Overrides __repr__ method since the values of SCC Nodes are sets."""
	
	def __repr__(self):
		"""
		Returns a string representing self.adjacencyMap and self.valueMap.
		Because the Nodes' values are sets of other Nodes, the number of Nodes in these sets 
		is displayed rather than the full values themselves.

		Return value is of the form:
			"Key: {key1}, number of components: {#}, parent: {parent1}, edges --> [(adjacentKey1, weight1), (adjacentKey2, weight2)...]
			"Key: {key2}, number of components: {#}, parent: {parent2}, edges --> [(adjacentKey1, weight1), (adjacentKey2, weight2)...]
		"""

		reprString = ""
		# keyList used to enforce consistent order on key iteration.
		keyList = list(self.adjacencyMap.keys())
		keyList.sort()
		for key in keyList:
			sortedAdjacencySet = list(self.adjacencyMap[key])
			sortedAdjacencySet.sort()
			# f-string expressions insist on not having line breaks.
			reprString += 	f"key: {key : >3}, number of components: {len(self.valueMap[key]) : >3}, "\
							f"parent: {self.vertexMap[key].parent.key if self.vertexMap[key].parent is not None else str(None) : >4}, "\
							f"edges --> {sortedAdjacencySet}"
			# Each key is unique since it came from a set.
			if key != keyList[-1]:
				reprString += "\n"

		return reprString


def strongly_connected_components(graph):
	"""
	Returns a new Graph object, SCCgraph, whose Nodes are the 
	strongly connected components of the input graph, with 
	corresponding edges between them. Edge weights are ignored.

	The value attributes of these SCC Nodes contains a set 
	of the Nodes in the input graph represented by that 
	strongly connected component. 
	The choice could be made to instead store the Nodes' keys, 
	but that requires the SCCgraph to track its parent graph as well.

	A strongly connected component is a maximal set of nodes with edges such that 
	every node in the set is reachable from any other node in the set.
	"""

	# Topologically sort the input graph.
	topologicalKeyList = topological_sort(graph)

	# Transpose the input graph, reversing all edges.
	# A copy of graph could be created and transposed so as not to mutate graph.
	# However, we will correct graph when we are finished.
	graph.reverse_edges()

	# Perform a depth-first search on the transposed graph and gather trees together into SCCs.

	graph.reset()
	
	# time is tracked to furnish the graph with discoveryTimes and finishTimes 
	# as a matter of good practice.
	global time
	time = 0

	# treeSet is the set of all Nodes in each tree, separately, found in the following DFS.
	global treeSet
	# SCC_value_set will contain all treeSets.
	SCC_value_set = set()

	for key in topologicalKeyList:
		if graph.vertexMap[key].searchStatus == "undiscovered":
			treeSet = set()
			SCC_visit(graph, key)
			# key is the root of a tree whose elements comprise an SCC.
			# treeSet now contains all Nodes in that tree.
			
			# Add treeSet to SCC_value_set
			SCC_value_set.add(frozenset(treeSet))
	# SCC_value_set now contains all treeSets, 
	# i.e. all sets of Nodes in each SCC, from the input graph.
	
	# Revert graph's edges.
	graph.reverse_edges()

	# Create a new Graph object of SCC Nodes and return it.
	return generate_SCC_graph(graph, SCC_value_set)


def SCC_visit(graph, thisKey):
	"""
	Visit the Node at key. Recursively visit all adjacent Nodes in a depth-first fashion.
	Add each Node encountered to the global variable treeSet.
	"""

	thisNode = graph.vertexMap[thisKey]
	treeSet.add(thisNode)

	# Discover thisNode.
	thisNode.searchStatus = "exploring"
	global time
	time += 1
	thisNode.discoveryTime = time

	# Explore all undiscovered adjacent Nodes.
	for adjacentKey in graph.adjacentKeys(thisKey):
		adjacentNode = graph.vertexMap[adjacentKey]
		if adjacentNode.searchStatus == "undiscovered":
			adjacentNode.parent = thisNode
			SCC_visit(graph, adjacentKey)

	# Finish thisNode.
	thisNode.searchStatus = "finished"
	time += 1
	thisNode.finishTime = time


def generate_SCC_graph(graph, SCC_value_set):
	"""
	SCC_value_set is the set of values for the Nodes in the Graph to be generated.
	Each element in SCC_value_set is a set of Nodes comprising a strongly connected component of graph.
	"""
	
	# Assign a key to each SCC Node value and store the pairings in SCC_valueMap.
	SCC_valueMap = dict()
	# newKey is used to create new key values for each SCC Node.
	newKey = 0
	# SCC_value_set is sorted to enforce easily predictable key assignment.
	sorted_SCC_values = sorted(SCC_value_set, key = lambda treeSet: sorted(map(lambda node: node.key, treeSet)))
	for treeSet in sorted_SCC_values:
		SCC_valueMap[newKey] = treeSet
		newKey += 1

	# Create a dictionary of keys to edges between SCC Nodes of the form 
	# SCC_adjacencyMap[SCCkey] = {adjacent_SCC_keys},
	# based on all edges between Nodes in each treeSet.
	# i.e. if there is an edge A --> B between a Node A in treeSet 1 and a Node B in treeSet 2, 
	# there will be a corresponding edge between SCC Node 1 and SCC Node 2.
	# Ignore edge weights.
	SCC_adjacencyMap = dict()

	for SCC_key, treeSet in SCC_valueMap.items():
		for treeNode in treeSet:
			for edge in graph.adjacencyMap[treeNode.key]:
				adjacentNode = graph.vertexMap[edge[0]]
				adjacent_SCC_key = find_SCC_key(SCC_valueMap, adjacentNode)
				if adjacent_SCC_key != SCC_key:
					# Add edge from SCC_key to adjacent_SCC_key. 
					# Edges are stored in sets so there is no need to check for membership.
					# When Graph objects are instantiated, edge weights are added automatically if not present.
					SCC_adjacencyMap.setdefault(SCC_key, set()).add(adjacent_SCC_key)

	# Create and return Graph.
	return SCCGraph(SCC_adjacencyMap, SCC_valueMap)


def find_SCC_key(SCC_valueMap, treeNode):
	"""
	SCC_valueMap is a dictionary of the form SCC_valueMap[SCC_key] = {Nodes within that SCC (treeSet)}.
	If treeNode is in one of these SCC treeSets, return the SCC_key of the corresponding treeSet.
	Else return None.
	"""

	for SCC_key, treeSet in SCC_valueMap.items():
		if treeNode in treeSet:
			return SCC_key

	# treeNode is not in any treeSet within SCC_valueMap.
	return None


class TestSCC(unittest.TestCase):

	def setUp(self):
		# Create a graph with 4 strongly connected components.
		self.graph = sample_undirected_forest(1, 4)
		# Each tree is currently isolated. Each tree contains 5 Nodes, 
		# and every 5th Node is a root, starting at 0.

		# Add edges to make the following connections: 
		# tree0 --> tree1
		# tree2 --> tree1
		# tree2 --> tree3
		self.graph.put({0: {5}, 10: {5, 15}})


	def test_strongly_connected_components(self):

		SCC_graph_from_sample_undirected_forest_1_4_with_connections = \
"""key:   0, number of components:   5, parent: None, edges --> [(1, 1)]
key:   1, number of components:   5, parent: None, edges --> []
key:   2, number of components:   5, parent: None, edges --> [(1, 1), (3, 1)]
key:   3, number of components:   5, parent: None, edges --> []"""

		SCCgraph = strongly_connected_components(self.graph)

		# Verify state of SCCgraph.
		self.assertEqual(repr(SCCgraph), SCC_graph_from_sample_undirected_forest_1_4_with_connections)

		# The value of each Node in SCCgraph should be a set of 5 Nodes.
		SCC_values = map(lambda node: node.value, SCCgraph.vertexMap.values())
		SCC_set_of_keys = {map(lambda node: node.key, SCC_value) for SCC_value in SCC_values}
		actual_keys_of_values = {frozenset(keySetGenerator) for keySetGenerator in SCC_set_of_keys}
		theoretical_keys_of_values = {frozenset(key for key in range(shift, shift+5)) \
										for shift in [0, 5, 10, 15]}

		# Verify values of components.
		self.assertEqual(theoretical_keys_of_values, actual_keys_of_values)


if __name__ == "__main__":
	unittest.main()
