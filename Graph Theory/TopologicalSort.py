from Graph import *
import unittest

"""
Node attributes:

key (unique)
value = None
parent = None
searchStatus = "undiscovered"
distance = math.inf
discoveryTime = math.inf
finishTime = math.inf

Graph attributes:

adjacencyMap --> dictionary of the form adjacencyMap[key] = {adjacent keys}.
valueMap ------> defaultdict(lambda: None) of the form valueMap[key] = value.
vertexMap -----> dictionary of the form vertexMap[key] = corresponding Node.

"""

def topological_sort(graph, rootKey = None):
	"""
	If graph is a dag (directional acyclic graph), 
	returns a list of the Nodes in graph such that 
	for each edge A --> B, A appears before B.

	If graph is not a dag, returns a list of the keys in graph 
	in decreasing order of their Nodes' finishTime as determined from a DFS.
	"""

	# Reset's the attribute values of all Nodes in graph to their initialization values.
	# Importantly, resets Node.searchStatus to "undiscovered" and Node.parent to None.
	graph.reset()

	topologicalKeyList = []

	# time is declared inside a function and so must be made global.
	global time; time = 0

	# If a starting root is specified, begin there.
	if rootKey is not None:
		topological_sort_visit(graph, rootKey, topologicalKeyList)

	# Visit each undiscovered Node.

	# The keys are ordered here to enforce an easily predictable traversal.
	# This is not necessary and reduces efficiency, but makes testing very straightforward. 
	# For the purposes of this program this loss in efficiency is acceptable.
	orderedKeys = list(graph.adjacencyMap.keys()); orderedKeys.sort()
	for key in orderedKeys:
		if graph.vertexMap[key].searchStatus == "undiscovered":
			topological_sort_visit(graph, key, topologicalKeyList)

	# Explored and created a forest within graph.
	return topologicalKeyList


def topological_sort_visit(graph, thisKey, topologicalKeyList):
	"""
	Recursively creates a tree rooted at thisKey. 
	Calls from outside this function create a 
	new tree rooted at thisKey.

	As each Node is finished, it('s key) is added to the front of a list 
	which is ultimately returned. The final list is ordered by decreasing finish times.

	Returns [latest key finished, ..., first key finished].
	"""

	# Discover the Node at thisKey.
	global time
	time += 1
	thisNode = graph.vertexMap[thisKey]
	thisNode.searchStatus = "exploring"

	# Explore each undiscovered adjacent Node and set their parent attributes.

	# The keys are ordered here to enforce an easily predictable traversal.
	# This is not necessary and reduces efficiency, but makes testing very straightforward. 
	# For the purposes of this program this loss in efficiency is acceptable.
	sortedAdjacentKeys = list(graph.adjacentKeys(thisKey)); sortedAdjacentKeys.sort()
	for adjacentKey in sortedAdjacentKeys:
		adjacentNode = graph.vertexMap[adjacentKey]
		if adjacentNode.searchStatus == "undiscovered":
			adjacentNode.parent = thisNode
			topological_sort_visit(graph, adjacentKey, topologicalKeyList)

	# All adjacent Nodes have been explored.
	time += 1
	thisNode.finishTime = time
	thisNode.searchStatus = "finished"
	topologicalKeyList.insert(0, thisKey)


class TestTopologicalSort(unittest.TestCase):

	def setUp(self):
		# self.graph contains a dag of 3 trees.
		self.graph = sample_dag_forest(2, 3)


	def test_topological_sort(self):
		shelLDepth2numTrees3SampleDAG_topologically_sorted = \
"""key:   0, value: None, parent: None, edges --> [(1, 1), (2, 1), (3, 1)]
key:   1, value: None, parent:    0, edges --> [(3, 1), (4, 1), (5, 1)]
key:   2, value: None, parent:    0, edges --> [(5, 1), (6, 1), (7, 1)]
key:   3, value: None, parent:    1, edges --> [(7, 1), (8, 1), (9, 1)]
key:   4, value: None, parent:    1, edges --> [(10, 1)]
key:   5, value: None, parent:    1, edges --> [(10, 1)]
key:   6, value: None, parent:    2, edges --> [(10, 1), (11, 1)]
key:   7, value: None, parent:    3, edges --> [(11, 1)]
key:   8, value: None, parent:    3, edges --> [(11, 1), (12, 1)]
key:   9, value: None, parent:    3, edges --> [(12, 1)]
key:  10, value: None, parent:    4, edges --> [(12, 1), (13, 1)]
key:  11, value: None, parent:    7, edges --> [(13, 1)]
key:  12, value: None, parent:    8, edges --> [(13, 1)]
key:  13, value: None, parent:   11, edges --> []
key:  14, value: None, parent: None, edges --> [(15, 1), (16, 1), (17, 1)]
key:  15, value: None, parent:   14, edges --> [(17, 1), (18, 1), (19, 1)]
key:  16, value: None, parent:   14, edges --> [(19, 1), (20, 1), (21, 1)]
key:  17, value: None, parent:   15, edges --> [(21, 1), (22, 1), (23, 1)]
key:  18, value: None, parent:   15, edges --> [(24, 1)]
key:  19, value: None, parent:   15, edges --> [(24, 1)]
key:  20, value: None, parent:   16, edges --> [(24, 1), (25, 1)]
key:  21, value: None, parent:   17, edges --> [(25, 1)]
key:  22, value: None, parent:   17, edges --> [(25, 1), (26, 1)]
key:  23, value: None, parent:   17, edges --> [(26, 1)]
key:  24, value: None, parent:   18, edges --> [(26, 1), (27, 1)]
key:  25, value: None, parent:   21, edges --> [(27, 1)]
key:  26, value: None, parent:   22, edges --> [(27, 1)]
key:  27, value: None, parent:   25, edges --> []
key:  28, value: None, parent: None, edges --> [(29, 1), (30, 1), (31, 1)]
key:  29, value: None, parent:   28, edges --> [(31, 1), (32, 1), (33, 1)]
key:  30, value: None, parent:   28, edges --> [(33, 1), (34, 1), (35, 1)]
key:  31, value: None, parent:   29, edges --> [(35, 1), (36, 1), (37, 1)]
key:  32, value: None, parent:   29, edges --> [(38, 1)]
key:  33, value: None, parent:   29, edges --> [(38, 1)]
key:  34, value: None, parent:   30, edges --> [(38, 1), (39, 1)]
key:  35, value: None, parent:   31, edges --> [(39, 1)]
key:  36, value: None, parent:   31, edges --> [(39, 1), (40, 1)]
key:  37, value: None, parent:   31, edges --> [(40, 1)]
key:  38, value: None, parent:   32, edges --> [(40, 1), (41, 1)]
key:  39, value: None, parent:   35, edges --> [(41, 1)]
key:  40, value: None, parent:   36, edges --> [(41, 1)]
key:  41, value: None, parent:   39, edges --> []"""

		topologicalKeyList = topological_sort(self.graph)

		# Verify resultant graph state.
		self.assertEqual(repr(self.graph), shelLDepth2numTrees3SampleDAG_topologically_sorted)

		# Verify returned list is in order of decreasing finish times.
		keyListByDescendingFinishTimes = [\
			28, 30, 34, 29, 33, 32, 38, 31, 37, 36, 40, 35, 39, 41, \
			14, 16, 20, 15, 19, 18, 24, 17, 23, 22, 26, 21, 25, 27, \
			0, 2, 6, 1, 5, 4, 10, 3, 9, 8, 12, 7, 11, 13]
		self.assertEqual(topologicalKeyList, keyListByDescendingFinishTimes)


if __name__ == "__main__":
	unittest.main()