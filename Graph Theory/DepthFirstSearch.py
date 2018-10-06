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

def depth_first_search(graph, rootKey = None):

	# Reset's the attribute values of all Nodes in graph to their initialization values.
	# Importantly, resets Node.searchStatus to "undiscovered" and Node.parent to None.
	graph.reset()

	time = 0

	# If a starting root is specified, begin there.
	if rootKey is not None:
		time = DFS_visit(graph, rootKey, time)

	# The keys are ordered here to enforce an easily predictable traversal.
	# This is not necessary and reduces efficiency, but makes testing very straightforward. 
	# For the purposes of this program this loss in efficiency is acceptable.
	orderedKeys = list(graph.adjacencyMap.keys()); orderedKeys.sort()
	for key in orderedKeys:
		if graph.vertexMap[key].searchStatus == "undiscovered":
			time = DFS_visit(graph, key, time)


def DFS_visit(graph, thisKey, time):
	# Discover node for the first time.
	time += 1
	thisNode = graph.vertexMap[thisKey]
	thisNode.discoveryTime = time
	thisNode.searchStatus = "exploring"

	# Explore all undiscovered adjacent nodes and set their parent attributes.

	# The keys are ordered here to enforce an easily predictable traversal.
	# This is not necessary and reduces efficiency, but makes testing very straightforward. 
	# For the purposes of this program this loss in efficiency is acceptable.
	orderedAdjacentKeys = list(graph.adjacentKeys(thisKey)); orderedAdjacentKeys.sort()
	for adjacentKey in orderedAdjacentKeys:
		adjacentNode = graph.vertexMap[adjacentKey]
		if adjacentNode.searchStatus == "undiscovered":
			adjacentNode.parent = thisNode
			time = DFS_visit(graph, adjacentKey, time)

	# All adjacent nodes have been explored.
	time += 1
	thisNode.finishTime = time
	thisNode.searchStatus = "finished"

	return time


class TestDFS(unittest.TestCase):
	"""
	By its nature, a depth-first search produces trees whose structure depends 
	on the order in which roots and adjacent nodes are visited. This implementation 
	uses sets but explores them in sorted order.
	"""

	def setUp(self):
		self.graph = sample_dag(3)


	def test_DFS(self):
		DFStraversedDAG = \
"""key:   0, value: None, parent: None, edges --> [(1, 1), (2, 1), (3, 1)]
key:   1, value: None, parent:    0, edges --> [(3, 1), (4, 1), (5, 1)]
key:   2, value: None, parent:    0, edges --> [(5, 1), (6, 1), (7, 1)]
key:   3, value: None, parent:    1, edges --> [(7, 1), (8, 1), (9, 1)]
key:   4, value: None, parent:    1, edges --> [(9, 1), (10, 1), (11, 1)]
key:   5, value: None, parent:    1, edges --> [(11, 1), (12, 1), (13, 1)]
key:   6, value: None, parent:    2, edges --> [(13, 1), (14, 1), (15, 1)]
key:   7, value: None, parent:    3, edges --> [(15, 1), (16, 1), (17, 1)]
key:   8, value: None, parent:    3, edges --> [(17, 1), (18, 1), (19, 1)]
key:   9, value: None, parent:    3, edges --> [(19, 1), (20, 1), (21, 1)]
key:  10, value: None, parent:    4, edges --> [(22, 1)]
key:  11, value: None, parent:    4, edges --> [(22, 1)]
key:  12, value: None, parent:    5, edges --> [(22, 1), (23, 1)]
key:  13, value: None, parent:    5, edges --> [(23, 1)]
key:  14, value: None, parent:    6, edges --> [(23, 1), (24, 1)]
key:  15, value: None, parent:    7, edges --> [(24, 1)]
key:  16, value: None, parent:    7, edges --> [(24, 1), (25, 1)]
key:  17, value: None, parent:    7, edges --> [(25, 1)]
key:  18, value: None, parent:    8, edges --> [(25, 1), (26, 1)]
key:  19, value: None, parent:    8, edges --> [(26, 1)]
key:  20, value: None, parent:    9, edges --> [(26, 1), (27, 1)]
key:  21, value: None, parent:    9, edges --> [(27, 1)]
key:  22, value: None, parent:   10, edges --> [(27, 1), (28, 1)]
key:  23, value: None, parent:   12, edges --> [(28, 1)]
key:  24, value: None, parent:   15, edges --> [(28, 1), (29, 1)]
key:  25, value: None, parent:   16, edges --> [(29, 1)]
key:  26, value: None, parent:   18, edges --> [(29, 1), (30, 1)]
key:  27, value: None, parent:   20, edges --> [(30, 1)]
key:  28, value: None, parent:   24, edges --> [(30, 1), (31, 1)]
key:  29, value: None, parent:   24, edges --> [(31, 1)]
key:  30, value: None, parent:   28, edges --> [(31, 1)]
key:  31, value: None, parent:   30, edges --> []"""
		depth_first_search(self.graph, 0)
		self.assertEqual(repr(self.graph), DFStraversedDAG)
		self.assertEqual(path_to_root_to_string(self.graph, 31), "[31, 30, 28, 24, 15, 7, 3, 1, 0]")
		self.assertEqual(path_to_root_to_string(self.graph, 25), "[25, 16, 7, 3, 1, 0]")
		self.assertEqual(path_to_root_to_string(self.graph, 27), "[27, 20, 9, 3, 1, 0]")


if __name__ == "__main__":
	unittest.main()
