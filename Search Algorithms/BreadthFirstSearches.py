from queue import Queue
import math
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

class GraphNode:
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.discovered = False
        self.distance = 0


class Graph:
    """
    Graph is a collection of GraphNode objects represented by 
    an array of vertices and an adjacency list of edges.
    """

    def __init__(self, vertexEdges):
        """
        vertexEdges is an ordered list of sets 
        of the form vertexEdges[vertexIndex] = {adjacentVertexIndices}.
        The length of vertexEdges is the number of vertices.
        """

        self.vertices = []
        self.adjacencyList = vertexEdges
        # Populate vertices
        for vertexIndex in range(len(vertexEdges)):
            self.vertices.append(GraphNode(vertexIndex))


class ListQueue:
    """Simple queue object that stores elements in an accessible list."""
    def __init__(self):
        self.elements = []
    def __len__(self):
        return len(self.elements)
    def put(self, newItem):
        self.elements.insert(0, newItem)
    def get(self):
        return self.elements.pop()
    def empty(self):
        return len(self.elements) == 0

r"""
   _____           _                              _     _                      
  / ____|         | |                            | |   (_)                     
 | (___    _   _  | |__    _ __    ___    _   _  | |_   _   _ __     ___   ___ 
  \___ \  | | | | | '_ \  | '__|  / _ \  | | | | | __| | | | '_ \   / _ \ / __|
  ____) | | |_| | | |_) | | |    | (_) | | |_| | | |_  | | | | | | |  __/ \__ \
 |_____/   \__,_| |_.__/  |_|     \___/   \__,_|  \__| |_| |_| |_|  \___| |___/
                                                                               
                                                                               
"""

def reset_graph(graph):
    # Set all vertices in graph to default unexplored state
    for vertex in graph.vertices:
        vertex.parent = None
        vertex.discovered = False
        vertex.distance = math.inf


def discover_vertex(parent, child):
    """Set child to discovered, set child's parent to parent, set child's distance to parent's + 1."""
    child.discovered = True
    child.parent = parent
    child.distance = parent.distance + 1


def explore_adjacent_vertices(graph, queue, targetIndices = []):
    """
    Dequeue next vertex index and explore the adjacent vertices, 
    adding yet undiscovered ones to the queue.
    """
    currentNodeIndex = queue.get()
    currentNode = graph.vertices[currentNodeIndex]
    for adjacentNodeIndex in graph.adjacencyList[currentNodeIndex]:
        adjacentNode = graph.vertices[adjacentNodeIndex]

        if not adjacentNode.discovered:
            discover_vertex(currentNode, adjacentNode)
            queue.put(adjacentNodeIndex)
        
        # Stop early and return the corresponding distance.
        if adjacentNodeIndex in targetIndices:
            # A return value other than None indicates early termination.
            # return successCode, collisionIndex, discovererIndex
            return True, adjacentNodeIndex, currentNodeIndex

    # Explored the vertices adjacent to the next element in the queue.
    # Did not encounter targetIndex.
    return False, None, None


def reverse_path(graph, predecessorIndex, successorIndex):
    """
    Reverse the GraphNode.parent pointers along the path 
    from predecessorIndex to successorIndex.
    Sets graph.vertices[successorIndex].parent to None.
    """

    finalNode = graph.vertices[predecessorIndex]

    # Check path from successor to predecessor
    tracerNode = graph.vertices[successorIndex]
    # Check for cycles
    pathNodes = [tracerNode]
    while tracerNode is not None and tracerNode != finalNode:
        tracerNode = tracerNode.parent
        if tracerNode not in pathNodes: 
            pathNodes.append(tracerNode)
        else: 
            raise Exception("Cycle detected.")
    if tracerNode != finalNode:
        raise LookupError(f"No established path from {predecessorIndex} to {successorIndex}.")
    # No longer needed
    del pathNodes, tracerNode

    # Create 3 adjacent vertices starting at successorIndex
    previousNode = graph.vertices[successorIndex]
    currentNode = previousNode.parent
    # If currentNode is None, nextNode is also None.
    nextNode = currentNode and currentNode.parent

    # The parent of the vertex at successorIndex is removed
    previousNode.parent = None

    # Reverse parent pointers along path
    while currentNode is not None and currentNode != finalNode:
        currentNode.parent = previousNode
        previousNode = currentNode
        currentNode = nextNode
        if nextNode: nextNode = nextNode.parent
    finalNode.parent = previousNode


def correct_distances_along_path(graph, predecessorIndex, successorIndex):
    """
    Recalculates and sets the distance attribute of each GraphNode 
    from predecessor to successor.
    """

    finalNode = graph.vertices[predecessorIndex]
    tracerNode = graph.vertices[successorIndex]

    # Verify path before making modifications,
    # check for cycles, and count the length of the path.
    edgesAway = 0
    pathNodes = [tracerNode]
    while tracerNode is not None and tracerNode != finalNode:
        tracerNode = tracerNode.parent
        edgesAway += 1
        if tracerNode not in pathNodes: 
            pathNodes.append(tracerNode)
        else: 
            raise Exception("Cycle detected.")
    if tracerNode != finalNode:
        raise LookupError(f"No established path from {successorIndex} to {predecessorIndex}.""")

    # A well-defined path exists between successor and predecessor through their parent attributes.

    tracerNode = graph.vertices[successorIndex]
    while tracerNode != finalNode:
        tracerNode.distance = edgesAway
        tracerNode = tracerNode.parent
        edgesAway -= 1
    # finalNode.distance = 0
    tracerNode.distance = edgesAway


def update_shell(graph, shellIndices, targetShell):
    """
    Replace shellIndices with the valid vertices adjacent to the current source shell.
    Checks each vertex adjacent to the current shell and adds them to the new shell only 
    if they are not yet discovered.

    If an element from targetShell is encountered, a collision has occurred.
    """
    collisionIndex = None
    discovererIndex = None
    newShellIndices = []
    for shellIndex in shellIndices:
        for adjacentIndex in graph.adjacencyList[shellIndex]:
            adjacentNode = graph.vertices[adjacentIndex]

            # Check for collision
            if adjacentIndex in targetShell:
                collisionIndex = adjacentIndex
                discovererIndex = shellIndex

            if not adjacentNode.discovered:
                discover_vertex(graph.vertices[shellIndex], adjacentNode)
                newShellIndices.append(adjacentIndex)
    return newShellIndices, collisionIndex, discovererIndex


def make_undirected(graph):
    """
    For each edge A --> B, ensure that the opposite edge B --> A is present.
    Runtime is O(|vertices| + |edges|).
    """

    # Alias graph.adjacencyList for convenience
    adjacencyList = graph.adjacencyList

    for vertexIndex in range(len(adjacencyList)):
        for adjacentNodeIndex in adjacencyList[vertexIndex]:
        # Since the elements of adjacencyList are sets, we need not check for membership.
            adjacencyList[adjacentNodeIndex].add(vertexIndex)


def construct_adjacent_to_list(graph):
    """
    Constructs and returns an "inverse" adjacencyList: 
    a list with an element for each vertex whose elements are 
    lists of the indices for all vertices that that vertex is adjacent to.
    That is, adjacentToList[vertexIndex] = [indices of vertices with an edge to that vertex]
    
    Worst case runtime is roughly O(|vertices|^2 * log|vertices|). Dreadful. Avoid use.
    (Not used.)
    """

    adjacentToList = []
    for vertexIndex in range(len(graph.vertices)):
        adjacentToList.append([])
        for adjacentToIndex in range(len(graph.vertices)):
            # If adjacentToIndex is adjacent to vertexIndex
            if vertexIndex in graph.adjacencyList[adjacentToIndex]:
                adjacentToList[vertexIndex].append(adjacentToIndex)
    return adjacentToList

r"""
  ____                              _   _     _                ______   _                _               _____                                _                  
 |  _ \                            | | | |   | |              |  ____| (_)              | |             / ____|                              | |                 
 | |_) |  _ __    ___    __ _    __| | | |_  | |__    ______  | |__     _   _ __   ___  | |_   ______  | (___     ___    __ _   _ __    ___  | |__     ___   ___ 
 |  _ <  | '__|  / _ \  / _` |  / _` | | __| | '_ \  |______| |  __|   | | | '__| / __| | __| |______|  \___ \   / _ \  / _` | | '__|  / __| | '_ \   / _ \ / __|
 | |_) | | |    |  __/ | (_| | | (_| | | |_  | | | |          | |      | | | |    \__ \ | |_            ____) | |  __/ | (_| | | |    | (__  | | | | |  __/ \__ \
 |____/  |_|     \___|  \__,_|  \__,_|  \__| |_| |_|          |_|      |_| |_|    |___/  \__|          |_____/   \___|  \__,_| |_|     \___| |_| |_|  \___| |___/
                                                                                                                                                                 
                                                                                                                                                                 
"""

def BFS(graph, rootIndex, terminationIndex = None):
    """
    Constructs a predecessor tree starting at the graph vertex specified by rootIndex. 
    If terminationIndex is provided and encountered during the tree creation, then 
    the tree creation is terminated early and the distance of the vertex at terminationIndex 
    is returned.
    """

    reset_graph(graph)

    # Discover the rootNode
    rootNode = graph.vertices[rootIndex]
    rootNode.discovered = True
    rootNode.distance = 0

    # searchQueue is a Queue of the vertexIndex values representing vertices to be explored
    searchQueue = Queue()
    # Seed searchQueue with rootIndex
    searchQueue.put(rootIndex)

    while not searchQueue.empty():
        # if terminationIndex was found, a path has been made to it from rootIndex.
        if explore_adjacent_vertices(graph, searchQueue, [terminationIndex])[0]:
            return graph.vertices[terminationIndex].distance

    # Explored all accessible vertices.
    return None


def bi_directional_BFS_queue_version(graph, sourceIndex, destinationIndex): 
    """
    Runs a breadth-first search from sourceIndex and destinationIndex concurrently, 
    and terminates when a connecting path is found. 
    Returns the distance (number of edges) between them.
    Assumes an undirected graph.

    Due to the nature of a bo-directional BFS, after running, the resultant state 
    of the graph is such that only the parent pointers and distances that are along 
    the first discovered shortest path can be assumed correct.
    """

    reset_graph(graph)

    sourceNode = graph.vertices[sourceIndex]
    sourceNode.discovered = True
    sourceNode.distance = 0
    sourceQueue = ListQueue()
    sourceQueue.put(sourceIndex)
    
    destinationNode = graph.vertices[destinationIndex]
    destinationNode.discovered = True
    destinationNode.distance = 0
    destinationQueue = ListQueue()
    destinationQueue.put(destinationIndex)

    currentDestinationIndex = destinationIndex
    # while sourceQueue and destinatinoQueue are both not empty,
    # search outwards from sourceNode and destinationNode 
    while not (sourceQueue.empty() or destinationQueue.empty()):
        collisionFound, collisionIndex, discovererIndex = explore_adjacent_vertices(graph, 
                                                        sourceQueue, 
                                                        destinationQueue.elements)
        # If source collided with destionation:
        if collisionFound:
            # collisionIndex is connected to destination.
            # discovererIndex is connected to source.
            reverse_path(graph, destinationIndex, collisionIndex)
            # The vertex at collisionIndex's parent is None, but should be discovererIndex in source.
            graph.vertices[collisionIndex].parent = graph.vertices[discovererIndex]
            correct_distances_along_path(graph, sourceIndex, destinationIndex)
            return graph.vertices[destinationIndex].distance

        collisionFound, collisionIndex, discovererIndex = explore_adjacent_vertices(graph, 
                                                        destinationQueue, 
                                                        sourceQueue.elements)
        # If destination collided with source:
        if collisionFound:
            # discovererIndex is connected to destination.
            # collisionIndex is connected to source.
            reverse_path(graph, destinationIndex, discovererIndex)
            # The vertex at discovererIndex's parent is None, but should be the collisionIndex in source.
            graph.vertices[discovererIndex].parent = graph.vertices[collisionIndex]
            correct_distances_along_path(graph, sourceIndex, destinationIndex)
            return graph.vertices[destinationIndex].distance
    # The trees rooted at source and destination cannot be connected.
    return None


def bi_directional_BFS_shell_version(graph, sourceIndex, destinationIndex):
    """
    Runs a breadth-first search from sourceIndex and destinationIndex concurrently, 
    and terminates when a connecting path is found. 
    Returns the distance (number of edges) between them.
    Assumes an undirected graph.

    Due to the nature of a bo-directional BFS, after running, the resultant state 
    of the graph is such that only the parent pointers and distances that are along 
    the first discovered shortest path can be assumed correct.
    """

    reset_graph(graph)

    sourceNode = graph.vertices[sourceIndex]
    sourceNode.discovered = True
    sourceNode.distance = 0
    sourceShellIndices = [sourceIndex]

    destinationNode = graph.vertices[destinationIndex]
    destinationNode.discovered = True
    destinationNode.distance = 0
    destinationShellIndices = [destinationIndex]

    collisionIndex = None
    discovererIndex = None
    # Each iteration handles a new shell centered on source and destination, until a tree completes.
    while sourceShellIndices and destinationShellIndices:
        # Replace shells with the valid vertices adjacent to the current shells and check for collisions.
        sourceShellIndices, collisionIndex, discovererIndex = update_shell(graph, 
                                                                sourceShellIndices, 
                                                                destinationShellIndices)
        # If the source shell collided with the destination shell:
        if collisionIndex is not None:
            # collisionIndex's parent is in the destination shell.
            # discovererIndex's parent is in the source shell.
            reverse_path(graph, destinationIndex, collisionIndex)
            # The vertex at collisionIndex's parent is None, 
            # but should be discovererIndex in the source shell.
            graph.vertices[collisionIndex].parent = graph.vertices[discovererIndex]
            correct_distances_along_path(graph, sourceIndex, destinationIndex)
            return graph.vertices[destinationIndex].distance

        destinationShellIndices, collisionIndex, discovererIndex = update_shell(graph, 
                                                                    destinationShellIndices, 
                                                                    sourceShellIndices)
        # If the destination shell collided with the source shell:
        if collisionIndex is not None:
            # discovererIndex's parent is in the destination shell.
            # collisionIndex's parent is in the source shell.
            reverse_path(graph, destinationIndex, discovererIndex)
            # The vertex at discovererIndex's parent is None, 
            # but should be the collisionIndex in the source shell.
            graph.vertices[discovererIndex].parent = graph.vertices[collisionIndex]
            correct_distances_along_path(graph, sourceIndex, destinationIndex)
            return graph.vertices[destinationIndex].distance
    
    # The trees rooted at source and destination cannot be connected.
    return None

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
        print(targetNode.value);
    elif targetNode.parent == None:
        print("No path from rootNode to targetNode.")
    else:
        print_path(rootNode, targetNode.parent)
        print(targetNode.value)


def print_edges(graph):
    """For each vertex, print_edges prints each outgoing edge on its own line."""

    for vertexIndex in range(len(graph.adjacencyList)):
        print()
        for adjacentIndex in graph.adjacencyList[vertexIndex]:
            print(f"vertexIndex: {vertexIndex}, adjacentIndex: {adjacentIndex}")


def print_path_to_root(graph, vertexIndex):
    """
    Follows the parent pointer of vertexIndex and its ancestors and prints each value 
    until a cycle is detected or None is encountered.
    """

    tracerNode = graph.vertices[vertexIndex]
    pathNodes = [tracerNode]
    while tracerNode != None:
        print(tracerNode.value)
        tracerNode = tracerNode.parent
        if tracerNode not in pathNodes:
            pathNodes.append(tracerNode)
        else:
            raise Exception("Cycle detected.")

r"""
  _    _           _   _       _______                _         
 | |  | |         (_) | |     |__   __|              | |        
 | |  | |  _ __    _  | |_       | |      ___   ___  | |_   ___ 
 | |  | | | '_ \  | | | __|      | |     / _ \ / __| | __| / __|
 | |__| | | | | | | | | |_       | |    |  __/ \__ \ | |_  \__ \
  \____/  |_| |_| |_|  \__|      |_|     \___| |___/  \__| |___/
                                                                
                                                                
"""

class Test_Breadth_First_Searches(unittest.TestCase):
    """General framework for testing Breadth-First Search functions."""

    def setUp(self):

        # Create graph. 
        # To understand it, I recommend drawing the first half based on the initialization of vertexEdges.
        # Note that the print_edges(graph) function above will show you each vertex's outgoing edges.
        # This can make constructing the graph quite straightforward.

        shellDepth = 3
        # numVertices = 1 + 3 + 3*2 + (3*2*2) + 3*2 + 3 + 1
        numCenterLine = 3 * 2**(shellDepth - 1)
        numVertices = (1 + sum([3 * 2**i for i in range(shellDepth - 1)])) * 2 + numCenterLine

        vertexEdges = [{2 * i + 1, 2 * i + 2, 2 * i + 3} for i in range(int((numVertices - numCenterLine) / 2))]
        vertexEdges += [set() for _ in range(numCenterLine)]
        vertexTaperingEdges = []
        for i in range(numVertices - 1, int((numVertices + numCenterLine) / 2) - 1, -1):
            lowerEdges = numVertices - 1 - 2 * (numVertices - 1 - i)
            vertexTaperingEdges.append({lowerEdges - 1, lowerEdges - 2, lowerEdges - 3})
        vertexEdges += vertexTaperingEdges[::-1]

        self.graph = Graph(vertexEdges)
        make_undirected(self.graph)


    def test_BFS(self):

        self.assertEqual(BFS(self.graph, 0, 31), 6)
        self.assertEqual(BFS(self.graph, 10, 21), 3)
        self.assertEqual(BFS(self.graph, 12, 13), 2)
        self.assertEqual(BFS(self.graph, 12, 14), 2)
        self.assertEqual(BFS(self.graph, 29, 13), 4)
        self.assertEqual(BFS(self.graph, 24, 24), 0)

        BFS(self.graph, 0)
        self.assertEqual(self.graph.vertices[31].distance, 6)
        self.assertEqual(self.graph.vertices[28].distance, 5)
        self.assertEqual(self.graph.vertices[27].distance, 4)
        self.assertEqual(self.graph.vertices[14].distance, 3)
        self.assertEqual(self.graph.vertices[9].distance, 2)
        self.assertEqual(self.graph.vertices[2].distance, 1)
        self.assertEqual(self.graph.vertices[0].distance, 0)

        BFS(self.graph, 2)
        self.assertEqual(self.graph.vertices[26].distance, 5)


    def test_bi_directional_BFS_queue_version(self):

        self.assertEqual(bi_directional_BFS_queue_version(self.graph, 0, 31), 6)
        self.assertEqual(bi_directional_BFS_queue_version(self.graph, 0, 1), 1)
        self.assertEqual(bi_directional_BFS_queue_version(self.graph, 10, 21), 3)
        self.assertEqual(bi_directional_BFS_queue_version(self.graph, 12, 13), 2)
        self.assertEqual(bi_directional_BFS_queue_version(self.graph, 12, 14), 2)
        self.assertEqual(bi_directional_BFS_queue_version(self.graph, 12, 23), 1)
        self.assertEqual(bi_directional_BFS_queue_version(self.graph, 29, 13), 4)
        self.assertEqual(bi_directional_BFS_queue_version(self.graph, 24, 24), 0)


    def test_bi_directional_BFS_shell_version(self):

        self.assertEqual(bi_directional_BFS_shell_version(self.graph, 0, 31), 6)
        self.assertEqual(bi_directional_BFS_shell_version(self.graph, 0, 1), 1)
        self.assertEqual(bi_directional_BFS_shell_version(self.graph, 10, 21), 3)
        self.assertEqual(bi_directional_BFS_shell_version(self.graph, 12, 13), 2)
        self.assertEqual(bi_directional_BFS_shell_version(self.graph, 12, 14), 2)
        self.assertEqual(bi_directional_BFS_shell_version(self.graph, 12, 23), 1)
        self.assertEqual(bi_directional_BFS_shell_version(self.graph, 29, 13), 4)
        self.assertEqual(bi_directional_BFS_shell_version(self.graph, 24, 24), 0)


if __name__ == "__main__":
    unittest.main()
