import numpy as np

from Heaps import N_ary_heap


class _Balltree_node:

    def __init__(self, coordinates, split_dimension, point_count=1, parent=None, radius=None, left_child=None, right_child=None):
        self.coordinates = coordinates
        self.split_dimension = split_dimension
        self.parent = parent
        self.radius = radius
        self.left_child = left_child
        self.right_child = right_child

    
    def attach_left_child(self, left_child):
        
        left_child.parent = self
        self.left_child = left_child

    
    def attach_right_child(self, right_child):
        
        right_child.parent = self
        self.right_child = right_child

    
    def set_radius(self, radius):
        
        self.radius = radius

    
    def is_leaf(self):

        return self.left_child is None and self.right_child is None


class Balltree:
    """A binary tree of _Balltree_node objects to support the k_nearest_neighbors_search method."""

    def __init__(self, array, spread_of=5, median_of=5):
        """
        Recursively constructs a binary tree of _Balltree_node objects whose root node is stored as self.root by making the first call to generate_Balltree.
        
        Args:
            array (np.ndarray): The array to be made into a balltree. Assumed to be of shape (number of points, number of dimensions for each point).
            spread_of (int, optional): The maximum number of points to check to estimate the spread of a subset of points along a particular dimension. Defaults to 5.
            median_of (int, optional): The maximum number of points to check to estimate the median of a subset of points along a particular dimension. Defaults to 5.
        """

        self.root = self.generate_Balltree(array, spread_of, median_of)


    @staticmethod
    def generate_Balltree(array, spread_of=5, median_of=5):
        """
        Recursively constructs a Balltree of _Balltree_node objects generated from array and returns the root.
        
        Args:
            array (np.ndarray): The array to be made into a balltree. Assumed to be of shape (number of points, number of dimensions for each point).
            spread_of (int, optional): The maximum number of points to check to estimate the spread of a subset of points along a particular dimension. Defaults to 5.
            median_of (int, optional): The maximum number of points to check to estimate the median of a subset of points along a particular dimension. Defaults to 5.
        
        Returns:
            _Balltree_node: The root of the constructed balltree.
        """

        # Determine pivot, or split point.

        pivot_dimension = Balltree._find_widest_dimension_approx(array, spread_of=spread_of)
        pivot_value = Balltree._find_median_approx(array, pivot_dimension, median_of=median_of)
        pivot_column = array[:, pivot_dimension]

        # Make this_node.

        pivot_points = array[pivot_column == pivot_value]
        pivot_coordinates = pivot_points[0]
        this_node = _Balltree_node(pivot_coordinates, pivot_dimension)

        # Partition array into left_array and right_array, reduced by the pivot point.

        left_array = array[pivot_column <= pivot_value]
        right_array = array[pivot_column > pivot_value]
        # Remove the first element of pivot_points from left_array.
        for index, point in enumerate(left_array):
            if point[pivot_dimension] == pivot_value:
                break
        left_array = np.append(left_array[:index], left_array[index + 1:], axis=0) # Side effect: breaks alias, makes copy.

        # Recursively make and attach children, terminating each child if it has size 0.

        if left_array.size > 0:
            left_child = Balltree.generate_Balltree(left_array, spread_of, median_of)
            this_node.attach_left_child(left_child)

        if right_array.size > 0:
            right_child = Balltree.generate_Balltree(right_array, spread_of, median_of)
            this_node.attach_right_child(right_child)

        # Compute and set radius as the largest distance from the pivot point to another point in array.
        
        radius = np.sqrt(max(map(lambda point: np.sum((point - pivot_coordinates)**2), array)))
        this_node.set_radius(radius)

        return this_node


    @staticmethod
    def _get_sample_indices(array, n_indices=5):
        """Return up to n_indices random indices into the 0th dimension of array."""
        
        try:
            point_indices = np.random.choice(range(len(array)), n_indices, replace=False)
        except ValueError:
            # len(array) < n_indices.
            point_indices = np.arange(len(array))
            
        return point_indices


    @staticmethod
    def _find_widest_dimension_approx(array, spread_of=5):
        """
        Return the dimension of array with estimably the greatest spread.
        Assumes array's 0th and 1st dimensions correspond to points and dimensions respectively.
        """
        
        point_indices = Balltree._get_sample_indices(array, n_indices=spread_of)
        
        array_sample = array[point_indices]
        
        return np.argmax(np.ptp(array_sample, axis=0))


    @staticmethod
    def _find_median_approx(array, dimension, median_of=5):
        """
        Return the median of median_of randomly chosen dimension-th coordinates in array.
        Always takes the median of an odd number of points.
        Assumes array's 0th and 1st dimensions correspond to points and dimensions respectively.
        """
        
        point_indices = Balltree._get_sample_indices(array, n_indices=median_of)
        if point_indices.size % 2 == 0:
            point_indices = np.random.choice(point_indices, size=point_indices.size - 1, replace=False)

        return np.median(array[point_indices, dimension])


    @staticmethod
    def _distance(point_x, point_y):
        """Return the quadratic sum of the two coordinates point_x and point_y."""

        return np.sqrt(np.sum((point_y - point_x)**2))


    def k_nearest_neighbors_search(self, target, k):
        """
        Create and return a max-first priority queue with capacity k to store all encountered nearest neighbors to target.

        Recursively follows the tree starting at self.root by making the first call to _k_nearest_neighbors_recursive. 
        
        The tree is searched selectively, terminating recursion at nodes whose subtree cannot contain nearer neighbors 
        than have already been encountered.
        
        Args:
            target (np.ndarray, seq): The coordinates of the point whose nearest neighbors are returned.
            k (int): The number of nearest-neighbors to search for.
        
        Raises:
            TypeError: Raised if target has values of types other than float or int.
            ValueError: Raised if target is not 1-dimensional.
            ValueError: raised if target does not have the same length as the coordinates stored in self.root.
            TypeError: Raised if k is not of type int.
            ValueError: Raised if k is not positive.
        
        Returns:
            N_ary_heap: The heap used as a max-first priority queue holding the (up to) k nearest neighbors to target in self.
        """

        # Validate inputs.
        target = np.array(target)
        if target.dtype not in [float, int]:
            raise TypeError(f"target must have values of type int or float.\n"
                            f"target.dtype: {target.dtype}.")
        if target.ndim != 1:
            raise ValueError(f"target must be 1-dimensional.\n"
                             f"target.ndim: {target.ndim}.")
        if len(target) != len(self.root.coordinates):
            raise ValueError(f"target must have the same length as the coordinates in self, the Balltree.\n"
                             f"len(target): {len(taraget)}, len(self.root.coordinates): {len(self.root.coordinates)}.")
        if not isinstance(k, int):
            raise TypeError(f"k must be of type int.\n"
                            f"type(k): {type(k)}.")
        if k < 1:
            raise ValueError(f"k must be positive.\n"
                             f"k: {k}.")

        queue = N_ary_heap(capacity=k, heap_type='max', satellites=True)

        return Balltree._k_nearest_neighbors_recursive(target, queue, self.root)

    
    @staticmethod
    def _k_nearest_neighbors_recursive(target, queue, node):
        """
        Recursively add the nearest neighbors to target in the subtree rooted at node to queue and return that queue.

        Selectively explore the subtree, terminating recursion at nodes whose subtree cannot contain a neighbor 
        nearer than one already in queue.
        """

        target_to_node_distance = Balltree._distance(target, node.coordinates)

        # If the queue is not yet full, 
        # or the minimum possible distance to a point within this node's radius is less than the greatest distance in the queue, 
        # then enqueue target_to_node_distance and recurse on this node's children.
        if not queue.is_full() or target_to_node_distance - node.radius < queue.peek():
            queue.push(target_to_node_distance, node.coordinates)

            # Recurse on both children if they exist, starting with the nearer one.

            # If node has both a left and a right child, recurse on the nearer one and then the further one.
            if node.left_child is not None and node.right_child is not None:
                # Calculate the distance to each child.
                target_to_left_child_distance = Balltree._distance(target, node.left_child.coordinates)
                target_to_right_child_distance = Balltree._distance(target, node.right_child.coordinates)
                # Recurse on the nearer child first.
                if target_to_left_child_distance <= target_to_right_child_distance:
                    Balltree._k_nearest_neighbors_recursive(target, queue, node.left_child)
                    Balltree._k_nearest_neighbors_recursive(target, queue, node.right_child)
                else:
                    Balltree._k_nearest_neighbors_recursive(target, queue, node.right_child)
                    Balltree._k_nearest_neighbors_recursive(target, queue, node.left_child)
            # Otherwise, recurse on any extant child, terminating recursion when both children are None.
            else:
                if node.left_child is not None:
                    Balltree._k_nearest_neighbors_recursive(target, queue, node.left_child)
                if node.right_child is not None:
                    Balltree._k_nearest_neighbors_recursive(target, queue, node.right_child)

        return queue
