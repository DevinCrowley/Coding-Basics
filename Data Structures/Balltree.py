import numpy as np


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

    def __init__(self, array, spread_of=5, median_of=5):

        self.root = self.generate_Balltree(array, spread_of, median_of)


    @staticmethod
    def generate_Balltree(array, spread_of=5, median_of=5):

        # Determine pivot, or split point.

        pivot_dimension = Balltree._find_widest_dimension_approx(array, spread_of=spread_of)
        pivot_value = Balltree._find_median_approx(array, pivot_dimension, median_of=median_of)
        pivot_column = array[array[:, pivot_dimension]]

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

        # Recursively make and attack children, terminating each child if it has size 0.

        if left_child.size > 0:
            left_child = Balltree.generate_Balltree(left_array, spread_of, median_of)
            this_node.attach_left_child(left_child)

        if right_child.size > 0:
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
        
        point_indices = _get_sample_indices(array, n_indices=spread_of)
        
        array_sample = array[point_indices]
        
        return np.argmax(np.ptp(array_sample, axis=0))


    @staticmethod
    def _find_median_approx(array, dimension, median_of=5):
        """
        Return the median of median_of randomly chosen dimension-th coordinates in array.
        Always takes the median of an odd number of points.
        Assumes array's 0th and 1st dimensions correspond to points and dimensions respectively.
        """
        
        point_indices = _get_sample_indices(array, n_indices=median_of)
        if point_indices.size % 2 == 0:
            point_indices = np.random.choice(point_indices, size=point_indices.size - 1, replace=False)

        return np.median(array[point_indices, dimension])


    def k_nearest_neighbors(self, k):

        raise NotImplementedError("Do it.")