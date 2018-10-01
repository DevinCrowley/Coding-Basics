// using System.Random;

namespace BinarySearchTree
{

	public class Node
	{
		int key;
		public int Key
		{
			get {return key;}
		}

		Node parent = null;
		public Node Parent
		{
			get {return parent;}
			set {parent = value;}
		}

		Node left = null;
		public Node Left
		{
			get {return left;}
			set {left = value;}
		}

		Node right = null;
		public Node Right
		{
			get {return right;}
			set {right = value;}
		}

		// For setting parent, left, & right properties at construction if desired.
		public Node(int key, params Node[] parent_left_right)
		{
			this.key = key;

			int numParams = parent_left_right.Length;

			switch (numParams)
			{
				case 3:
					this.right = parent_left_right[2];
					goto case 2;
				case 2:
					this.left = parent_left_right[1];
					goto case 1;
				case 1:
					this.parent = parent_left_right[0];
					break;
				case 0:
					break;
				// If numParams > 3, assume the first 3 were correct and ignore the rest.
				default:
					goto case 3;
			}
		}



	}

	public class BinarySearchTree 
	{
		Node root = null;
		public Node Root
		{
			get {return root;}
			set {root = value;}
		}


		public Node findMinimum(Node node)
		{
			// Handle null case
			if (node == null) {return null;}

			while (node.Left != null)
			{
				node = node.Left;
			}

			return node;
		}


		public Node findMaximum(Node node)
		{
			// Handle null case
			if (node == null) {return null;}

			while (node.Right != null)
			{
				node = node.Right;
			}

			return node;
		}


		public Node findSuccessor(Node node)
		{
			// Handle null case
			if (node == null) {return null;}

			if (node.Right != null) {return findMinimum(node.Right);}

			// Since node has no right child, the successor is the parent of 
			// the nearest (highest) left-child.
			Node parent = node.Parent;
			while (parent != null && node == parent.Right)
			{
				node = parent;
				parent = parent.Parent;
			}
			// node is now either the root of the whole tree, or it is a left child.

			return parent;
		}


		public Node findPredecessor(Node node)
		{
			// Handle null case
			if (node == null) {return null;}

			if (node.Left != null) {return findMaximum(node.Left);}

			// Since node has no left child, the predecessor is the parent of 
			// the nearest (highest) right-child.
			Node parent = node.Parent;
			while (parent != null && node == parent.Left)
			{
				node = parent;
				parent = parent.Parent;
			}
			// node is now either the root of the whole tree, or it is a right child.

			return parent;
		}


		public void inOrderTreeWalk(Node root)
		{
			if (root != null)
			{
				inOrderTreeWalk(root.Left);
				System.Console.WriteLine(root.Key);
				inOrderTreeWalk(root.Right);
			}
		}


		public void insert(Node newNode)
		{
			// Find node's appropriate location in the tree
			Node tracerNode = this.root;
			Node tracerNodeParent = null;
			while (tracerNode != null)
			{
				tracerNodeParent = tracerNode;
				if (newNode.Key < tracerNode.Key)
				{
					tracerNode = tracerNode.Left;
				}
				else 
				{
					tracerNode = tracerNode.Right;
				}
			}
			// tracerNodeParent is now the appropriate, available parent for newNode.

			// if tracerNodeParent is null, we are adding the root to an empty tree.
			if (tracerNodeParent == null)
			{
				this.root = newNode;
				return;
			}

			// Perform insertion
			newNode.Parent = tracerNodeParent;
			if (newNode.Key < tracerNodeParent.Key)
			{
				tracerNodeParent.Left = newNode;
			}
			else
			{
				tracerNodeParent.Right = newNode;
			}
		}


		void transplant(Node oldNode, Node newNode)
		{
			// Set root if applicable
			if (oldNode.Parent == null)
				{this.root = newNode;}
			// Set parent's child
			else if (oldNode == oldNode.Parent.Left)
				{oldNode.Parent.Left = newNode;}
			else 
				{oldNode.Parent.Right = newNode;}

			// Set newNode's parent
			if (newNode != null)
				{newNode.Parent = oldNode.Parent;}
		}


		void delete(Node removeThisNode)
		{
			// Case 1: removeThisNode has less than two children.
			if (removeThisNode.Left == null)
			{
				transplant(removeThisNode, removeThisNode.Right);
			}
			else if (removeThisNode.Right == null)
			{
				transplant(removeThisNode, removeThisNode.Left);
			}

			// Case 2: removeThisNode has two children. 
			// Successor necessarily has no left child.
			Node successor = findSuccessor(removeThisNode);

			// If successor is not the child of removeThisNode, 
			// then insert successor above removeThisNode's right child.
			if (successor != removeThisNode.Right)
			{
				// Gracefully remove successor from its position.
				transplant(successor, successor.Right);

				// Attach successor above removeThisNode's right child.
				removeThisNode.Right.Parent = successor;
				successor.Right = removeThisNode.Right;
			}

			// Attach successor to removeThisNode's parent.
			transplant(removeThisNode, successor);

			// Attach removeThisNode's left child.
			removeThisNode.Left.Parent = successor;
			successor.Left = removeThisNode.Left;

		}


	} // end class BinarySearchTree


	// Testing required.


} // end namespace BinarySearchTree


