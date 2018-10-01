"""
This file contains the following instantiable classes: UniversalHashTable, MultiplicationHashTable, and DoubleHashedOpenAddressTable.

Since there are many ways to produce a hash table, much of the work has been 
generalized into abstract classes so that new classes can be made easily.

UniversalHashTable and MultiplicationHashTable inherit from AbstractHashTable. 
DoubleHashedOpenAddressTable inherits from AbstractOpenAddressTable.

This file is organized as follows:
- Utility Functions
- Abstract Classes
- Instantiables
- Unit Tests
"""

from abc import ABC, abstractmethod
from sympy import sieve
import numpy as np
import unittest

r"""
  _    _   _     _   _   _   _               ______                          _     _                       
 | |  | | | |   (_) | | (_) | |             |  ____|                        | |   (_)                      
 | |  | | | |_   _  | |  _  | |_   _   _    | |__     _   _   _ __     ___  | |_   _    ___    _ __    ___ 
 | |  | | | __| | | | | | | | __| | | | |   |  __|   | | | | | '_ \   / __| | __| | |  / _ \  | '_ \  / __|
 | |__| | | |_  | | | | | | | |_  | |_| |   | |      | |_| | | | | | | (__  | |_  | | | (_) | | | | | \__ \
  \____/   \__| |_| |_| |_|  \__|  \__, |   |_|       \__,_| |_| |_|  \___|  \__| |_|  \___/  |_| |_| |___/
                                    __/ |                                                                  
                                   |___/                                                                   
"""

def next_prime(value):
	"""Returns the nearest prime at or after value, calculated using the sieve of Erastosthenes."""
	return sieve[sieve.search(value)[1]]


def bounding_powers_of_2(value):
	"""Returns the nearest powers of two bounding value on both sides."""
	root = np.log2(value)
	if root % 1 == 0: # If value is a power of 2
		return value, value
	else:
		truncatedRoot = int(root)
		return np.power(2, truncatedRoot), np.power(2, truncatedRoot + 1)


def position_between_powers_of_2(value):
	"""Returns the ratio of value to the nearest previous power of 2."""
	root = np.log2(value)
	if root % 1 == 0: # If value is a power of 2
		return 0
	else:
		actualRange = np.power(2, int(root))
		return (value - actualRange) / actualRange


def next_prime_away_from_powers_of_2(value, threshold = 0.3):
	"""Returns the nearest prime number at or after value that is not within threshold * (distance between bounding powers of 2) of a power of 2."""
	# If threshold == 0.5, at most one value between the bounding powers of 2 will be allowed.
	if threshold > 0.5: 
		raise ValueError("threshold must be in [0.0, 0.5].")

	newValue = next_prime(value)
	closenessToPowerOf2 = position_between_powers_of_2(newValue) 

	# While newValue is not within threshold of a bounding power of 2
	while closenessToPowerOf2 < threshold or closenessToPowerOf2 > 1 - threshold:
		truncatedRoot = int(np.log2(newValue))
		# Try a prime starting at threshold
		if closenessToPowerOf2 < threshold:
			newValue = next_prime(np.power(2, truncatedRoot) * (1 + threshold))
		# Try a prime starting at threshold after the next power of 2
		else: # closenessToPowerOf2 > 1 - threshold
			newValue = next_prime(np.power(2, truncatedRoot + 1) * (1 + threshold))

		closenessToPowerOf2 = position_between_powers_of_2(newValue)

	return newValue

r"""
             _             _                           _        _____   _                                  
     /\     | |           | |                         | |      / ____| | |                                 
    /  \    | |__    ___  | |_   _ __    __ _    ___  | |_    | |      | |   __ _   ___   ___    ___   ___ 
   / /\ \   | '_ \  / __| | __| | '__|  / _` |  / __| | __|   | |      | |  / _` | / __| / __|  / _ \ / __|
  / ____ \  | |_) | \__ \ | |_  | |    | (_| | | (__  | |_    | |____  | | | (_| | \__ \ \__ \ |  __/ \__ \
 /_/    \_\ |_.__/  |___/  \__| |_|     \__,_|  \___|  \__|    \_____| |_|  \__,_| |___/ |___/  \___| |___/
                                                                                                           
                                                                                                           
"""

class AbstractHashTable(ABC):
	
	def __init__(self, numSlots):
		self.numSlots = numSlots
		self.table = [[] for _ in range(numSlots)]


	@abstractmethod
	def hash(self, key): pass

	
	def find_slot(self, key):
		hashCode = self.hash(key)
		slot = self.table[hashCode]
		return slot

	
	def find_key_in_slot(self, key, slot):
		for index in range(len(slot)):
			if slot[index][0] == key:
				return index
		# Key not found in slot.
		return None

	
	def insert(self, key, value):
		slot = self.find_slot(key)
		index = self.find_key_in_slot(key, slot)
		if index is None:
			# If key is not already in the slot, append it to slot.
			slot.append((key, value))
		else:
			# If key is already there, overwrite the old value.
			slot[index] = (key, value)

	
	def search(self, key):
		slot = self.find_slot(key)
		index = self.find_key_in_slot(key, slot)
		if index is not None:
			return slot[index][1]
		else:
			return None

	
	def delete(self, key):
		slot = self.find_slot(key)
		index = self.find_key_in_slot(key, slot)
		if index is not None:
			value = slot[index][1]
			del slot[index]
			return value
		else:
			return None


class AbstractOpenAddressTable(ABC):
	
	def __init__(self, numSlots):
		self.numSlots = next_prime_away_from_powers_of_2(numSlots)
		self.table = [None for _ in range(self.numSlots)]

	@ abstractmethod
	
	def hash(self, key, iteration): pass

	
	def insert(self, key, value):
		for iteration in range(self.numSlots):
			hashCode = self.hash(key, iteration)
			element = self.table[hashCode]
			if element in [None, 'DELETED ELEMENT']:
				self.table[hashCode] = (key, value)
				return hashCode
		raise Exception("Overflow: Open Address Table is already full to capacity.")


	def findHashCode(self, key):
		for iteration in range(self.numSlots):
			hashCode = self.hash(key, iteration)
			element = self.table[hashCode]
			# Hash sequence not probed up to this point: terminate early
			if element is None:
				return None
			if element != 'DELETED ELEMENT' and element[0] == key:
				return hashCode
		# key not found
		return None


	def search(self, key):
		hashCode = self.findHashCode(key)
		if hashCode is not None:
			return self.table[hashCode][1]
		else:
			return None

	def delete(self, key):
		hashCode = self.findHashCode(key)
		if hashCode is not None:
			value = self.table[hashCode][1]
			self.table[hashCode] = 'DELETED ELEMENT'
			return value
		else:
			return None

r"""
  _____                 _                     _     _           _       _              
 |_   _|               | |                   | |   (_)         | |     | |             
   | |    _ __    ___  | |_    __ _   _ __   | |_   _    __ _  | |__   | |   ___   ___ 
   | |   | '_ \  / __| | __|  / _` | | '_ \  | __| | |  / _` | | '_ \  | |  / _ \ / __|
  _| |_  | | | | \__ \ | |_  | (_| | | | | | | |_  | | | (_| | | |_) | | | |  __/ \__ \
 |_____| |_| |_| |___/  \__|  \__,_| |_| |_|  \__| |_|  \__,_| |_.__/  |_|  \___| |___/
                                                                                       
                                                                                       
"""

class UniversalHashTable(AbstractHashTable):
	
	def __init__(self, numSlots, universalKeyMax):
		super().__init__(numSlots)
		self.universalPrime = next_prime(universalKeyMax)
		self.a = np.random.randint(1, self.universalPrime)
		self.b = np.random.randint(0, self.universalPrime)

	
	def hash(self, key):
		return (self.a * key + self.b) % self.universalPrime % self.numSlots


class MultiplicationHashTable(AbstractHashTable):
	
	def __init__(self, numSlots):
		super().__init__(numSlots)
		# Multiplication constant
		self.A = (np.sqrt(5) - 1) / 2

	
	def hash(self, key):
		return int((self.A * key) % 1 * self.numSlots)


class DoubleHashedOpenAddressTable(AbstractOpenAddressTable):
	
	def hash(self, key, iteration):
		return ((key % self.numSlots) + 
				iteration * (1 + key % (self.numSlots - 1))) % self.numSlots

r"""
  _   _           _   _       _____                _         
 | | | |  _ __   (_) | |_    |_   _|   ___   ___  | |_   ___ 
 | | | | | '_ \  | | | __|     | |    / _ \ / __| | __| / __|
 | |_| | | | | | | | | |_      | |   |  __/ \__ \ | |_  \__ \
  \___/  |_| |_| |_|  \__|     |_|    \___| |___/  \__| |___/
                                                             
"""

class AbstractTestHashTables(ABC):
	"""Enforces a standard battery of tests for inheritors. 
	Ultimately this is unnecessary, but the order it imposes may be convenient 
	if more subclasses are made from the same abstract classes, 
	and additional, very similar unittest classes are required."""

	@abstractmethod
	def setUp(self): pass


	@abstractmethod
	def test_init(self): pass


	@abstractmethod
	def test_hash(self): pass


	@abstractmethod
	def test_insert(self): pass


	@abstractmethod
	def test_search(self): pass


	@abstractmethod
	def test_delete(self): pass


class TestUniversalHashTable(unittest.TestCase, AbstractTestHashTables):

	def setUp(self):
		# Create table
		np.random.seed(0)
		self.hashTable = UniversalHashTable(15, 60)

		# Input keys for tests
		self.testKeys = [0, 1, 2, 3, 4]
		# testValues = testKeys**2: [0, 1, 4, 9, 16]

		# Insert key-value pairs into table
		for testKey in self.testKeys: 
			self.hashTable.insert(testKey, testKey**2)

		# Expected hashvalues
		self.expectedHashes = [2, 1, 0, 0, 14]

		# Expected resultant table after insertions
		self.expectedTable = [[] for _ in range(15)]
		for thisKey in self.testKeys:
			thisHash = self.expectedHashes[thisKey]
			self.expectedTable[thisHash].append((thisKey, thisKey**2))


	def test_init(self):
		self.assertEqual(self.hashTable.a, 45)
		self.assertEqual(self.hashTable.b, 47)
		self.assertEqual(self.hashTable.universalPrime, 61)
		self.assertEqual(self.hashTable.numSlots, 15)


	def test_hash(self): 
		hashValues = list(map(self.hashTable.hash, self.testKeys))
		self.assertEqual(hashValues, self.expectedHashes)


	def test_insert(self):
		self.assertEqual(self.hashTable.table, self.expectedTable)


	def test_search(self):
		# search for keys 0-5. Only keys 0-4 have been inserted.
		returnedValues = list(map(self.hashTable.search, range(6)))
		expectedReturnedValues = [i**2 for i in range(5)] + [None]
		self.assertEqual(returnedValues, expectedReturnedValues)


	def test_delete(self):
		# Alias self.expectedTable for clarity
		modifiedTable = self.expectedTable
		# The key 2 was the first key inserted that hashed to slot 0.
		del modifiedTable[0][0]

		# delete entry with the key 2
		deletedValue = self.hashTable.delete(2)
		self.assertEqual(deletedValue, 4)
		
		self.assertEqual(self.hashTable.table, modifiedTable)


class TestDoubleHashedOpenAddressTable(unittest.TestCase, AbstractTestHashTables):

	def setUp(self):
		self.hashTable = DoubleHashedOpenAddressTable(10)
		testKeys = [2, 5, 13, 16, 27]
		# testValues = testKeys**2: [4, 25, 169, 256, 729]

		# Insert key-value pairs into table
		for testKey in testKeys: 
			self.hashTable.insert(testKey, testKey**2)

		# Expected value of numSlots: 11

		# Expected hash sequences: key --> hash sequence
		"""
		2 --> 2, 5, 8, 0, 3, 6, 9, 1, 4, 7, 10
		5 --> 5, 0, 6, 1, 7, 2, 8, 3, 9, 4, 10
		13 --> 2, 6, 10, 3, 7, 0, 4, 8, 1, 5, 9
		16 --> 5, 1, 8, 4, 0, 7, 3, 10, 6, 2, 9
		27 --> 5, 2, 10, 7, 4, 1, 9, 6, 3, 0, 8
		"""
		# Expected hashes
		expectedHashes = [2, 5, 6, 1, 10]

		self.keyHashPairs = dict(zip(testKeys, expectedHashes))

		# Expected resultant table after insertions
		self.expectedTable = [None for _ in range(11)]
		for thisKey in self.keyHashPairs.keys():
			thisHash = self.keyHashPairs[thisKey]
			self.expectedTable[thisHash] = (thisKey, thisKey**2)


	def test_init(self):
		self.assertEqual(self.hashTable.numSlots, 11)


	def test_hash(self):
		hashValues = list(map(self.hashTable.hash, self.keyHashPairs.keys(), [0, 0, 1, 1, 2]))
		self.assertEqual(hashValues, list(self.keyHashPairs.values()))


	def test_insert(self):
		self.assertEqual(self.hashTable.table, self.expectedTable)


	def test_search(self):
		# search for keys [2, 5, 13, 16, 27, 30]. Only the first 5 have been inserted.
		returnedValues = list(map(self.hashTable.search, list(self.keyHashPairs.keys()) + [30]))
		expectedReturnedValues = [key**2 for key in self.keyHashPairs.keys()] + [None]
		self.assertEqual(returnedValues, expectedReturnedValues)


	def test_delete(self):
		# Copy table before deletion
		modifiedTable = list(self.hashTable.table)
		# Perform manual deletion of key 13, hashed to index 6
		modifiedTable[6] = 'DELETED ELEMENT'

		# delete entry with key 13
		deletedValue = self.hashTable.delete(13)
		self.assertEqual(deletedValue, 13**2)
		
		self.assertEqual(self.hashTable.table, modifiedTable)


	def test_insert_after_delete(self):
		# Copy table before modification
		modifiedTable = list(self.hashTable.table)

		# delete keys 2 and 13
		self.hashTable.delete(2)
		self.hashTable.delete(13)

		# re-insert 13 - index 2, the value passed over in it's initial hash sequence, 
		# should be marked 'DELETED ELEMENT' and be available for insertion.
		self.hashTable.insert(13, 13**2)

		modifiedTable[self.keyHashPairs[2]] = (13, 13**2)
		modifiedTable[self.keyHashPairs[13]] = 'DELETED ELEMENT'

		# self.hashTable.table should be: 
		# [None, (16, 256), (13, 169), None, None, (5, 25), 'DELETED ELEMENT', 
		# None, None, None, (27, 729)]


		self.assertEqual(self.hashTable.table, modifiedTable)


if __name__ == "__main__":
	unittest.main()
