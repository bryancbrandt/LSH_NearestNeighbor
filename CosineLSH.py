# Similarity Search in High Dimensions via Hashing Implementation
# Derived from the paper by Gionis, Indyk, Motwani, 1999
# Bryan Brandt
# CS738 2023

import ast
import linecache
import pickle
from collections import defaultdict
from os.path import exists
import numpy as np
from PriorityQueue import PriorityQueue


class CosineLSH:
    n = 0  # Number of data points
    l = 32  # Number of hash tables
    bit_depth = 24
    d = 0  # Number of dimensions
    file = ""  # The csv file to be loaded
    tau = list(defaultdict(list))
    verbose = False
    hyperplanes = []
    DIST_THRESH = 100

    def __init__(self, file: str, num_tables: int = 32, bit_depth: int = 24, verbose: bool = False):
        assert num_tables > 0, "Number of hash tables must be greater than 0"
        assert bit_depth > 0, "Bit Depth must be greater than 0"
        self.verbose = verbose
        self.l = num_tables
        self.bit_depth = bit_depth

        if self.verbose:
            print("================================================")
            print("Cosine LSH Initializing")
            print("================================================")

        file_exists = exists(file)
        if not file_exists:
            raise IOError(f"{file} does not exist!")
        self.file = file

        # Initialize the hash tables
        for i in range(self.l):
            self.tau.append(defaultdict(list))

    def load_csv_file(self):
        """
        Loads the csv file, and sets all the class attributes
        """

        # Load the Data
        if self.verbose:
            print(f"Loading data file {self.file}")
        n_size = 0
        with open(self.file, 'r') as f:
            for i in f:
                line_data = np.array(ast.literal_eval(i.strip()))
                n_size += 1

        # Get the dimension of the data
        self.d = line_data.shape[0]
        if self.verbose:
            print(f"Dimensionality of data: {self.d}")

        # Get the number of data points
        self.n = n_size
        if self.verbose:
            print(f"Number of data points {self.n}")

        # Generate Hyperplanes for hashing
        for i in range(self.l):
            self.hyperplanes.append(np.random.normal(size=(self.bit_depth, self.d)))

    def preprocess(self, save_pickle: bool = True):
        """
        Preprocesses the data
        :param save_pickle: True if we want to save the self.tau dict to a file
        """
        if self.verbose:
            print()
            print("================================================")
            print("Preprocessing...")
            print("================================================")

        # Open the file, so that we can begin adding the points to the hash tables
        # the points identifier will be the row number in the file, beginning with 1
        if self.verbose:
            print("Hashing datapoints from the file into the hashtable...")
        line_index = 0
        with open(self.file, 'r') as f:
            for i in f:
                line_data = np.array(ast.literal_eval(i.strip()))
                line_index += 1
                for j in range(self.l):
                    hash_code = self.generate_hash_code(line_data, j)
                    self.tau[j][hash_code].append(line_index)

        # Save the file if required to
        if save_pickle:
            tau_filename = "pickle_files/" + self.file[5:-4] + "_cosine.obj"
            fileObj = open(tau_filename, 'wb')
            save_object = {"tau": self.tau, "l": self.l, "hyperplanes": self.hyperplanes}
            pickle.dump(save_object, fileObj)
            fileObj.close()
            print(f"Tau saved to file {tau_filename} successfully")

    def generate_hash_code(self, vector: np.ndarray, l: int) -> str:
        """
        Generates a bit string hash code based on the passed vector
        :param l: Determines which hyperplane set to use from the list of hyperplanes
        :param vector: a vector with n-dimensions that a hash code will be generated for
        :return: A hashcode string of length self.bit_depth using 1 and 0
        """
        hash_matrix = np.array(vector * self.hyperplanes[l])
        hash_code = str()
        for item in hash_matrix:
            result = item.sum()
            if result >= 0:
                hash_code += '1'
            else:
                hash_code += '0'
        return hash_code

    def load_pickle(self, file: str):
        """
        Loads a formerly saved tau dictionary from disk, and sets it as the self.tau attribute
        :param file: the filename to be used
        """
        file_exists = exists(file)
        if not file_exists:
            raise IOError(f"Pickle file: {file} does not exist!")

        if self.verbose:
            print(f"Loading pickle file: {file}")

        fileObj = open(file, 'rb')
        load_object = pickle.load(fileObj)
        self.tau = load_object["tau"]
        self.l = load_object["l"]
        self.hyperplanes = load_object["hyperplanes"]
        fileObj.close()

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Cosine Similarity
        :param vec1: a vector to compare to vector 2
        :param vec2: a vector to compare to vector 1
        :return: Distance between vector 1 and 2 measured as cosine similarity
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)

    def query(self, q, k):
        """
        Find the k approximate nearest neighbors of q
        :param q: an integer identifying the index number or line number in the file
        :param k: an integer specifying the number of nearest neighbors
        :return: A set containing tuples in the form of (line_index #, distance)
        """
        assert q > 0, "Error, query point index number must > 0"
        assert k > 1, "Error, number of neighbors must be > 1"

        if self.verbose:
            print("================================================")
            print(f"Querying value {q}")
            print("================================================")

        neighbors = set()
        buckets = []

        # Extract the vector q from the file
        q_line = linecache.getline(self.file, q).strip()
        q_values = np.array(ast.literal_eval(q_line))

        # Get the hash values for the point q, and add them to the list
        for i in range(self.l):
            hash_val = self.generate_hash_code(q_values, i)
            buckets.append(hash_val)

        pq = PriorityQueue(True)

        # Search through the list of candidates and add them to the priority queue
        if self.verbose:
            print(f"Retrieving nearest {k} neighbors for index {q}")
        for i in range(self.l):
            for item in self.tau[i][buckets[i]]:
                neighbors.add(item)
                item_line = linecache.getline(self.file, item).strip()
                item_values = np.array(ast.literal_eval(item_line))
                distance = self.cosine_similarity(q_values, item_values)
                if 0 < distance < self.DIST_THRESH:
                    pq.add(distance, item)

        if self.verbose:
            print(f"Length of set S: {len(neighbors)}, Number of neighbors that pass threshold: {self.DIST_THRESH}")

        # Get the k nearest neighbors
        result = set()
        while len(result) < k:
            result.add(pq.poll())

        return result
