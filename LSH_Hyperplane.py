# Similarity Search in High Dimensions via Hashing Implementation
# Derived from the paper by Gionis, Indyk, Motwani, 1999
# Bryan Brandt
# CS738 2023

import ast
import linecache
import numpy as np
import pickle
from collections import defaultdict
from os.path import exists
from PriorityQueue import PriorityQueue
from scipy.spatial.distance import cityblock


class LSH_Hyperplane:
    ALPHA = 2  # Memory utilization parameter
    m = 0  # Size of the hash table
    n = 0  # Number of data points
    B = 0  # Maximum bucket size, calculated by block size / dimensions
    d = 0  # Number of dimensions
    l = 32  # Number of hash tables
    min = 0  # Minimum value of the data
    max = 0  # Maximum value of the data
    file = ""  # The csv file to be loaded
    num_dim_hash = 24  # Number of dimensions for the hash tables
    tau = list(defaultdict(list))  # The set of hashtable
    l_subset_dimensions = []  # Dimensional subsets for I1,...,Il
    l_subset_a = []  # A values for h(x) hash calculation
    DIST_THRESH = 100  # The distance threshold for l1 normalization
    hyperplanes = []  # The list of all np array hyperplanes
    hash_bit_size = 0  # The bit-size of the hash table code
    verbose = True

    def __init__(self, file: str, hash_bit_size: int = 32, verbose: bool = True):
        assert hash_bit_size > 0, "Error hash bit size must be greater than 0"
        self.verbose = verbose

        if self.verbose:
            print("================================================")
            print("LSH Hyperplane Initializing")
            print("================================================")

        file_exists = exists(file)
        if not file_exists:
            raise IOError(f"{file} does not exist!")
        self.file = file

        self.hash_bit_size = hash_bit_size

        # Initialize the hash tables
        if self.verbose:
            print("Initializing hash tables...")
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
                if self.max < line_data.max():
                    self.max = line_data.max()

        # Get the max value
        # self.max = self.data.max()
        if self.verbose:
            print(f"Max value: {self.max}")

        # Get the dimension of the data
        self.d = line_data.shape[0]
        if self.verbose:
            print(f"Dimensionality of data: {self.d}")

        # Get the number of data points
        self.n = n_size
        if self.verbose:
            print(f"Number of data points {self.n}")

        # Calculate M (First calculate B)
        self.B = int(8192 / self.d)
        self.m = int(self.ALPHA * (self.n / self.B))
        if self.verbose:
            print(f"Bucket size: {self.B}")
            print(f"M value: {self.m}")

    def preprocess(self, save_pickle: bool = False):
        """
        Pre-process the Data.  This is the Preprocessing Algorithm from the paper
        :param save_pickle: True if we want to save the self.tau dict to a file
        """
        if self.verbose:
            print()
            print("================================================")
            print("Preprocessing...")
            print("================================================")

        # Create a matrix of random values of [0, 1) size l x d and normalize them to [-1, 1)
        if self.verbose:
            print("Creating Hyperplanes...")
        for r in range(self.l):
            hyperplanes = np.random.random_sample((self.hash_bit_size, self.d))
            hyperplanes = (hyperplanes * 2) - 1
            self.hyperplanes.append(hyperplanes)

        line_index = 0
        with open(self.file, 'r') as f:
            for i in f:
                line_data = np.array(ast.literal_eval(i.strip()))
                line_index += 1
                # Iterate through each hash function with each line from the file
                for j in range(self.l):
                    hash_code = self.generate_hash_code(line_data, j)
                    self.tau[j][hash_code].append(line_index)

        # Pickle needs to save self.d, self.m, self.hyperplanes,
        if save_pickle:
            tau_filename = "pickle_files/" + self.file[5:-4] + "_hyper.obj"
            fileObj = open(tau_filename, 'wb')
            save_object = {"tau": self.tau, "d": self.d, "m": self.m, "hyperplanes": self.hyperplanes}
            pickle.dump(save_object, fileObj)
            fileObj.close()
            if self.verbose:
                print(f"Saved to pickle file {tau_filename} successfully")

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
        self.d = load_object["d"]
        self.m = load_object["m"]
        self.hyperplanes = load_object["hyperplanes"]
        fileObj.close()

        if self.verbose:
            print("File loaded successfully")

    def generate_hash_code(self, array: np.ndarray, index: int) -> int:
        assert isinstance(array, np.ndarray) or array.shape[0] != self.d, "Must be ndarray with same dimensions!"
        product = []
        for planes in self.hyperplanes[index]:
            product.append(np.dot(planes, array))
        bits = [1 if x > 0 else 0 for x in product]
        integer = sum([j * (2 ** i) for i, j in list(enumerate(reversed(bits)))])
        integer = integer % self.m
        return integer

    def query(self, q: int, k: int) -> set:
        """
        Approximate Nearest Neighbor Query.
        :param q: The line number from the file representing the data point to be queried
        :param k: The number of nearest neighbors to return
        :return: List of tuples formatted: (distance, point number)
        """
        assert q > 0, "Value of q must be greater than 0."
        assert k >= 1, "Value of k must be greater or equal to 1."

        if self.verbose:
            print()
            print("================================================")
            print(f"Querying value {q}")
            print("================================================")

        S = set()
        buckets = []

        # Extract the dimensional values of q
        q_line = linecache.getline(self.file, q).strip()
        q_values = np.array(ast.literal_eval(q_line))

        # Hash the q_values to get all the bucket numbers and add them to buckets list
        if self.verbose:
            print(f"Getting hash values for {q}")
        for i in range(self.l):
            hash_code = self.generate_hash_code(q_values, i)
            buckets.append(hash_code)

        pq = PriorityQueue(True)

        # Iterate through the bucket list.  For each bucket entry, get all the points from that bucket
        if self.verbose:
            print(f"Retrieving nearest neighbors from buckets")
        for i in range(self.l):
            for item in self.tau[i][buckets[i]]:
                S.add(item)
                item_line = linecache.getline(self.file, item).strip()
                item_values = np.array(ast.literal_eval(item_line))
                manhat_distance = cityblock(q_values, item_values)
                if 0 < manhat_distance < self.DIST_THRESH:
                    # d[item] = manhat_distance
                    pq.add(manhat_distance, item)

        if self.verbose:
            print(f"Length of set S: {len(S)}, Number of neighbors that pass threshold: {self.DIST_THRESH}")
        result = set()
        while len(result) <= k:
            result.add(pq.poll())

        return result
