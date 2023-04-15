import ast
import heapq
import linecache
import pickle

import numpy as np
from collections import defaultdict
from os.path import exists

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
    DIST_THRESH = 5  # The distance threshold for l1 normalization
    hyperplanes = []  # The list of all np array hyperplanes
    hash_bit_size = 0  # The bit-size of the hash table code

    def __init__(self, file: str, hash_bit_size: int):

        print("================================================")
        print("LSH Hyperplane Initializing")
        print("================================================")

        file_exists = exists(file)
        if not file_exists:
            raise IOError(f"{file} does not exist!")
        self.file = file

        self.hash_bit_size = hash_bit_size

        # Initialize the hash tables
        print("Initializing hash tables...")
        for i in range(self.l):
            self.tau.append(defaultdict(list))

    def load_csv_file(self):
        """
               Loads the csv file, and sets all the class attributes

               :return:
               """

        # Load the Data
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
        print(f"Max value: {self.max}")

        # Get the dimension of the data
        self.d = line_data.shape[0]
        print(f"Dimensionality of data: {self.d}")

        # Get the number of data points
        self.n = n_size
        print(f"Number of data points {self.n}")

        # Calculate M (First calculate B)
        self.B = int(8192 / self.d)
        self.m = int(self.ALPHA * (self.n / self.B))
        print(f"Bucket size: {self.B}")
        print(f"M value: {self.m}")

    def preprocess(self, save_pickle: bool):
        """
        Pre-process the Data.  This is the Preprocessing Algorithm from the paper
        :param save_pickle: True if we want to save the self.tau dict to a file
        :return: nothing
        """
        print()
        print("================================================")
        print("Preprocessing...")
        print("================================================")

        rng = np.random.default_rng()

        # Create a matrix of random values of [0, 1) size l x d and normalize them to [-1, 1)
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
            tau_filename = self.file[7:-4] + "_hyper.obj"
            fileObj = open(tau_filename, 'wb')
            save_object = {"tau": self.tau, "d": self.d, "m": self.m, "hyperplanes": self.hyperplanes}
            pickle.dump(save_object, fileObj)
            fileObj.close()
            print(f"Saved to pickle file {tau_filename} successfully")

    def load_pickle(self, file: str):
        """
        Loads a formerly saved tau dictionary from disk, and sets it as the self.tau attribute
        :param file: the filename to be used
        :return: null
        """
        file_exists = exists(file)
        if not file_exists:
            raise IOError(f"Pickle file: {file} does not exist!")

        print(f"Loading pickle file: {file}")

        fileObj = open(file, 'rb')
        load_object = pickle.load(fileObj)
        self.tau = load_object["tau"]
        self.d = load_object["d"]
        self.m = load_object["m"]
        self.hyperplanes = load_object["hyperplanes"]
        fileObj.close()

        print("File loaded successfully")

    def generate_hash_code(self, array: np.ndarray, index: int) -> int:
        assert isinstance(array, np.ndarray) or array.shape[0] != self.d, "Must be ndarray with same dimensions!"
        product = []
        for planes in self.hyperplanes[index]:
            product.append(np.dot(planes, array))
        bits = [1 if x > 0 else 0 for x in product]
        # hash_code = int("".join(list(map(str, bits))))
        integer = sum([j * (2 ** i) for i, j in list(enumerate(reversed(bits)))])
        integer = integer % self.m
        return integer

    def ann_query(self, q: int, k: int) -> list:
        """
        Approximate Nearest Neighbor Query.

        :param q: The line number from the file representing the data point to be queried
        :param k: The number of nearest neighbors to return
        :return: List of tuples formatted: (distance, point number)
        """
        assert q > 0, "Value of q must be greater than 0."
        assert k >= 1, "Value of k must be greater or equal to 1."

        # print()
        # print("================================================")
        # print(f"Querying value {q}")
        # print("================================================")

        S = set()
        buckets = []
        d = {}

        # Extract the dimensional values of q
        q_line = linecache.getline(self.file, q).strip()
        q_values = np.array(ast.literal_eval(q_line))

        # Hash the q_values to get all the bucket numbers and add them to buckets list
        # print(f"Getting hash values for {q}")
        for i in range(self.l):
            hash_code = self.generate_hash_code(q_values, i)
            buckets.append(hash_code)

        # Iterate through the bucket list.  For each bucket entry, get all the points from that bucket
        # print(f"Retrieving nearest neighbors from buckets")
        for i in range(self.l):
            for item in self.tau[i][buckets[i]]:
                S.add(item)
                item_line = linecache.getline(self.file, item).strip()
                item_values = np.array(ast.literal_eval(item_line))
                manhat_distance = cityblock(q_values, item_values)
                if 0 < manhat_distance < self.DIST_THRESH:
                    d[item] = manhat_distance

        # print(f"Length of set S: {len(S)}, Number of neighbors that pass threshold: {len(d)}")
        k_items = heapq.nsmallest(k, d.items(), key=lambda item: item[1])
        return k_items