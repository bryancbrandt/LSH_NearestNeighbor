# Similarity Search in High Dimensions via Hashing Implementation
# Derived from the paper by Gionis, Indyk, Motwani, 1999
# Bryan Brandt
# CS7-- 2023
# Similarity Search in High Dimension File

import ast
import linecache
import numpy as np
import pickle
from collections import defaultdict
from os.path import exists
from PriorityQueue import PriorityQueue
from scipy.spatial.distance import cityblock


class SSHD_Hash:

    ALPHA = 2  # Memory utilization parameter
    m = 0  # Size of the hash table
    n = 0  # Number of data points
    B = 0  # Maximum bucket size, calculated by block size / dimensions
    d = 0  # Number of dimensions
    l = 32  # Number of hash tables
    min = 0  # Minimum value of the data
    max = 0  # Maximum value of the data
    file = ""  # The csv file to be loaded
    num_dim_hash = 24  # Bit depth of the hash values
    tau = list(defaultdict(list))  # The set of hashtable
    l_subset_dimensions = []  # Dimensional subsets for I1,...,Il
    l_subset_a = []  # A values for h(x) hash calculation
    DIST_THRESH = 5  # The distance threshold for l1 normalization

    def __init__(self, file: str):
        print("================================================")
        print("Similarity Search in High Dimension Initializing")
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

        # Generate the dimensional subsets of {1...d'} for each l
        print("Generating dimensional subsets for l...")
        for i in range(self.l):
            dimensions = rng.integers(low=0, high=self.d, size=self.num_dim_hash)
            while len(np.unique(dimensions)) != self.num_dim_hash:
                dimensions = rng.integers(low=0, high=self.d, size=self.num_dim_hash)
            self.l_subset_dimensions.append(dimensions)

        # Generate the {a1...ak} values for hash value calculation
        print("Generating a values for each hash table function...")
        for i in range(self.l):
            a_set = rng.integers(low=0, high=self.m - 1, size=self.num_dim_hash)
            self.l_subset_a.append(a_set)

        # Open the file, so that we can begin adding the points to the hash tables
        # the points identifier will be the row number in the file, beginning with 1
        print("Hashing datapoints from the file into the hashtable...")
        line_index = 0
        with open(self.file, 'r') as f:
            for i in f:
                line_data = np.array(ast.literal_eval(i.strip()))
                line_index += 1
                # Iterate through each hash function using the line from the file
                for j in range(self.l):
                    hash_code = self.generate_hash_code(line_data, j)
                    self.tau[j][hash_code].append(line_index)

        # Save the self.tau dict, self.l_subset_dimension, and self.l_subset_a to a file
        if save_pickle:
            tau_filename = "pickle_files/" + self.file[7:-4] + "_sshd.obj"
            fileObj = open(tau_filename, 'wb')
            save_object = {"tau": self.tau, "l_subset_dimension": self.l_subset_dimensions,
                           "l_subset_a": self.l_subset_a, "m": self.m}
            pickle.dump(save_object, fileObj)
            fileObj.close()
            print(f"Tau saved to file {tau_filename} successfully")

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
        self.l_subset_dimensions = load_object["l_subset_dimension"]
        self.l_subset_a = load_object["l_subset_a"]
        self.m = load_object["m"]
        fileObj.close()

        print("File loaded successfully")

    def query(self, q: int, k: int):
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

        pq = PriorityQueue(True)

        # Iterate through the bucket list.  For each bucket entry, get all the points from that bucket
        # print(f"Retrieving nearest neighbors from buckets")
        for i in range(self.l):
            for item in self.tau[i][buckets[i]]:
                S.add(item)
                item_line = linecache.getline(self.file, item).strip()
                item_values = np.array(ast.literal_eval(item_line))
                manhat_distance = cityblock(q_values, item_values)
                if 0 < manhat_distance < self.DIST_THRESH:
                    #  d[item] = manhat_distance
                    pq.add(manhat_distance, item)

        # print(f"Length of set S: {len(S)}, Number of neighbors that pass threshold: {len(d)}")
        # k_items = heapq.nsmallest(k, d.items(), key=lambda item: item[1])
        result = set()
        while len(result) < k:
            result.add(pq.poll())

        return result  # k_items

    def generate_hash_code(self, array: np.ndarray, index: int):
        """
        Generate the has code for a bucket
        array : 1 x d numpy array (a row from the data file with all dimensional data included)
        index : the index number for selection of hash function and dimension selection
        """
        dim_reduced_data = np.array(array[self.l_subset_dimensions[index]]) * self.l_subset_a[index]
        hash_code = int(dim_reduced_data.sum() % self.m)
        return hash_code
