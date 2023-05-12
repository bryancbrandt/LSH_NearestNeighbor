# Similarity Search in High Dimensions via Hashing Implementation
# Derived from the paper by Gionis, Indyk, Motwani, 1999
# Bryan Brandt
# CS738 2023

import ast
import linecache
import numpy as np
import pickle
from datasketch import MinHash, MinHashLSH
from os.path import exists
from PriorityQueue import PriorityQueue
from scipy.spatial.distance import cityblock


class MinHash_Continuous:
    file = "" # File name to be loaded
    threshold = 0  # Jaccard similarity threshold.
    num_bands = 0  #
    num_rows = 0
    decimal_round = 0  # What degree to round the decimal values to
    lsh = None  # The location of the MinHashLSH object
    DIST_THRESH = 1000  # Distance threshold value
    verbose = True

    def __init__(self, file: str, threshold: int, num_bands: int = 10, num_rows: int = 4, decimal_round: int = 4,
                 verbose: bool = True):
        assert num_bands > 0, "Error number of bands must be greater than 0"
        assert num_rows > 0, "Error number of rows must be greater than 0"
        assert decimal_round > 0, "Error decimal round must be greater than 0"
        self.verbose = verbose
        self.file = file
        self.threshold = threshold
        self.num_bands = num_bands
        self.num_rows = num_rows
        self.decimal_round = decimal_round

        if self.verbose:
            print("================================================")
            print("MinHash Initializing")
            print("================================================")

        # Load the file, and initialize the MinHashLSH
        file_exists = exists(file)
        if not file_exists:
            raise IOError(f"{file} does not exist!")
        self.file = file

        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=128)

    def preprocess(self, save_pickle: bool = False):
        """
        Preprocess the data, creating min hash classes for each line, inserting them into MinHashLSH
        """

        if self.verbose:
            print(f"Loading data file {self.file}")
        line_index = 0
        with open(self.file, 'r') as f:
            for i in f:
                # Retrieve the data, and round the numbers to the self.decimal_round
                line_data = np.array(ast.literal_eval(i.strip()))
                line_data = np.round(line_data, self.decimal_round)
                line_index += 1
                # MinHash each line from the file and add it to hash function dict
                m = MinHash(num_perm=128)
                for j in range(line_data.shape[0]):
                    m.update(str(line_data[j]).encode('utf8'))
                self.lsh.insert(str(line_index), m)

        # Save the object as a pickle file for fast re-loads.
        if save_pickle:
            tau_filename = "pickle_files/" + self.file[5:-4] + "_minHash.obj"
            fileObj = open(tau_filename, 'wb')
            pickle.dump(self.lsh, fileObj)
            fileObj.close()
            if self.verbose:
                print(f"Saved to pickle file {tau_filename} successfully")

    def load_pickle(self, file: str):
        """
        Load the MinHashLSH object from the pickled file
        :param file: the filename the object is saved into
        """
        file_exists = exists(file)
        if not file_exists:
            raise IOError(f"Pickle file: {file} does not exist!")

        if self.verbose:
            print(f"Loading pickle file: {file}")

        fileObj = open(file, 'rb')
        self.lsh = pickle.load(fileObj)
        fileObj.close()

        if self.verbose:
            print("File loaded successfully")

    def query(self, q: int, k: int) -> set:
        """
        Query a point q, and find the k nearest neighbors of it
        :param q: the point to used to find the nearest neighbors of
        :param k: an integer value of how many nearest neighbors to find
        :return: a set of nearest neighbors containing a tuple of (distance, line index)
        """
        # Get the datapoint from the file, and round it to it's set values, and assign it to a MinHash Object
        q_line = linecache.getline(self.file, q).strip()
        q_values = np.array(ast.literal_eval(q_line))
        q_values = np.round(q_values, self.decimal_round)
        m_query = MinHash(num_perm=128)

        # For each dimension of point q, encode it into the MinhHash object as a utf8 string
        for i in range(q_values.shape[0]):
            m_query.update(str(q_values[i]).encode('utf8'))

        # Query the MinHashLSH using the q MinHash object
        result = self.lsh.query(m_query)

        # Creat ea priority queue
        pq = PriorityQueue(True)

        # For each item in the resulting MinHash query:
        # Calculate its manhattan distance to point q
        # Add it to the priority queue based on its distance value
        for item in result:
            item_value = linecache.getline(self.file, int(item)).strip()
            item_values = np.array(ast.literal_eval(item_value))
            manhat_distance = cityblock(q_values, item_values)
            if 0 < manhat_distance < self.DIST_THRESH:
                pq.add(manhat_distance, item)

        # Create a set for the results so any redundant points are excluded
        result = set()
        while len(result) <= k:
            result.add(pq.poll())

        # Return the K most results
        return result

