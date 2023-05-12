# Similarity Search in High Dimensions via Hashing Implementation
# Derived from the paper by Gionis, Indyk, Motwani, 1999
# Bryan Brandt
# CS738 2023

import ast
import linecache
import sys
from collections import defaultdict
import hashlib
import numpy as np
import pickle
from os.path import exists


def jaccard_similarity(S1: set, S2: set) -> int:
    intersect = len(S1.intersection(S2))
    union = len(S1.union(S2))
    return intersect / union


class MinHash_Set:
    file = ""  # File name to be loaded
    threshold = 0  # Jaccard similarity threshold.
    decimal_round = 0  # What degree to round the decimal values to
    lsh = defaultdict(list)  # A dictionary of sets.
    verbose = True
    tau = list(defaultdict(list))  # The set of hashtable
    l = 8  # The number of hash tables.  Note: Message digest is 64 bit, so increasing this will impact greatly

    def __init__(self, file: str, threshold: int = .01, decimal_round: int = 4, verbose: bool = True):
        assert threshold > 0 or threshold < 1, "Error in threshold value"
        assert decimal_round > 0, "Error decimal round must be greater than 0"
        self.verbose = verbose
        self.file = file
        self.threshold = threshold
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

        # Initialize the list of hash tables
        for i in range(self.l):
            self.tau.append(defaultdict(list))

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

                # For each hash table, get the hash code, and add it to the dictionary defined
                # at list[i].  The key will be the hash_code, the value will be the line_number
                # from the file.
                hash_codes = self.generate_hash_codes(line_data)
                for j in range(self.l):
                    self.tau[j][hash_codes[j]].append(line_index)

        # Save the object as a pickle file for fast re-loads.
        if save_pickle:
            tau_filename = "pickle_files/" + self.file[5:-4] + "_minHashset.obj"
            fileObj = open(tau_filename, 'wb')
            pickle.dump(self.tau, fileObj)
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
        self.tau = pickle.load(fileObj)
        fileObj.close()

        if self.verbose:
            print("File loaded successfully")

    def generate_hash_codes(self, arr: np.ndarray) -> list:
        min_table = []

        for h in range(self.l):
            min_table.append(sys.maxsize)

        for i, value in enumerate(arr):
            # Hash each word (dimensional number) in the document (array of values) using SHA256
            value = arr[i].tobytes()
            m = hashlib.sha256()
            m.update(value)
            digest = m.hexdigest()

            # Split those values into l segments and convert the hex values to integers
            segment_length = len(digest) // self.l  # Calculate the length of each segment
            segments = [int(digest[i:i + segment_length], 16) for i in range(0, len(digest), segment_length)]

            # Add the min value to the min table
            for j in range(self.l):
                if segments[j] < min_table[j]:
                    min_table[j] = segments[j]

        return min_table

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

        # Retrieve the hash_code of the q value
        q_hash_codes = self.generate_hash_codes(q_values)
        S = set()

        # For each hash code in the min table for q:
        for i, code in enumerate(q_hash_codes):
            # Get a list of the other line numbers in the same bucket for the given hashcode at index i
            values_in_bucket = self.tau[i][code]

            # We have to rehash the values for each value in the bucket to calculate the jaccard similarity
            # Unfortunately there was no other way I could do this without storing all the min hash tables
            # in memory.  This shouldn't increase computation time by much with low k values
            for value in values_in_bucket:
                # Get the value from the file and calculate the hash code
                bucket_value_line = linecache.getline(self.file, value).strip()
                bucket_value = np.array(ast.literal_eval(bucket_value_line))
                bucket_value = np.round(bucket_value, self.decimal_round)
                bucket_value_hash_code = self.generate_hash_codes(bucket_value)

                # If the jaccard similarity is greater than the threshold, then add it to the set.
                # If the set is full, return the set
                if jaccard_similarity(set(q_hash_codes), set(bucket_value_hash_code)) > self.threshold:
                    S.add(value)
                    if len(S) >= k:
                        return S

        return S

