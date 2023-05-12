# Similarity Search in High Dimensions via Hashing Implementation
# Derived from the paper by Gionis, Indyk, Motwani, 1999
# Bryan Brandt
# CS738 2023

import ast
import linecache
import numpy as np
import pickle
from os.path import exists
from sklearn.neighbors import KDTree


class KD:
    file = ""
    tree = None
    leaf_size = 0
    DIST_THRESH = 1000000  # The distance threshold for l1 normalization
    verbose = True

    def __init__(self, file: str, leaf_size: int = 2, verbose: bool = True):
        assert leaf_size > 0, "Leaf size must be greater than 0"
        self.verbose = verbose
        self.file = file
        self.leaf_size = leaf_size

        if self.verbose:
            print("================================================")
            print("KDTree Initializing")
            print("================================================")

        file_exists = exists(file)
        if not file_exists:
            raise IOError(f"{file} does not exist!")
        self.file = file

    def preprocess(self, save_pickle: bool = False):
        """
        Preprocess the csv file and generate the KDTree
        """

        if self.verbose:
            print(f"Loading data file {self.file}")
        data = np.genfromtxt(self.file, delimiter=",", dtype=float)

        self.tree = KDTree(data, self.leaf_size, metric='cityblock')

        if save_pickle:
            tau_filename = "pickle_files/" + self.file[5:-4] + "_tree.obj"
            fileObj = open(tau_filename, 'wb')
            pickle.dump(self.tree, fileObj)
            fileObj.close()
            if self.verbose:
                print(f"Saved to pickle file {tau_filename} successfully")

    def load_pickle(self, file: str):
        """
        Load the KDTree from the pickle file and initialize the object
        :param file: the filename for the saved pickle object
        """
        file_exists = exists(file)
        if not file_exists:
            raise IOError(f"Pickle file: {file} does not exist!")

        if self.verbose:
            print(f"Loading pickle file: {file}")

        fileObj = open(file, 'rb')
        self.tree = pickle.load(fileObj)
        fileObj.close()

        if self.verbose:
            print("File loaded successfully")

    def query(self, q: int, k: int) -> tuple:
        """
        Query a point q for it's k nearest neighbors
        :param q: the point to be queried
        :param k: an integer representing the number of neighbors to find
        :return: A tuple of lists containing the distance of the k nearest points, and their index numbers
        """
        q_line = linecache.getline(self.file, q).strip()
        q_values = np.array(ast.literal_eval(q_line))
        q_values = q_values.reshape((1, -1))

        dist, index = self.tree.query(q_values, k)

        return dist
