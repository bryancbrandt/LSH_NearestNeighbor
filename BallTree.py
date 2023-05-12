# Similarity Search in High Dimensions via Hashing Implementation
# Derived from the paper by Gionis, Indyk, Motwani, 1999
# Bryan Brandt
# CS738 2023

import ast
import linecache
import numpy as np
import pickle
from os.path import exists
from sklearn.neighbors import BallTree


class BallT:
    file = ""
    tree = None
    leaf_size = 0
    DIST_THRESH = 100000 # The distance threshold for l1 normalization
    verbose = True

    def __init__(self, file: str, leaf_size: int = 2, verbose: bool = True):
        assert leaf_size > 0, "Leaf size must be greater than 0"
        self.verbose = verbose
        self.file = file
        self.leaf_size = leaf_size

        if self.verbose:
            print("================================================")
            print("BallTree Initializing")
            print("================================================")

        file_exists = exists(file)
        if not file_exists:
            raise IOError(f"{file} does not exist!")
        self.file = file

    def preprocess(self, save_pickle: bool = False):
        """
        Preprocess the data, and generate the BallTree.  If the information needs to be saved
        then save the information
        """

        if self.verbose:
            print(f"Loading data file {self.file}")
        data = np.genfromtxt(self.file, delimiter=",", dtype=float)

        self.tree = BallTree(data, self.leaf_size, metric='cityblock')

        if save_pickle:
            tau_filename = "pickle_files/" + self.file[5:-4] + "_ball.obj"
            fileObj = open(tau_filename, 'wb')
            pickle.dump(self.tree, fileObj)
            fileObj.close()
            if self.verbose:
                print(f"Saved to pickle file {tau_filename} successfully")

    def load_pickle(self, file: str):
        """
        Load a pickle file as a Ball Tree
        :param file:
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
        Query a point q, and find the k nearest neighbors
        :param q: the point to find the nearest neighbors for
        :param k: how many of the nearest neighbors should be returned
        :return: A tuple of tables containing the distance of each k neighbors and their indexes
        """
        q_line = linecache.getline(self.file, q).strip()
        q_values = np.array(ast.literal_eval(q_line))
        q_values = q_values.reshape((1, -1))

        dist, index = self.tree.query(q_values, k)

        return dist
