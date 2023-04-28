import ast
import linecache
import numpy as np
import pickle
from os.path import exists
from scipy.spatial import KDTree


class KD:
    file = ""
    tree = None
    leaf_size = 0
    DIST_THRESH = 5  # The distance threshold for l1 normalization

    def __init__(self, file: str, leaf_size: int = 2):
        assert leaf_size > 0, "Leaf size must be greater than 0"
        self.file = file
        self.leaf_size = leaf_size

        print("================================================")
        print("KDTree Initializing")
        print("================================================")

        file_exists = exists(file)
        if not file_exists:
            raise IOError(f"{file} does not exist!")
        self.file = file

    def preprocess(self, save_pickle: bool = False):
        """

        :return:
        """

        print(f"Loading data file {self.file}")
        data = np.genfromtxt(self.file, delimiter=",", dtype=float)

        self.tree = KDTree(data, self.leaf_size)

        if save_pickle:
            tau_filename = "pickle_files/" + self.file[7:-4] + "_tree.obj"
            fileObj = open(tau_filename, 'wb')
            pickle.dump(self.tree, fileObj)
            fileObj.close()
            print(f"Saved to pickle file {tau_filename} successfully")

    def load_pickle(self, file: str):
        """

        :param file:
        :return:
        """
        file_exists = exists(file)
        if not file_exists:
            raise IOError(f"Pickle file: {file} does not exist!")

        print(f"Loading pickle file: {file}")

        fileObj = open(file, 'rb')
        self.tree = pickle.load(fileObj)
        fileObj.close()

        print("File loaded successfully")

    def query(self, q: int, k: int) -> list:
        """

        :param q:
        :param k:
        :return:
        """
        q_line = linecache.getline(self.file, q).strip()
        q_values = np.array(ast.literal_eval(q_line))


        dist, index = self.tree.query(q_values, k)

        return dist, index
