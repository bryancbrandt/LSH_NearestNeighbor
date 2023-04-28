import ast
import linecache
import numpy as np
import pickle
from datasketch import MinHash, MinHashLSH
from os.path import exists
from PriorityQueue import PriorityQueue
from scipy.spatial.distance import cityblock


class MinHash_Continuous:
    file = ""
    threshold = 0
    num_bands = 0
    num_rows = 0
    decimal_round = 0
    lsh = None
    DIST_THRESH = 5

    def __init__(self, file: str, threshold: int, num_bands: int = 10, num_rows: int = 4, decimal_round: int = 4):
        self.file = file
        self.threshold = threshold
        self.num_bands = num_bands
        self.num_rows = num_rows
        self.decimal_round = decimal_round

        print("================================================")
        print("MinHash Initializing")
        print("================================================")

        file_exists = exists(file)
        if not file_exists:
            raise IOError(f"{file} does not exist!")
        self.file = file

        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=128)

    def preprocess(self, save_pickle: bool = False):
        """

        :return:
        """

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
                self.lsh.insert(line_index, m)

        if save_pickle:
            tau_filename = "pickle_files/" + self.file[7:-4] + "_minHash.obj"
            fileObj = open(tau_filename, 'wb')
            pickle.dump(self.lsh, fileObj)
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
        self.lsh = pickle.load(fileObj)
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
        q_values = np.round(q_values, self.decimal_round)
        m_query = MinHash(num_perm=128)

        for i in range(q_values.shape[0]):
            m_query.update(str(q_values[i]).encode('utf8'))

        result = self.lsh.query(m_query)

        pq = PriorityQueue(True)

        for item in result:
            item_value = linecache.getline(self.file, item).strip()
            item_values = np.array(ast.literal_eval(item_value))
            manhat_distance = cityblock(q_values, item_values)
            if 0 < manhat_distance < self.DIST_THRESH:
                pq.add(manhat_distance, item)

        result = set()
        while len(result) <= k:
            result.add(pq.poll())
        return result

