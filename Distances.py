import ast
import linecache
import numpy as np
from os.path import exists
from PriorityQueue import PriorityQueue
from scipy.spatial.distance import cityblock


class distances():
    """
    Clas used for validating the distances to a point using l1 normalization.
    """
    file = ""
    pq = PriorityQueue(True)

    def __init__(self, file: str):
        file_exists = exists(file)
        if not file_exists:
            raise IOError(f"{file} does not exist!")
        self.file = file

    def process(self, q: int):
        """
        Process the self.file.  For each data point within the file, calculate the l1 norm with q and add it
        to the self.pq priority queue
        :param q: integer value of the point to be searched.
        :return: null
        """
        assert q > 0, "Q must be greater than 0"

        q_line = linecache.getline(self.file, q).strip()
        q_values = np.array(ast.literal_eval(q_line))

        n_size = 0
        with open(self.file, 'r') as f:
            for i in f:
                line_data = np.array(ast.literal_eval(i.strip()))
                n_size += 1
                manhat_distance = cityblock(q_values, line_data)
                self.pq.add(manhat_distance, n_size)

    def compare(self, data: set):
        """
        Polls the priority queue to determine if the polled item exists with the passed set data.  If a match
        is found, it returns the position of the polled item, which is a measure of proximity to the q
        value passed to process
        :param data: a set consisting of tuples with values for (data point number, l1 norm distance)
        :return: null
        """
        count = 0
        position = 0
        while count < len(data):
            popped = self.pq.poll()
            position += 1
            if popped in data:
                count += 1
                print(f"{popped} found at position {position}")
