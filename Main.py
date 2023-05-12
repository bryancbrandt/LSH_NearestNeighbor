import ast
import linecache
import timeit

from scipy.spatial.distance import cityblock

import EvaluationFunctions
from BallTree import BallT
from Distances import distances
from LSH_Hyperplane import LSH_Hyperplane
from MinHash_Set import MinHash_Set
from SSHD_Hash import SSHD_Hash
from CosineLSH import CosineLSH
from MinHash import MinHash_Continuous
from KDTree import KD
import numpy as np
import statistics


def main():
    CSV_INDEX = 0
    PICKLE_IDX = 6
    K = 10

    query_points = [
        [12531, 31316, 107032, 139436, 35233, 91785, 122160, 9091, 12972, 7000],  # 60_270,000
        [1690, 1807, 12580, 7529, 18204, 9596, 5204, 17594, 11107, 7000],  # 64_20,000
        [47949, 46231, 93, 21288, 38130, 22383, 59572, 22222, 9607, 7000],  # 1024_60,000
        [260043, 772087, 634236, 329774, 298119, 768993, 471134, 461915, 564244, 7000]  # 100_1,000,000
    ]

    suffix = [
        "_sshd.obj",
        "_hyper.obj",
        "_cosine.obj",
        "_minHash.obj",
        "_ball.obj",
        "_tree.obj",
        "_minHashset.obj"
    ]

    files = [
        "data/60_270000_RandDist_8Clusters.csv",
        "data/64_20000_RandDist_8Clusters.csv",
        "data/1024_60000_RandDist_4Clusters.csv",
        "data/100_1000000_RandDist_4Clusters.csv"
    ]

    pickle_files = [
        "pickle_files/60_270000_RandDist_8Clusters" + suffix[PICKLE_IDX],
        "pickle_files/64_20000_RandDist_8Clusters" + suffix[PICKLE_IDX],
        "pickle_files/1024_60000_RandDist_4Clusters" + suffix[PICKLE_IDX],
        "pickle_files/100_1000000_RandDist_4Clusters" + suffix[PICKLE_IDX]
    ]

    file_name = files[CSV_INDEX]

    # sshd = SSHD_Hash(file_name, verbose=False)
    # sshd.load_csv_file()
    # sshd.preprocess(save_pickle=True)
    # sshd.load_pickle(pickle_files[CSV_INDEX])

    # hyperplane = LSH_Hyperplane(file_name, verbose=False)
    # hyperplane.load_csv_file()
    # hyperplane.preprocess(save_pickle=True)
    # hyperplane.load_pickle(pickle_files[CSV_INDEX])
    # hyperplane.query(7000, K)

    # cos = CosineLSH(file_name, verbose=False)
    # cos.load_csv_file()
    # cos.preprocess(save_pickle=True)
    # cos.load_pickle(pickle_files[CSV_INDEX])
    # cos.query(7000, K)

    # minhash = MinHash_Continuous(file_name, threshold=.001, verbose=False)
    # minhash.preprocess(save_pickle=True)
    # minhash.load_pickle(pickle_files[CSV_INDEX])
    # minhash.query(query_points[CSV_INDEX][i], K)

    minhash_set = MinHash_Set(file_name, threshold=.01, verbose=False)
    minhash_set.preprocess(save_pickle=True)
    # minhash_set.load_pickle(pickle_files[CSV_INDEX])
    # minhash_set.query(query_points[CSV_INDEX][i], K)

    # b_tree = BallT(file_name, verbose=False)
    # b_tree.preprocess(save_pickle=True)
    # b_tree.load_pickle(pickle_files[CSV_INDEX])
    # b_tree.query(query_points[CSV_INDEX][i], K)

    # kdtree = KD(file_name, verbose=False)
    # kdtree.preprocess(save_pickle=True)
    # kdtree.load_pickle(pickle_files[CSV_INDEX])
    # kdtree.query(query_points[CSV_INDEX][i], K)

    # dist = distances(file_name)
    # dist.process(query_points[CSV_INDEX][i])
    # print(dist.max_distance(7000))
    # Test the query points

    for i in range(len(query_points[CSV_INDEX])):
        tic = timeit.default_timer()
        result = minhash_set.query(query_points[CSV_INDEX][i], K)
        toc = timeit.default_timer()
        print(toc - tic)

        print("===")

        if i == 9:
            for j in result:
                print(get_manhattan(file_name, query_points[CSV_INDEX][i], j))


def get_manhattan(file_name: str, p: int, q: int):
    # Get the p_value
    p_value = linecache.getline(file_name, p).strip()
    p_values = np.array(ast.literal_eval(p_value))

    # Get the q_value
    q_value = linecache.getline(file_name, q).strip()
    q_values = np.array(ast.literal_eval(q_value))

    # Determine the distance
    manhat_distance = cityblock(q_values, p_values)

    return manhat_distance


if __name__ == "__main__":
    main()
