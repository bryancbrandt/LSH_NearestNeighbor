import EvaluationFunctions
from Distances import distances
from SSHD_Hash import SSHD_Hash
from CosineLSH import CosineLSH
import numpy as np


# TODO: Change the hyper planes to a normal distribution instead of random
def main():
    # eval_sshdHash_mdc_csv("mdcgen/64_20000_RandDist_8Clusters.csv", True, 25, 7000, 5)
    # EvaluationFunctions.evaluate_q_proximity_hyper("mdcgen/64_20000_RandDist_8Clusters.csv", 7000, 10)
    pass
    data = np.genfromtxt("mdcgen/64_20000_RandDist_8Clusters.csv", delimiter=',', dtype=float)
    cosine_lsh = CosineLSH(data, 32, 8)
    print(cosine_lsh.query(7000, 10))


if __name__ == "__main__":
    main()
