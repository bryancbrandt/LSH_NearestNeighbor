import EvaluationFunctions
from Distances import distances
from SSHD_Hash import SSHD_Hash


def main():
    # eval_sshdHash_mdc_csv("mdcgen/64_20000_RandDist_8Clusters.csv", True, 25, 7000, 5)
    EvaluationFunctions.evaluate_q_proximity_hyper("mdcgen/64_20000_RandDist_8Clusters.csv", 7000, 10)


if __name__ == "__main__":
    main()
