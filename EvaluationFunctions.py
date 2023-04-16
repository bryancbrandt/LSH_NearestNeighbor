# Similarity Search in High Dimensions via Hashing Implementation
# Derived from the paper by Gionis, Indyk, Motwani, 1999
# Bryan Brandt
# CS7-- 2023
# File for evaluation of algorithms
from Distances import distances
from LSH_Hyperplane import LSH_Hyperplane
from os.path import exists
from SSHD_Hash import SSHD_Hash
import timeit


def eval_sshdHash_mdc_csv(file: str = "", from_pickle: bool = False, num_iterations: int = 25, q: int = 1, k: int = 10):
    """
    Evaluation function for the SSHD class that uses MDCGen Cluster data stored in /mdcgen folder
    :param file:
    :param from_pickle:
    :param num_iterations:
    :param q: the point to be queried to search for the ann (must be >= 1)
    :param k: the number of approximate nearest neighbors to return (must be >=1)
    :return:
    """
    assert num_iterations > 0, "num_iterations must be an integer greater than 0"
    assert q >= 1, "Q must be a point number greater than or equal to 1"
    assert k >= 1, "k must be greater than or equal to 1"

    file_exists = exists(file)
    if not file_exists:
        raise IOError(f"{file} does not exist!")

    sshdHash = SSHD_Hash(file)

    if from_pickle:
        pickle_filename = "pickle_files/" + file[7:-4] + "_sshd.obj"
        sshdHash.load_pickle(pickle_filename)
    else:
        sshdHash.load_csv_file()
        sshdHash.preprocess(save_pickle=True)

    print()

    for i in range(num_iterations):
        tic = timeit.default_timer()
        sshdHash.ann_query(q, k)
        toc = timeit.default_timer()
        print(toc - tic)


def eval_lshHyperplane_mdc_csv(file: str = "", from_pickle: bool = False, num_iterations: int = 25, bit_depth: int = 8,
                               q: int = 1, k: int = 10):
    """
    Evaluation function for the LSH Hyperplane class that uses the MDCGen Cluster data stored in /mdcgen folder
    :param file: The filename of the .csv file used for processing
    :param from_pickle: If true, load the tau table from the passed pickle file, otherwise preprocess the file
    :param num_iterations: The number of iterations to execute ann.
    :param bit_depth: The number of hyperplanes to generate per hash table.  What the bit depth of the binary value is
    :param q: the point to be queried to search for the ann (must be >= 1)
    :param k: the number of approximate nearest neighbors to return (must be >=1)
    """
    assert num_iterations > 0, "num_iterations must be an integer greater than 0"
    assert bit_depth >= 2, "Bith depth must be greater than or equal to 2"
    assert q >= 1, "Q must be a point number greater than or equal to 1"
    assert k >= 1, "k must be greater than or equal to 1"

    file_exists = exists(file)
    if not file_exists:
        raise IOError(f"{file} does not exist!")

    lshHyperplane = LSH_Hyperplane(file, bit_depth)

    if from_pickle:
        pickle_filename = "pickle_files/" + file[7:-4] + "_hyper.obj"
        lshHyperplane.load_pickle(pickle_filename)
    else:
        lshHyperplane.load_csv_file()
        lshHyperplane.preprocess(True)

    print()

    for i in range(100):
        tic = timeit.default_timer()
        lshHyperplane.ann_query(q, k)
        toc = timeit.default_timer()
        print(toc - tic)


def evaluate_q_proximity_sshd(file: str = "", q: int = 1, k: int = 10):
    """
    Takes the point q and finds the k approximate nearest neighbors using sshd.  It then checks these k neighbors
    against a sorted distance measure that comprises all points sorted from the shortest distance to the
    longest distance from q.
    :param file: The filename of the .csv file used for processing
    :param q: the point to be queried to search for the ann (must be >= 1)
    :param k: the number of approximate nearest neighbors to return (must be >=1)
    :return: null
    """

    assert q > 0, "Q value must be greater than 0"
    assert k > 0, "K value must be greater than 0"
    file_exists = exists(file)
    if not file_exists:
        raise IOError(f"{file} does not exist!")

    sshdHash = SSHD_Hash(file)
    pickle_filename = "pickle_files/" + file[7:-4] + "_sshd.obj"
    sshdHash.load_pickle(pickle_filename)
    dist_list = sshdHash.ann_query(q, k)
    dist = distances(file)
    dist.process(q)
    dist.compare(dist_list)


def evaluate_q_proximity_hyper(file: str = "", q: int = 1, k: int = 10):
    """
    Evaluates the order of the k approximate nearest neighbors returned from the LSH hyperplane algorithm against
    a sorted list of all distances from smallest to largest to point q.
    :param file: The filename of the .csv file used for processing
    :param q: the point to be queried to search for the ann (must be >= 1)
    :param k: the number of approximate nearest neighbors to return (must be >=1)
    :return: null
    """

    assert q > 0, "Q value must be greater than 0"
    assert k > 0, "K value must be greater than 0"
    file_exists = exists(file)
    if not file_exists:
        raise IOError(f"{file} does not exist!")

    hyper = lshHyperplane = LSH_Hyperplane(file, 8)
    pickle_filename = "pickle_files/" + file[7:-4] + "_hyper.obj"
    hyper.load_pickle(pickle_filename)
    dist_list = hyper.ann_query(q, k)
    dist = distances(file)
    dist.process(q)
    dist.compare(dist_list)