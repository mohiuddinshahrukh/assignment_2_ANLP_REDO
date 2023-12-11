import numpy as np
import numpy.typing as npt

from typing import Tuple, List, Set


def bag_of_words_matrix(sentences: List[str]) -> npt.ArrayLike:
    """
    Convert the dataset into V x M matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE

    bag_of_words = {}
    for i, sentence in enumerate(sentences):
        split_sentence = sentence.split()
        for j, word in enumerate(split_sentence):
            if word in bag_of_words:
                bag_of_words[word]['f'] += 1
            else:
                bag_of_words[word] = {'f': 1, 'w': word}

    # assign each word an index and an index to each word
    for i, word in enumerate(bag_of_words):
        bag_of_words[word]['i'] = i

    for i, word in enumerate(bag_of_words):
        if bag_of_words[word]['f'] < 2:
            bag_of_words[word]['w'] = "<UNK>"
    # Matrix V x M, 1762 X 1000
    matrix = np.zeros((len(bag_of_words), len(sentences)))
    for j, sentence in enumerate(sentences):
        split_sentence = sentence.split()
        for word in split_sentence:
            if word in bag_of_words:
                row_index = bag_of_words[word]['i']
                matrix[row_index, j] += 1

    return matrix
    #########################################################################


def labels_matrix(data: Tuple[List[str], Set[str]]) -> npt.ArrayLike:
    """
    Convert the dataset into K x M matrix.
    k = number of classes
    m = number of examples
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE

    examples = data[0]
    classes = data[1]
    class_to_index = {class_label: i for i, class_label in enumerate(classes)}

    matrix = np.zeros((len(classes), len(examples)))

    # Populating the labels matrix
    for i, intent in enumerate(examples):
        # Assuming intent is the class label for the corresponding example
        class_label = intent
        if class_label in class_to_index:
            matrix[class_to_index[class_label], i] = 1

    return matrix
    #########################################################################


def softmax(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Softmax function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    #########################################################################


def relu(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Rectified Linear Unit function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    return np.maximum(0, z)
    #########################################################################


def relu_prime(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    First derivative of ReLU function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    return np.where(z > 0, 1, 0)
    #########################################################################