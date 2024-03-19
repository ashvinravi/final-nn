# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import pandas as pd

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """

    sequences = pd.DataFrame(seqs)
    sequences['labels'] = labels

    sequences['seq'] = sequences[0]
    sequences = sequences.drop(columns=0)

    positive_sequences = sequences[sequences['labels'] == True]
    
    negative_sequences = sequences[sequences['labels'] == False]

    randomly_permuted_negative = negative_sequences.sample(n=len(positive_sequences), random_state=88)

    # randomly permute positive and negative sequences
    sampled_sequences = pd.concat([positive_sequences, randomly_permuted_negative]).sample(frac=1) 
    sampled_seqs = list(sampled_sequences['seq'])
    sampled_labels = list(sampled_sequences['labels'])

    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    nucleotide_dict = {'A': np.array([1, 0, 0, 0]), 
                       'T': np.array([0, 1, 0, 0]), 
                       'C': np.array([0, 0, 1, 0]), 
                       'G': np.array([0, 0, 0, 1])}
    
    encoded_seq = []

    for nucleotide in seq_arr:
        encoded = nucleotide_dict[nucleotide]
        encoded_seq.append(encoded)
    
    return np.array(encoded_seq).flatten()

