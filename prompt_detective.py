# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer


def permutation_test_means_cosine(group1: np.ndarray, group2: np.ndarray, n_permutations: int = 10_000,
                                  n_task_prompts: int = None, k_responses: int = None) -> float:
    """
        Perform a permutation test for the difference in means between two groups,
        using the cosine similarity between the mean vectors as the test statistic.
        This version allows for more general permutations of the responses across the two groups,
        while preserving the task prompt structure.

        Parameters:
        - group1, group2: numpy arrays of high-dimensional vectors representing the two groups.
          Each group should have the shape (n_task_prompts * k_responses, embedding_dim),
          where n_task_prompts is the number of task prompts, k_responses is the number of responses
          per task prompt, and embedding_dim is the dimensionality of the embeddings.
        - n_task_prompts: the number of task prompts.
        - k_responses: the number of responses per task prompt.
        - n_permutations: the number of permutations to perform.

        Returns:
        - p-value for the observed difference in means.
        """
    # Calculate the observed mean vectors
    observed_mean1 = np.mean(group1, axis=0)
    observed_mean2 = np.mean(group2, axis=0)

    # Calculate the observed cosine similarity between mean vectors
    observed_cosine_sim = np.dot(observed_mean1, observed_mean2) / (
        np.linalg.norm(observed_mean1) * np.linalg.norm(observed_mean2)
    )

    # Initialize a variable to count how many times we see a similarity
    # as extreme as the observed similarity
    count_extreme = 0

    for _ in range(n_permutations):
        # Create copies of the original groups
        new_group1 = group1.copy()
        new_group2 = group2.copy()

        # Perform permutations for each task prompt
        for i in range(n_task_prompts):
            # Concatenate the responses from both groups for this task prompt
            combined_responses = np.concatenate([new_group1[i*k_responses:(i+1)*k_responses],
                                                 new_group2[i*k_responses:(i+1)*k_responses]])

            # Randomly permute the combined responses
            np.random.shuffle(combined_responses)

            # Split the permuted responses into two parts
            new_group1[i*k_responses:(i+1)*k_responses] = combined_responses[:k_responses]
            new_group2[i*k_responses:(i+1)*k_responses] = combined_responses[k_responses:]

        # Calculate the new mean vectors
        new_mean1 = np.mean(new_group1, axis=0)
        new_mean2 = np.mean(new_group2, axis=0)

        # Calculate the new cosine similarity between mean vectors
        new_cosine_sim = np.dot(new_mean1, new_mean2) / (
            np.linalg.norm(new_mean1) * np.linalg.norm(new_mean2)
        )

        # Check if the new similarity is as extreme as the observed one
        if new_cosine_sim <= observed_cosine_sim:
            count_extreme += 1

    # Calculate the p-value
    p_value = count_extreme / n_permutations

    return p_value

class PromptDetective(object):
    def __init__(self, model_id: str = "bert-base-uncased", n_task_prompts: int = None, k_responses: int = None):
        """
        Initialize the PromptDetective instance.

        Parameters:
        - model_id: The identifier of the sentence transformer model to use.
        - n_task_prompts: The number of task prompts.
        - k_responses: The number of responses per task prompt.
        """

        self.n_task_prompts = n_task_prompts
        self.k_responses = k_responses
        self.model = SentenceTransformer(model_id)

    def compute_embeddings_transformers(self, max_length: int, input_text: List[str]) -> np.ndarray:
        """
        Compute the embeddings for a list of input texts using the sentence transformer model.

        Parameters:
        - max_length: The maximum length of the input texts.
        - input_text: A list of input texts.

        Returns:
        - A numpy array of embeddings for the input texts.
        """

        self.model.max_seq_length = max_length
        embeddings = self.model.encode(input_text)

        return embeddings

    def __call__(self, generations_group_1: List[str], generations_group_2: List[str], max_length: int = 512) -> float:
        """
        Compute the p-value for the difference between two groups of text generations.

        Parameters:
        - generations_group_1: A list of text generations for group 1.
        - generations_group_2: A list of text generations for group 2.
        - max_length: The maximum length of the input texts.

        Returns:
        - The p-value for the observed difference in means.
        """
        mean_embed_group_1 = self.compute_embeddings_transformers(max_length, generations_group_1)
        mean_embed_group_2 = self.compute_embeddings_transformers(max_length, generations_group_2)

        p_value = permutation_test_means_cosine(group1=mean_embed_group_1,
                                                group2=mean_embed_group_2,
                                                n_task_prompts=self.n_task_prompts,
                                                k_responses=self.k_responses)

        return p_value
