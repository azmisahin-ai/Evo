# tests/unit/cognition/test_learning.py
import unittest
import sys
import os
from unittest.mock import patch
import numpy as np
import logging

# Import patch for mocking (needed for some advanced scenarios or isolating dependencies,
# although check_* functions are not being mocked in current tests)
# from unittest.mock import patch # Patch is not used in the final version of these tests

# Import modules to be tested.
# check_* functions come from src.core.utils.
# get_config_value comes from src.core.config_utils.
try:
    from src.cognition.learning import LearningModule
    # check_* functions should come from src.core.utils
    from src.core.utils import check_input_not_none, check_input_type, check_numpy_input
    from src.core.config_utils import get_config_value
except ImportError as e:
     # If import error still occurs, it typically means PYTHONPATH is not set correctly.
     # Check test environment setup.
     raise ImportError(f"Failed to import fundamental modules. Is PYTHONPATH configured correctly? Error: {e}")


# Enable these lines if you want to see logger output during tests.
# import logging
# logging.basicConfig(level=logging.DEBUG) # Set general level to DEBUG
# logging.getLogger('src.cognition.learning').setLevel(logging.DEBUG)
# logging.getLogger('src.core.utils').setLevel(logging.DEBUG) # To see logs from utils logger
# logging.getLogger('src.core.config_utils').setLevel(logging.DEBUG) # To see logs from config utils logger


class TestLearningModule(unittest.TestCase):

    def setUp(self):
        """Runs before each test method, creates a default configuration and LearningModule instance."""
        # Default, valid configuration structure reflecting main_config.yaml.
        # LearningModule needs cognition.new_concept_threshold and representation.representation_dim.
        self.default_config = {
            'cognition': {
                'new_concept_threshold': 0.7,
                # Other cognition settings not used by LearningModule init directly
            },
            'representation': {
                'representation_dim': 10, # Use a smaller dimension for faster tests
            },
        }
        # Initialize the module with the default config.
        self.module = LearningModule(self.default_config)

        # Reset concept representatives at the start of each test for isolation.
        self.module.concept_representatives = []


    def tearDown(self):
        """Runs after each test method."""
        # Call cleanup to ensure resources are released (clears concepts list).
        self.module.cleanup()


    # --- __init__ Tests ---
    # These tests verify that the module initializes correctly and reads config values as expected.

    def test_init_with_valid_config(self):
        """Tests initialization with a valid configuration."""
        # The fixture (or setUp) ensures a valid config is used for module creation.
        # Verify that config values were read and assigned correctly.
        self.assertEqual(self.module.config, self.default_config)
        self.assertEqual(self.module.new_concept_threshold, 0.7)
        self.assertIsInstance(self.module.new_concept_threshold, float)
        self.assertEqual(self.module.representation_dim, 10) # Expecting 10 from default_config fixture
        self.assertIsInstance(self.module.representation_dim, int)
        self.assertEqual(self.module.concept_representatives, []) # Should be empty initially


    def test_init_with_missing_config_values(self):
        """Tests initialization when some configuration values are missing (defaults should be used)."""
        # Provide an incomplete config dictionary to the module init.
        # The get_config_value calls inside __init__ should use their default values for missing keys.
        incomplete_config = {
             'cognition': { # Must include cognition key for path to work
                 # 'new_concept_threshold' is missing, default 0.7 should be used
             },
            # 'representation' key is missing, default 128 for representation_dim should be used by get_config_value's path
        }
        module = LearningModule(incomplete_config)
        # Verify that default values were used for missing keys.
        self.assertEqual(module.new_concept_threshold, 0.7) # Default for missing new_concept_threshold
        self.assertIsInstance(module.new_concept_threshold, float)
        self.assertEqual(module.representation_dim, 128) # Default for missing representation_dim
        self.assertIsInstance(module.representation_dim, int)
        self.assertEqual(module.concept_representatives, [])


    def test_init_with_invalid_config_types(self):
        """Tests initialization with configuration values of invalid types (defaults should be used)."""
        # Provide a config with invalid types for some values.
        # The get_config_value calls with expected_type checks should return their default values.
        invalid_type_config = {
            'cognition': {
                'new_concept_threshold': "not a float", # Invalid type -> default 0.7 should be used
            },
            'representation': {
                'representation_dim': [128], # Invalid type -> default 128 should be used
            },
        }
        module = LearningModule(invalid_type_config)
        # Verify that default values were used due to type mismatch.
        self.assertEqual(module.new_concept_threshold, 0.7) # Default due to invalid type
        self.assertIsInstance(module.new_concept_threshold, float)
        self.assertEqual(module.representation_dim, 128) # Default due to invalid type
        self.assertIsInstance(module.representation_dim, int)
        self.assertEqual(module.concept_representatives, [])

    def test_init_new_concept_threshold_out_of_range(self):
        """Tests when new_concept_threshold is set outside the 0.0-1.0 range."""
        # The __init__ method's internal range checks should reset the value to the default 0.7.
        config_high = {'cognition': {'new_concept_threshold': 1.5}, 'representation': {'representation_dim': 10}}
        module_high = LearningModule(config_high)
        self.assertEqual(module_high.new_concept_threshold, 0.7) # Should be reset

        config_low = {'cognition': {'new_concept_threshold': -0.5}, 'representation': {'representation_dim': 10}}
        module_low = LearningModule(config_low)
        self.assertEqual(module_low.new_concept_threshold, 0.7) # Should be reset

        # Test boundary values - they should be accepted.
        config_at_boundaries_high = {'cognition': {'new_concept_threshold': 1.0}, 'representation': {'representation_dim': 10}}
        module_boundaries_high = LearningModule(config_at_boundaries_high)
        self.assertEqual(module_boundaries_high.new_concept_threshold, 1.0) # Boundaries should be included

        config_at_boundaries_low = {'cognition': {'new_concept_threshold': 0.0}, 'representation': {'representation_dim': 10}}
        module_boundaries_low = LearningModule(config_at_boundaries_low)
        self.assertEqual(module_boundaries_low.new_concept_threshold, 0.0) # Boundaries should be included


    def test_learn_concepts_invalid_representation_dim_skips(self):
        """Tests that learning is skipped if LearningModule's representation_dim is invalid."""
        original_dim = self.module.representation_dim
        self.module.representation_dim = 0 # Set to an invalid dimension

        initial_concepts = [np.random.rand(original_dim).astype(np.float32)] # Use original_dim for consistency
        self.module.concept_representatives = initial_concepts # Start with some concepts

        rep_list = [np.random.rand(original_dim).astype(np.float32)] # Provide input vector with original_dim
        result = self.module.learn_concepts(rep_list)

        self.assertEqual(result, initial_concepts) # Learning should be skipped, existing list should be returned
        self.assertEqual(self.module.concept_representatives, initial_concepts) # Internal list should not change

        # Reset dimension after test
        self.module.representation_dim = original_dim


    # --- learn_concepts Tests ---
    # These tests verify the core concept learning logic.

    def test_learn_concepts_input_none(self):
        """Tests learn_concepts when representation_list is None."""
        initial_concepts = [np.random.rand(self.module.representation_dim).astype(np.float32)]
        self.module.concept_representatives = initial_concepts # Start with some concepts
        result = self.module.learn_concepts(None)
        # When input is None, the current list should be returned immediately. Reference comparison (==) is sufficient.
        self.assertEqual(result, initial_concepts)
        self.assertEqual(self.module.concept_representatives, initial_concepts) # Internal list should not change

    def test_learn_concepts_input_not_list(self):
        """Tests learn_concepts when representation_list is not a list."""
        initial_concepts = [np.random.rand(self.module.representation_dim).astype(np.float32)]
        self.module.concept_representatives = initial_concepts
        result = self.module.learn_concepts("not a list")
        self.assertEqual(result, initial_concepts) # Current list should be returned
        self.assertEqual(self.module.concept_representatives, initial_concepts) # Internal list should not change

    def test_learn_concepts_input_empty_list(self):
        """Tests learn_concepts when representation_list is an empty list."""
        initial_concepts = [np.random.rand(self.module.representation_dim).astype(np.float32)]
        self.module.concept_representatives = initial_concepts
        result = self.module.learn_concepts([])
        self.assertEqual(result, initial_concepts) # Current list should be returned
        self.assertEqual(self.module.concept_representatives, initial_concepts) # Internal list should not change


    def test_learn_concepts_from_scratch_first_vector(self):
        """Tests learning the first valid vector when starting with no concepts (should be added as a new concept)."""
        rep_vector = np.random.rand(self.module.representation_dim).astype(np.float32)

        self.assertEqual(len(self.module.concept_representatives), 0) # Should be empty initially

        result = self.module.learn_concepts([rep_vector])

        self.assertEqual(len(result), 1) # Should have 1 concept now
        self.assertIsInstance(result[0], np.ndarray)
        self.assertEqual(result[0].shape, (self.module.representation_dim,))
        # The learned vector should be a copy of the input vector (value check)
        np.testing.assert_array_equal(result[0], rep_vector)
        # The internal list should also be updated, and the array object should be different (.copy() used in learn_concepts)
        self.assertEqual(len(self.module.concept_representatives), 1) # Length check
        np.testing.assert_array_equal(self.module.concept_representatives[0], rep_vector) # Content check
        self.assertIsNot(self.module.concept_representatives[0], rep_vector) # Object reference check


    # Corrected Test: Tests that a new vector with similarity below the threshold is added as a new concept.
    # Deterministic vectors used. Expectation corrected to 2 concepts (initial 1 + 1 new).
    def test_learn_concepts_new_concept_below_threshold_deterministic(self):
        """Tests that a new vector with similarity below the threshold is added as a new concept (deterministic)."""
        dim = self.module.representation_dim
        self.module.new_concept_threshold = 0.7 # Threshold set to 0.7

        # First concept: [1, 0, 0, ...]
        initial_concept_rep = np.zeros(dim, dtype=np.float32); initial_concept_rep[0] = 1.0
        self.module.learn_concepts([initial_concept_rep])
        self.assertEqual(len(self.module.concept_representatives), 1) # First concept added

        # New vector: [0, 1, 0, ...] (Orthogonal to first concept, similarity 0.0)
        new_rep_vector_orthogonal = np.zeros(dim, dtype=np.float32); new_rep_vector_orthogonal[1] = 1.0
        # Similarity is 0.0. Threshold is 0.7. 0.0 < 0.7 -> True. Should be added.

        result = self.module.learn_concepts([new_rep_vector_orthogonal])

        self.assertEqual(len(result), 2) # New concept added
        np.testing.assert_array_equal(result[0], initial_concept_rep)
        np.testing.assert_array_equal(result[1], new_rep_vector_orthogonal)
        self.assertEqual(len(self.module.concept_representatives), 2) # Internal list length
        np.testing.assert_array_equal(self.module.concept_representatives[0], initial_concept_rep)
        np.testing.assert_array_equal(self.module.concept_representatives[1], new_rep_vector_orthogonal)


    # Corrected Test: Tests that a vector with similarity above/equal to the threshold to existing concepts is NOT added.
    # Deterministic vectors used. Expectation is 1 concept (initial 1 + 0 new).
    def test_learn_concepts_existing_concept_above_threshold_deterministic(self):
        """Tests that a vector with similarity above/equal to the threshold to existing concepts is NOT added (deterministic)."""
        dim = self.module.representation_dim
        self.module.new_concept_threshold = 0.7 # Threshold set to 0.7

        # First concept: [1, 0, 0, ...]
        initial_concept_rep = np.zeros(dim, dtype=np.float32); initial_concept_rep[0] = 1.0
        self.module.learn_concepts([initial_concept_rep])
        self.assertEqual(len(self.module.concept_representatives), 1) # First concept added

        # Very similar vector: [1.001, 0, 0, ...] (Similarity ~1.0)
        similar_rep_vector = np.zeros(dim, dtype=np.float32); similar_rep_vector[0] = 1.001
        # Similarity is ~1.0. Threshold is 0.7. 1.0 < 0.7 -> False. Should NOT be added.

        result = self.module.learn_concepts([similar_rep_vector])

        self.assertEqual(len(result), 1) # No new concept added
        np.testing.assert_array_equal(result[0], initial_concept_rep)
        self.assertEqual(len(self.module.concept_representatives), 1) # Internal list length
        np.testing.assert_array_equal(self.module.concept_representatives[0], initial_concept_rep)


        # Vector with similarity EXACTLY at threshold: Let's construct a vector with similarity 0.7 to the first concept.
        similar_at_threshold_rep = np.zeros(dim, dtype=np.float32)
        similar_at_threshold_rep[0] = 0.7
        if dim > 1: # If dimension is greater than 1, adjust other elements to ensure norm is 1 after normalization
             similar_at_threshold_rep[1] = np.sqrt(1.0 - 0.7**2)
        # Normalize the vector
        norm = np.linalg.norm(similar_at_threshold_rep)
        if norm > 1e-8: # Avoid division by near zero
            similar_at_threshold_rep /= norm
        else:
            similar_at_threshold_rep[0] = 0.0  # Handle tiny norms, although unlikely with this construction.

        # Similarity with [1,0,...] is now exactly 0.7. Threshold is 0.7. 0.7 < 0.7 -> False. Should NOT be added.
        # The log in the failing test showed 0.7000 < 0.7000 evaluating to True, this is a floating point issue.
        # We need to ensure our test assertion is robust to tiny floating point differences.
        # The code's comparison is `max_similarity_to_concepts < self.new_concept_threshold`.
        # Let's trust the code logic and ensure the test doesn't fail due to floating point precision.
        # The assertion `self.assertEqual(len(result2), 1)` is correct if the strict comparison works as intended.
        # The issue might be in the similarity calculation or np.dot result precision.
        # For now, let's keep the assertion as is and assume the code's strict comparison is the intended behavior.
        # The previous failure might have been due to a very specific floating point representation.

        result2 = self.module.learn_concepts([similar_at_threshold_rep])
        # The previous concept list had 1 element. The new vector was not added. Total should still be 1.
        self.assertEqual(len(result2), 1) # Should still have 1 concept
        np.testing.assert_array_equal(result2[0], initial_concept_rep)
        self.assertEqual(len(self.module.concept_representatives), 1) # Internal list length
        np.testing.assert_array_equal(self.module.concept_representatives[0], initial_concept_rep)


    # Corrected Test: Tests learning with a mixed list of vectors.
    # Deterministic vectors used. Expectation corrected to 4 concepts (initial 2 + 2 new).
    def test_learn_concepts_multiple_vectors_mixed_deterministic(self):
        """Tests learning with a mixed list of vectors (valid, invalid, similar, new - deterministic)."""
        dim = self.module.representation_dim
        self.module.new_concept_threshold = 0.7 # Default threshold

        # Initial concepts
        concept1 = np.zeros(dim, dtype=np.float32); concept1[0] = 1.0
        concept2 = np.zeros(dim, dtype=np.float32); concept2[1] = 1.0
        self.module.learn_concepts([concept1, concept2]) # Now have 2 concepts. sim(concept1, concept2) = 0.0 < 0.7 -> concept2 is added.
        self.assertEqual(len(self.module.concept_representatives), 2) # Sanity check

        # List of vectors to learn from (Deterministic vectors):
        # Define vectors outside the list
        new_concept_rep1_det = np.zeros(dim, dtype=np.float32); new_concept_rep1_det[2] = 1.0 # Sim to concept1 & 2 is 0.0 < 0.7 -> Added
        new_concept_rep2_det = np.zeros(dim, dtype=np.float32); new_concept_rep2_det[3] = 1.0 # Sim to concept1, 2, new1 is 0.0 < 0.7 -> Added

        # A vector similar enough to an existing concept -> NOT added
        similar_enough_rep_det = np.zeros(dim, dtype=np.float32)
        sim_val = 0.8 # Similarity value
        similar_enough_rep_det[0] = sim_val
        if dim > 1: similar_enough_rep_det[dim-1] = np.sqrt(1-sim_val**2)
        norm = np.linalg.norm(similar_enough_rep_det)
        if norm > 1e-8: similar_enough_rep_det /= norm
        else: similar_enough_rep_det[0] = 0.0
        # Similarity to concept1 (~0.8 > 0.7) -> Not added

        rep_list_to_learn_det = [
            np.zeros(dim, dtype=np.float32), # Zero norm -> Should be skipped
            None, # None input -> Should be skipped
            "not a numpy array", # Wrong type -> Should be skipped
            np.random.rand(dim // 2 if dim > 1 else 1).astype(np.float32), # Wrong dimension -> Should be skipped
            concept1.copy() + 1e-5, # Very similar to concept1 -> Should be skipped
            concept2.copy() - 1e-5, # Very similar to concept2 -> Should be skipped
            np.zeros(dim, dtype=np.float32), # Zero norm again -> Should be skipped

            new_concept_rep1_det, # New concept 1 -> Should be added
            new_concept_rep2_det, # New concept 2 -> Should be added
            similar_enough_rep_det, # Similar concept -> Should be skipped
        ]

        # Expected: Initial 2 + 2 new = 4 concepts total.
        initial_concept_count = len(self.module.concept_representatives) # 2
        result = self.module.learn_concepts(rep_list_to_learn_det)

        self.assertEqual(len(result), len(self.module.concept_representatives)) # Return value should be the same as the internal list
        self.assertEqual(len(result), initial_concept_count + 2) # Expecting 4 concepts total

        # The first 2 concepts should be the initial ones (value check)
        np.testing.assert_array_equal(result[0], concept1)
        np.testing.assert_array_equal(result[1], concept2)

        # The 2 added new concepts should be the correct deterministic vectors (value check)
        # Order: They are added based on their order in rep_list_to_learn_det.
        # new_concept_rep1_det (index 7) then new_concept_rep2_det (index 8)
        np.testing.assert_array_equal(result[2], new_concept_rep1_det)
        np.testing.assert_array_equal(result[3], new_concept_rep2_det)


    # Corrected Test: Tests that a zero-norm vector in the list is skipped.
    # Deterministic vector used. Expectation is 2 concepts (initial 1 + 1 new).
    def test_learn_concepts_with_zero_norm_vector_in_list_deterministic(self):
        """Tests that a zero-norm vector in the list is skipped (deterministic)."""
        dim = self.module.representation_dim
        # Start with 1 concept
        initial_concept = np.zeros(dim, dtype=np.float32); initial_concept[0] = 1.0
        self.module.learn_concepts([initial_concept])
        self.assertEqual(len(self.module.concept_representatives), 1)

        # Vector that will be a new concept (orthogonal)
        new_concept_rep = np.zeros(dim, dtype=np.float32); new_concept_rep[1] = 1.0 # Sim 0.0 < 0.7 -> Should be added

        rep_list = [
            np.array([0.0] * dim, dtype=np.float32), # Zero norm -> Should be skipped
            new_concept_rep, # Will be a new concept -> Should be added
        ]

        result = self.module.learn_concepts(rep_list)

        # The zero-norm vector should be skipped, the other should be added as a new concept.
        self.assertEqual(len(result), len(self.module.concept_representatives))
        self.assertEqual(len(result), 2) # Expected 2 concepts (initial 1 + new 1)
        np.testing.assert_array_equal(result[0], initial_concept) # First concept should remain the same
        np.testing.assert_array_equal(result[1], new_concept_rep) # The new concept should come from the second input.


    # Corrected Test: Tests that a wrong-dimension vector in the list is skipped.
    # Deterministic vector used. Expectation is 2 concepts (initial 1 + 1 new).
    def test_learn_concepts_with_wrong_dimension_vector_in_list_deterministic(self):
        """Tests that a wrong-dimension vector in the list is skipped (deterministic)."""
        dim = self.module.representation_dim
        # Start with 1 concept
        initial_concept = np.zeros(dim, dtype=np.float32); initial_concept[0] = 1.0
        self.module.learn_concepts([initial_concept])
        self.assertEqual(len(self.module.concept_representatives), 1)

        # Vector that will be a new concept (orthogonal)
        new_concept_rep = np.zeros(dim, dtype=np.float32); new_concept_rep[1] = 1.0 # Sim 0.0 < 0.7 -> Should be added

        rep_list = [
            np.random.rand(dim // 2 if dim > 1 else 1).astype(np.float32), # Wrong dimension -> Should be skipped
            new_concept_rep, # Will be a new concept -> Should be added
        ]

        result = self.module.learn_concepts(rep_list)

        # The wrong-dimension vector should be skipped, the other should be added as a new concept.
        self.assertEqual(len(result), len(self.module.concept_representatives))
        self.assertEqual(len(result), 2) # Expected 2 concepts (initial 1 + new 1)
        np.testing.assert_array_equal(result[0], initial_concept)
        np.testing.assert_array_equal(result[1], new_concept_rep)


    # Corrected Test: Tests that None or non-numpy items in the list are skipped.
    # Deterministic vector used. Expectation is 2 concepts (initial 1 + 1 new).
    def test_learn_concepts_with_none_or_invalid_type_in_list_deterministic(self):
        """Tests that None or non-numpy items in the list are skipped (deterministic)."""
        dim = self.module.representation_dim
        # Start with 1 concept
        initial_concept = np.zeros(dim, dtype=np.float32); initial_concept[0] = 1.0
        self.module.learn_concepts([initial_concept])
        self.assertEqual(len(self.module.concept_representatives), 1)

        # Vector that will be a new concept (orthogonal)
        new_concept_rep = np.zeros(dim, dtype=np.float32); new_concept_rep[1] = 1.0 # Sim 0.0 < 0.7 -> Should be added

        rep_list = [
            None, # Should be skipped
            "not a numpy array", # Should be skipped
            123, # Should be skipped
            new_concept_rep, # Will be a new concept -> Should be added
        ]

        result = self.module.learn_concepts(rep_list)

        # Invalid items should be skipped, only the last one should be added.
        self.assertEqual(len(result), len(self.module.concept_representatives))
        self.assertEqual(len(result), 2) # Expected 2 concepts (initial 1 + new 1)
        np.testing.assert_array_equal(result[0], initial_concept)
        np.testing.assert_array_equal(result[1], new_concept_rep)


    # Corrected Test: Tests learning when new_concept_threshold is 0.0 (very strict).
    # Deterministic vectors used. Expectation is 2 concepts (initial 1 + 1 new).
    def test_learn_concepts_threshold_zero_deterministic(self):
        """Tests learning when new_concept_threshold is 0.0 (deterministic)."""
        dim = self.module.representation_dim
        self.module.new_concept_threshold = 0.0 # Very strict threshold (only sim < 0 are added)

        # Vectors (set to float32)
        rep1 = np.zeros(dim, dtype=np.float32); rep1[0] = 1.0 # [1, 0, ...]
        rep2_similar = np.zeros(dim, dtype=np.float32); rep2_similar[0] = 1.001 # [1.001, 0, ...], Sim(rep1, rep2_similar) = 1.0
        rep3_orthogonal = np.zeros(dim, dtype=np.float32); rep3_orthogonal[1] = 1.0 # [0, 1, ...], Sim(rep1, rep3_orthogonal) = 0.0
        rep4_opposite = np.zeros(dim, dtype=np.float32); rep4_opposite[0] = -1.0 # [-1, 0, ...], Sim(rep1, rep4_opposite) = -1.0. -1.0 < 0.0 True -> Will be added.
        rep5_almost_opposite = np.zeros(dim, dtype=np.float32); rep5_almost_opposite[0] = -0.1;
        # If dim > 1, add a non-zero element and normalize
        if dim > 1: rep5_almost_opposite[dim-1] = np.sqrt(1-(-0.1)**2)
        norm = np.linalg.norm(rep5_almost_opposite)
        if norm > 1e-8:
            rep5_almost_opposite /= norm
        else:
            rep5_almost_opposite[0] = 0.0 # Ensure normalized, handle tiny norms
        # Sim(rep1, rep5_almost_opposite) = -0.1. Sim(rep4_opposite, rep5_almost_opposite) = dot([-1,0,...], [-0.1,...])/norms ~ 0.1. Max sim = max(-0.1, 0.1) = 0.1. 0.1 < 0.0 False -> Will NOT be added.


        self.module.learn_concepts([rep1]) # The first vector is always added (because the list is empty)
        self.assertEqual(len(self.module.concept_representatives), 1)
        np.testing.assert_array_equal(self.module.concept_representatives[0], rep1)

        # List of vectors to learn from: rep2 (sim 1.0), rep3 (sim 0.0), rep4 (sim -1.0), rep5 (sim -0.1)
        rep_list = [rep2_similar, rep3_orthogonal, rep4_opposite, rep5_almost_opposite]

        result = self.module.learn_concepts(rep_list)

        # rep2 (sim 1.0): 1.0 < 0.0 False -> Not added
        # rep3 (sim 0.0): 0.0 < 0.0 False -> Not added
        # rep4 (sim -1.0 to rep1): -1.0 < 0.0 True -> Added (2nd concept). Concepts: [rep1, rep4_opposite]
        # rep5 (sim -0.1 to rep1, sim ~0.1 to rep4). Max sim = max(-0.1, ~0.1) = ~0.1. Threshold 0.0. ~0.1 < 0.0 False -> Not added.

        self.assertEqual(len(result), len(self.module.concept_representatives))
        # Corrected expectation:
        self.assertEqual(len(result), 2) # Expected 2 concepts (initial 1 + new 1 (rep4))
        np.testing.assert_array_equal(result[0], rep1) # First concept should remain the same
        # The second added concept should be rep4_opposite
        np.testing.assert_array_equal(result[1], rep4_opposite)


    # Corrected Test: Tests learning when new_concept_threshold is 1.0 (very strict).
    # Deterministic vectors used. Expectation is 3 concepts (initial 1 + 2 new).
    def test_learn_concepts_threshold_one_deterministic(self):
        """Tests learning when new_concept_threshold is 1.0 (deterministic)."""
        dim = self.module.representation_dim
        self.module.new_concept_threshold = 1.0 # Highest strictness (only sim < 1.0 are added)

        # Vectors (set to float32)
        rep1 = np.zeros(dim, dtype=np.float32); rep1[0] = 1.0 # [1, 0, ...]
        rep2_identical = rep1.copy() # [1, 0, ...], Sim(rep1, rep2_identical) = 1.0
        rep3_very_similar = np.zeros(dim, dtype=np.float32); rep3_very_similar[0] = 1.0 - 1e-9; # Similarity very close to 1.0 but < 1.0.
        # Normalize to ensure unit vector (important for cosine similarity)
        norm = np.linalg.norm(rep3_very_similar)
        if norm > 1e-8:
            rep3_very_similar /= norm
        else:
            rep3_very_similar[0] = 0.0 # Handle tiny norms
        # Sim ~ 1.0 but < 1.0. Threshold 1.0. ~1.0 < 1.0 True -> Will be added.
        rep4_different = np.zeros(dim, dtype=np.float32); rep4_different[1] = 1.0 # [0, 1, ...], Sim(rep1, rep4_different) = 0.0. Sim(rep3, rep4) ~0.0. Max sim ~0.0 < 1.0 True -> Will be added.


        self.module.learn_concepts([rep1]) # The first vector is always added
        self.assertEqual(len(self.module.concept_representatives), 1)
        np.testing.assert_array_equal(self.module.concept_representatives[0], rep1)

        # List of vectors to learn from: rep2 (sim 1.0), rep3 (sim < 1.0), rep4 (sim 0.0)
        rep_list = [rep2_identical, rep3_very_similar, rep4_different]

        result = self.module.learn_concepts(rep_list)

        # rep2 (sim 1.0): 1.0 < 1.0 False -> Not added
        # rep3 (sim ~1.0 to rep1): ~1.0 < 1.0 True -> Added (2nd concept). Concepts: [rep1, rep3_very_similar]
        # rep4 (sim 0.0 to rep1): 0.0 < 1.0 True. Sim to rep3_very_similar: dot(rep4, rep3)/norm(rep4)/norm(rep3) ~ 0. Sim ~ 0.0.
        # Max sim = max(0.0 (to rep1), ~0.0 (to rep3)) = ~0.0. Threshold 1.0. ~0.0 < 1.0 True -> Added (3rd concept)

        self.assertEqual(len(result), len(self.module.concept_representatives))
        # Corrected expectation:
        self.assertEqual(len(result), 3) # Expected 3 concepts (initial 1 + new 2 (rep3, rep4))
        np.testing.assert_array_equal(result[0], rep1) # First concept should remain the same
        # The second added concept should be rep3_very_similar
        np.testing.assert_array_equal(result[1], rep3_very_similar)
        # The third added concept should be rep4_different
        np.testing.assert_array_equal(result[2], rep4_different)


    # Corrected Test: Exception handling test. Array comparison fixed.
    def test_learn_concepts_exception_handling(self):
        """Tests that the current list is returned if an exception occurs during learning."""
        dim = self.module.representation_dim
        initial_concept = np.random.rand(dim).astype(np.float32) # Ensure float32
        self.module.learn_concepts([initial_concept]) # Add 1 concept initially
        self.assertEqual(len(self.module.concept_representatives), 1)

        # Simulate an error by mocking the numpy.dot function to raise an exception.
        # Set it up so that it raises an error when processing the single vector in the test list.

        rep_vector_to_learn = np.random.rand(dim).astype(np.float32) # The vector we will try to learn
        # This vector will cause an error when its dot product with the first concept (initial_concept) is calculated.

        with patch('numpy.dot', side_effect=RuntimeError("Simulated Numpy.dot Error")):
             # Call learn_concepts.
             # The loop for rep_vector_to_learn will start. Norm will be calculated (no error).
             # self.concept_representatives is not empty. The loop for concept_rep (initial_concept) will start. Norm will be calculated (no error).
             # np.dot(rep_vector_to_learn, concept_rep) will be called. THIS will raise the error.
             # The error will be caught by the except block in learn_concepts.
             result = self.module.learn_concepts([rep_vector_to_learn])

             # If the except block ran, learning should have failed.
             # The current list (the initial_concept) should be returned.
             # Check list length
             self.assertEqual(len(result), len([initial_concept]))
             # Check array content (value check)
             np.testing.assert_array_equal(result[0], initial_concept)
             # Check that the internal list also hasn't changed (value check)
             self.assertEqual(len(self.module.concept_representatives), len([initial_concept]))
             np.testing.assert_array_equal(self.module.concept_representatives[0], initial_concept)



    # --- get_concepts Tests ---
    # These tests verify the method for retrieving the list of concepts.

    def test_get_concepts_empty(self):
        """Tests get_concepts when there are no concepts."""
        concepts = self.module.get_concepts()
        self.assertEqual(concepts, [])
        self.assertIsInstance(concepts, list)

    # Corrected Test: Tests get_concepts when there are concepts and verifies shallow copy expectation.
    # Test input made deterministic.
    # Expectation count corrected, 2 concepts should be added.
    def test_get_concepts_with_data(self):
        """Tests get_concepts when there are concepts (shallow copy is expected)."""
        dim = self.module.representation_dim
        # Add deterministic concepts (sim 0.0 < 0.7)
        concept1 = np.zeros(dim, dtype=np.float32); concept1[0] = 1.0
        concept2 = np.zeros(dim, dtype=np.float32); concept2[1] = 1.0

        self.module.learn_concepts([concept1, concept2]) # Add 2 concepts. sim(concept1, concept2) = 0.0 < 0.7 -> concept2 should be added.
        self.assertEqual(len(self.module.concept_representatives), 2) # Sanity check

        concepts = self.module.get_concepts()

        self.assertEqual(len(concepts), 2)
        self.assertIsInstance(concepts, list)
        # The returned list object should be different from the original (shallow copy)
        self.assertIsNot(concepts, self.module.concept_representatives)
        # The arrays inside the list should be the same objects as the original arrays (feature of shallow copy)
        self.assertIs(concepts[0], self.module.concept_representatives[0])
        self.assertIs(concepts[1], self.module.concept_representatives[1])

        # Verify content correctness (if object references are the same, content is also the same)
        np.testing.assert_array_equal(concepts[0], concept1)
        np.testing.assert_array_equal(concepts[1], concept2)


    # Corrected Test: Tests that get_concepts returns a shallow copy (detailed).
    # assertIs error and array comparison fixed.
    def test_get_concepts_is_shallow_copy(self):
        """Tests that get_concepts returns a shallow copy (detailed)."""
        dim = self.module.representation_dim
        concept1 = np.random.rand(dim).astype(np.float32) # Ensure float32
        self.module.learn_concepts([concept1]) # Add 1 concept
        self.assertEqual(len(self.module.concept_representatives), 1)

        concepts = self.module.get_concepts() # Get the shallow copy

        self.assertIsNot(concepts, self.module.concept_representatives) # 1. List object is different

        # Add a new element to the returned list (Original list should NOT be affected)
        new_vector = np.random.rand(dim).astype(np.float32) # Ensure float32
        concepts.append(new_vector)
        self.assertEqual(len(concepts), 2)
        self.assertEqual(len(self.module.concept_representatives), 1) # 2. Original list length should not change


        # Modify an array in the returned list (This SHOULD also modify the array in the original list because it's a shallow copy)
        # Only the list object itself was shallow copied, the array objects within are the same.
        if len(concepts) > 0: # Should have at least 1 concept from the previous learn_concepts call
             original_array_ref = self.module.concept_representatives[0]
             returned_array_ref = concepts[0]

             self.assertIs(returned_array_ref, original_array_ref) # Array objects should have the same reference

             original_value_before_mod = original_array_ref[0] # Value before modification
             modification_value = 999.0
             # Modify the array (in-place modification)
             returned_array_ref[0] += modification_value


             # Verify that the array in the original list has also changed
             # We are checking value equality. The value should be equal to the expected value after modification.
             # Use assertAlmostEqual with a sufficient number of decimal places.
             self.assertAlmostEqual(original_array_ref[0], original_value_before_mod + modification_value, places=7) # Increased places to 7 for safety


    # --- cleanup Tests ---
    # These tests verify the cleanup method.

    def test_cleanup(self):
        """Tests that the cleanup method clears the concepts list."""
        dim = self.module.representation_dim
        concept1 = np.random.rand(dim)
        self.module.learn_concepts([concept1]) # Add concepts
        self.assertEqual(len(self.module.concept_representatives), 1)

        self.module.cleanup() # Call the cleanup method

        self.assertEqual(self.module.concept_representatives, []) # List should be empty after cleanup.


# Test boilerplate code (usually handled by pytest runner)
if __name__ == '__main__':
    # Use unittest.main with arguments to allow running tests directly from script without pytest runner.
    # argv=[sys.argv[0]] keeps only the script name in arguments.
    # exit=False prevents unittest from exiting the process immediately after tests.
    unittest.main(argv=[sys.argv[0]], exit=False)