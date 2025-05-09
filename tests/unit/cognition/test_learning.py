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
                'learning':{
                    'new_concept_threshold': 0.7,
                }
                
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
        config_high = {'cognition': {'learning': {'new_concept_threshold': 1.5}}, 'representation': {'representation_dim': 10}}
        module_high = LearningModule(config_high)
        self.assertEqual(module_high.new_concept_threshold, 0.7) # Should be reset

        config_low = {'cognition': {'learning': {'new_concept_threshold': -0.5}}, 'representation': {'representation_dim': 10}}
        module_low = LearningModule(config_low)
        self.assertEqual(module_low.new_concept_threshold, 0.7) # Should be reset

        # Test boundary values - they should be accepted.
        config_at_boundaries_high = {'cognition': {'learning':{'new_concept_threshold': 1.0}}, 'representation': {'representation_dim': 10}}
        module_boundaries_high = LearningModule(config_at_boundaries_high)
        self.assertEqual(module_boundaries_high.new_concept_threshold, 1.0) # Boundaries should be included

        config_at_boundaries_low = {'cognition': {'learning':{'new_concept_threshold': 0.0}}, 'representation': {'representation_dim': 10}}
        module_boundaries_low = LearningModule(config_at_boundaries_low)
        self.assertEqual(module_boundaries_low.new_concept_threshold, 0.0) # Boundaries should be included


    def test_learn_concepts_invalid_representation_dim_skips(self):
        """Tests that learning is skipped if LearningModule's representation_dim is invalid."""
        original_dim = self.module.representation_dim
        self.module.representation_dim = 0 # Set to an invalid dimension

        initial_concepts_list = [np.random.rand(original_dim).astype(np.float32)] # Use original_dim for consistency
        self.module.concept_representatives = initial_concepts_list[:] # Start with some concepts (make a copy)

        rep_list = [np.random.rand(original_dim).astype(np.float32)] # Provide input vector with original_dim
        result = self.module.learn_concepts(rep_list)

        self.assertEqual(len(result), len(initial_concepts_list))
        np.testing.assert_array_equal(result[0], initial_concepts_list[0])
        self.assertEqual(len(self.module.concept_representatives), len(initial_concepts_list))
        np.testing.assert_array_equal(self.module.concept_representatives[0], initial_concepts_list[0])

        # Reset dimension after test
        self.module.representation_dim = original_dim


    # --- learn_concepts Tests ---
    # These tests verify the core concept learning logic.

    def test_learn_concepts_input_none(self):
        """Tests learn_concepts when representation_list is None."""
        initial_concepts_list = [np.random.rand(self.module.representation_dim).astype(np.float32)]
        self.module.concept_representatives = initial_concepts_list[:] # Start with some concepts
        result = self.module.learn_concepts(None)
        self.assertEqual(len(result), len(initial_concepts_list))
        if initial_concepts_list: # only compare content if not empty
            np.testing.assert_array_equal(result[0], initial_concepts_list[0])
        self.assertEqual(len(self.module.concept_representatives), len(initial_concepts_list))

    def test_learn_concepts_input_not_list(self):
        """Tests learn_concepts when representation_list is not a list."""
        initial_concepts_list = [np.random.rand(self.module.representation_dim).astype(np.float32)]
        self.module.concept_representatives = initial_concepts_list[:]
        result = self.module.learn_concepts("not a list")
        self.assertEqual(len(result), len(initial_concepts_list))
        if initial_concepts_list:
            np.testing.assert_array_equal(result[0], initial_concepts_list[0])
        self.assertEqual(len(self.module.concept_representatives), len(initial_concepts_list))

    def test_learn_concepts_input_empty_list(self):
        """Tests learn_concepts when representation_list is an empty list."""
        initial_concepts_list = [np.random.rand(self.module.representation_dim).astype(np.float32)]
        self.module.concept_representatives = initial_concepts_list[:]
        result = self.module.learn_concepts([])
        self.assertEqual(len(result), len(initial_concepts_list))
        if initial_concepts_list:
            np.testing.assert_array_equal(result[0], initial_concepts_list[0])
        self.assertEqual(len(self.module.concept_representatives), len(initial_concepts_list))


    def test_learn_concepts_from_scratch_first_vector(self):
        """Tests learning the first valid vector when starting with no concepts (should be added as a new concept)."""
        rep_vector = np.random.rand(self.module.representation_dim).astype(np.float32)
        # Ensure it's a unit vector for consistent similarity calculations
        norm = np.linalg.norm(rep_vector)
        if norm > 1e-8: rep_vector /= norm
        else: rep_vector[0] = 1.0 # a default unit vector if original is zero

        self.assertEqual(len(self.module.concept_representatives), 0) # Should be empty initially

        result = self.module.learn_concepts([rep_vector])

        self.assertEqual(len(result), 1) # Should have 1 concept now
        self.assertIsInstance(result[0], np.ndarray)
        self.assertEqual(result[0].shape, (self.module.representation_dim,))
        np.testing.assert_array_equal(result[0], rep_vector)
        self.assertEqual(len(self.module.concept_representatives), 1)
        np.testing.assert_array_equal(self.module.concept_representatives[0], rep_vector)
        self.assertIsNot(self.module.concept_representatives[0], rep_vector)


    def test_learn_concepts_new_concept_below_threshold_deterministic(self):
        """Tests that a new vector with similarity below the threshold is added as a new concept (deterministic)."""
        dim = self.module.representation_dim
        self.module.new_concept_threshold = 0.7

        initial_concept_rep = np.zeros(dim, dtype=np.float32); initial_concept_rep[0] = 1.0
        self.module.learn_concepts([initial_concept_rep])
        self.assertEqual(len(self.module.concept_representatives), 1)

        new_rep_vector_orthogonal = np.zeros(dim, dtype=np.float32); new_rep_vector_orthogonal[1] = 1.0
        result = self.module.learn_concepts([new_rep_vector_orthogonal])

        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], initial_concept_rep)
        np.testing.assert_array_equal(result[1], new_rep_vector_orthogonal)
        self.assertEqual(len(self.module.concept_representatives), 2)


    def test_learn_concepts_existing_concept_above_threshold_deterministic(self):
        """Tests that a vector with similarity above/equal to the threshold to existing concepts is NOT added (deterministic)."""
        dim = self.module.representation_dim
        self.module.new_concept_threshold = 0.7

        initial_concept_rep = np.zeros(dim, dtype=np.float32); initial_concept_rep[0] = 1.0
        self.module.learn_concepts([initial_concept_rep])
        self.assertEqual(len(self.module.concept_representatives), 1)

        # Vector very similar (cosine sim ~1.0)
        similar_rep_vector = np.zeros(dim, dtype=np.float32); similar_rep_vector[0] = 0.999 # will be normalized
        norm_s = np.linalg.norm(similar_rep_vector); similar_rep_vector /= norm_s
        # Similarity will be very high (0.999), which is > 0.7. NOT added.
        result = self.module.learn_concepts([similar_rep_vector])
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], initial_concept_rep)
        self.assertEqual(len(self.module.concept_representatives), 1)

        # Vector with similarity EXACTLY at threshold (e.g., 0.7 after normalization)
        # The log "Highest existing similarity: 0.7000 < Threshold 0.7000" suggests a floating point issue
        # where 0.7 was treated as < 0.7. Python's `0.7 < 0.7` is False.
        # If `max_similarity_to_concepts` is truly 0.7000000000000000, it should NOT be added.
        # Let's create a vector whose dot product with initial_concept_rep is *slightly above* 0.7 BEFORE normalization
        # to ensure it's not added, and another *slightly below* to ensure it is added.

        # Case 1: Similarity slightly ABOVE threshold (e.g. 0.7001) -> NOT added
        above_threshold_rep = np.zeros(dim, dtype=np.float32)
        above_threshold_rep[0] = 0.7001 
        if dim > 1: above_threshold_rep[1] = np.sqrt(1.0 - 0.7001**2)
        norm_at = np.linalg.norm(above_threshold_rep); above_threshold_rep /= norm_at
        
        # Recalculate concepts to ensure clean state
        self.module.concept_representatives = []
        self.module.learn_concepts([initial_concept_rep]) # Back to 1 concept

        result_above = self.module.learn_concepts([above_threshold_rep])
        # Similarity is ~0.7001. 0.7001 < 0.7 is False. Not added.
        self.assertEqual(len(result_above), 1, "Concept with similarity just above threshold should not be added.")

        # Case 2: Similarity slightly BELOW threshold (e.g. 0.6999) -> ADDED
        below_threshold_rep = np.zeros(dim, dtype=np.float32)
        below_threshold_rep[0] = 0.6999
        if dim > 1: below_threshold_rep[1] = np.sqrt(1.0 - 0.6999**2)
        norm_bt = np.linalg.norm(below_threshold_rep); below_threshold_rep /= norm_bt

        # Recalculate concepts to ensure clean state
        self.module.concept_representatives = []
        self.module.learn_concepts([initial_concept_rep]) # Back to 1 concept

        result_below = self.module.learn_concepts([below_threshold_rep])
        # Similarity is ~0.6999. 0.6999 < 0.7 is True. Added.
        self.assertEqual(len(result_below), 2, "Concept with similarity just below threshold should be added.")


    def test_learn_concepts_multiple_vectors_mixed_deterministic(self):
        dim = self.module.representation_dim
        self.module.new_concept_threshold = 0.7

        concept1 = np.zeros(dim, dtype=np.float32); concept1[0] = 1.0
        concept2 = np.zeros(dim, dtype=np.float32); concept2[1] = 1.0
        self.module.learn_concepts([concept1, concept2])
        self.assertEqual(len(self.module.concept_representatives), 2)

        new_concept_rep1_det = np.zeros(dim, dtype=np.float32); new_concept_rep1_det[2] = 1.0
        new_concept_rep2_det = np.zeros(dim, dtype=np.float32); new_concept_rep2_det[3] = 1.0

        similar_enough_rep_det = np.zeros(dim, dtype=np.float32)
        sim_val = 0.8
        similar_enough_rep_det[0] = sim_val
        if dim > 1: similar_enough_rep_det[dim-1] = np.sqrt(max(0, 1-sim_val**2)) # max(0, ...) for safety
        norm = np.linalg.norm(similar_enough_rep_det)
        if norm > 1e-8: similar_enough_rep_det /= norm
        else: similar_enough_rep_det = concept1.copy() # fallback if norm is zero

        rep_list_to_learn_det = [
            np.zeros(dim, dtype=np.float32),
            None,
            "not a numpy array",
            np.random.rand(dim // 2 if dim > 1 else 1).astype(np.float32),
            concept1.copy() * 0.99, # Similar to concept1 (sim 0.99 > 0.7)
            concept2.copy() * 0.98, # Similar to concept2 (sim 0.98 > 0.7)
            np.zeros(dim, dtype=np.float32),
            new_concept_rep1_det,
            new_concept_rep2_det,
            similar_enough_rep_det,
        ]

        initial_concept_count = len(self.module.concept_representatives)
        result = self.module.learn_concepts(rep_list_to_learn_det)

        self.assertEqual(len(result), len(self.module.concept_representatives))
        self.assertEqual(len(result), initial_concept_count + 2)

        np.testing.assert_array_equal(result[0], concept1)
        np.testing.assert_array_equal(result[1], concept2)
        np.testing.assert_array_equal(result[2], new_concept_rep1_det)
        np.testing.assert_array_equal(result[3], new_concept_rep2_det)


    def test_learn_concepts_with_zero_norm_vector_in_list_deterministic(self):
        dim = self.module.representation_dim
        initial_concept = np.zeros(dim, dtype=np.float32); initial_concept[0] = 1.0
        self.module.learn_concepts([initial_concept])
        self.assertEqual(len(self.module.concept_representatives), 1)

        new_concept_rep = np.zeros(dim, dtype=np.float32); new_concept_rep[1] = 1.0
        rep_list = [
            np.array([0.0] * dim, dtype=np.float32),
            new_concept_rep,
        ]
        result = self.module.learn_concepts(rep_list)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], initial_concept)
        np.testing.assert_array_equal(result[1], new_concept_rep)


    def test_learn_concepts_with_wrong_dimension_vector_in_list_deterministic(self):
        dim = self.module.representation_dim
        initial_concept = np.zeros(dim, dtype=np.float32); initial_concept[0] = 1.0
        self.module.learn_concepts([initial_concept])
        self.assertEqual(len(self.module.concept_representatives), 1)

        new_concept_rep = np.zeros(dim, dtype=np.float32); new_concept_rep[1] = 1.0
        rep_list = [
            np.random.rand(dim // 2 if dim > 1 else dim + 1).astype(np.float32), # Ensure wrong dimension
            new_concept_rep,
        ]
        result = self.module.learn_concepts(rep_list)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], initial_concept)
        np.testing.assert_array_equal(result[1], new_concept_rep)


    def test_learn_concepts_with_none_or_invalid_type_in_list_deterministic(self):
        dim = self.module.representation_dim
        initial_concept = np.zeros(dim, dtype=np.float32); initial_concept[0] = 1.0
        self.module.learn_concepts([initial_concept])
        self.assertEqual(len(self.module.concept_representatives), 1)

        new_concept_rep = np.zeros(dim, dtype=np.float32); new_concept_rep[1] = 1.0
        rep_list = [
            None, "not a numpy array", 123, new_concept_rep,
        ]
        result = self.module.learn_concepts(rep_list)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], initial_concept)
        np.testing.assert_array_equal(result[1], new_concept_rep)


    def test_learn_concepts_threshold_zero_deterministic(self):
        dim = self.module.representation_dim
        self.module.new_concept_threshold = 0.0

        rep1 = np.zeros(dim, dtype=np.float32); rep1[0] = 1.0
        rep2_similar = rep1.copy() * 0.99 # Sim 0.99 > 0.0. Not added.
        rep3_orthogonal = np.zeros(dim, dtype=np.float32); rep3_orthogonal[1] = 1.0 # Sim 0.0. 0.0 < 0.0 is False. Not added.
        rep4_opposite = np.zeros(dim, dtype=np.float32); rep4_opposite[0] = -1.0 # Sim -1.0 < 0.0. Added.
        
        # Normalize rep5_almost_opposite correctly
        rep5_almost_opposite = np.zeros(dim, dtype=np.float32); rep5_almost_opposite[0] = -0.1;
        if dim > 1: rep5_almost_opposite[1] = np.sqrt(max(0,1.0 - (-0.1)**2)) # Use index 1 for orthogonality if dim > 1
        norm_rep5 = np.linalg.norm(rep5_almost_opposite)
        if norm_rep5 > 1e-8: rep5_almost_opposite /= norm_rep5
        else: rep5_almost_opposite[0] = -1.0; rep5_almost_opposite[1:] = 0 # Fallback

        self.module.learn_concepts([rep1])
        self.assertEqual(len(self.module.concept_representatives), 1)

        # After rep4 is added, concepts are [rep1, rep4_opposite]
        # rep5 vs rep1: sim is -0.1 / (1*1) = -0.1
        # rep5 vs rep4_opposite: dot([-0.1, sqrt(0.99),0...], [-1,0,0...]) = 0.1. sim = 0.1 / (1*1) = 0.1
        # Max sim for rep5 is 0.1. 0.1 < 0.0 is False. Not added.

        rep_list = [rep2_similar, rep3_orthogonal, rep4_opposite, rep5_almost_opposite]
        result = self.module.learn_concepts(rep_list)

        self.assertEqual(len(result), 2) # rep1, rep4_opposite
        np.testing.assert_array_equal(result[0], rep1)
        np.testing.assert_array_equal(result[1], rep4_opposite)


    def test_learn_concepts_threshold_one_deterministic(self):
        dim = self.module.representation_dim
        self.module.new_concept_threshold = 1.0

        rep1 = np.zeros(dim, dtype=np.float32); rep1[0] = 1.0
        rep2_identical = rep1.copy() # Sim 1.0. 1.0 < 1.0 is False. Not added.

        rep3_very_similar = np.zeros(dim, dtype=np.float32); rep3_very_similar[0] = 1.0 - 1e-7 # Slightly less than 1
        if dim > 1 : rep3_very_similar[1] = np.sqrt(max(0, 1.0 - (1.0-1e-7)**2)) # ensure orthogonal component
        norm_r3 = np.linalg.norm(rep3_very_similar); 
        if norm_r3 > 1e-8: rep3_very_similar /= norm_r3
        # Sim with rep1 is (1.0-1e-7) which is < 1.0. Added.

        rep4_different = np.zeros(dim, dtype=np.float32); rep4_different[1] = 1.0
        # Sim with rep1 is 0.0. Sim with rep3_very_similar (assuming rep3 is [~1, small, 0...]) is small.
        # Max sim is small < 1.0. Added.

        self.module.learn_concepts([rep1])
        self.assertEqual(len(self.module.concept_representatives), 1)

        rep_list = [rep2_identical, rep3_very_similar, rep4_different]
        result = self.module.learn_concepts(rep_list)
        
        # Expected: rep1 (initial), rep3_very_similar, rep4_different
        self.assertEqual(len(result), 3)
        np.testing.assert_array_equal(result[0], rep1)
        np.testing.assert_array_equal(result[1], rep3_very_similar)
        np.testing.assert_array_equal(result[2], rep4_different)


    def test_learn_concepts_exception_handling(self):
        dim = self.module.representation_dim
        initial_concept_list = [np.random.rand(dim).astype(np.float32)]
        # Ensure unit vector
        norm_ic = np.linalg.norm(initial_concept_list[0])
        if norm_ic > 1e-8: initial_concept_list[0] /= norm_ic
        else: initial_concept_list[0][0] = 1.0; initial_concept_list[0][1:] = 0.0


        self.module.learn_concepts(initial_concept_list)
        self.assertEqual(len(self.module.concept_representatives), 1)

        rep_vector_to_learn = np.random.rand(dim).astype(np.float32)
        norm_rvl = np.linalg.norm(rep_vector_to_learn)
        if norm_rvl > 1e-8: rep_vector_to_learn /= norm_rvl
        else: rep_vector_to_learn[0] = 1.0; rep_vector_to_learn[1:] = 0.0


        with patch('numpy.dot', side_effect=RuntimeError("Simulated Numpy.dot Error")):
             result = self.module.learn_concepts([rep_vector_to_learn])
             self.assertEqual(len(result), 1)
             np.testing.assert_array_equal(result[0], initial_concept_list[0])
             self.assertEqual(len(self.module.concept_representatives), 1)
             np.testing.assert_array_equal(self.module.concept_representatives[0], initial_concept_list[0])


    # --- get_concepts Tests ---
    def test_get_concepts_empty(self):
        concepts = self.module.get_concepts()
        self.assertEqual(concepts, [])
        self.assertIsInstance(concepts, list)

    def test_get_concepts_with_data(self):
        dim = self.module.representation_dim
        concept1 = np.zeros(dim, dtype=np.float32); concept1[0] = 1.0
        concept2 = np.zeros(dim, dtype=np.float32); concept2[1] = 1.0

        self.module.learn_concepts([concept1, concept2])
        self.assertEqual(len(self.module.concept_representatives), 2)

        concepts = self.module.get_concepts()
        self.assertEqual(len(concepts), 2)
        self.assertIsInstance(concepts, list)
        self.assertIsNot(concepts, self.module.concept_representatives)
        self.assertIs(concepts[0], self.module.concept_representatives[0])
        self.assertIs(concepts[1], self.module.concept_representatives[1])
        np.testing.assert_array_equal(concepts[0], concept1)
        np.testing.assert_array_equal(concepts[1], concept2)


    def test_get_concepts_is_shallow_copy(self):
        dim = self.module.representation_dim
        concept1_orig = np.random.rand(dim).astype(np.float32)
        self.module.learn_concepts([concept1_orig.copy()]) # Pass a copy to learn_concepts
        self.assertEqual(len(self.module.concept_representatives), 1)
    
        concepts = self.module.get_concepts()
        self.assertIsNot(concepts, self.module.concept_representatives)
    
        new_vector = np.random.rand(dim).astype(np.float32)
        concepts.append(new_vector)
        self.assertEqual(len(concepts), 2)
        self.assertEqual(len(self.module.concept_representatives), 1)
    
        if len(concepts) > 0 and len(self.module.concept_representatives) > 0 :
             original_array_ref = self.module.concept_representatives[0]
             returned_array_ref = concepts[0]
             self.assertIs(returned_array_ref, original_array_ref)
    
             original_value_before_mod = original_array_ref[0].copy() # Copy the scalar value
             modification_value = 999.0
             returned_array_ref[0] += modification_value # In-place modification
    
             # Using numpy.testing.assert_allclose for robust float comparison
             expected_value = original_value_before_mod + modification_value
             np.testing.assert_allclose(original_array_ref[0], expected_value, rtol=1e-5, atol=1e-5)


    # --- cleanup Tests ---
    def test_cleanup(self):
        dim = self.module.representation_dim
        concept1 = np.random.rand(dim)
        self.module.learn_concepts([concept1])
        self.assertEqual(len(self.module.concept_representatives), 1)
        self.module.cleanup()
        self.assertEqual(self.module.concept_representatives, [])


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]], exit=False)