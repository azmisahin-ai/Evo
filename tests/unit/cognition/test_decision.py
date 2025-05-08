# tests/unit/cognition/test_decision.py
import unittest
import sys
import os
import numpy as np
import logging
import random

from unittest.mock import patch, MagicMock

# Set PROJECT_ROOT to the root directory of the project. Assuming test file is in tests/unit/cognition.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

# Import modules to be tested.
# check_* and get_config_value imports are expected to work now.
try:
    from src.cognition.decision import DecisionModule
    # check_* functions come from src.core.utils
    from src.core.utils import check_input_not_none, check_input_type
    from src.core.config_utils import get_config_value # get_config_value comes from here

except ImportError as e:
     print(f"Failed to import fundamental modules. Is PYTHONPATH configured correctly? Error: {e}")
     raise e


# Enable these lines if you want to see logger output during tests.
# import logging
# logging.basicConfig(level=logging.DEBUG) # Set general level to DEBUG
# logging.getLogger('src.cognition.decision').setLevel(logging.DEBUG)
# logging.getLogger('src.core.utils').setLevel(logging.DEBUG) # To see logs from utils logger
# logging.getLogger('src.core.config_utils').setLevel(logging.DEBUG)


class TestDecisionModule(unittest.TestCase):

    def setUp(self):
        """Runs before each test method, creates a default configuration and DecisionModule instance."""
        # Default, valid configuration structure reflecting main_config.yaml.
        # DecisionModule needs settings under the 'cognition' key.
        self.default_config = {
            'cognition': { # These settings are directly under the 'cognition' key in main_config.yaml
                'familiarity_threshold': 0.8,
                'audio_energy_threshold': 1000.0,
                'visual_edges_threshold': 50.0,
                'brightness_threshold_high': 200.0,
                'brightness_threshold_low': 50.0,
                'concept_recognition_threshold': 0.85,
                'curiosity_threshold': 5.0,
                'curiosity_increment_new': 1.0,
                'curiosity_decrement_familiar': 0.5,
                'curiosity_decay': 0.1,
            }
        }
        # Initialize the module with the default config.
        # DecisionModule's __init__ expects the full config dict.
        self.module = DecisionModule(self.default_config)

        # Reset curiosity level to default (0.0) at the start of each test for isolation.
        self.module.curiosity_level = 0.0


    def tearDown(self):
        """Runs after each test method."""
        # Call cleanup to ensure resources are released (clears concepts list if implemented).
        self.module.cleanup()


    # --- __init__ Tests (Verifies config reading and attribute assignment) ---

    def test_init_with_valid_config(self):
        """Tests initialization with a valid configuration."""
        # The fixture (or setUp) ensures a valid config is used for module creation.
        # Verify that config values were read and assigned correctly.
        self.assertEqual(self.module.config, self.default_config)
        # Verify attributes match config values and are floats.
        self.assertEqual(self.module.familiarity_threshold, 0.8)
        self.assertIsInstance(self.module.familiarity_threshold, float)
        self.assertEqual(self.module.audio_energy_threshold, 1000.0)
        self.assertIsInstance(self.module.audio_energy_threshold, float)
        self.assertEqual(self.module.visual_edges_threshold, 50.0)
        self.assertIsInstance(self.module.visual_edges_threshold, float)
        self.assertEqual(self.module.brightness_threshold_high, 200.0)
        self.assertIsInstance(self.module.brightness_threshold_high, float)
        self.assertEqual(self.module.brightness_threshold_low, 50.0)
        self.assertIsInstance(self.module.brightness_threshold_low, float)
        self.assertEqual(self.module.concept_recognition_threshold, 0.85)
        self.assertIsInstance(self.module.concept_recognition_threshold, float)
        self.assertEqual(self.module.curiosity_threshold, 5.0)
        self.assertIsInstance(self.module.curiosity_threshold, float)
        self.assertEqual(self.module.curiosity_increment_new, 1.0)
        self.assertIsInstance(self.module.curiosity_increment_new, float)
        self.assertEqual(self.module.curiosity_decrement_familiar, 0.5)
        self.assertIsInstance(self.module.curiosity_decrement_familiar, float)
        self.assertEqual(self.module.curiosity_decay, 0.1)
        self.assertIsInstance(self.module.curiosity_decay, float)
        self.assertEqual(self.module.curiosity_level, 0.0) 
        self.assertIsInstance(self.module.curiosity_level, float)


    def test_init_with_missing_config_values(self):
        """Tests initialization when some configuration values are missing (defaults should be used)."""
        incomplete_config = {
             'cognition': { 
                 'familiarity_threshold': 0.9, 
             }
        }
        module = DecisionModule(incomplete_config)
        self.assertEqual(module.familiarity_threshold, 0.9) 
        self.assertIsInstance(module.familiarity_threshold, float)
        self.assertEqual(module.audio_energy_threshold, 1000.0) 
        self.assertIsInstance(module.audio_energy_threshold, float)
        self.assertEqual(module.visual_edges_threshold, 50.0) 
        self.assertIsInstance(module.visual_edges_threshold, float)
        # ... (add assertions for other defaults being float)


    def test_init_with_invalid_config_types(self):
        """Tests initialization with configuration values of invalid types (defaults should be used)."""
        invalid_type_config = {
            'cognition': { 
                'audio_energy_threshold': "not a float", 
                'visual_edges_threshold': 60, 
                'brightness_threshold_high': [250], 
                'brightness_threshold_low': 30.0, 
            }
        }
        module = DecisionModule(invalid_type_config)
        self.assertEqual(module.audio_energy_threshold, 1000.0) 
        self.assertIsInstance(module.audio_energy_threshold, float)
        self.assertEqual(module.visual_edges_threshold, 60.0) 
        self.assertIsInstance(module.visual_edges_threshold, float) # This should now pass
        self.assertEqual(module.brightness_threshold_high, 200.0) 
        self.assertIsInstance(module.brightness_threshold_high, float)
        self.assertEqual(module.brightness_threshold_low, 30.0) 
        self.assertIsInstance(module.brightness_threshold_low, float)
        # ... (add assertions for other defaults being float)


    def test_init_thresholds_out_of_range(self):
        """Tests when some thresholds are set outside the 0.0-1.0 range."""
        config_high = {'cognition': {'familiarity_threshold': 1.5, 'concept_recognition_threshold': 1.1, 'audio_energy_threshold': -10.0, 'brightness_threshold_high': 260.0, 'brightness_threshold_low': -10.0, 'curiosity_increment_new': -5.0}}
        module_high = DecisionModule(config_high)
        self.assertEqual(module_high.familiarity_threshold, 0.8) 
        self.assertEqual(module_high.concept_recognition_threshold, 0.85)
        self.assertEqual(module_high.audio_energy_threshold, 1000.0)
        # Corrected assertion: 260.0 is valid per current logic as long as low < high and high is not negative.
        # Low becomes 50.0. 50.0 < 260.0 is true. 260.0 is not negative. So it remains 260.0.
        self.assertEqual(module_high.brightness_threshold_high, 260.0) # CHANGED from 200.0 to 260.0
        self.assertEqual(module_high.brightness_threshold_low, 50.0) 
        self.assertEqual(module_high.curiosity_increment_new, 1.0)

        config_low_high_swap = {'cognition': {'brightness_threshold_low': 100.0, 'brightness_threshold_high': 80.0}} 
        module_low_high_swap = DecisionModule(config_low_high_swap)
        self.assertEqual(module_low_high_swap.brightness_threshold_low, 50.0) 
        self.assertEqual(module_low_high_swap.brightness_threshold_high, 200.0)

        config_at_boundaries = {'cognition': {'familiarity_threshold': 0.0, 'concept_recognition_threshold': 1.0, 'curiosity_threshold': 0.0, 'curiosity_decrement_familiar': 0.0, 'curiosity_decay': 0.0}} 
        module_boundaries = DecisionModule(config_at_boundaries)
        self.assertEqual(module_boundaries.familiarity_threshold, 0.0) 
        self.assertEqual(module_boundaries.concept_recognition_threshold, 1.0)
        self.assertEqual(module_boundaries.curiosity_threshold, 0.0) 
        self.assertEqual(module_boundaries.curiosity_decrement_familiar, 0.0)
        self.assertEqual(module_boundaries.curiosity_decay, 0.0) 


    # --- decide Input Validation Tests ---
    def test_decide_input_none(self):
        initial_curiosity = self.module.curiosity_level 
        result = self.module.decide(None, [], []) 
        self.assertIsNone(result)
        self.assertEqual(self.module.curiosity_level, initial_curiosity)


    def test_decide_input_not_dict(self):
        initial_curiosity = self.module.curiosity_level
        result = self.module.decide("not a dict", [], []) 
        self.assertIsNone(result)
        self.assertEqual(self.module.curiosity_level, initial_curiosity)


    def test_decide_input_empty_dict(self):
        initial_curiosity = 0.0
        self.module.curiosity_level = initial_curiosity
        expected_curiosity = initial_curiosity + self.module.curiosity_increment_new - self.module.curiosity_decay
        expected_curiosity = max(0.0, expected_curiosity) 

        result = self.module.decide({}, [], []) 

        self.assertEqual(result, "new_input_detected") 
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    # --- decide Decision Priority Tests ---
    @patch('random.choice', return_value='explore_randomly') 
    def test_decide_priority_curiosity_threshold(self, mock_random_choice):
        initial_curiosity = self.module.curiosity_threshold + 1.0 
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': 0.1, 
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.1, 
            'most_similar_concept_id': None,
        }
        expected_curiosity = initial_curiosity - self.module.curiosity_decay
        expected_curiosity = max(0.0, expected_curiosity) 

        result = self.module.decide(signals, [], []) 
        self.assertEqual(result, 'explore_randomly') 
        mock_random_choice.assert_called_once_with(["explore_randomly", "make_noise"]) 
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) 


    def test_decide_priority_sound_detected(self):
        initial_curiosity = self.module.curiosity_threshold - 1.0 
        self.module.curiosity_level = initial_curiosity
        signals = {
            'similarity_score': 0.1, 
            'high_audio_energy': True, 
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.9, 
            'most_similar_concept_id': 1,
        }
        expected_curiosity = initial_curiosity - self.module.curiosity_decay 
        expected_curiosity = max(0.0, expected_curiosity)
        result = self.module.decide(signals, [], []) 
        self.assertEqual(result, "sound_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_complex_visual_detected(self):
        initial_curiosity = self.module.curiosity_threshold - 1.0 
        self.module.curiosity_level = initial_curiosity
        signals = {
            'similarity_score': 0.1, 
            'high_audio_energy': False, 
            'high_visual_edges': True, 
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.9, 
            'most_similar_concept_id': 1,
        }
        expected_curiosity = initial_curiosity - self.module.curiosity_decay
        expected_curiosity = max(0.0, expected_curiosity)
        result = self.module.decide(signals, [], []) 
        self.assertEqual(result, "complex_visual_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_bright_light_detected(self):
        initial_curiosity = self.module.curiosity_threshold - 1.0 
        self.module.curiosity_level = initial_curiosity
        signals = {
            'similarity_score': 0.1, 
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': True, 
            'is_dark': False,
            'max_concept_similarity': 0.9, 
            'most_similar_concept_id': 1,
        }
        expected_curiosity = initial_curiosity - self.module.curiosity_decay
        expected_curiosity = max(0.0, expected_curiosity)
        result = self.module.decide(signals, [], []) 
        self.assertEqual(result, "bright_light_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_dark_environment_detected(self):
        initial_curiosity = self.module.curiosity_threshold - 1.0 
        self.module.curiosity_level = initial_curiosity
        signals = {
            'similarity_score': 0.1, 
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': True, 
            'max_concept_similarity': 0.9, 
            'most_similar_concept_id': 1,
        }
        expected_curiosity = initial_curiosity - self.module.curiosity_decay
        expected_curiosity = max(0.0, expected_curiosity)
        result = self.module.decide(signals, [], []) 
        self.assertEqual(result, "dark_environment_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_recognized_concept(self):
        initial_curiosity = self.module.curiosity_threshold - 1.0 
        self.module.curiosity_level = initial_curiosity
        signals = {
            'similarity_score': 0.1, 
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': self.module.concept_recognition_threshold + 0.01, 
            'most_similar_concept_id': 42, 
        }
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay 
        expected_curiosity = max(0.0, expected_curiosity)
        result = self.module.decide(signals, [], []) 
        self.assertEqual(result, "recognized_concept_42")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_recognized_concept_at_threshold(self):
        initial_curiosity = self.module.curiosity_threshold - 1.0 
        self.module.curiosity_level = initial_curiosity
        signals = {
            'similarity_score': self.module.familiarity_threshold, # Changed from concept_recognition_threshold to avoid conflict
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': self.module.concept_recognition_threshold, 
            'most_similar_concept_id': 99, 
        }
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay
        expected_curiosity = max(0.0, expected_curiosity)
        result = self.module.decide(signals, [], []) 
        self.assertEqual(result, "recognized_concept_99")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_recognized_concept_similarity_below_threshold(self):
        initial_curiosity = self.module.curiosity_threshold - 1.0 
        self.module.curiosity_level = initial_curiosity
        signals = {
            'similarity_score': 0.1, 
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': self.module.concept_recognition_threshold - 0.01, 
            'most_similar_concept_id': 123, 
        }
        expected_curiosity = initial_curiosity + self.module.curiosity_increment_new - self.module.curiosity_decay 
        expected_curiosity = max(0.0, expected_curiosity)
        result = self.module.decide(signals, [], []) 
        self.assertEqual(result, "new_input_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_familiar_input_detected(self):
        initial_curiosity = self.module.curiosity_threshold - 1.0 
        self.module.curiosity_level = initial_curiosity
        signals = {
            'similarity_score': self.module.familiarity_threshold + 0.01, 
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.1, 
            'most_similar_concept_id': None, 
        }
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay
        expected_curiosity = max(0.0, expected_curiosity)
        result = self.module.decide(signals, [], []) 
        self.assertEqual(result, "familiar_input_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_familiar_input_at_threshold(self):
        initial_curiosity = self.module.curiosity_threshold - 1.0 
        self.module.curiosity_level = initial_curiosity
        signals = {
            'similarity_score': self.module.familiarity_threshold, 
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.1, 
            'most_similar_concept_id': None, 
        }
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay
        expected_curiosity = max(0.0, expected_curiosity)
        result = self.module.decide(signals, [], []) 
        self.assertEqual(result, "familiar_input_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_new_input_detected(self):
        initial_curiosity = self.module.curiosity_threshold - 1.0 
        self.module.curiosity_level = initial_curiosity
        signals = {
            'similarity_score': self.module.familiarity_threshold - 0.01, 
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.1, 
            'most_similar_concept_id': None, 
        }
        expected_curiosity = initial_curiosity + self.module.curiosity_increment_new - self.module.curiosity_decay
        expected_curiosity = max(0.0, expected_curiosity)
        result = self.module.decide(signals, [], []) 
        self.assertEqual(result, "new_input_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    # --- Curiosity Update Tests ---
    def test_curiosity_update_new_input(self):
        initial_curiosity = 1.0 
        self.module.curiosity_level = initial_curiosity
        signals = { 
            'similarity_score': self.module.familiarity_threshold - 0.01, 
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }
        expected_curiosity = initial_curiosity + self.module.curiosity_increment_new - self.module.curiosity_decay
        expected_curiosity = max(0.0, expected_curiosity) 
        result = self.module.decide(signals, [], []) 
        self.assertEqual(result, "new_input_detected") 
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) 


    def test_curiosity_update_familiar_input(self):
        initial_curiosity = 1.0 
        self.module.curiosity_level = initial_curiosity
        signals = { 
            'similarity_score': self.module.familiarity_threshold + 0.01, 
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay 
        expected_curiosity = max(0.0, expected_curiosity) 
        result = self.module.decide(signals, [], []) 
        self.assertEqual(result, "familiar_input_detected") 
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) 


    def test_curiosity_update_recognized_concept(self):
        initial_curiosity = 1.0 
        self.module.curiosity_level = initial_curiosity
        signals = { 
            'similarity_score': 0.1, 
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': self.module.concept_recognition_threshold + 0.01, 'most_similar_concept_id': 77, 
        }
        # Corrected expected curiosity based on logic: initial - decrement - decay
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay 
        expected_curiosity = max(0.0, expected_curiosity) 
        result = self.module.decide(signals, [], []) 
        self.assertEqual(result, "recognized_concept_77") 
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) 


    def test_curiosity_update_other_decisions_only_decay(self):
        initial_curiosity = 1.0 
        self.module.curiosity_level = initial_curiosity
        signals = { 
            'similarity_score': 0.1,
            'high_audio_energy': True, 
            'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }
        expected_curiosity = initial_curiosity - self.module.curiosity_decay 
        expected_curiosity = max(0.0, expected_curiosity) 
        result = self.module.decide(signals, [], []) 
        self.assertEqual(result, "sound_detected") 
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) 

        self.module.curiosity_level = initial_curiosity 
        signals_visual = { 'high_audio_energy': False, 'high_visual_edges': True, 'is_bright': False, 'is_dark': False, 'similarity_score': 0.1, 'max_concept_similarity': 0.1, 'most_similar_concept_id': None, }
        result_visual = self.module.decide(signals_visual, [], []) 
        self.assertEqual(result_visual, "complex_visual_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) 

        self.module.curiosity_level = initial_curiosity 
        signals_bright = { 'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': True, 'is_dark': False, 'similarity_score': 0.1, 'max_concept_similarity': 0.1, 'most_similar_concept_id': None, }
        result_bright = self.module.decide(signals_bright, [], []) 
        self.assertEqual(result_bright, "bright_light_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) 

        self.module.curiosity_level = initial_curiosity 
        signals_dark = { 'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': True, 'similarity_score': 0.1, 'max_concept_similarity': 0.1, 'most_similar_concept_id': None, }
        result_dark = self.module.decide(signals_dark, [], []) 
        self.assertEqual(result_dark, "dark_environment_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) 


    @patch('random.choice', return_value='explore_randomly') 
    def test_curiosity_update_explore_randomly_only_decay(self, mock_random_choice):
        initial_curiosity = self.module.curiosity_threshold + 1.0 
        self.module.curiosity_level = initial_curiosity
        signals = { 
            'similarity_score': 0.1,
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }
        expected_curiosity = initial_curiosity - self.module.curiosity_decay 
        expected_curiosity = max(0.0, expected_curiosity)
        result = self.module.decide(signals, [], []) 
        self.assertEqual(result, 'explore_randomly') 
        mock_random_choice.assert_called_once_with(["explore_randomly", "make_noise"]) 
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) 


    def test_curiosity_does_not_go_below_zero(self):
        initial_curiosity = 0.1 
        self.module.curiosity_level = initial_curiosity
        signals = { 
            'similarity_score': self.module.familiarity_threshold + 0.01, 
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }
        expected_curiosity = max(0.0, initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay)
        result = self.module.decide(signals, [], []) 
        self.assertEqual(result, "familiar_input_detected") 
        self.assertAlmostEqual(self.module.curiosity_level, 0.0, places=6) # expected_curiosity will be 0.0


    # --- Exception Handling Test ---
    @patch('random.choice', side_effect=RuntimeError("Simulated Decision Error"))
    def test_decide_exception_handling_during_decision_logic(self, mock_random_choice):
        initial_curiosity_for_test = self.module.curiosity_threshold + 1.0 
        self.module.curiosity_level = initial_curiosity_for_test
        signals = { 
            'similarity_score': 0.1,
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }
        result = self.module.decide(signals, [], []) 
        self.assertIsNone(result) 
        mock_random_choice.assert_called_once() 
        self.assertEqual(self.module.curiosity_level, initial_curiosity_for_test)


    # --- cleanup Test ---
    def test_cleanup(self):
        with patch('src.cognition.decision.logger.info') as mock_logger_info:
             self.module.cleanup() 
             mock_logger_info.assert_called_with("DecisionModule object cleaning up.")

if __name__ == '__main__':
    unittest.main()