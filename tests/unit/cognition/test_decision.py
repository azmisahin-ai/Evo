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
        # Verify attributes match config values.
        self.assertEqual(self.module.familiarity_threshold, 0.8)
        self.assertEqual(self.module.audio_energy_threshold, 1000.0)
        self.assertEqual(self.module.visual_edges_threshold, 50.0)
        self.assertEqual(self.module.brightness_threshold_high, 200.0)
        self.assertEqual(self.module.brightness_threshold_low, 50.0)
        self.assertEqual(self.module.concept_recognition_threshold, 0.85)
        self.assertEqual(self.module.curiosity_threshold, 5.0)
        self.assertEqual(self.module.curiosity_increment_new, 1.0)
        self.assertEqual(self.module.curiosity_decrement_familiar, 0.5)
        self.assertEqual(self.module.curiosity_decay, 0.1)
        self.assertEqual(self.module.curiosity_level, 0.0) # Should be initialized to 0.0


    def test_init_with_missing_config_values(self):
        """Tests initialization when some configuration values are missing (defaults should be used)."""
        # Provide an incomplete config dictionary to the module init.
        # The get_config_value calls inside __init__ should use their default values for missing keys.
        # The structure must match the expected path, even if values are missing.
        incomplete_config = {
             'cognition': { # Must include cognition key for path to work
                 'familiarity_threshold': 0.9, # This value should be read
                 # Other cognition settings are missing, defaults should be used by get_config_value.
             }
        }
        # Initialize the module with the incomplete config.
        module = DecisionModule(incomplete_config)
        # Verify that specified values were read and default values were used for missing keys.
        self.assertEqual(module.familiarity_threshold, 0.9) # Should be read from the provided incomplete config
        self.assertEqual(module.audio_energy_threshold, 1000.0) # Default for missing key
        self.assertEqual(module.visual_edges_threshold, 50.0) # Default
        self.assertEqual(module.brightness_threshold_high, 200.0) # Default
        self.assertEqual(module.brightness_threshold_low, 50.0) # Default
        self.assertEqual(module.concept_recognition_threshold, 0.85) # Default
        self.assertEqual(module.curiosity_threshold, 5.0) # Default
        self.assertEqual(module.curiosity_increment_new, 1.0) # Default
        self.assertEqual(module.curiosity_decrement_familiar, 0.5) # Default
        self.assertEqual(module.curiosity_decay, 0.1) # Default
        self.assertEqual(module.curiosity_level, 0.0)


    def test_init_with_invalid_config_types(self):
        """Tests initialization with configuration values of invalid types (defaults should be used)."""
        # Provide a config with invalid types for some values.
        # The get_config_value calls with expected_type checks should return their default values.
        invalid_type_config = {
            'cognition': { # Must include cognition key for path to work
                'audio_energy_threshold': "not a float", # Invalid type -> default 1000.0 should be used
                'visual_edges_threshold': 60, # Valid int -> should be read and converted to float by init code
                'brightness_threshold_high': [250], # Invalid type -> default 200.0 should be used
                'brightness_threshold_low': 30.0, # Valid float -> should be read
            }
        }
        # get_config_value logs a WARNING on type mismatch and returns default.
        # __init__'s internal float casting doesn't affect this if default is already float.
        module = DecisionModule(invalid_type_config)
        # Verify default values were used due to type mismatch and valid values were read.
        self.assertEqual(module.audio_energy_threshold, 1000.0) # Default due to invalid type
        self.assertIsInstance(module.audio_energy_threshold, float)
        self.assertEqual(module.visual_edges_threshold, 60.0) # Read as int 60, converted to float 60.0
        self.assertIsInstance(module.visual_edges_threshold, float)
        self.assertEqual(module.brightness_threshold_high, 200.0) # Default due to invalid type
        self.assertIsInstance(module.brightness_threshold_high, float)
        self.assertEqual(module.brightness_threshold_low, 30.0) # Read as float 30.0
        self.assertIsInstance(module.brightness_threshold_low, float)

        # Check other attributes (defaults should apply if not in invalid_type_config)
        self.assertEqual(module.familiarity_threshold, 0.8)
        self.assertEqual(module.concept_recognition_threshold, 0.85)
        self.assertEqual(module.curiosity_threshold, 5.0)
        self.assertEqual(module.curiosity_increment_new, 1.0)
        self.assertEqual(module.curiosity_decrement_familiar, 0.5)
        self.assertEqual(module.curiosity_decay, 0.1)
        self.assertEqual(module.curiosity_level, 0.0)


    def test_init_thresholds_out_of_range(self):
        """Tests when some thresholds are set outside the 0.0-1.0 range."""
        # The __init__ method's internal range checks should reset the value to the default 0.7.
        # Provide configs with out-of-range values and the expected nested structure.
        config_high = {'cognition': {'familiarity_threshold': 1.5, 'concept_recognition_threshold': 1.1, 'audio_energy_threshold': -10.0, 'brightness_threshold_high': 260.0, 'brightness_threshold_low': -10.0, 'curiosity_increment_new': -5.0}} # Provide relevant out-of-range values
        module_high = DecisionModule(config_high)
        self.assertEqual(module_high.familiarity_threshold, 0.8) # Should be reset
        self.assertEqual(module_high.concept_recognition_threshold, 0.85) # Should be reset
        self.assertEqual(module_high.audio_energy_threshold, 1000.0) # Should be reset
        self.assertEqual(module_high.brightness_threshold_high, 200.0) # Should be reset
        self.assertEqual(module_high.brightness_threshold_low, 50.0) # Should be reset (negative)
        self.assertEqual(module_high.curiosity_increment_new, 1.0) # Should be reset (negative)

        # Test the low > high brightness case
        config_low_high_swap = {'cognition': {'brightness_threshold_low': 100.0, 'brightness_threshold_high': 80.0}} # lower > higher
        module_low_high_swap = DecisionModule(config_low_high_swap)
        self.assertEqual(module_low_high_swap.brightness_threshold_low, 50.0) # Both should be reset to defaults
        self.assertEqual(module_low_high_swap.brightness_threshold_high, 200.0)

        # Test boundary values - they should be accepted.
        config_at_boundaries = {'cognition': {'familiarity_threshold': 0.0, 'concept_recognition_threshold': 1.0, 'curiosity_threshold': 0.0, 'curiosity_decrement_familiar': 0.0, 'curiosity_decay': 0.0}} # Test 0 boundaries
        module_boundaries = DecisionModule(config_at_boundaries)
        self.assertEqual(module_boundaries.familiarity_threshold, 0.0) # Boundary should be accepted
        self.assertEqual(module_boundaries.concept_recognition_threshold, 1.0) # Boundary should be accepted
        self.assertEqual(module_boundaries.curiosity_threshold, 0.0) # Boundary should be accepted
        self.assertEqual(module_boundaries.curiosity_decrement_familiar, 0.0) # Boundary should be accepted
        self.assertEqual(module_boundaries.curiosity_decay, 0.0) # Boundary should be accepted


    # --- decide Input Validation Tests ---
    # These tests verify that decide handles invalid input signals gracefully.

    def test_decide_input_none(self):
        """Tests the decide method when understanding_signals is None."""
        initial_curiosity = self.module.curiosity_level # 0.0
        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(None, [], []) # Add empty list for current_concepts
        self.assertIsNone(result)
        # Curiosity level should not be updated if decision is None (handled by 'if decision is not None' in finally).
        self.assertEqual(self.module.curiosity_level, initial_curiosity)


    def test_decide_input_not_dict(self):
        """Tests the decide method when understanding_signals is not a dictionary."""
        initial_curiosity = self.module.curiosity_level # 0.0
        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide("not a dict", [], []) # Add empty list for current_concepts
        self.assertIsNone(result)
        # Curiosity level should not be updated if decision is None.
        self.assertEqual(self.module.curiosity_level, initial_curiosity)


    def test_decide_input_empty_dict(self):
        """Tests the decide method when understanding_signals is an empty dictionary (defaults should be used)."""
        # An empty dict means all flags are False and scores are 0.0. No high-priority thresholds are met.
        # The fundamental state becomes 'new'. The final decision should be "new_input_detected".
        # Curiosity level: Starts at 0.0 -> +increment due to "new_input_detected" -> -decay.
        initial_curiosity = 0.0
        self.module.curiosity_level = initial_curiosity # Ensure starting value
        # Expected curiosity after update: initial + increment - decay
        expected_curiosity = initial_curiosity + self.module.curiosity_increment_new - self.module.curiosity_decay
        expected_curiosity = max(0.0, expected_curiosity) # Curiosity cannot be negative.

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide({}, [], []) # Pass empty dict for signals, empty lists for others.

        # The fallback decision should be "new_input_detected".
        self.assertEqual(result, "new_input_detected") # Fallback is now "new_input_detected".
        # Curiosity level should have been updated (inc + decay).
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    # --- decide Decision Priority Tests ---
    # These tests verify that the decision logic follows the specified priority order.
    # Curiosity levels are started at 0.0 in setUp, and expected curiosity is calculated including inc/dec and decay based on the decision.

    @patch('random.choice', return_value='explore_randomly') # Mock random.choice to control curiosity decision
    def test_decide_priority_curiosity_threshold(self, mock_random_choice):
        """Tests the decide method when the curiosity threshold is exceeded."""
        initial_curiosity = self.module.curiosity_threshold + 1.0 # Set curiosity above threshold (e.g., 6.0)
        self.module.curiosity_level = initial_curiosity

        # Other signals should not trigger any higher priority decision than curiosity.
        signals = {
            'similarity_score': 0.1, # Below memory familiarity threshold
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.1, # Below concept recognition threshold
            'most_similar_concept_id': None,
        }

        # Curiosity level update: Start > threshold -> Decision is 'explore_randomly' (mocked) -> This decision type doesn't trigger inc/dec -> Only decay applies.
        expected_curiosity = initial_curiosity - self.module.curiosity_decay # e.g., 6.0 - 0.1 = 5.9
        expected_curiosity = max(0.0, expected_curiosity) # Ensure curiosity is not negative.

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        self.assertEqual(result, 'explore_randomly') # Should be the value returned by mocked random.choice
        mock_random_choice.assert_called_once_with(["explore_randomly", "make_noise"]) # Verify random.choice was called with correct options
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) # Curiosity should only decay.


    def test_decide_priority_sound_detected(self):
        """Tests the decide method when high audio energy is detected."""
        # Ensure curiosity is below the threshold so the curiosity decision is not triggered.
        initial_curiosity = self.module.curiosity_threshold - 1.0 # e.g., 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': 0.1, # Below memory familiarity threshold
            'high_audio_energy': True, # Audio detected - high priority
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.9, # Above concept recognition threshold (but audio has higher priority)
            'most_similar_concept_id': 1,
        }

        # Curiosity level update: Start < threshold -> Decision is "sound_detected" -> This decision type doesn't trigger inc/dec -> Only decay applies.
        expected_curiosity = initial_curiosity - self.module.curiosity_decay # e.g., 4.0 - 0.1 = 3.9
        expected_curiosity = max(0.0, expected_curiosity)

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        self.assertEqual(result, "sound_detected")
        # Curiosity should only decay.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_complex_visual_detected(self):
        """Tests the decide method when high visual edge density is detected."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # e.g., 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': 0.1, # Below memory familiarity threshold
            'high_audio_energy': False, # No audio
            'high_visual_edges': True, # Visual edge detected - priority
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.9, # Above concept recognition threshold (but visual edge has higher priority)
            'most_similar_concept_id': 1,
        }

        # Curiosity level update: Start < threshold -> Decision is "complex_visual_detected" -> Doesn't trigger inc/dec -> Only decay applies.
        expected_curiosity = initial_curiosity - self.module.curiosity_decay # e.g., 4.0 - 0.1 = 3.9
        expected_curiosity = max(0.0, expected_curiosity)

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        self.assertEqual(result, "complex_visual_detected")
        # Curiosity should only decay.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_bright_light_detected(self):
        """Tests the decide method when a bright environment is detected."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # e.g., 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': 0.1, # Below memory familiarity threshold
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': True, # Bright detected - priority
            'is_dark': False,
            'max_concept_similarity': 0.9, # Above concept recognition threshold (but brightness has higher priority)
            'most_similar_concept_id': 1,
        }

        # Curiosity level update: Start < threshold -> Decision is "bright_light_detected" -> Doesn't trigger inc/dec -> Only decay applies.
        expected_curiosity = initial_curiosity - self.module.curiosity_decay # e.g., 4.0 - 0.1 = 3.9
        expected_curiosity = max(0.0, expected_curiosity)

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        self.assertEqual(result, "bright_light_detected")
        # Curiosity should only decay.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_dark_environment_detected(self):
        """Tests the decide method when a dark environment is detected."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # e.g., 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': 0.1, # Below memory familiarity threshold
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': True, # Dark detected - priority
            'max_concept_similarity': 0.9, # Above concept recognition threshold (but darkness has higher priority)
            'most_similar_concept_id': 1,
        }

        # Curiosity level update: Start < threshold -> Decision is "dark_environment_detected" -> Doesn't trigger inc/dec -> Only decay applies.
        expected_curiosity = initial_curiosity - self.module.curiosity_decay # e.g., 4.0 - 0.1 = 3.9
        expected_curiosity = max(0.0, expected_curiosity)

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        self.assertEqual(result, "dark_environment_detected")
        # Curiosity should only decay.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_recognized_concept(self):
        """Tests the decide method when a concept is recognized."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # e.g., 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': 0.1, # Below memory familiarity threshold
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': self.module.concept_recognition_threshold + 0.01, # Above concept recognition threshold (e.g., 0.86)
            'most_similar_concept_id': 42, # Concept ID exists - priority (after Process signals)
        }

        # Curiosity level update: Start < threshold -> Decision is "recognized_concept_42" -> Triggers decrement -> Decay applies.
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay # e.g., 4.0 - 0.5 - 0.1 = 3.4
        expected_curiosity = max(0.0, expected_curiosity) # Ensure curiosity is not negative.

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        self.assertEqual(result, "recognized_concept_42")
        # Curiosity should decrease and decay.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_recognized_concept_at_threshold(self):
        """Tests the decide method when concept recognition similarity score is exactly at the threshold."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # e.g., 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': self.module.concept_recognition_threshold, # Exactly at threshold (0.85)
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': self.module.concept_recognition_threshold, # Exactly at threshold (0.85)
            'most_similar_concept_id': 99, # Concept ID exists
        }

        # If similarity is equal to the threshold, it should still be recognized (using >=).
        # Curiosity level update: Start < threshold -> Decision is "recognized_concept_99" -> Triggers decrement -> Decay applies.
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay # e.g., 4.0 - 0.5 - 0.1 = 3.4
        expected_curiosity = max(0.0, expected_curiosity)

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        self.assertEqual(result, "recognized_concept_99")
        # Curiosity should decrease and decay.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_recognized_concept_similarity_below_threshold(self):
        """Tests the decide method when concept recognition similarity score is below the threshold."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # e.g., 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': 0.1, # Below memory familiarity threshold
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': self.module.concept_recognition_threshold - 0.01, # Below threshold (e.g., 0.84)
            'most_similar_concept_id': 123, # ID exists but similarity is low
        }

        # Concept recognition condition is not met. Decision should fall through to the next priority condition (Memory Familiarity or New).
        # similarity_score is 0.1, familiarity_threshold is 0.8 -> 0.1 < 0.8 -> Fundamental state is is_fundamentally_new.
        # Decision should be "new_input_detected".
        # Curiosity level update: Start < threshold -> Decision is "new_input_detected" -> Triggers increment -> Decay applies.
        expected_curiosity = initial_curiosity + self.module.curiosity_increment_new - self.module.curiosity_decay # e.g., 4.0 + 1.0 - 0.1 = 4.9
        expected_curiosity = max(0.0, expected_curiosity)

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        # Fallback to "new_input_detected" as no higher priority was detected.
        self.assertEqual(result, "new_input_detected")
        # Curiosity should increase and decay.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_familiar_input_detected(self):
        """Tests the decide method when memory similarity score exceeds the threshold."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # e.g., 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': self.module.familiarity_threshold + 0.01, # Above familiarity threshold (e.g., 0.81)
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.1, # Below concept recognition threshold
            'most_similar_concept_id': None, # No ID
        }

        # All previous priorities are False. similarity_score >= familiarity_threshold is True -> "familiar_input_detected"
        # Curiosity level update: Start < threshold -> Decision is "familiar_input_detected" -> Triggers decrement -> Decay applies.
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay # e.g., 4.0 - 0.5 - 0.1 = 3.4
        expected_curiosity = max(0.0, expected_curiosity)

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        self.assertEqual(result, "familiar_input_detected")
        # Curiosity should decrease and decay.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_familiar_input_at_threshold(self):
        """Tests the decide method when memory similarity score is exactly at the threshold."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # e.g., 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': self.module.familiarity_threshold, # Exactly at threshold (0.8)
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.1, # Below concept recognition threshold
            'most_similar_concept_id': None, # No ID
        }

        # If similarity is equal to the threshold, it should be considered familiar (using >=).
        # Curiosity level update: Start < threshold -> Decision is "familiar_input_detected" -> Triggers decrement -> Decay applies.
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay # e.g., 4.0 - 0.5 - 0.1 = 3.4
        expected_curiosity = max(0.0, expected_curiosity)

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        self.assertEqual(result, "familiar_input_detected")
        # Curiosity should decrease and decay.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_new_input_detected(self):
        """Tests the decide method when no higher priority or familiar condition is met."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # e.g., 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': self.module.familiarity_threshold - 0.01, # Below familiarity threshold (e.g., 0.79)
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.1, # Below concept recognition threshold
            'most_similar_concept_id': None, # No ID
        }

        # All previous priorities are False. similarity_score < familiarity_threshold is True -> Fundamental state 'new'.
        # Decision falls through to the default "new_input_detected".
        # Curiosity level update: Start < threshold -> Decision is "new_input_detected" -> Triggers increment -> Decay applies.
        expected_curiosity = initial_curiosity + self.module.curiosity_increment_new - self.module.curiosity_decay # e.g., 4.0 + 1.0 - 0.1 = 4.9
        expected_curiosity = max(0.0, expected_curiosity)

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        # Fallback to "new_input_detected" as no higher priority was detected.
        self.assertEqual(result, "new_input_detected")
        # Curiosity should increase and decay.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    # --- Curiosity Update Tests ---
    # These tests verify the logic for updating the curiosity level.
    # These tests overlap with the priority tests above but focus specifically on curiosity update.
    # They are kept for clarity and isolation of curiosity logic testing.

    def test_curiosity_update_new_input(self):
        """Tests that the 'new_input_detected' decision increments curiosity and applies decay."""
        initial_curiosity = 1.0 # Starting curiosity level
        self.module.curiosity_level = initial_curiosity

        signals = { # Signals that lead to a "new_input_detected" decision (all priorities False, sim < threshold)
            'similarity_score': self.module.familiarity_threshold - 0.01, # 0.79
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }

        # Expected curiosity: initial + increment - decay
        expected_curiosity = initial_curiosity + self.module.curiosity_increment_new - self.module.curiosity_decay # 1.0 + 1.0 - 0.1 = 1.9
        expected_curiosity = max(0.0, expected_curiosity) # Curiosity cannot be negative.

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        self.assertEqual(result, "new_input_detected") # Decision should be "new_input_detected"
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) # Curiosity should increment and decay.


    def test_curiosity_update_familiar_input(self):
        """Tests that the 'familiar_input_detected' decision decrements curiosity and applies decay (not below zero)."""
        initial_curiosity = 1.0 # Starting curiosity level
        self.module.curiosity_level = initial_curiosity

        signals = { # Signals that lead to a "familiar_input_detected" decision (priorities False, sim >= threshold)
            'similarity_score': self.module.familiarity_threshold + 0.01, # 0.81
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }

        # Expected curiosity: initial - decrement - decay
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay # 1.0 - 0.5 - 0.1 = 0.4
        expected_curiosity = max(0.0, expected_curiosity) # Ensure curiosity is not negative.

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        self.assertEqual(result, "familiar_input_detected") # Decision should be "familiar_input_detected"
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) # Curiosity should decrease and decay.


    def test_curiosity_update_recognized_concept(self):
        """Tests that the 'recognized_concept_X' decision decrements curiosity and applies decay (not below zero)."""
        initial_curiosity = 1.0 # Starting curiosity level
        self.module.curiosity_level = initial_curiosity

        signals = { # Signals that lead to a "recognized_concept_X" decision (process false, sim < threshold, concept_sim >= threshold)
            'similarity_score': 0.1, # Ensure memory sim is below threshold so familiar is not triggered
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': self.module.concept_recognition_threshold + 0.01, 'most_similar_concept_id': 77, # 0.86
        }

        # Curiosity level update: Start < threshold -> Decision is "recognized_concept_77" -> Triggers decrement -> Decay applies.
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay # 1.0 - 0.5 - 0.1 = 0.4
        expected_curiosity = max(0.0, expected_curiosity) # Ensure curiosity is not negative.

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        self.assertEqual(result, "recognized_concept_77") # Decision should be "recognized_concept_77"
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) # Curiosity should decrease and decay.


    def test_curiosity_update_other_decisions_only_decay(self):
        """Tests that process-based decisions (audio, visual, etc.) only apply decay to curiosity."""
        initial_curiosity = 1.0 # Starting curiosity level
        self.module.curiosity_level = initial_curiosity

        signals = { # Signals that lead to a "sound_detected" decision (high priority, others false)
            'similarity_score': 0.1,
            'high_audio_energy': True, # This decision is triggered
            'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }

        # Expected curiosity: initial - decay (no inc/dec)
        expected_curiosity = initial_curiosity - self.module.curiosity_decay # 1.0 - 0.1 = 0.9
        expected_curiosity = max(0.0, expected_curiosity) # Ensure curiosity is not negative.

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        self.assertEqual(result, "sound_detected") # Decision should be correct
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) # Curiosity should only decay.

        # The same behavior is expected for other process-based decisions.
        self.module.curiosity_level = initial_curiosity # Reset for next check
        signals_visual = { 'high_audio_energy': False, 'high_visual_edges': True, 'is_bright': False, 'is_dark': False, 'similarity_score': 0.1, 'max_concept_similarity': 0.1, 'most_similar_concept_id': None, }
        # Corrected: Provide the missing 'current_concepts' argument.
        result_visual = self.module.decide(signals_visual, [], []) # Add empty list

        self.assertEqual(result_visual, "complex_visual_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) # Should only decay (1.0 -> 0.9)

        self.module.curiosity_level = initial_curiosity # Reset
        signals_bright = { 'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': True, 'is_dark': False, 'similarity_score': 0.1, 'max_concept_similarity': 0.1, 'most_similar_concept_id': None, }
        # Corrected: Provide the missing 'current_concepts' argument.
        result_bright = self.module.decide(signals_bright, [], []) # Add empty list

        self.assertEqual(result_bright, "bright_light_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) # Should only decay (1.0 -> 0.9)


        self.module.curiosity_level = initial_curiosity # Reset
        signals_dark = { 'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': True, 'similarity_score': 0.1, 'max_concept_similarity': 0.1, 'most_similar_concept_id': None, }
        # Corrected: Provide the missing 'current_concepts' argument.
        result_dark = self.module.decide(signals_dark, [], []) # Add empty list

        self.assertEqual(result_dark, "dark_environment_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) # Should only decay (1.0 -> 0.9)


    @patch('random.choice', return_value='explore_randomly') # Mock random.choice to control curiosity decision
    def test_curiosity_update_explore_randomly_only_decay(self, mock_random_choice):
        """Tests that the 'explore_randomly' decision only applies decay to curiosity."""
        initial_curiosity = self.module.curiosity_threshold + 1.0 # Set curiosity above threshold (e.g., 6.0)
        self.module.curiosity_level = initial_curiosity

        signals = { # Signals that will trigger the curiosity decision (others are low priority)
            'similarity_score': 0.1,
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }

        # Expected curiosity: initial - decay (no inc/dec)
        expected_curiosity = initial_curiosity - self.module.curiosity_decay # e.g., 6.0 - 0.1 = 5.9
        expected_curiosity = max(0.0, expected_curiosity)

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        self.assertEqual(result, "explore_randomly") # Should be the value returned by mocked random.choice
        mock_random_choice.assert_called_once_with(["explore_randomly", "make_noise"]) # Verify random.choice was called with correct options
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) # Curiosity should only decay.


    def test_curiosity_does_not_go_below_zero(self):
        """Tests that the curiosity level does not go below zero."""
        initial_curiosity = 0.1 # Very low starting value
        self.module.curiosity_level = initial_curiosity

        signals = { # Signals that lead to a "familiar_input_detected" decision (triggers decrement)
            'similarity_score': self.module.familiarity_threshold + 0.01, # 0.81
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }

        # Expected curiosity: initial (0.1) - decrement (0.5) - decay (0.1) = -0.5. Should be capped at 0.0.
        expected_curiosity = max(0.0, initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay)

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        self.assertEqual(result, "familiar_input_detected") # Decision should be familiar
        self.assertAlmostEqual(self.module.curiosity_level, 0.0, places=6) # Should be capped at zero.


    # --- Exception Handling Test ---

    # Mock random.choice (called when curiosity threshold is exceeded) to raise an error during decide.
    @patch('random.choice', side_effect=RuntimeError("Simulated Decision Error"))
    def test_decide_exception_handling_during_decision_logic(self, mock_random_choice):
        """Tests that the decide method returns None if an error occurs during decision making logic."""
        initial_curiosity_for_test = self.module.curiosity_threshold + 1.0 # Set curiosity above threshold to ensure random.choice is called (e.g., 6.0)
        self.module.curiosity_level = initial_curiosity_for_test

        signals = { # Signals that will trigger the curiosity decision (others are low priority)
            'similarity_score': 0.1,
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }

        # Expectation: An error will be raised during decision making (due to the mock).
        # The except block will catch it and return None.
        # The finally block will run. The 'if decision is not None' check in finally will be false (as decision remained None).
        # Thus, curiosity will NOT be updated (no increment/decrement or decay). The initial curiosity level should remain unchanged.
        # Let's confirm the finally block logic: yes, update happens inside `if decision is not None:`.

        # Corrected: Provide the missing 'current_concepts' argument.
        result = self.module.decide(signals, [], []) # Add empty list for current_concepts

        self.assertIsNone(result) # Should return None on error
        mock_random_choice.assert_called_once() # Verify the mocked function was called

        # Verify curiosity level remains unchanged (no update due to error).
        self.assertEqual(self.module.curiosity_level, initial_curiosity_for_test)


    # --- cleanup Test ---
    # This test verifies the cleanup method runs without raising an exception.

    def test_cleanup(self):
        """Tests that the cleanup method runs without issues (currently just logs)."""
        # cleanup currently doesn't change state that can be easily asserted, just logs.
        # We can use mocking to verify that the logger.info method is called.
        with patch('src.cognition.decision.logger.info') as mock_logger_info:
             self.module.cleanup() # Call the cleanup method
             # Verify the correct log message was called.
             mock_logger_info.assert_called_with("DecisionModule object cleaning up.")
        # Curiosity level is reset in setUp and after tearDown by the fixture,
        # and cleanup doesn't actually change its state currently.