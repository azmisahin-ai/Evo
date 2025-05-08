# src/cognition/decision.py
#
# Evo's decision making module.
# Makes an action decision using signals from the understanding module, memory entries, and internal state.

import logging
import numpy as np
import random

# Import utility functions (input checks from src/core/utils, config from src/core/config_utils)
# check_* functions come from src.core.utils
# get_config_value comes from src.core.config_utils
# Assuming these imports are successful.
from src.core.utils import check_input_not_none, check_input_type
from src.core.config_utils import get_config_value

logger = logging.getLogger(__name__)

class DecisionModule:
    """
    Evo's decision making capability class (Phase 3/4 implementation).

    Receives signals from the Understanding module, memory entries, and internal state (curiosity level).
    Applies priority logic to these signals to select an action decision.
    Updates its internal state (e.g., curiosity level) based on the decision made.
    """
    def __init__(self, config):
        """
        Initializes the DecisionModule.

        Reads configuration settings for decision thresholds and curiosity dynamics.

        Args:
            config (dict): Cognitive core configuration settings (full config dict).
                           DecisionModule will read its relevant section from this dict,
                           specifically settings under 'cognition'.
        """
        self.config = config # DecisionModule receives the full config
        logger.info("DecisionModule initializing (Phase 3/4)...")

        # Get thresholds and curiosity settings from config using get_config_value.
        # These settings are under the 'cognition' key in the main config.
        # Ensure all these attributes are stored as floats.
        self.familiarity_threshold = float(get_config_value(config, 'cognition', 'familiarity_threshold', default=0.8, expected_type=(float, int), logger_instance=logger))
        self.audio_energy_threshold = float(get_config_value(config, 'cognition', 'audio_energy_threshold', default=1000.0, expected_type=(float, int), logger_instance=logger))
        self.visual_edges_threshold = float(get_config_value(config, 'cognition', 'visual_edges_threshold', default=50.0, expected_type=(float, int), logger_instance=logger))
        self.brightness_threshold_high = float(get_config_value(config, 'cognition', 'brightness_threshold_high', default=200.0, expected_type=(float, int), logger_instance=logger))
        self.brightness_threshold_low = float(get_config_value(config, 'cognition', 'brightness_threshold_low', default=50.0, expected_type=(float, int), logger_instance=logger))
        self.concept_recognition_threshold = float(get_config_value(config, 'cognition', 'concept_recognition_threshold', default=0.85, expected_type=(float, int), logger_instance=logger))
        self.curiosity_threshold = float(get_config_value(config, 'cognition', 'curiosity_threshold', default=5.0, expected_type=(float, int), logger_instance=logger))
        self.curiosity_increment_new = float(get_config_value(config, 'cognition', 'curiosity_increment_new', default=1.0, expected_type=(float, int), logger_instance=logger))
        self.curiosity_decrement_familiar = float(get_config_value(config, 'cognition', 'curiosity_decrement_familiar', default=0.5, expected_type=(float, int), logger_instance=logger))
        self.curiosity_decay = float(get_config_value(config, 'cognition', 'curiosity_decay', default=0.1, expected_type=(float, int), logger_instance=logger))


        # Simple value checks for thresholds (0.0-1.0 range for similarity, non-negative for others)
        # Attributes are now floats, so direct comparison is fine.
        if not (0.0 <= self.familiarity_threshold <= 1.0):
             logger.warning(f"DecisionModule: Config 'familiarity_threshold' out of expected range ({self.familiarity_threshold}). Using default 0.8.")
             self.familiarity_threshold = 0.8
        if self.audio_energy_threshold < 0.0:
             logger.warning(f"DecisionModule: Config 'audio_energy_threshold' is negative ({self.audio_energy_threshold}). Using default 1000.0.")
             self.audio_energy_threshold = 1000.0
        if self.visual_edges_threshold < 0.0:
             logger.warning(f"DecisionModule: Config 'visual_edges_threshold' is negative ({self.visual_edges_threshold}). Using default 50.0.")
             self.visual_edges_threshold = 50.0
        # Brightness threshold check: Must be positive and low < high
        if self.brightness_threshold_high < 0.0:
             logger.warning(f"DecisionModule: Config 'brightness_threshold_high' is negative ({self.brightness_threshold_high}). Using default 200.0.")
             self.brightness_threshold_high = 200.0
        if self.brightness_threshold_low < 0.0:
             logger.warning(f"DecisionModule: Config 'brightness_threshold_low' is negative ({self.brightness_threshold_low}). Using default 50.0.")
             self.brightness_threshold_low = 50.0
        # After previous checks, check the current values
        if self.brightness_threshold_low >= self.brightness_threshold_high:
             logger.warning(f"DecisionModule: Config 'brightness_threshold_low' ({self.brightness_threshold_low}) is greater than or equal to 'brightness_threshold_high' ({self.brightness_threshold_high}). Using defaults 50.0 and 200.0.")
             self.brightness_threshold_low = 50.0
             self.brightness_threshold_high = 200.0 # Both low and high reset to ensure low < high.

        if not (0.0 <= self.concept_recognition_threshold <= 1.0):
             logger.warning(f"DecisionModule: Config 'concept_recognition_threshold' out of expected range ({self.concept_recognition_threshold}). Expected between 0.0 and 1.0. Using default 0.85.")
             self.concept_recognition_threshold = 0.85
        # Curiosity threshold and update amounts must not be negative
        if self.curiosity_threshold < 0.0:
             logger.warning(f"DecisionModule: Config 'curiosity_threshold' is negative ({self.curiosity_threshold}). Using default 5.0.")
             self.curiosity_threshold = 5.0
        if self.curiosity_increment_new < 0.0:
             logger.warning(f"DecisionModule: Config 'curiosity_increment_new' is negative ({self.curiosity_increment_new}). Using default 1.0.")
             self.curiosity_increment_new = 1.0
        if self.curiosity_decrement_familiar < 0.0:
             logger.warning(f"DecisionModule: Config 'curiosity_decrement_familiar' is negative ({self.curiosity_decrement_familiar}). Using default 0.5.")
             self.curiosity_decrement_familiar = 0.5
        # Curiosity decay must not be negative
        if self.curiosity_decay < 0.0:
             logger.warning(f"DecisionModule: Config 'curiosity_decay' is negative ({self.curiosity_decay}). Using default 0.1.")
             self.curiosity_decay = 0.1


        # Internal state variables (for now, just curiosity level)
        self.curiosity_level = 0.0 # Curiosity level (starts at 0.0).

        logger.info(f"DecisionModule initialized. Familiarity Threshold: {self.familiarity_threshold}, Audio Threshold: {self.audio_energy_threshold}, Visual Threshold: {self.visual_edges_threshold}, Brightness High Threshold: {self.brightness_threshold_high}, Brightness Low Threshold: {self.brightness_threshold_low}, Concept Recognition Threshold: {self.concept_recognition_threshold}, Curiosity Threshold: {self.curiosity_threshold}")
        logger.debug(f"DecisionModule: Curiosity Increment (New): {self.curiosity_increment_new}, Decrement (Familiar): {self.curiosity_decrement_familiar}, Decay: {self.curiosity_decay}")


    def decide(self, understanding_signals, relevant_memory_entries, current_concepts):
        """
        Makes an action decision based on understanding signals and internal state (curiosity_level).

        Args:
            understanding_signals (dict or None): Dictionary of signals from the Understanding module.
                                                Expected keys: 'similarity_score', 'high_audio_energy', 'high_visual_edges',
                                                               'is_bright', 'is_dark', 'max_concept_similarity', 'most_similar_concept_id'.
                                                Can be None if understanding failed.
            relevant_memory_entries (list or None): List of relevant memory entries. Not directly used in current decision logic
                                                    but passed as context.
                                                    Expected format: list of dicts or None.
            current_concepts (list): List of current concept representative vectors. Not directly used in current decision logic
                                  (concept recognition uses signals), but passed as context.
                                  Expected format: list of np.ndarray.
            # internal_state (any, optional): Placeholder for other internal state information. Defaults to None.

        Returns:
            str or None: The decided action (string) or None if understanding signals are invalid or decision making failed.
                         Decision strings like "sound_detected", "familiar_input_detected", "new_input_detected", etc.
        """
        # Input validation. Are understanding_signals a valid dictionary?
        if not check_input_not_none(understanding_signals, input_name="understanding_signals for DecisionModule", logger_instance=logger):
             logger.debug("DecisionModule.decide: understanding_signals is None. Cannot make a decision.")
             # Curiosity level is not updated if decision is None (handled in finally block).
             return None

        if not isinstance(understanding_signals, dict):
             logger.error(f"DecisionModule.decide: understanding_signals has unexpected type: {type(understanding_signals)}. Dictionary or None expected. Cannot make a decision.")
             # Curiosity level is not updated if decision is None.
             return None


        decision = None # Variable to hold the decision. Starts as None.

        # Safely get understanding signals from the dictionary. Use default values if keys are missing.
        # Use get() method. Values obtained might be None or wrong type if UnderstandingModule had issues,
        # so be defensive in decision logic below.
        similarity_score_input = understanding_signals.get('similarity_score', 0.0)
        high_audio_energy = understanding_signals.get('high_audio_energy', False)
        high_visual_edges = understanding_signals.get('high_visual_edges', False)
        is_bright = understanding_signals.get('is_bright', False)
        is_dark = understanding_signals.get('is_dark', False)
        max_concept_similarity_input = understanding_signals.get('max_concept_similarity', 0.0)
        most_similar_concept_id = understanding_signals.get('most_similar_concept_id', None)
        
        # Ensure numeric inputs are float for reliable comparison
        similarity_score = 0.0
        if isinstance(similarity_score_input, (int, float)):
            similarity_score = float(similarity_score_input)
        else:
            logger.warning(f"DecisionModule.decide: 'similarity_score' received non-numeric value {similarity_score_input}. Defaulting to 0.0.")

        max_concept_similarity = 0.0
        if isinstance(max_concept_similarity_input, (int, float)):
            max_concept_similarity = float(max_concept_similarity_input)
        else:
            logger.warning(f"DecisionModule.decide: 'max_concept_similarity' received non-numeric value {max_concept_similarity_input}. Defaulting to 0.0.")


        logger.debug(f"DecisionModule.decide: Received understanding signals - Sim:{similarity_score:.4f}, Audio:{high_audio_energy}, Visual:{high_visual_edges}, Bright:{is_bright}, Dark:{is_dark}, ConceptSim:{max_concept_similarity:.4f}, ConceptID:{most_similar_concept_id}. Current Curiosity: {self.curiosity_level:.2f}. Making decision.")

        is_fundamentally_familiar = False
        # self.familiarity_threshold is guaranteed to be float
        if similarity_score >= self.familiarity_threshold:
             is_fundamentally_familiar = True

        is_fundamentally_new = not is_fundamentally_familiar


        try:
            # Decision Making Logic (Phase 3/4): Priority-based Logic
            # Priority Order: Curiosity > Audio > Visual Edge > Brightness/Darkness > Concept Recognition > Memory Familiarity > Default (New)

            # Check conditions in order of priority. Use elif so only the first met condition sets the decision.
            # self.curiosity_level and self.curiosity_threshold are guaranteed to be float
            if self.curiosity_level >= self.curiosity_threshold:
                 decision = random.choice(["explore_randomly", "make_noise"])
                 logger.debug(f"DecisionModule.decide: Decision: '{decision}'. Curiosity threshold ({self.curiosity_level:.2f} >= {self.curiosity_threshold:.2f}) exceeded.")

            elif isinstance(high_audio_energy, bool) and high_audio_energy:
                 decision = "sound_detected"
                 logger.debug(f"DecisionModule.decide: Decision: '{decision}'. High audio energy detected.")

            elif isinstance(high_visual_edges, bool) and high_visual_edges:
                 decision = "complex_visual_detected"
                 logger.debug(f"DecisionModule.decide: Decision: '{decision}'. High visual edge density detected.")

            elif isinstance(is_bright, bool) and is_bright:
                 decision = "bright_light_detected"
                 logger.debug(f"DecisionModule.decide: Decision: '{decision}'. Environment detected as bright.")

            elif isinstance(is_dark, bool) and is_dark:
                 decision = "dark_environment_detected"
                 logger.debug(f"DecisionModule.decide: Decision: '{decision}'. Environment detected as dark.")
            
            # self.concept_recognition_threshold is guaranteed to be float
            elif max_concept_similarity >= self.concept_recognition_threshold and most_similar_concept_id is not None:
                 if isinstance(most_similar_concept_id, int):
                      decision = f"recognized_concept_{most_similar_concept_id}"
                      logger.debug(f"DecisionModule.decide: Decision: '{decision}'. Concept recognized (Similarity: {max_concept_similarity:.4f} >= Threshold {self.concept_recognition_threshold:.4f}, ID: {most_similar_concept_id}).")
                 else:
                      logger.debug(f"DecisionModule.decide: Concept recognition similarity high ({max_concept_similarity:.4f}) but ConceptID is not a valid integer ({type(most_similar_concept_id)}). Skipping concept recognition decision.")

            elif is_fundamentally_familiar:
                 decision = "familiar_input_detected"
                 logger.debug(f"DecisionModule.decide: Decision: '{decision}'. Memory similarity score ({similarity_score:.4f}) >= Threshold ({self.familiarity_threshold:.4f}).")

            if decision is None:
                 decision = "new_input_detected" 
                 logger.debug(f"DecisionModule.decide: Decision: '{decision}'. No higher priority condition detected.")

        except Exception as e:
            logger.error(f"DecisionModule.decide: Unexpected error during decision making: {e}", exc_info=True)
            return None

        finally:
            # --- Update Curiosity Level ---
            # self.curiosity_level, self.curiosity_increment_new, self.curiosity_decrement_familiar, self.curiosity_decay are floats.
            if decision is not None: # curiosity_level is always float
                try:
                    curiosity_before_update = self.curiosity_level # Already float

                    if decision == "new_input_detected" or decision == "new_input_detected_fallback":
                         curiosity_before_update += self.curiosity_increment_new
                         logger.debug(f"DecisionModule: Curiosity increment ({self.curiosity_increment_new:.2f}) based on decision: '{decision}'.")
                    elif decision == "familiar_input_detected" or (isinstance(decision, str) and decision.startswith("recognized_concept_")):
                         curiosity_before_update -= self.curiosity_decrement_familiar
                         logger.debug(f"DecisionModule: Curiosity decrement ({self.curiosity_decrement_familiar:.2f}) based on decision: '{decision}'.")
                    else:
                         logger.debug(f"DecisionModule: No curiosity change (only decay). Decision: '{decision}'.")
                    
                    self.curiosity_level = max(0.0, curiosity_before_update - self.curiosity_decay)

                    logger.debug(f"DecisionModule: Current Curiosity Level: {self.curiosity_level:.2f}")

                except Exception as e:
                     logger.error(f"DecisionModule: Unexpected error while updating curiosity level: {e}", exc_info=True)
        return decision

    def cleanup(self):
        logger.info("DecisionModule object cleaning up.")
        pass