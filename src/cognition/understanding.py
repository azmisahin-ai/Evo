# src/cognition/understanding.py
#
# Evo'nın anlama modülünü temsil eder.
# İşlenmiş temsilleri, bellek girdilerini, anlık duyu özelliklerini ve öğrenilmiş kavramları kullanarak dünyayı anlamaya çalışır.

import logging
import numpy as np # For similarity calculation, array operations and mean

# Import utility functions (input checks from src/core/utils, config from src/core/config_utils)
try:
    from src.core.utils import check_input_not_none, check_input_type, check_numpy_input
    from src.core.config_utils import get_config_value
except ImportError as e:
    logging.critical(f"Fundamental utility modules could not be imported: {e}. Please ensure src/core/utils.py and src/core/config_utils.py exist and PYTHONPATH is configured correctly.")
    # Placeholder functions (only used in case of import error)
    def get_config_value(config, *keys, default=None, expected_type=None, logger_instance=None):
         return default
    def check_input_not_none(input_data, input_name="Input", logger_instance=None):
         return input_data is not None
    def check_input_type(input_data, expected_type, input_name="Input", logger_instance=None):
         return isinstance(input_data, expected_type)
    def check_numpy_input(input_data, expected_dtype=None, expected_ndim=None, input_name="Input", logger_instance=None):
         return isinstance(input_data, np.ndarray)


# Create a logger for this module
logger = logging.getLogger(__name__)

class UnderstandingModule:
    """
    Evo's understanding capability module class (Phase 3/4 implementation).

    Receives Representation from RepresentationLearner, relevant memory entries from Memory,
    instantaneous sensory features from Process (low-level features), and learned concept
    representatives from LearningModule. It uses these inputs to generate a primitive
    "understanding" output (currently a dictionary).
    Current implementation: Calculates the highest memory similarity score, boolean flags
    based on instantaneous sensory features, AND the highest similarity to learned concept
    representatives.
    More complex understanding algorithms will be implemented in the future.
    """
    def __init__(self, config):
        """
        Initializes the UnderstandingModule.

        Args:
            config (dict): Full configuration settings for the system.
                           UnderstandingModule will read its relevant sections from this dict,
                           specifically settings under 'cognition'.
                           'audio_energy_threshold': Threshold for detecting high audio energy (float, default 1000.0).
                           'visual_edges_threshold': Threshold for detecting high visual edge density (float, default 50.0).
                           'brightness_threshold_high': Threshold for detecting a bright environment (float, default 200.0).
                           'brightness_threshold_low': Threshold for detecting a dark environment (float, default 50.0).
        """
        self.config = config # UnderstandingModule receives the full config
        logger.info("UnderstandingModule initializing (Phase 3/4)...")

        # Get thresholds from config using get_config_value with keyword arguments.
        # Based on config, these settings are directly under the 'cognition' key.
        self.audio_energy_threshold = get_config_value(config, 'cognition', 'audio_energy_threshold', default=1000.0, expected_type=(float, int), logger_instance=logger)
        self.visual_edges_threshold = get_config_value(config, 'cognition', 'visual_edges_threshold', default=50.0, expected_type=(float, int), logger_instance=logger)
        self.brightness_threshold_high = get_config_value(config, 'cognition', 'brightness_threshold_high', default=200.0, expected_type=(float, int), logger_instance=logger)
        self.brightness_threshold_low = get_config_value(config, 'cognition', 'brightness_threshold_low', default=50.0, expected_type=(float, int), logger_instance=logger)

        # Cast thresholds to float to ensure correct comparison arithmetic later.
        # While expected_type checks type, explicit cast ensures float.
        self.audio_energy_threshold = float(self.audio_energy_threshold)
        self.visual_edges_threshold = float(self.visual_edges_threshold)
        self.brightness_threshold_high = float(self.brightness_threshold_high)
        self.brightness_threshold_low = float(self.brightness_threshold_low)


        logger.info(f"UnderstandingModule initialized. Audio Energy Threshold: {self.audio_energy_threshold}, Visual Edge Threshold: {self.visual_edges_threshold}, Brightness High Threshold: {self.brightness_threshold_high}, Brightness Low Threshold: {self.brightness_threshold_low}")


    def process(self, processed_inputs, learned_representation, relevant_memory_entries, current_concepts):
        """
        Performs the understanding process using incoming Representation, relevant memory entries,
        instantaneous Process outputs, and learned concepts.

        Args:
            processed_inputs (dict or None): Processed instantaneous sensory inputs.
                                            Expected format: {'visual': dict, 'audio': np.ndarray} or None/empty dict.
                                            Visual dict expected keys: 'grayscale' (np.ndarray), 'edges' (np.ndarray).
            learned_representation (numpy.ndarray or None): The latest learned representation vector
                                                         or None if processing failed.
                                                         Expected format: shape (D,), numerical dtype, or None.
            relevant_memory_entries (list or None): List of relevant memory entries retrieved from Memory.
                                            Can be an empty list `[]` if memory is empty or retrieval failed.
                                            Expected format: list of dicts, or None.
            current_concepts (list): List of current concept representative vectors from the LearningModule.
                                  Can be an empty list `[]` if no concepts learned or retrieval failed.
                                  Expected format: list of np.ndarray.

        Returns:
            dict: A dictionary containing understanding signals.
                  E.g., {'similarity_score': float, 'high_audio_energy': bool, 'high_visual_edges': bool,
                       'is_bright': bool, 'is_dark': bool,
                       'max_concept_similarity': float, 'most_similar_concept_id': int or None,
                       ... any other derived signals ...
                      }.
                  Returns a default dictionary of signals on error or if inputs are insufficient.
        """
        # Default understanding signals dictionary
        understanding_signals = {
            'similarity_score': 0.0,
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.0,
            'most_similar_concept_id': None,
            # Add other default signals here as implemented in the module
            # 'novelty_score': 0.0, # Example
            # 'sensory_conflict': False, # Example
        }

        # Input validity checks.
        # check_input_not_none and check_input_type come from src/core/utils.
        # check_numpy_input also comes from src/core/utils.
        is_valid_memory_list = check_input_not_none(relevant_memory_entries, "relevant_memory_entries for UnderstandingModule", logger) and \
                               (isinstance(relevant_memory_entries, list) or relevant_memory_entries is None) # Check for list or None

        # Representation is valid if not None AND is a numpy array AND 1D AND numerical.
        is_valid_representation = check_input_not_none(learned_representation, "learned_representation for UnderstandingModule", logger) and \
                                  check_numpy_input(learned_representation, expected_dtype=np.number, expected_ndim=1, input_name="learned_representation for UnderstandingModule", logger_instance=logger)

        is_valid_processed_inputs = check_input_not_none(processed_inputs, "processed_inputs for UnderstandingModule", logger) and \
                                    isinstance(processed_inputs, dict) # isinstance check is more flexible than check_input_type for dicts


        # Concepts are valid if not None AND is a list.
        is_valid_concepts_list = check_input_not_none(current_concepts, "current_concepts for UnderstandingModule", logger) and \
                                  (isinstance(current_concepts, list) or current_concepts is None) # Check for list or None


        logger.debug(f"UnderstandingModule.process: Understanding process started. Input validity - Repr:{is_valid_representation}, Mem:{is_valid_memory_list}, Proc:{is_valid_processed_inputs}, Concepts:{is_valid_concepts_list}")

        # If Representation is valid, calculate its norm, otherwise skip Representation-based calculations.
        query_norm = 0.0
        if is_valid_representation:
             query_norm = np.linalg.norm(learned_representation)
             if query_norm < 1e-8: # If norm is near zero, similarity is undefined.
                 logger.warning("UnderstandingModule.process: Learned representation has near zero norm. Similarity and Concept recognition will be skipped.")
                 is_valid_representation = False # Treat as invalid for similarity/concept checks

        # This try block catches errors that might arise from operations like numpy calculations or indexing.
        # The check_* utility functions do not raise exceptions when they return False,
        # so this except block will primarily catch errors from np.dot, np.linalg.norm, np.mean, or indexing.
        try:
            # 1. Calculate Memory Similarity Score (If Representation and Memory are valid)
            max_memory_similarity = 0.0
            # Check if memory_entries is a list and not empty before iterating
            if is_valid_representation and isinstance(relevant_memory_entries, list) and relevant_memory_entries:
                for memory_entry in relevant_memory_entries:
                    # Ensure each memory entry is a dictionary before getting the representation.
                    if isinstance(memory_entry, dict):
                         stored_representation = memory_entry.get('representation') # Safe access with .get()
                         # Ensure stored representation is a valid numerical 1D numpy array with correct dimension.
                         if stored_representation is not None \
                            and isinstance(stored_representation, np.ndarray) \
                            and stored_representation.ndim == 1 \
                            and np.issubdtype(stored_representation.dtype, np.number) \
                            and stored_representation.shape[0] == learned_representation.shape[0]: # Dimension must match query

                              stored_norm = np.linalg.norm(stored_representation) # Use np.linalg.norm
                              if stored_norm > 1e-8: # Check if stored vector is near zero to avoid division errors.
                                   # Calculate cosine similarity: (dot product) / (norm1 * norm2)
                                   similarity = np.dot(learned_representation, stored_representation) / (query_norm * stored_norm)
                                   if not np.isnan(similarity): # Ignore NaN similarity scores.
                                        max_memory_similarity = max(max_memory_similarity, similarity)
                              # else: logger.debug("UM.process: Stored rep near zero norm, skipping similarity.")
                         # else: logger.debug("UM.process: Invalid stored rep format/type/shape, skipping.")
                    # else: logger.warning("UM.process: Memory list element is not a dict, skipping.")
                understanding_signals['similarity_score'] = max_memory_similarity


            # 2. Calculate Similarity with Learned Concepts (If Representation and Concepts are valid)
            max_concept_similarity = 0.0
            most_similar_concept_id = None

            # Check if concepts list is a list and not empty before iterating.
            if is_valid_representation and isinstance(current_concepts, list) and current_concepts:
                 for i, concept_rep in enumerate(current_concepts):
                      # Ensure each concept representative is a valid numerical 1D numpy array with correct dimension.
                      if concept_rep is not None \
                         and isinstance(concept_rep, np.ndarray) \
                         and concept_rep.ndim == 1 \
                         and np.issubdtype(concept_rep.dtype, np.number) \
                         and concept_rep.shape[0] == learned_representation.shape[0]: # Dimension must match query

                           concept_norm = np.linalg.norm(concept_rep)
                           if concept_norm > 1e-8: # Check if concept vector is near zero to avoid division errors.
                                similarity = np.dot(learned_representation, concept_rep) / (query_norm * concept_norm)
                                if not np.isnan(similarity): # Ignore NaN similarity scores.
                                     if similarity > max_concept_similarity:
                                          max_concept_similarity = similarity
                                          most_similar_concept_id = i
                           # else: logger.debug("UM.process: Concept rep near zero norm, skipping similarity.")
                      # else: logger.warning("UM.process: Invalid concept rep format/type/shape, skipping.")

                 understanding_signals['max_concept_similarity'] = max_concept_similarity
                 understanding_signals['most_similar_concept_id'] = most_similar_concept_id


            # 3. Simple Understanding Based on Process Outputs (If Processed Inputs are valid)
            # check_numpy_input returns False on failure and doesn't raise exception, this is correct.
            # Exceptions might come from indexing or np.mean operations.
            if is_valid_processed_inputs:
                 # Audio Energy Check
                 audio_features = processed_inputs.get('audio')
                 # check_numpy_input already checks dtype. Check for numpy array, 1D, and sufficient size.
                 if isinstance(audio_features, np.ndarray) and \
                    check_numpy_input(audio_features, expected_dtype=np.number, expected_ndim=1, input_name="processed_inputs['audio']", logger_instance=logger) and \
                    audio_features.shape[0] >= 1: # Check size > 0 like audio_features.shape[0] >= 1
                      try:
                           audio_energy = float(audio_features[0]) # Could raise IndexError for empty array
                           if audio_energy >= self.audio_energy_threshold:
                                understanding_signals['high_audio_energy'] = True
                                logger.debug(f"UnderstandingModule.process: High Audio Energy Detected: {audio_energy:.4f} >= Threshold {self.audio_energy_threshold:.4f}")
                           # Add else logging for energy below threshold? Or only log when above? Policy decision.
                           # else: logger.debug(f"UnderstandingModule.process: Audio Energy Below Threshold: {audio_energy:.4f} < Threshold {self.audio_energy_threshold:.4f}")
                      except IndexError:
                           logger.warning("UnderstandingModule.process: Processed audio features array is empty or too small for energy check.")
                      except Exception as ex:
                           logger.error(f"UnderstandingModule.process: Error processing audio energy: {ex}", exc_info=True)


                 # Visual Edge Density Check
                 visual_features = processed_inputs.get('visual')
                 if isinstance(visual_features, dict):
                      edges_data = visual_features.get('edges')
                      # check_numpy_input already checks dtype. Check for numpy array, 2D, and non-empty size.
                      if isinstance(edges_data, np.ndarray) and \
                         check_numpy_input(edges_data, expected_dtype=np.number, expected_ndim=(1,2), input_name="processed_inputs['visual']['edges']", logger_instance=logger) and \
                         edges_data.size > 0: # size > 0 is important for mean calculation
                           try:
                                visual_edges_mean = np.mean(edges_data) # Could raise error for empty array
                                if visual_edges_mean >= self.visual_edges_threshold:
                                     understanding_signals['high_visual_edges'] = True
                                     # logger.debug(...) # Log already done in VisionProcessor? Or log here again? Policy.
                           except Exception as ex:
                                logger.error(f"UnderstandingModule.process: Error processing visual edges: {ex}", exc_info=True)


                 # Visual Brightness/Darkness Check
                 if isinstance(visual_features, dict):
                      grayscale_data = visual_features.get('grayscale')
                      # check_numpy_input already checks dtype. Check for numpy array, 2D, and non-empty size.
                      if isinstance(grayscale_data, np.ndarray) and \
                         check_numpy_input(grayscale_data, expected_dtype=np.number, expected_ndim=(1,2), input_name="processed_inputs['visual']['grayscale']", logger_instance=logger) and \
                         grayscale_data.size > 0: # size > 0 is important for mean calculation
                           try:
                                visual_brightness_mean = np.mean(grayscale_data) # Could raise error for empty array

                                if visual_brightness_mean >= self.brightness_threshold_high:
                                     understanding_signals['is_bright'] = True
                                     # logger.debug(...) # Policy on where to log this.
                                elif visual_brightness_mean <= self.brightness_threshold_low:
                                     understanding_signals['is_dark'] = True
                                     # logger.debug(...) # Policy on where to log this.
                                # Add else logging for brightness between thresholds? Policy.
                                # else: logger.debug(f"UnderstandingModule.process: Brightness within thresholds: {visual_brightness_mean:.2f}")
                           except Exception as ex:
                                logger.error(f"UnderstandingModule.process: Error processing visual brightness: {ex}", exc_info=True)


            logger.debug(f"UnderstandingModule.process: Generated understanding signals: {understanding_signals}")

        except Exception as e:
            # Catch any unexpected error during calculation or input processing within the try block.
            # This block is NOT triggered when check_* functions return False.
            # It's triggered by errors in numpy operations (norm, dot, mean) or indexing.
            logger.error(f"UnderstandingModule.process: Unexpected error during understanding process: {e}", exc_info=True)
            # In case of error, return the default understanding signals dictionary.
            return {
                'similarity_score': 0.0,
                'high_audio_energy': False,
                'high_visual_edges': False,
                'is_bright': False,
                'is_dark': False,
                'max_concept_similarity': 0.0,
                'most_similar_concept_id': None,
                # Include defaults for any other implemented signals
            }

        return understanding_signals

    def cleanup(self):
        """
        Cleans up UnderstandingModule resources.

        Currently, this module does not use specific resources and does not require
        a cleanup step.
        Includes an informational log.
        Called by module_loader.py when the program terminates (if it exists).
        """
        logger.info("UnderstandingModule object cleaning up.")
        pass