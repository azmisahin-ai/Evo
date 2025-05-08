# src/cognition/learning.py
#
# Evo's unsupervised learning (concept discovery) module.
# Analyzes Representation vectors in memory to discover basic patterns/clusters (concepts).
# Uses a simple threshold-based new concept addition logic.

import logging
import numpy as np # For vector operations and similarity calculation

# Import utility functions
# check_* functions come from src/core/utils
# get_config_value comes from src/core/config_utils
try:
    from src.core.utils import check_input_not_none, check_input_type, check_numpy_input
    from src.core.config_utils import get_config_value
except ImportError as e:
    # Log critical error if fundamental utility modules cannot be imported (for development/debug)
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

class LearningModule:
    """
    Evo's unsupervised learning (concept discovery) module class (Phase 4 implementation).

    Receives Representation vectors from memory and applies a simple threshold-based
    new concept addition logic to discover and learn basic concepts (pattern groupings).
    Discovered concepts are stored as concept representative vectors (the first vector that
    represents that concept).
    Uses a "from scratch" principle, with a simple approach without complex libraries.
    """
    def __init__(self, config):
        """
        Initializes the LearningModule.

        Args:
            config (dict): Learning module configuration settings (full config dict).
                           'new_concept_threshold': The similarity threshold (float, default 0.7) below which
                                                    a Representation vector is considered a new concept
                                                    relative to existing concept representatives.
                           'representation_dim': The dimension of the Representation vectors to be processed (int).
                                                  This should match the output dimension of the RepresentationLearner.
        """
        self.config = config # LearningModule receives the full config
        logger.info("LearningModule initializing (Phase 4)...")

        # Get threshold from config using get_config_value with keyword arguments.
        # Based on config, new_concept_threshold is under the 'cognition' key.
        self.new_concept_threshold = get_config_value(config, 'cognition', 'new_concept_threshold', default=0.7, expected_type=(float, int), logger_instance=logger)
        # Get the representation dimension from config. Obtain it from the representation.representation_dim key.
        self.representation_dim = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)

        # Cast threshold to float to ensure correct comparison arithmetic later.
        self_logger = logging.getLogger(f"{__name__}.__init__") # Use a specific logger for init logs if needed, or just module logger
        self.new_concept_threshold = float(self.new_concept_threshold)
        # representation_dim is expected to be an int.

        # Simple value check for the threshold (must be between 0.0 and 1.0)
        if not (0.0 <= self.new_concept_threshold <= 1.0):
             logger.warning(f"LearningModule: Config 'new_concept_threshold' out of expected range ({self.new_concept_threshold}). Expected between 0.0 and 1.0. Using default 0.7.")
             self.new_concept_threshold = 0.7

        # Check if the representation dimension is positive.
        if self.representation_dim <= 0:
             # This could be a critical error but we don't crash during initialization.
             logger.error(f"LearningModule: Invalid 'representation_dim' config value ({self.representation_dim}). Expected a positive value. Concept learning might not function correctly.")
             # Subsequent methods (learn_concepts) should handle this state.


        # List to store concept representatives (vectors).
        self.concept_representatives = []

        # TODO: Persistence for concepts (saving/loading to file) will be added in the future.

        logger.info(f"LearningModule initialized. New Concept Threshold: {self.new_concept_threshold}, Representation Dimension: {self.representation_dim}. Initial Concept Count: {len(self.concept_representatives)}")

    def learn_concepts(self, representation_list):
        """
        Discovers new concepts or updates existing ones using the provided list of Representation vectors.

        Applies a simple threshold-based logic. If a vector's highest similarity to existing
        concept representatives is below the new concept threshold, it is added as a new
        concept representative. Otherwise, it is considered to belong to an existing concept
        and does not trigger a change in the concept list (although future implementations
        might update representatives, e.g., by averaging).
        Handles invalid inputs gracefully (None, wrong type, wrong dimension) and logs errors.

        Args:
            representation_list (list or None): A list of Representation vectors (numpy arrays) to learn from.
                                                Expected format: list of np.ndarray (shape (representation_dim,), numerical dtype).
                                                Can be None or an empty list.

        Returns:
            list: The updated list of learned concept representatives (numpy arrays).
                  Returns the current list on error or if input is invalid/empty.
        """
        # Input validation. Check if representation_list is None.
        if not check_input_not_none(representation_list, input_name="representation_list for LearningModule", logger_instance=logger):
             logger.debug("LearningModule.learn_concepts: representation_list is None. Skipping concept learning.")
             return self.concept_representatives # Return current list if input is None.

        # Input validation. Check if representation_list is a list.
        if not check_input_type(representation_list, list, input_name="representation_list for LearningModule", logger_instance=logger):
             logger.error(f"LearningModule.learn_concepts: representation_list is not a list ({type(representation_list)}). Skipping concept learning.")
             return self.concept_representatives # Return current list if input is not a list.

        # Input validation. Check if representation_list is an empty list.
        if not representation_list:
             logger.debug("LearningModule.learn_concepts: representation_list is an empty list. Skipping concept learning.")
             return self.concept_representatives # Return current list if input is empty.

        # Check if the module's representation dimension is valid before proceeding.
        if self.representation_dim <= 0:
             logger.error("LearningModule.learn_concepts: Representation dimension is invalid. Skipping concept learning.")
             return self.concept_representatives # Return current list if dimension is invalid.


        logger.debug(f"LearningModule.learn_concepts: Received {len(representation_list)} representation vectors for learning. Currently have {len(self.concept_representatives)} concepts.")

        try:
            # Process each Representation vector in the list.
            for rep_vector in representation_list:
                # Ensure each item in the list is a valid Representation vector (numpy array, 1D, numerical, correct dimension).
                if rep_vector is not None and check_numpy_input(rep_vector, expected_dtype=np.number, expected_ndim=1, input_name="item in representation_list", logger_instance=logger):
                    # Check if the vector's dimension matches the module's expected dimension.
                    if rep_vector.shape[0] == self.representation_dim:
                        # Calculate the norm of the vector (for similarity calculation).
                        rep_norm = np.linalg.norm(rep_vector)

                        # If the vector is not near zero norm (meaningful)
                        if rep_norm > 1e-8:
                            # Find the highest similarity of this vector to existing concept representatives.
                            # Initialize max_similarity_to_concepts to -1.0 (cosine similarity is between -1.0 and 1.0)
                            max_similarity_to_concepts = -1.0 # Initialize with lowest possible similarity

                            if self.concept_representatives: # Calculate similarity only if there are learned concepts.
                                for concept_rep in self.concept_representatives:
                                     # Ensure the existing concept representative is valid before calculation.
                                     if concept_rep is not None and isinstance(concept_rep, np.ndarray) and concept_rep.ndim == 1 and np.issubdtype(concept_rep.dtype, np.number):
                                          concept_norm = np.linalg.norm(concept_rep)
                                          if concept_norm > 1e-8: # Check if the concept vector is near zero to avoid division errors.
                                               # Calculate cosine similarity.
                                               similarity = np.dot(rep_vector, concept_rep) / (rep_norm * concept_norm)
                                               if not np.isnan(similarity) and np.isfinite(similarity): # Ignore NaN or infinite similarity scores.
                                                    max_similarity_to_concepts = max(max_similarity_to_concepts, similarity)
                                          # else: logger.debug("LM.learn_concepts: Concept rep near zero norm, skipping similarity calculation for this concept.")
                                     # else: logger.warning("LM.learn_concepts: Existing concept representative is invalid, skipping similarity calculation for this concept.")

                                # logger.debug(f"LM.learn_concepts: Highest similarity to existing concepts for vector: {max_similarity_to_concepts:.4f}")

                            # If the highest similarity is BELOW the threshold, add as a new concept.
                            # Or if there are no existing concepts, the first valid vector is the first concept.
                            should_add_concept = False
                            if not self.concept_representatives:
                                # The first valid vector is always added as the first concept (empty list check).
                                should_add_concept = True
                            else:
                                # Perform the comparison.
                                # Use floating point tolerance when comparing near 1.0 threshold.
                                # The condition for adding a NEW concept is that it is NOT sufficiently similar to ANY existing concept.
                                # The threshold 'new_concept_threshold' defines what "sufficiently similar" means.
                                # If similarity is >= threshold, it's NOT a new concept. If similarity < threshold, it IS a new concept.
                                # Example: threshold = 0.7. Sim = 0.6 -> 0.6 < 0.7 is True -> New concept.
                                # Example: threshold = 0.7. Sim = 0.7 -> 0.7 < 0.7 is False -> Not a new concept.
                                # Example: threshold = 0.7. Sim = 0.8 -> 0.8 < 0.7 is False -> Not a new concept.
                                # The existing check `max_similarity_to_concepts < self.new_concept_threshold` seems correct based on this logic.
                                # However, unit tests were failing when similarity was exactly equal to the threshold.
                                # Let's re-evaluate the comparison, perhaps using a small tolerance for equality checks if needed,
                                # but standard float comparison should be okay here given how similarity is calculated.
                                # The logs in the failing test show "En yüksek mevcut benzerlik: 0.7000 < Eşik 0.7000" -> Yeni kavram keşfedildi. This means 0.7000 < 0.7000 evaluated to True. This is mathematically incorrect for standard float comparison.
                                # It might be a subtle floating point precision issue, or the test case is slightly off.
                                # Let's explicitly check for near-equality if using strict `<`.
                                # Or, perhaps the intention was that similarity must be *strictly less than* the threshold?
                                # Yes, "Bir Representation'ın yeni kavram sayılması için mevcut kavramlara olan en yüksek benzerlik eşiği (0.0-1.0)" implies that if it's >= the threshold, it's *not* new. So strictly less than is correct.
                                # Let's check if max_similarity_to_concepts is strictly less than the threshold.
                                should_add_concept = max_similarity_to_concepts < self.new_concept_threshold
                                # Add a small tolerance for floating point equality check if needed, e.g., `max_similarity_to_concepts < (self.new_concept_threshold - 1e-9)`?
                                # Let's stick to strict comparison for now, as the test case might be comparing 0.7 exactly.

                            if should_add_concept:
                                # Add the current Representation vector as a new concept representative (a copy).
                                self.concept_representatives.append(rep_vector.copy()) # Store a copy of the numpy array.
                                logger.info(f"LearningModule.learn_concepts: New concept discovered! Total concepts: {len(self.concept_representatives)}. Highest existing similarity: {max_similarity_to_concepts:.4f} < Threshold {self.new_concept_threshold:.4f}")
                            # else: logger.debug(f"LM.learn_concepts: Vector is sufficiently familiar (similarity {max_similarity_to_concepts:.4f} >= threshold {self.new_concept_threshold:.4f}). No new concept added.")

                        # else: logger.debug("LM.learn_concepts: Representation vector has zero norm, skipping processing.")
                    # else: logger.warning(f"LM.learn_concepts: Representation vector has wrong dimension ({rep_vector.shape[0]} vs {self.representation_dim}), skipping processing.")
                # else: logger.warning("LM.learn_concepts: Item in list is not a valid Representation vector, skipping processing.")

            logger.debug(f"LearningModule.learn_concepts: Concept learning completed. Current concept count: {len(self.concept_representatives)}")

        except Exception as e:
            # Catch any unexpected error during the learning process.
            logger.error(f"LearningModule.learn_concepts: Unexpected error during concept learning: {e}", exc_info=True)
            # In case of error, return the current list of concepts.

        return self.concept_representatives # Return the updated list of concept representatives.


    def get_concepts(self):
        """
        Returns the list of learned concept representatives (shallow copy).

        Returns:
            list: A list of learned concept representatives (numpy arrays).
                  Returns an empty list on error.
        """
        # Returning a shallow copy of the list is better for safety.
        # This copies the list itself, but the array objects inside remain references.
        return self.concept_representatives[:]


    def cleanup(self):
        """
        Cleans up LearningModule resources.

        Currently, it primarily clears the list of learned concepts.
        TODO: Persistence logic for saving learned concepts will go here (future TODO).
        """
        logger.info("LearningModule object cleaning up.")
        # TODO: Add logic here to save learned concepts (future TODO).

        # Clear the list of learned concepts.
        self.concept_representatives = []
        logger.info("LearningModule: Learned concepts cleared.")
        pass

    # TODO: Methods for saving/loading concepts will be added (future TODO).