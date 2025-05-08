# src/cognition/core.py
#
# Evo's cognitive core.
# Attempts to understand the world, learn concepts, and make action decisions
# using incoming representations, memory entries, and instantaneous sensory features.
# Coordinates the UnderstandingModule, DecisionModule, and LearningModule sub-modules.

import logging # For logging.
# numpy is required (for parameter types)
import numpy as np
import random # For sampling for LearningModule
import inspect # For debugging method signatures

# Import utility functions
from src.core.config_utils import get_config_value
# check_input_not_none, check_numpy_input, check_input_type are currently not used in cognition/core, can be removed.
# from src.core.utils import check_input_not_none, check_numpy_input, check_input_type # <<< Utils imports

# Import sub-module classes
from .understanding import UnderstandingModule
from .decision import DecisionModule
from .learning import LearningModule # Import LearningModule


# Create a logger for this module
logger = logging.getLogger(__name__)


class CognitionCore:
    """
    Evo's cognitive core class.

    Receives information flow (processed inputs, Representation, memory entries).
    Passes this information to the Understanding module to get understanding output (dictionary signals).
    Passes the understanding output and memory entries to the Decision module to make a decision.
    Periodically triggers the LearningModule to learn concepts based on Representations in memory.
    Coordinates the UnderstandingModule, DecisionModule, and LearningModule sub-modules.
    Logs errors and prevents program crashes.
    """
    def __init__(self, config, module_objects): # Receive the module_objects dictionary during init.
        """
        Initializes the CognitionCore module.

        Initializes sub-modules (UnderstandingModule, DecisionModule, LearningModule).
        Stores reference to the Memory module.
        Sub-module objects might remain None if initialization fails.

        Args:
            config (dict): Full configuration settings for the system.
                           CognitionCore and its sub-modules will read their relevant sections from this dict.
                           (e.g., settings under 'cognition', 'representation', etc.)
            module_objects (dict): Dictionary containing references to other initialized modules (from module_loader).
                                   Used to get references to Memory, etc.
        """
        self.config = config # CognitionCore receives the full config
        logger.info("Cognition module initializing...")

        self.understanding_module = None # Understanding module object.
        self.decision_module = None # Decision making module object.
        self.learning_module = None # Learning module object.

        # Store the Memory module reference.
        # Get the Memory object from the module_objects dict if it is a valid dict. It might be None.
        # The module_objects dictionary will be passed to the CognitiveCore init when initialize_modules is called by run_evo.
        self.memory_instance = module_objects.get('memories', {}).get('core_memory')
        if self.memory_instance is None:
             logger.warning("CognitionCore: Could not obtain Memory module reference. Learning (LearningModule) and some decision mechanisms (Memory-based) may not function.")


        # Learning Module's operating frequency and the number of Representations to sample from Memory.
        # Get from config using get_config_value.
        # These settings are under the 'cognition.learning' key.
        self.learning_frequency = get_config_value(config, 'cognition', 'learning', 'learning_frequency', default=100, expected_type=int, logger_instance=logger)
        self.learning_memory_sample_size = get_config_value(config, 'cognition', 'learning', 'learning_memory_sample_size', default=50, expected_type=int, logger_instance=logger)

        self._loop_counter = 0 # Loop counter to trigger the LearningModule.


        # Try to initialize sub-modules. Initialization errors are logged internally by the sub-modules.
        try:
            # Initialize the understanding module, passing the full config.
            # UnderstandingModule init will read its relevant section from this config.
            understanding_config = config # Pass the full config
            self.understanding_module = UnderstandingModule(understanding_config)
            # If UnderstandingModule init does not raise an exception and does not return None, it is considered initialized.
            # Sub-modules are expected to log their own initialization errors.
            # if self.understanding_module is None: # This check can be removed, __init__ should ideally raise exception or set state.
            #      logger.error("CognitionCore: UnderstandingModule initialization failed.")


            # Initialize the decision making module, passing the full config.
            # DecisionModule init will read its relevant section from this config.
            decision_config = config # Pass the full config
            self.decision_module = DecisionModule(decision_config)
            # if self.decision_module is None: # This check can be removed.
            #      logger.error("CognitionCore: DecisionModule initialization failed.")


            # Initialize the learning module, passing the full config.
            # LearningModule init will read its relevant sections (cognition.learning, representation) from this config.
            # The entire main config should be passed here so LearningModule can access all relevant parts.
            learning_config_for_module = config # Pass the full config
            self.learning_module = LearningModule(learning_config_for_module)
            # if self.learning_module is None: # This check can be removed.
            #      logger.error("CognitionCore: LearningModule initialization failed.")


        except Exception as e:
             # If an unexpected error occurs during sub-module initialization.
             logger.critical(f"CognitionCore: Error during sub-module initialization: {e}", exc_info=True)
             # In case of error, sub-module objects might remain None.


        # Learning frequency and sample size checks (must be positive)
        if self.learning_frequency <= 0:
             logger.warning(f"CognitionCore: Invalid 'learning_frequency' config value ({self.learning_frequency}). Using default 100.")
             self.learning_frequency = 100
        if self.learning_memory_sample_size <= 0:
             logger.warning(f"CognitionCore: Invalid 'learning_memory_sample_size' config value ({self.learning_memory_sample_size}). Using default 50.")
             self.learning_memory_sample_size = 50


        logger.info(f"Cognition module initialized. Learning Frequency: {self.learning_frequency} cycles, Memory Sample Size: {self.learning_memory_sample_size}")


    # run_evo.py calls this method. It receives processed_inputs, learned_representation, relevant_memory_entries, current_concepts arguments.
    # The signature of the decide method matches the call in run_evo.py.
    def decide(self, processed_inputs, learned_representation, relevant_memory_entries, current_concepts):
        """
        Makes an action decision based on processed inputs, learned representation, and relevant memory entries.

        Args:
            processed_inputs (dict or None): Processed instantaneous sensory inputs.
            learned_representation (numpy.ndarray or None): The latest learned representation vector.
            relevant_memory_entries (list or None): List of relevant memory entries retrieved from Memory.
            current_concepts (list): List of current concept representative vectors from the LearningModule.

        Returns:
            str or None: The decided action (string) or None if decision making failed.
        """
        # Increment the loop counter.
        self._loop_counter += 1

        # The current_concepts information is passed to the decide method externally (from run_evo.py).
        # The CognitionCore decide method itself does not retrieve concepts from the LearningModule at the start of the cycle.
        # This makes the decide method easier to test in isolation, as concepts can be mocked.

        # LearningModule trigger check. If the LearningModule and Memory module exist, and it's time based on frequency.
        # The learning step should not interrupt the main decision flow, so it runs in its own try/except block.
        # This try/except block covers calls to get_all_representations, random.sample, and learn_concepts.
        # The LearningModule instance is held as self.learning_module.
        if self.learning_module is not None and self.memory_instance is not None and self._loop_counter % self.learning_frequency == 0:
             logger.info(f"CognitionCore: Learning cycle triggered (cycle #{self._loop_counter}).")
             try:
                  # Memory'den öğrenme için Representation örneklemi al.
                  if hasattr(self.memory_instance, 'get_all_representations'):
                      all_memory_representations = self.memory_instance.get_all_representations()

                      # Ensure the retrieved list of Representations is a list of numpy arrays.
                      # Filter for representations that are valid and match the LearningModule's expected dimension.
                      # Access the representation_dim attribute from the LearningModule instance.
                      learning_rep_dim = self.learning_module.representation_dim if self.learning_module else None

                      valid_representations_for_learning = [
                          rep for rep in all_memory_representations
                          if rep is not None
                          and isinstance(rep, np.ndarray)
                          and np.issubdtype(rep.dtype, np.number) # Check for numeric dtype
                          and rep.ndim == 1
                          and learning_rep_dim is not None # Check if LearningModule's rep dim is available
                          and rep.shape[0] == learning_rep_dim # Dimension check
                      ]

                      if valid_representations_for_learning:
                           # Take a random sample for learning (if memory is very large).
                           # Use min() to handle cases where memory size is smaller than the population.
                           # random.sample raises ValueError if the sample size is larger than the population.
                           learning_sample = random.sample(valid_representations_for_learning, min(self.learning_memory_sample_size, len(valid_representations_for_learning)))
                           logger.debug(f"CognitionCore: Sampled {len(learning_sample)} representations from Memory for learning.")
                           # Call the LearningModule with the sample of Representations. Exceptions from learn_concepts are also caught here.
                           self.learning_module.learn_concepts(learning_sample)
                      else:
                           logger.debug("CognitionCore: Not enough valid Representations in Memory for learning.")
                  else:
                      logger.warning("CognitionCore: Memory module does not have 'get_all_representations' method. Cannot obtain Representations for LearningModule.")

             except Exception as e:
                  # Catch errors during the learning cycle but do not interrupt the main decide flow.
                  logger.error(f"CognitionCore: Unexpected error during learning cycle: {e}", exc_info=True)


        # If critical sub-modules of CognitionCore were not initialized, do not proceed with decision making.
        # Understanding and Decision modules are required for decision making. Learning is optional.
        # Check for None instances of self.understanding_module and self.decision_module.
        if self.understanding_module is None or self.decision_module is None:
            logger.error("CognitionCore.decide: Critical sub-modules (Understanding/Decision) are not initialized or are None. Cannot make a decision.")
            return None


        understanding_signals = None # Dictionary of signals from the understanding module.
        decision = None # The decided action from the decision making module.

        # Place the calls to Understanding and Decision modules in the main try/except block.
        # Errors in these modules are considered critical and will stop the decision making process for this cycle.
        try:
            # 1. Pass the incoming information to the understanding module.
            # The UnderstandingModule.process method returns a dictionary of signals.
            # We are passing processed_inputs, learned_representation, relevant_memory_entries, AND current_concepts as arguments.
            understanding_signals = self.understanding_module.process(
                processed_inputs, # Processed instantaneous sensory inputs (dict/None)
                learned_representation, # Learned Representation (None/array)
                relevant_memory_entries, # Relevant memories from Memory (list/None)
                current_concepts # Current concepts from LearningModule (list), passed from run_evo.py
            )
            # DEBUG Log: The understanding signals dictionary (already logged inside UnderstandingModule).
            # if isinstance(understanding_signals, dict): ...


            # 2. Pass the understanding output and memory entries to the decision making module.
            # The DecisionModule.decide expects understanding_signals (dict/None) and relevant_memory_entries (list/None).
            # It manages its own internal state (curiosity). current_concepts is also passed.
            decision = self.decision_module.decide(
                understanding_signals, # Output of the understanding module (dictionary signals)
                relevant_memory_entries, # Memory entries (list/None)
                current_concepts # Concepts (list), can be used by the decision module
            )

            # DEBUG Log: The decision result (already logged inside DecisionModule).
            # if decision is not None: ...


        except Exception as e:
            # Catch any unexpected error when calling methods of Understanding or Decision modules,
            # or if an unhandled exception occurs within those methods.
            # This is a critical error and will halt the decision making process for this cycle.
            logger.error(f"CognitionCore.decide: Critical error during understanding or decision making: {e}", exc_info=True)
            return None # Return None in case of error.

        # Add a final DEBUG log for the decision result, if it's not None, before returning.
        if decision is not None:
            logger.debug(f"CognitionCore.decide: Final decision reached: '{decision}'.")
        else:
            logger.debug("CognitionCore.decide: No specific decision reached (result is None).")

        return decision # Return the decision string or None.

    # ... (cleanup method - same as before) ...


def cleanup(self):
        """
        CognitionCore modülü kaynaklarını temizler.

        Alt modülleri (UnderstandingModule, DecisionModule, LearningModule) cleanup metotlarını (varsa) çağırır.
        module_loader.py bu metotu program sonlanırken çağrır (varsa).
        """
        logger.info("Cognition modülü objesi siliniyor...")
        # Alt modüllerin cleanup metotlarını çağır (varsa).
        if self.understanding_module and hasattr(self.understanding_module, 'cleanup'):
             try:
                 self.understanding_module.cleanup()
             except Exception as e:
                 logger.error(f"CognitionCore Cleanup: UnderstandingModule cleanup sırasında hata: {e}", exc_info=True)

        if self.decision_module and hasattr(self.decision_module, 'cleanup'):
             try:
                 self.decision_module.cleanup()
             except Exception as e:
                 logger.error(f"CognitionCore Cleanup: DecisionModule cleanup sırasında hata: {e}", exc_info=True)

        if self.learning_module and hasattr(self.learning_module, 'cleanup'):
             try:
                 self.learning_module.cleanup()
             except Exception as e:
                  logger.error(f"CognitionCore Cleanup: LearningModule cleanup sırasında hata: {e}", exc_info=True)


        logger.info("Cognition modülü objesi silindi.")    