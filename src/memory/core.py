# src/memory/core.py
#
# Evo's core memory system.
# Stores learned representations and retrieves them as needed.
# Adds file-based persistence to memory.
# Will coordinate sub-modules like episodic and semantic memory in the future.

import numpy as np # For representation vectors (numpy array).
import time # For memory entry timestamps.
import random # For placeholder retrieve (random selection).
import logging # For logging.
import pickle # For saving and loading memory to file
import os # For filesystem operations (checking existence, creating directories)

# Import utility functions
# setup_logging will not be called here. isinstance is used instead of check_* functions.
from src.core.utils import run_safely, cleanup_safely # Only run_safely and cleanup_safely are used here
from src.core.config_utils import get_config_value # get_config_value is imported from here

# Import sub-memory modules (Placeholder classes)
# Commenting out imports for now if they are just placeholders and might cause import errors
# if the files/classes don't fully exist or are not intended to be tested/used yet.
# from .episodic import EpisodicMemory
# from .semantic import SemanticMemory


# Create a logger for this module
# Returns a logger named 'src.memory.core'.
# Logging level and handlers are configured externally (by conftest.py or the main run script).
logger = logging.getLogger(__name__)


class Memory:
    """
    Evo's primary memory system class (Coordinator/Manager).

    Receives learned representation vectors from the RepresentationLearner.
    Directs and manages these representations and/or related information to different memory types (core/working,
    episodic, semantic).
    Includes a basic list-based storage implementation (core/working memory) with file-based persistence (pickle).
    Logs errors and prevents program crashes.
    """
    def __init__(self, config):
        """
        Initializes the Memory module.

        Initializes the basic storage structure (currently a list), loads from persistent memory,
        and attempts to initialize sub-memory modules (EpisodicMemory, SemanticMemory) in the future.

        Args:
            config (dict): Memory system configuration settings (full config dict).
                           Settings for this module are read from the 'memory' section,
                           and representation dimension from the 'representation' section.
        """
        self.config = config # Memory module now receives the full config
        logger.info("Memory module initializing...")

        # Get configuration settings using get_config_value
        # Pass logger_instance=logger to each call to ensure logs within get_config_value are visible.
        # These settings are under the 'memory' key.
        self.max_memory_size = get_config_value(config, 'memory', 'max_memory_size', default=1000, expected_type=int, logger_instance=logger)
        self.num_retrieved_memories = get_config_value(config, 'memory', 'num_retrieved_memories', default=5, expected_type=int, logger_instance=logger)
        # The representation dimension is under the 'representation' key in the main config.
        # The Memory module needs to know this dimension to store/retrieve correctly.
        self.representation_dim = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)

        # memory_file_path is retrieved from config, under the 'memory' key.
        self.memory_file_path = get_config_value(config, 'memory', 'memory_file_path', default='data/core_memory.pkl', expected_type=str, logger_instance=logger)


        # Check for negative values for num_retrieved_memories
        if self.num_retrieved_memories < 0:
             logger.warning(f"Memory: Invalid num_retrieved_memories config value ({self.num_retrieved_memories}). Using default 5.")
             self.num_retrieved_memories = 5

        # Check for negative values for max_memory_size
        if self.max_memory_size < 0:
             logger.warning(f"Memory: Invalid max_memory_size config value ({self.max_memory_size}). Using default 1000.")
             self.max_memory_size = 1000

        # Check for non-positive values for representation_dim
        if self.representation_dim <= 0:
             logger.warning(f"Memory: Invalid representation_dim config value ({self.representation_dim}). Expected a positive value. Using default 128.")
             self.representation_dim = 128


        # Log the memory_file_path obtained from config - important for DEBUG logs!
        # This log will clearly show the result of the get_config_value call.
        logger.info(f"Memory __init__: memory_file_path set from config: {self.memory_file_path}")


        # Memory storage structures. For now, just a simple list-based core/working memory.
        # Each element is a dictionary: {'representation': numpy_array, 'metadata': dict, 'timestamp': float}
        self.core_memory_storage = [] # Core / Working Memory Storage

        # Sub-memory module objects (if they exist)
        self.episodic_memory = None
        self.semantic_memory = None

        # Logic to load from persistent storage
        self._load_from_storage()


        # Try to initialize sub-memory modules (Future TODO).
        # TODO: Add logic here to initialize sub-memory modules.
        # try:
        #     episodic_config = config.get('episodic', {}) # Get sub-config for episodic
        #     # If EpisodicMemory class is imported and exists, initialize it
        #     if 'EpisodicMemory' in globals() and EpisodicMemory:
        #         self.episodic_memory = EpisodicMemory(episodic_config) # Pass sub-config
        #     if self.episodic_memory is None: logger.error("Memory: EpisodicMemory initialization failed.")
        # except Exception as e: logger.error(f"Memory: Error during EpisodicMemory initialization: {e}", exc_info=True); self.episodic_memory = None

        # try:
        #     semantic_config = config.get('semantic', {}) # Get sub-config for semantic
        #     # If SemanticMemory class is imported and exists, initialize it
        #     if 'SemanticMemory' in globals() and SemanticMemory:
        #         self.semantic_memory = SemanticMemory(semantic_config) # Pass sub-config
        #     if self.semantic_memory is None: logger.error("Memory: SemanticMemory initialization failed.")
        # except Exception as e: logger.error(f"Memory: Error during SemanticMemory initialization: {e}", exc_info=True); self.semantic_memory = None


        logger.info(f"Memory module initialized. Maximum Core Memory size: {self.max_memory_size}. Default retrieval count: {self.num_retrieved_memories}. Persistence file: {self.memory_file_path}. Loaded memories count: {len(self.core_memory_storage)}")


    def _load_from_storage(self):
        """
        Loads the memory state (core_memory_storage) from the specified file.
        If the file does not exist, cannot be read, or is corrupted, initializes memory as empty.
        """
        # Log that the loading process is starting and the path being used - important for DEBUG logs!
        logger.info(f"Memory._load_from_storage: Loading process starting. Path used: {self.memory_file_path}")


        # Check if the file path is a valid string and not empty
        if not isinstance(self.memory_file_path, str) or not self.memory_file_path:
             logger.warning("Memory._load_from_storage: Invalid or empty memory file path specified. Loading skipped.")
             self.core_memory_storage = [] # Initialize memory as empty if loading is skipped.
             return

        # Check if the file exists
        if not os.path.exists(self.memory_file_path):
            logger.info(f"Memory._load_from_storage: Memory file not found: {self.memory_file_path}. Initializing memory as empty.")
            self.core_memory_storage = [] # Initialize as empty if the file doesn't exist.
            return

        # Try to read the file and load with pickle
        try:
            with open(self.memory_file_path, 'rb') as f: # 'rb' binary read mode
                # Load the data from the file using pickle.load
                # Security Note: Do not load pickle files from unknown or untrusted sources.
                # In this project context, we trust files saved by ourselves.
                loaded_data = pickle.load(f)

            # Check if the loaded data is a list as expected.
            # A more robust check would also involve verifying that each item in the list
            # is in the expected {'representation': np.ndarray, 'metadata': dict, 'timestamp': float} format,
            # but this is sufficient for a start.
            if isinstance(loaded_data, list):
                # Check if each item in the loaded list is in the minimum expected format
                # This helps catch corrupted or incompatible pickle files.
                valid_loaded_data = []
                for item in loaded_data:
                     if isinstance(item, dict) and 'representation' in item and 'metadata' in item and 'timestamp' in item:
                           # Optionally, check the format of the Representation itself (np.ndarray, 1D, numeric, correct dim)
                           rep = item['representation']
                           if isinstance(rep, np.ndarray) and rep.ndim == 1 and np.issubdtype(rep.dtype, np.number) and rep.shape[0] == self.representation_dim:
                                valid_loaded_data.append(item)
                           else:
                                logger.warning("Memory._load_from_storage: Found memory entry with invalid representation format, skipping.")
                     else:
                         logger.warning("Memory._load_from_storage: Found memory entry with unexpected format, skipping.")

                self.core_memory_storage = valid_loaded_data

                logger.info(f"Memory._load_from_storage: Memory loaded successfully: {self.memory_file_path} ({len(self.core_memory_storage)} entries, {len(loaded_data)-len(valid_loaded_data)} invalid entries skipped).")

                # If the number of loaded memories exceeds max_memory_size, delete older memories (cleanup after loading)
                if len(self.core_memory_storage) > self.max_memory_size:
                    logger.warning(f"Memory._load_from_storage: Number of loaded memories ({len(self.core_memory_storage)}) exceeds maximum size ({self.max_memory_size}). Deleting older memories.")
                    # Keep only the last max_memory_size memories.
                    # Negative index slicing is safe.
                    self.core_memory_storage = self.core_memory_storage[-self.max_memory_size:]


            else:
                # If the loaded data is not in list format
                logger.error(f"Memory._load_from_storage: Loaded memory file has unexpected format: {self.memory_file_path}. Expected a list, got: {type(loaded_data)}. Initializing memory as empty.", exc_info=True)
                self.core_memory_storage = [] # Initialize as empty if format is wrong.

        except FileNotFoundError:
            # Although os.path.exists check is done, catching here adds robustness.
            logger.warning(f"Memory._load_from_storage: Memory file not found (after re-check): {self.memory_file_path}. Initializing memory as empty.")
            self.core_memory_storage = []

        except (pickle.UnpicklingError, EOFError, ImportError, IndexError) as e:
            # Errors that might occur during pickle loading (corrupted file, incompatible pickle version, etc.)
            logger.error(f"Memory._load_from_storage: Pickle error while loading memory file: {self.memory_file_path}. Initializing memory as empty.", exc_info=True)
            self.core_memory_storage = [] # Initialize as empty if loading fails.

        except Exception as e:
            # Catch all other unexpected errors.
            logger.error(f"Memory._load_from_storage: Unexpected error while loading memory file: {self.memory_file_path}. Initializing memory as empty.", exc_info=True)
            self.core_memory_storage = []


    def _save_to_storage(self):
        """
        Saves the current memory state (core_memory_storage) to the specified file (in pickle format).
        Skips the save operation if memory is empty. Logs errors if they occur during saving.
        """
        # Skip saving if memory is empty.
        if not self.core_memory_storage:
            logger.info("Memory._save_to_storage: Core memory is empty. Saving skipped.")
            return

        # Check if the file path is a valid string and not empty
        if not isinstance(self.memory_file_path, str) or not self.memory_file_path:
             logger.warning("Memory._save_to_storage: Invalid or empty memory file path specified. Saving skipped.")
             return

        # Create the directory for the file path if it doesn't exist
        save_dir = os.path.dirname(self.memory_file_path)
        # Only create the directory if the directory path is not empty (e.g., if just a filename was given) and the directory doesn't exist.
        if save_dir and not os.path.exists(save_dir):
             try:
                  os.makedirs(save_dir, exist_ok=True) # exist_ok=True does not raise error if directory already exists
                  logger.info(f"Memory._save_to_storage: Save directory created: {save_dir}")
             except OSError as e:
                  logger.error(f"Memory._save_to_storage: Error creating save directory: {save_dir}. Saving skipped.", exc_info=True)
                  return # Skip saving if directory cannot be created.
             except Exception as e:
                  logger.error(f"Memory._save_to_storage: Unexpected error while creating save directory: {save_dir}. Saving skipped.", exc_info=True)
                  return # Skip saving if directory cannot be created.


        # Try to write to the file and save with pickle
        try:
            with open(self.memory_file_path, 'wb') as f: # 'wb' binary write mode
                # Save the data to the file using pickle.dump
                # Saving a copy might be safer, but Representations are numpy arrays, arguably immutable enough.
                pickle.dump(self.core_memory_storage, f)
            logger.info(f"Memory._save_to_storage: Memory saved successfully to: {self.memory_file_path} ({len(self.core_memory_storage)} entries).")

        except (pickle.PicklingError, IOError, OSError) as e:
            # Errors that might occur during pickle saving or file writing
            logger.error(f"Memory._save_to_storage: Error while saving memory file: {self.memory_file_path}.", exc_info=True)

        except Exception as e:
             # Catch all other unexpected errors.
             logger.error(f"Memory._save_to_storage: Unexpected error while saving memory file: {self.memory_file_path}.", exc_info=True)


    def store(self, representation, metadata=None):
        """
        Stores a learned representation (and associated metadata) in memory.

        Decides which memory type (core/working, episodic, semantic) is appropriate
        for the incoming representation and metadata, and saves it to the relevant
        memory structure and/or sub-module.
        Currently, it only saves to the basic list-based core memory.
        Skips the store operation if the representation is None or an unexpected numpy array type.
        If the core memory size exceeds self.max_memory_size, it deletes the oldest memory (FIFO principle).
        Logs errors on failure.

        Args:
            representation (numpy.ndarray or None): The representation vector to be stored in memory.
                                                    Typically comes from the RepresentationLearner.
                                                    Expected format: shape (self.representation_dim,), numerical dtype.
            metadata (dict, optional): Additional information associated with the representation (e.g., source, time range, state, etc.).
                                       Defaults to None. If None, stored as an empty dictionary.
                                       Expected type: dict or None.
        """
        # Error handling: Is the representation to be stored None?
        if representation is None:
             logger.debug("Memory.store: Representation input is None. Skipping storage.")
             return # Skip storing if None.

        # Error handling: Check if the representation is a numpy array, 1D, and has a numerical dtype.
        # expected_ndim=1 because representation is usually a 1D vector.
        # dtype should be numerical, likely float64 from RepresentationLearner output.
        if not isinstance(representation, np.ndarray) or representation.ndim != 1 or not np.issubdtype(representation.dtype, np.number):
            logger.error(f"Memory.store: Representation input is not a numpy array or has wrong dtype/dimension. Expected: numpy array (1D, numerical), Received: {type(representation)}, ndim: {getattr(representation, 'ndim', 'N/A')}, dtype: {getattr(representation, 'dtype', 'N/A')}. Skipping storage.")
            return # Skip storage if invalid type, dtype, or dimensions.

        # Check the representation dimension (must match representation_dim from config)
        # This verifies that the RepresentationLearner output matches the expected dimension for memory.
        if representation.shape != (self.representation_dim,):
             logger.warning(f"Memory.store: Attempted to add memory with unexpected representation dimension ({representation.shape}). Expected: {(self.representation_dim,)}. Skipping storage.")
             return # Skip storage if dimensions don't match.

        # Error handling: Is metadata None or a dict?
        if metadata is not None and not isinstance(metadata, dict):
             # If metadata is not None but also not a dict, log a warning and set metadata to None.
             logger.warning(f"Memory.store: Metadata has unexpected type ({type(metadata)}), dictionary expected. Ignoring metadata.")
             metadata = None # Set to None so it's stored as {} later.

        try:
            # TODO: In the future: Based on incoming representation and metadata, decide which memory type (core/working, episodic, semantic) to save to.
            # For example, entries with clear timestamps or specific context might go to episodic memory,
            # Recurring or relational entries might go to semantic memory.
            # For now, only saving to the basic list-based core memory.

            # Simply save to core memory (FIFO - First-In, First-Out)
            # Create a new memory entry dictionary.
            memory_entry = {
                'representation': representation,
                'metadata': metadata if metadata is not None else {}, # Store an empty dictionary if metadata was None.
                'timestamp': time.time() # Current time as a float (epoch time).
            }

            # Append the new entry to the core memory storage list (at the end).
            self.core_memory_storage.append(memory_entry)
            # DEBUG log: Info that the storage was successful and the current size.
            logger.debug(f"Memory.store: Representation successfully stored in core memory. Current size: {len(self.core_memory_storage)}")


            # If the maximum core memory size is exceeded, delete the oldest entry (FIFO).
            if len(self.core_memory_storage) > self.max_memory_size:
                # This check might be redundant if max_memory_size is 0 or negative, but we assume >= 0.
                # Remove the first (oldest) element from the list.
                # Index 0 is valid if core_memory_storage is not empty.
                removed_entry = self.core_memory_storage.pop(0)
                # DEBUG log: Info about the memory entry that was deleted.
                logger.debug(f"Memory.store: Maximum core memory size ({self.max_memory_size}) exceeded. Deleted oldest memory entry (timestamp: {removed_entry['timestamp']:.2f}).")

            # TODO: In the future: If sub-memory modules were initialized, save relevant data to them as well.
            # if self.episodic_memory and hasattr(self.episodic_memory, 'store_event'):
            #      # Context from metadata could be used for episodic memory.
            #      self.episodic_memory.store_event(representation, memory_entry['timestamp'], context=memory_entry['metadata'])
            # if self.semantic_memory and hasattr(self.semantic_memory, 'store_concept'):
            #      # Representation and relations (derived from metadata or perceived separately) could be used for semantic memory.
            #      self.semantic_memory.store_concept(representation, relations=...)


        except Exception as e:
            # Catch any unexpected error that might occur during the storage process.
            logger.error(f"Memory.store: Unexpected error during memory storage: {e}", exc_info=True)
            # Policy: Prevent program crash, just log and continue.


    def retrieve(self, query_representation, num_results=None):
        """
        Retrieves relevant memory entries from memory.

        Calculates vector similarity between the incoming query (query_representation)
        and entries in the core memory (based on their representation vectors) and
        returns the most relevant ones.
        Returns an empty list if query_representation is None or invalid, or if memory is empty.
        Returns an empty list on error.

        Args:
            query_representation (numpy.ndarray or None): The representation vector used for the query.
                                                         Typically comes from the RepresentationLearner.
                                                         Expected format: shape (self.representation_dim,), numerical dtype, or None.
            num_results (int, optional): The maximum number of memory entries to retrieve.
                                         Defaults to self.num_retrieved_memories.
                                         If None, the default is used. If invalid int, default or 5 is used.

        Returns:
            list: A list of retrieved memory entries.
                  Each entry is a dictionary: {'representation': numpy_array, 'metadata': dict, 'timestamp': float}.
                  Returns an empty list `[]` on error or if memory is empty.
        """
        # Set the default value for num_results (if it came in as None).
        if num_results is None:
            num_results = self.num_retrieved_memories

        # Error handling: Check if num_results is a valid integer (>= 0).
        # Use isinstance check. get_config_value's expected_type check should ensure it's int.
        if not isinstance(num_results, int) or num_results < 0:
             logger.warning(f"Memory.retrieve: Invalid num_results value or type ({num_results}). Will use default ({self.num_retrieved_memories}) or 5.")
             # Fallback to self.num_retrieved_memories if it's valid, otherwise use a hardcoded 5.
             fallback_num = self.num_retrieved_memories if isinstance(self.num_retrieved_memories, int) and self.num_retrieved_memories >= 0 else 5
             num_results = fallback_num # Use the determined valid number of results.

        # Determine the actual number of results to retrieve (should not exceed memory size).
        actual_num_results = min(num_results, len(self.core_memory_storage))

        retrieved_list = [] # List to hold retrieved memories/information.

        # Check if the query representation is valid (not None, numpy array, 1D, numerical, correct dimension).
        # Added dimension check.
        valid_query = query_representation is not None \
                      and isinstance(query_representation, np.ndarray) \
                      and query_representation.ndim == 1 \
                      and np.issubdtype(query_representation.dtype, np.number) \
                      and query_representation.shape[0] == self.representation_dim # Dimension check added


        if not self.core_memory_storage or actual_num_results <= 0:
            # If memory is empty or the requested number of results is 0 or negative, return an empty list.
            logger.debug("Memory.retrieve: Core memory is empty or effective num_results is non-positive. Returning empty list.")
            return []

        if not valid_query:
             # If the query is invalid (None, wrong type, wrong dimension, etc.), we cannot perform similarity search. Return an empty list.
             logger.warning(f"Memory.retrieve: Invalid query representation input. Type: {type(query_representation)}, ndim: {getattr(query_representation, 'ndim', 'N/A')}, dtype: {getattr(query_representation, 'dtype', 'N/A')}, shape: {getattr(query_representation, 'shape', 'N/A')}. Skipping similarity search.")
             return [] # Or could fallback to random retrieval based on policy. Returning empty list for now.


        # --- Similarity-based Retrieval Logic ---
        logger.debug(f"Memory.retrieve: Valid query representation provided (Shape: {query_representation.shape}, Dtype: {query_representation.dtype}). Performing similarity search.")
        similarities = []
        query_norm = np.linalg.norm(query_representation) # Use np.linalg.norm

        # If the query vector has zero norm, similarity cannot be calculated meaningfully.
        if query_norm < 1e-8: # Check for near-zero norm
             logger.warning("Memory.retrieve: Query representation has near-zero norm. Cannot calculate cosine similarity meaningfully. Returning empty list.")
             return []


        try:
            # Calculate similarity between the query and each memory entry in core memory.
            for memory_entry in self.core_memory_storage:
                # Get the representation from the memory entry and check its validity.
                # Ensure the stored representation is a valid numerical 1D numpy array.
                # And check that its dimension is correct.
                if memory_entry is not None and isinstance(memory_entry, dict): # Check if memory_entry is a dict
                     stored_representation = memory_entry.get('representation') # Safe access with .get()

                     if stored_representation is not None \
                        and isinstance(stored_representation, np.ndarray) \
                        and stored_representation.ndim == 1 \
                        and np.issubdtype(stored_representation.dtype, np.number) \
                        and stored_representation.shape[0] == self.representation_dim: # Dimension check added

                          stored_norm = np.linalg.norm(stored_representation) # Use np.linalg.norm
                          if stored_norm > 1e-8: # Check if the stored vector is near zero to avoid division errors.
                               # Calculate cosine similarity: (dot product) / (norm1 * norm2)
                               similarity = np.dot(query_representation, stored_representation) / (query_norm * stored_norm)
                               if not np.isnan(similarity): # Ignore NaN similarity scores.
                                    # Store the similarity score along with the memory entry.
                                    # Storing the original index of the memory might be useful, but not needed for now.
                                    similarities.append((float(similarity), memory_entry)) # Cast similarity to float
                               else:
                                    logger.debug("Memory.retrieve: Calculated NaN similarity, skipping entry.")
                          # else: logger.debug("Memory.retrieve: Stored rep near zero norm, skipping similarity.")
                     # else: logger.debug("Memory.retrieve: Invalid stored rep format/type/shape, skipping.")
                # else: logger.warning("Memory.retrieve: Core memory list element is not a dict, skipping.")


            # Sort by similarity in descending order.
            # If the similarities list is empty, sort does nothing.
            similarities.sort(key=lambda item: item[0], reverse=True)

            # Get the top 'actual_num_results' memories by similarity.
            # If the similarities list has fewer items than requested, it takes all items up to the list length.
            retrieved_list = [item[1] for item in similarities[:actual_num_results]]

            logger.debug(f"Memory.retrieve: Found {len(similarities)} memories with valid representations for similarity check. Retrieved top {len(retrieved_list)} by similarity.")


            # TODO: In the future: If sub-memory modules were initialized, retrieve relevant results from them too and combine with retrieved_list.
            # if hasattr(self.episodic_memory, 'retrieve_event'):
            #      episodic_results = self.episodic_memory.retrieve_event(query_representation, ...)
            #      retrieved_list.extend(episodic_results)
            # if hasattr(self.semantic_memory, 'retrieve_concept'):
            #      semantic_results = self.semantic_memory.retrieve_concept(query_representation, ...)
            #      retrieved_list.extend(semantic_results)

            # TODO: In the future: Prioritize or sort results from different memory types (based on similarity, relevance, timestamp, etc.).

        except Exception as e:
            # Catch any unexpected error that might occur during the retrieval process.
            # Vector operations (np.dot, np.linalg.norm) errors can be caught here.
            logger.error(f"Memory.retrieve: Unexpected error during memory retrieval: {e}", exc_info=True)
            return [] # Return an empty list in case of error to allow the main loop to continue.

        # Return the list of retrieved memories on success.
        # logger.debug(f"Memory.retrieve: Retrieved memory list size: {len(retrieved_list)}") # Logged in run_evo.py
        return retrieved_list


    def get_all_representations(self):
        """
        Returns a list of all Representation vectors stored in core memory.
        Used by external modules like LearningModule for learning.

        Returns:
            list: A list of numpy arrays. Returns an empty list on error.
        """
        logger.debug("Memory.get_all_representations called.")
        representations = []
        try:
            # Return a list containing only valid numpy array Representations.
            # Assumes each item in core_memory_storage is in the {'representation': np.ndarray, ...} format.
            representations = [entry.get('representation')
                               for entry in self.core_memory_storage
                               if isinstance(entry, dict) # Ensure the item is a dictionary
                                  and entry.get('representation') is not None # Ensure the 'representation' key is not None
                                  and isinstance(entry.get('representation'), np.ndarray) # Ensure it's a numpy array
                                  and entry.get('representation').ndim == 1 # Ensure it's a 1D vector
                                  and np.issubdtype(entry.get('representation').dtype, np.number) # Ensure it's a numerical dtype
                                  and entry.get('representation').shape[0] == self.representation_dim] # Ensure the dimension is correct


            logger.debug(f"Memory.get_all_representations: Found {len(representations)} valid Representations.")
            # Returning a copy might be safer, but returning a reference is sufficient for simple unit tests.
            return representations

        except Exception as e:
            # If an error occurs during the process (e.g., core_memory_storage has an unexpected format)
            logger.error(f"Memory.get_all_representations failed: {e}", exc_info=True)
            return [] # Return an empty list on error.


    def cleanup(self):
        """
        Cleans up Memory module resources.

        Saves the core memory list to persistent storage
        and calls the cleanup methods of sub-memory modules (EpisodicMemory, SemanticMemory)
        if they exist.
        Called by module_loader.py when the program terminates (if it exists).
        """
        logger.info("Memory module object cleaning up.")

        # Logic to save memory to persistent storage
        # Using run_safely is more robust in case of saving errors.
        run_safely(self._save_to_storage, logger_instance=logger, error_message="Memory: Error during _save_to_storage cleanup")

        # Clear the memory list in core memory (after it's saved).
        # Setting the list to None or an empty list helps the garbage collector collect the objects.
        self.core_memory_storage = [] # Or self.core_memory_storage = None
        logger.info("Memory: Core memory cleared (RAM).") # Specify that the RAM copy is cleared.

        # Call the cleanup methods of sub-memory modules (if they exist).
        # We can use the cleanup_safely helper function.
        # cleanup_safely should only be passed the method reference.
        # Call cleanup if the sub-module objects are not None and have a cleanup method.
        # Checking isinstance(self.episodic_memory, EpisodicMemory) is also an option.
        # TODO: Sub-module cleanup calls go here
        # if hasattr(self.episodic_memory, 'cleanup'):
        #      cleanup_safely(self.episodic_memory.cleanup, logger_instance=logger, error_message="Memory: Error during EpisodicMemory cleanup")
        # if hasattr(self.semantic_memory, 'cleanup'):
        #      cleanup_safely(self.semantic_memory.cleanup, logger_instance=logger, error_message="Memory: Error during SemanticMemory cleanup")


        logger.info("Memory module object cleaned up.")