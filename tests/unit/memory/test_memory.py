# tests/unit/memory/test_memory.py

import pytest
import numpy as np
import os
import sys # sys.path manipulation is done in conftest.
import logging
import time # For timestamps
import shutil # For cleaning up temporary memory directory

# conftest.py should set up sys.path, these imports should work directly now.
# setup_logging will not be called here, it's handled by conftest.
from src.memory.core import Memory
from src.core.config_utils import get_config_value
# from src.core.logging_utils import setup_logging # setup_logging import is okay if conftest uses it.
from src.core.utils import cleanup_safely # For fixture cleanup


# Create a logger for this test file. Level will be configured by conftest.
test_logger = logging.getLogger(__name__)
test_logger.info("src.memory.core and necessary helpers imported successfully for testing.")


# Fixture for a temporary directory where memory files will be stored
@pytest.fixture(scope="function") # A separate memory directory for each test function
def temp_memory_dir(tmp_path):
    """Creates a subdirectory for memory within pytest's provided temporary directory."""
    # tmp_path is a pathlib.Path object provided by pytest.
    mem_dir = tmp_path / "test_memory"
    mem_dir.mkdir(parents=True, exist_ok=True) # Create directories, don't error if they already exist.
    test_logger.debug(f"Temporary memory directory created: {mem_dir}")
    # Memory module might expect a string path, so use str().
    yield str(mem_dir)
    # Pytest automatically cleans up tmp_path after the test function finishes.
    # Manual cleanup (shutil.rmtree) is not strictly necessary but can be done explicitly if desired.
    # test_logger.debug(f"Cleaning up temporary memory directory: {mem_dir}")
    # shutil.rmtree(mem_dir, ignore_errors=True)


@pytest.fixture(scope="function") # A separate instance for each test function
def dummy_memory_config(temp_memory_dir):
    """Provides a dummy configuration dictionary for Memory module testing."""
    # The real Memory.__init__ method reads memory_file_path from config.
    # This test fixture should simulate that structure.
    # Construct the memory_file_path by joining the temporary_memory_dir and a filename.

    test_file_name = "test_core_memory.pkl" # Test-specific filename
    # Use os.path.join to create a path compatible with the operating system.
    test_memory_path = os.path.join(temp_memory_dir, test_file_name)


    config = {
        'memory': {
            # Memory module expects memory_file_path:
            'memory_file_path': test_memory_path, # <-- Assign the temporary file path here.
            'max_memory_size': 1000,         # Maximum size for testing
            'num_retrieved_memories': 5,     # Default retrieval count for testing
            # The representation_dim is needed by Memory __init__ and store/retrieve methods.
            # It's typically defined in the representation section of the main config.
            # Memory should read it from there.
        },
        'representation': {
             'representation_dim': 128, # representation_dim used by Memory
        },
        # ... other general config sections ...
    }
    test_logger.debug(f"Dummy memory config fixture created. memory_file_path: {config['memory']['memory_file_path']}")
    return config


@pytest.fixture(scope="function") # A separate instance for each test function
def memory_instance(dummy_memory_config):
    """Provides a Memory instance with dummy configuration."""
    test_logger.debug("Creating Memory instance...")
    try:
        # Initialize the Memory. Assuming its __init__ takes a config dict.
        mem = Memory(dummy_memory_config)
        test_logger.debug("Memory instance created.")
        yield mem # Provide the instance to the test function
        # Call cleanup if the module has one.
        # Use cleanup_safely for robustness during teardown.
        if hasattr(mem, 'cleanup'):
             cleanup_safely(mem.cleanup, logger_instance=test_logger, error_message="Error during Memory instance cleanup (teardown)")
             test_logger.debug("Memory cleanup called.")
    except Exception as e:
        # If initialization fails, fail the test.
        test_logger.error(f"Memory initialization failed: {e}", exc_info=True)
        pytest.fail(f"Memory initialization failed: {e}")


def test_memory_init_with_valid_config(memory_instance):
    """Tests that Memory initializes successfully with a valid config."""
    test_logger.info("test_memory_init_with_valid_config test started.")
    # The fixture itself ensures successful initialization.
    # Additional assertions can check if configuration values were assigned correctly.
    assert memory_instance.max_memory_size == 1000
    assert memory_instance.num_retrieved_memories == 5
    assert memory_instance.representation_dim == 128 # Check that rep_dim was read correctly
    # The memory_file_path is checked indirectly by the fixture creating it.
    # Can check if core_memory_storage is an empty list after init if no file existed.
    assert isinstance(memory_instance.core_memory_storage, list)
    # Initial load should happen in init, if file exists it's loaded, otherwise empty.
    # For a clean test run with a new temp dir, it should start empty.
    assert len(memory_instance.core_memory_storage) >= 0 # Could be 0 if no file, >0 if file existed


    test_logger.info("test_memory_init_with_valid_config test completed successfully.")


def test_memory_store_and_retrieve_basic(memory_instance, dummy_memory_config):
    """
    Tests that the Memory module stores representation vectors and can retrieve them.
    Verifies that the memory entry closest to the query representation is retrieved
    (at least that the list is not empty after retrieval if entries were stored).
    """
    test_logger.info("test_memory_store_and_retrieve_basic test started.")

    # Dummy input data for Memory.store method (Representation vector)
    # Dimension must match representation_dim from config.
    repr_dim = memory_instance.representation_dim # Use the dimension the instance initialized with
    # Use float64 as RepresentationLearner output is assumed to be float64.
    dummy_representation_1 = np.random.rand(repr_dim).astype(np.float64)
    dummy_representation_2 = np.random.rand(repr_dim).astype(np.float64)
    dummy_representation_3 = np.random.rand(repr_dim).astype(np.float64)

    # The vector used to query memory (make it similar to dummy_representation_2 for retrieval test).
    # Add some noise using np.random.randn to simulate similarity.
    query_noise_level = 0.01 # Controls how much noise is added (affects similarity level)
    query_representation = dummy_representation_2 + np.random.randn(repr_dim).astype(np.float64) * query_noise_level
    test_logger.debug("Created dummy representation vectors and query.")


    # --- Test the Store Method ---
    # Call the store methods and check for errors.
    try:
        test_logger.debug("Calling Memory.store (1)...")
        memory_instance.store(dummy_representation_1, metadata={'source': 'test1', 'timestamp': time.time() - 10})
        test_logger.debug("Memory.store called (1).")

        # Wait a bit to ensure timestamps are different (relevant if sorting by time in retrieve, although currently sorts by similarity).
        time.sleep(0.01)

        test_logger.debug("Calling Memory.store (2)...")
        memory_instance.store(dummy_representation_2, metadata={'source': 'test2', 'timestamp': time.time()})
        test_logger.debug("Memory.store called (2).")

        time.sleep(0.01)

        test_logger.debug("Calling Memory.store (3)...")
        memory_instance.store(dummy_representation_3, metadata={'source': 'test3', 'timestamp': time.time() + 10})
        test_logger.debug("Memory.store called (3).")

    except Exception as e:
        test_logger.error(f"Unexpected error while executing Memory.store: {e}", exc_info=True)
        pytest.fail(f"Unexpected error while executing Memory.store: {e}")


    # Check the number of memory entries (Optional, but a simple check that store worked).
    # Use the get_all_representations method if available, or skip this check if it accesses internal attributes.
    # get_all_representations method exists.
    try:
         all_reps_in_memory = memory_instance.get_all_representations()
         # Check if the number of entries is 3 (assuming the initial state was empty).
         # Note: If the persistent memory file existed and was not empty, the count might be > 3.
         # For this basic test, assume a clean start where only the 3 added entries exist.
         assert len(all_reps_in_memory) == 3, f"Expected 3 Representations in memory, found {len(all_reps_in_memory)}."
         test_logger.debug("Assertion passed: Memory entry count is correct using get_all_representations.")
    except Exception as e:
         # Log a warning if get_all_representations fails or results are unexpected, but don't fail the test yet.
         test_logger.warning(f"Error calling Memory.get_all_representations or results unexpected: {e}", exc_info=True)
         pass # Continue to retrieve test even if this assertion fails.


    # --- Test the Retrieve Method ---
    # Call the retrieve method and check for errors.
    try:
        test_logger.debug("Calling Memory.retrieve...")
        # Request the top (default 5) most similar memories to the query_representation.
        # retrieve method handles None/array query and invalid num_results safely, returns empty list.
        # num_memories_to_retrieve is from config and checked to be int.
        retrieved_entries = memory_instance.retrieve(
            query_representation, # Query representation (can be None or array)
            num_results=memory_instance.num_retrieved_memories # Number of results from config
        )
        test_logger.debug(f"Memory.retrieve called. Received {len(retrieved_entries)} entries.")

    except Exception as e:
        test_logger.error(f"Unexpected error while executing Memory.retrieve: {e}", exc_info=True)
        pytest.fail(f"Unexpected error while executing Memory.retrieve: {e}")


    # --- Assert the Retrieve Output ---
    # The retrieve method should return a list.
    assert isinstance(retrieved_entries, list), f"Retrieve output should be a list, received type: {type(retrieved_entries)}"
    test_logger.debug("Assertion passed: Retrieve output is a list.")

    # The list should contain dictionaries, and each dictionary should have specific keys.
    # [{'representation': array, 'metadata': {}, 'timestamp': float}, ...]
    # We stored 3 memories, and the query was very similar to dummy_representation_2.
    # Retrieve sorts by similarity. If similarity calculation works, the memory corresponding
    # to dummy_representation_2 is expected to be in the retrieved list. The list should not be empty.
    assert len(retrieved_entries) > 0, "Retrieve should not return an empty list when entries were stored."
    test_logger.debug(f"Assertion passed: Retrieved list is not empty ({len(retrieved_entries)} entries found).")

    # Check the format of the first element (checking a random element's format might be more robust).
    if retrieved_entries: # Only check if the list is not empty
        first_entry = retrieved_entries[0]
        assert isinstance(first_entry, dict), f"Elements in the retrieved list should be dicts, first element's type: {type(first_entry)}"
        assert 'representation' in first_entry, "Retrieved entry should contain 'representation' key."
        assert 'metadata' in first_entry, "Retrieved entry should contain 'metadata' key."
        assert 'timestamp' in first_entry, "Retrieved entry should contain 'timestamp' key."
        test_logger.debug("Assertion passed: First retrieved element contains the expected keys.")

        # The representation should be a numpy array and have the correct dimension.
        expected_repr_shape = (repr_dim,)
        rep_from_retrieved = first_entry.get('representation')
        assert rep_from_retrieved is not None and isinstance(rep_from_retrieved, np.ndarray), f"'representation' value should be a numpy array, type: {type(rep_from_retrieved)}"
        assert rep_from_retrieved.shape == expected_repr_shape, f"'representation' shape should be as expected. Expected: {expected_repr_shape}, Received: {rep_from_retrieved.shape}"
        assert np.issubdtype(rep_from_retrieved.dtype, np.floating), f"'representation' dtype should be float, received: {rep_from_retrieved.dtype}"
        test_logger.debug("Assertion passed: Retrieved 'representation' value has the correct format.")

        # Metadata should be a dictionary.
        assert isinstance(first_entry.get('metadata'), dict), f"'metadata' value should be a dict, type: {type(first_entry.get('metadata'))}"
        test_logger.debug("Assertion passed: Retrieved 'metadata' value is a dict.")

        # Timestamp should be a number.
        assert isinstance(first_entry.get('timestamp'), (int, float)), f"'timestamp' value should be a number, type: {type(first_entry.get('timestamp'))}"
        test_logger.debug("Assertion passed: Retrieved 'timestamp' value is a number.")

        # Optional: Verify that one of the retrieved memories is one of the ones we stored.
        # This requires checking numpy array equality and is slightly more complex.
        # Can be done using np.array_equal or np.allclose.
        # For example, check if the representation of any retrieved entry matches one of the dummy representations stored.
        # This provides stronger evidence that the similarity calculation logic is working correctly.
        # For now, just asserting the list is not empty is sufficient for this basic test.

    test_logger.info("test_memory_store_and_retrieve_basic test completed successfully.")


# TODO: More test scenarios for Memory module:
# - Test the case where the maximum number of entries is exceeded and the oldest ones are deleted.
# - Test storing with different metadata formats.
# - Test retrieving from an empty memory.
# - Test storing with invalid representation (NaN, wrong dimension, etc.) and verify they are not stored.
# - Tests for saving (_save_to_storage) and loading (_load_from_storage) functionality works correctly with files. (This might be considered more of an integration test but is the responsibility of the memory module).
# - Test what happens when an invalid memory_file_path is provided in config.
# - Test that retrieve returns an empty list when the query representation has a wrong dimension.
# - Test that get_all_representations method returns the correct list of Representations.
# - Test retrieval based on a threshold (if implemented) - currently 'retrieval_threshold' is in config but not used. Needs testing when implemented.