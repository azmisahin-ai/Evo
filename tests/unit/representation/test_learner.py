# tests/unit/representation/test_learner.py

import pytest
import numpy as np
import os
import sys
import logging

# Import the module to be tested
try:
    from src.representation.models import RepresentationLearner
    # Import necessary config and utils helpers
    from src.core.config_utils import get_config_value, load_config_from_yaml # load_config_from_yaml not strictly needed but okay
    from src.core.utils import check_numpy_input # Used for input validation in the module
except ImportError as e:
    pytest.fail(f"Failed to import src modules. Ensure you are running pytest from the project root or PYTHONPATH is configured correctly. Error: {e}")

# Create a logger for this test file. Level will be configured by conftest.
test_logger = logging.getLogger(__name__)
test_logger.info("src.representation.models and necessary helpers imported successfully for testing.")

@pytest.fixture(scope="module")
def dummy_learner_config():
    """Provides a dummy configuration dictionary for RepresentationLearner testing."""
    # Configuration values that RepresentationLearner's init or learn might need.
    # These should reflect the structure in main_config.yaml.
    config = {
        'processors': { # Need processor output dimensions to calculate expected input_dim
             'vision': {
                 'output_width': 64,
                 'output_height': 64,
             },
             'audio': {
                 'output_dim': 2, # Based on AudioProcessor's current implementation
             }
        },
        'representation': { # RepresentationLearner's own config section
             'input_dim': (64*64) + (64*64) + 2, # Explicitly calculate based on dummy processor sizes
             'representation_dim': 128, # Expected output vector dimension
             # ... other representation config if RepresentationLearner uses them ...
        },
        # ... other general config sections ...
    }
    test_logger.debug("Dummy learner config fixture created.")
    return config


@pytest.fixture(scope="module")
def representation_learner_instance(dummy_learner_config):
    """Provides a RepresentationLearner instance with dummy configuration."""
    test_logger.debug("Creating RepresentationLearner instance...")
    try:
        # Initialize the RepresentationLearner. Assuming its __init__ takes a config dict.
        learner = RepresentationLearner(dummy_learner_config)
        test_logger.debug("RepresentationLearner instance created.")
        yield learner # Provide the instance to the test function
        # Optional: Call cleanup if the module has one.
        # RepresentationLearner has a cleanup that calls its layers' cleanup.
        if hasattr(learner, 'cleanup'):
             learner.cleanup()
             test_logger.debug("RepresentationLearner cleanup called.")
    except Exception as e:
        # If initialization fails, fail the test.
        test_logger.error(f"RepresentationLearner initialization failed: {e}", exc_info=True)
        pytest.fail(f"RepresentationLearner initialization failed: {e}")


def test_representation_learner_init_with_valid_config(representation_learner_instance, dummy_learner_config):
    """Tests that RepresentationLearner initializes successfully with a valid config."""
    test_logger.info("test_representation_learner_init_with_valid_config test started.")
    # The fixture itself ensures successful initialization.
    # Additional assertions can check if configuration values were assigned correctly.
    assert representation_learner_instance.input_dim == (64*64) + (64*64) + 2 # Check calculated input_dim
    assert representation_learner_instance.representation_dim == 128
    # Check if internal layers were created (basic check)
    assert hasattr(representation_learner_instance, 'encoder') and representation_learner_instance.encoder is not None
    assert hasattr(representation_learner_instance, 'decoder') and representation_learner_instance.decoder is not None
    # Check layer dimensions (more detailed check)
    assert representation_learner_instance.encoder.input_dim == representation_learner_instance.input_dim
    assert representation_learner_instance.encoder.output_dim == representation_learner_instance.representation_dim
    assert representation_learner_instance.decoder.input_dim == representation_learner_instance.representation_dim
    assert representation_learner_instance.decoder.output_dim == representation_learner_instance.input_dim


    test_logger.info("test_representation_learner_init_with_valid_config test completed successfully.")


def test_representation_learner_basic_learn(representation_learner_instance, dummy_learner_config):
    """
    Tests that the RepresentationLearner's learn method produces the correct output format
    with basic dummy processed inputs.
    """
    test_logger.info("test_representation_learner_basic_learn test started.")

    # Dummy input data for RepresentationLearner.learn method.
    # This input should be in the format of Processors' output: {'visual': dict, 'audio': np.ndarray}.
    # The dictionary should contain data in the format expected by RepresentationLearner's input combination logic.

    # Dummy VisionProcessor output dictionary
    vis_out_w = get_config_value(dummy_learner_config, 'processors', 'vision', 'output_width', default=64, expected_type=int)
    vis_out_h = get_config_value(dummy_learner_config, 'processors', 'vision', 'output_height', default=64, expected_type=int)
    # Dummy grayscale and edges arrays (in VisionProcessor output format)
    dummy_processed_visual_gray = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
    dummy_processed_visual_edges = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
    dummy_visual_input = {'grayscale': dummy_processed_visual_gray, 'edges': dummy_processed_visual_edges}
    test_logger.debug(f"Created dummy visual input for learn: {list(dummy_visual_input.keys())}")


    # Dummy AudioProcessor output array
    audio_out_dim = get_config_value(dummy_learner_config, 'processors', 'audio', 'output_dim', default=2, expected_type=int)
    dummy_audio_input = np.random.rand(audio_out_dim).astype(np.float32) # Use float32 as specified in AudioProcessor output
    test_logger.debug(f"Created dummy audio input for learn: {dummy_audio_input.shape}, {dummy_audio_input.dtype}")


    # Combined input for RepresentationLearner.learn method
    dummy_processed_inputs = {
        'visual': dummy_visual_input,
        'audio': dummy_audio_input
    }
    test_logger.debug(f"Created dummy learn method input processed_inputs: {list(dummy_processed_inputs.keys())}")


    # Call the learn method
    try:
        learned_representation = representation_learner_instance.learn(dummy_processed_inputs)
        test_logger.debug(f"RepresentationLearner.learn called. Output type: {type(learned_representation)}")

    except Exception as e:
        test_logger.error(f"Unexpected error while executing RepresentationLearner.learn: {e}", exc_info=True)
        pytest.fail(f"Unexpected error while executing RepresentationLearner.learn: {e}")


    # --- Assert the Output ---
    # RepresentationLearner is expected to return a Representation vector (numpy array).
    # Its dimension should match the representation_dim from config.
    expected_representation_dim = representation_learner_instance.representation_dim

    # 1. Is the output a numpy array?
    assert isinstance(learned_representation, np.ndarray), f"Learn output should be a numpy array, received type: {type(learned_representation)}"
    test_logger.debug("Assertion passed: Output type is numpy array.")

    # 2. Does it have the expected shape? (1D representation vector expected)
    expected_representation_shape = (expected_representation_dim,)
    assert learned_representation.shape == expected_representation_shape, f"Representation vector should have the expected shape. Expected: {expected_representation_shape}, Received: {learned_representation.shape}"
    test_logger.debug("Assertion passed: Output has expected shape.")

    # 3. Does it have the expected dtype? (Generally float expected)
    # Check if it's a floating-point type.
    assert np.issubdtype(learned_representation.dtype, np.floating), f"Representation vector should be of a float type, received dtype: {learned_representation.dtype}"
    test_logger.debug("Assertion passed: Output has a float dtype.")

    # 4. Is the output not None?
    assert learned_representation is not None, "Representation vector should not be None."
    test_logger.debug("Assertion passed: Output is not None.")

    # 5. Are values reasonable? (e.g., should not contain NaN or Inf)
    assert not np.isnan(learned_representation).any(), "Representation vector contains NaN values."
    assert not np.isinf(learned_representation).any(), "Representation vector contains Inf values."
    test_logger.debug("Assertion passed: Output does not contain NaN or Inf.")


    test_logger.info("test_representation_learner_basic_learn test completed successfully.")


# TODO: More test scenarios for RepresentationLearner:
# - Test with empty or invalid input formats.
# - Test with different config values (e.g., representation_dim).
# - Verify that the input combination logic works correctly with missing modalities (e.g., only visual, only audio).
# - Test error handling scenarios.