# tests/unit/processing/test_audio.py

import pytest
import numpy as np
import os
import sys
import logging

# Import the module to be tested
try:
    from src.processing.audio import AudioProcessor
    # Import necessary config and utils helpers
    from src.core.config_utils import get_config_value
    from src.core.utils import check_numpy_input
except ImportError as e:
    pytest.fail(f"Failed to import src modules. Ensure you are running pytest from the project root or PYTHONPATH is configured correctly. Error: {e}")

# Create a logger for this test file. Level will be configured by conftest.
test_logger = logging.getLogger(__name__)
test_logger.info("src.processing.audio and necessary helpers imported successfully for testing.")


@pytest.fixture(scope="module")
def dummy_audio_processor_config():
    """Provides a dummy configuration dictionary for AudioProcessor testing."""
    # Configuration values that AudioProcessor's init or process might need.
    # These should reflect the structure in main_config.yaml.
    config = {
         'processors': {
             'audio': {
                 'audio_rate': 44100,        # Audio sample rate
                 'audio_chunk_size': 1024,   # Audio chunk size (used by sensor config, but processor might need it too)
                 'output_dim': 2, # Process output dimension (e.g., feature vector dimension)
                 # 'n_mfcc': 13,     # Config for MFCC calculation (if implemented)
             }
         },
         'cognition': { # Audio energy threshold is under cognition in main_config.yaml
             'audio_energy_threshold': 1000.0,
         },
        # ... other general config sections ...
    }
    test_logger.debug("Dummy audio processor config fixture created.")
    return config


@pytest.fixture(scope="module")
def audio_processor_instance(dummy_audio_processor_config):
    """Provides an AudioProcessor instance with dummy configuration."""
    test_logger.debug("Creating AudioProcessor instance...")
    try:
        # Initialize the AudioProcessor. Assuming its __init__ takes a config dict.
        processor = AudioProcessor(dummy_audio_processor_config)
        test_logger.debug("AudioProcessor instance created.")
        yield processor # Provide the instance to the test function
        # Optional: Call cleanup if the module has one.
        if hasattr(processor, 'cleanup'):
             processor.cleanup()
             test_logger.debug("AudioProcessor cleanup called.")
    except Exception as e:
        # If initialization fails, fail the test.
        test_logger.error(f"AudioProcessor initialization failed: {e}", exc_info=True)
        pytest.fail(f"AudioProcessor initialization failed: {e}")


def test_audio_processor_init_with_valid_config(audio_processor_instance):
    """Tests that AudioProcessor initializes successfully with a valid config."""
    test_logger.info("test_audio_processor_init_with_valid_config test started.")
    # The fixture itself ensures successful initialization.
    # Additional assertions can check if configuration values were assigned correctly.
    assert audio_processor_instance.audio_rate == 44100
    assert audio_processor_instance.output_dim == 2
    # Note: audio_energy_threshold is used by UnderstandingModule, not AudioProcessor itself,
    # so no need to check it here unless AudioProcessor uses it internally for some reason.

    test_logger.info("test_audio_processor_init_with_valid_config test completed successfully.")


def test_audio_processor_process_basic(audio_processor_instance, dummy_audio_processor_config):
    """
    Tests that the AudioProcessor's process method produces the correct output format
    with a basic dummy audio chunk input.
    """
    test_logger.info("test_audio_processor_process_basic test started.")

    # Dummy input data for AudioProcessor.process method (int16 numpy array).
    # Use chunk size from config.
    chunk_size = dummy_audio_processor_config['processors']['audio']['audio_chunk_size'] # Use processor config if available, else audio config
    if chunk_size is None:
         chunk_size = get_config_value(dummy_audio_processor_config, 'audio', 'audio_chunk_size', default=1024, expected_type=int) # Fallback to audio config if needed

    sample_rate = audio_processor_instance.audio_rate # Use the rate the processor initialized with

    # Create dummy int16 audio data (e.g., a simple tone)
    frequency = 440 # A4 note
    amplitude = np.iinfo(np.int16).max * 0.1
    t = np.linspace(0., chunk_size / sample_rate, chunk_size)
    dummy_chunk = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)

    test_logger.debug(f"Created dummy input audio chunk: {dummy_chunk.shape}, {dummy_chunk.dtype}")

    # Call the process method
    try:
        processed_output = audio_processor_instance.process(dummy_chunk)
        test_logger.debug(f"AudioProcessor.process called. Output type: {type(processed_output)}")

    except Exception as e:
        test_logger.error(f"Unexpected error while executing AudioProcessor.process: {e}", exc_info=True)
        pytest.fail(f"Unexpected error while executing AudioProcessor.process: {e}")


    # --- Assert the Output ---
    # AudioProcessor.process is expected to return a numpy array feature vector.
    # The format expected by RepresentationLearner is typically a 1D array of floats.

    # Get the expected output dimension from config.
    expected_output_dim = audio_processor_instance.output_dim # Use the dimension the processor initialized with

    # 1. Is the output a numpy array?
    assert isinstance(processed_output, np.ndarray), f"Process output should be a numpy array, received type: {type(processed_output)}"
    test_logger.debug("Assertion passed: Output type is numpy array.")

    # 2. Does it have the expected shape? (1D feature vector expected)
    expected_output_shape = (expected_output_dim,)
    assert processed_output.shape == expected_output_shape, f"Process output should have the expected shape. Expected: {expected_output_shape}, Received: {processed_output.shape}"
    test_logger.debug("Assertion passed: Output has expected shape.")

    # 3. Does it have the expected dtype? (Generally float expected)
    # Let's check if it's a floating-point type.
    assert np.issubdtype(processed_output.dtype, np.floating), f"Process output should be of a float type, received dtype: {processed_output.dtype}"
    test_logger.debug("Assertion passed: Output has a float dtype.")

    # 4. Is the output not None?
    assert processed_output is not None, "Process output should not be None."
    test_logger.debug("Assertion passed: Output is not None.")

    test_logger.info("test_audio_processor_basic_processing test completed successfully.")


# TODO: More test scenarios for AudioProcessor:
# - Test with different input sizes (if process is flexible).
# - Test with None or empty input (e.g., 0-size array).
# - Test with invalid input (e.g., wrong dtype input).
# - Test scenarios where config values (e.g., n_mfcc for future features) affect the output.
# - Test with simple deterministic signals (e.g., silence, pure tone) to verify energy and centroid calculation results.