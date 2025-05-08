# tests/unit/processing/test_vision.py

import pytest
import numpy as np
import os
import sys
import logging
import cv2 # Import cv2 as VisionProcessor uses it internally

# Import the module to be tested
try:
    from src.processing.vision import VisionProcessor
    # Import necessary config and utils helpers
    from src.core.config_utils import get_config_value
    from src.core.utils import check_numpy_input
except ImportError as e:
    pytest.fail(f"Failed to import src modules. Ensure you are running pytest from the project root or PYTHONPATH is configured correctly. Error: {e}")

# Create a logger for this test file. Level will be configured by conftest.
test_logger = logging.getLogger(__name__)
test_logger.info("src.processing.vision and necessary helpers imported successfully for testing.")


@pytest.fixture(scope="module")
def dummy_vision_processor_config():
    """Provides a dummy configuration dictionary for VisionProcessor testing."""
    # Configuration values that VisionProcessor's init or process might need.
    # These should reflect the structure in main_config.yaml.
    config = {
        'processors': {
             'vision': {
                 'output_width': 64, # Expected output dimensions
                 'output_height': 64,
                 'canny_low_threshold': 50, # Canny thresholds
                 'canny_high_threshold': 150,
             }
        },
        'cognition': { # Brightness thresholds are under cognition in main_config.yaml
             'brightness_threshold_high': 200.0,
             'brightness_threshold_low': 50.0,
             'visual_edges_threshold': 50.0 # Edge threshold is also under cognition for UnderstandingModule
        },
        'vision': { # Vision sensor dummy dimensions are needed for creating dummy input frame
            'dummy_height': 480,
            'dummy_width': 640,
        }
        # ... other general config sections if VisionProcessor needs them ...
    }
    test_logger.debug("Dummy vision processor config fixture created.")
    return config


@pytest.fixture(scope="module")
def vision_processor_instance(dummy_vision_processor_config):
    """Provides a VisionProcessor instance with dummy configuration."""
    test_logger.debug("Creating VisionProcessor instance...")
    try:
        # Initialize the VisionProcessor. Assuming its __init__ takes a config dict.
        processor = VisionProcessor(dummy_vision_processor_config)
        test_logger.debug("VisionProcessor instance created.")
        yield processor # Provide the instance to the test function
        # Optional: Call cleanup if the module has one.
        if hasattr(processor, 'cleanup'):
             processor.cleanup()
             test_logger.debug("VisionProcessor cleanup called.")
    except Exception as e:
        # If initialization fails, fail the test.
        test_logger.error(f"VisionProcessor initialization failed: {e}", exc_info=True)
        pytest.fail(f"VisionProcessor initialization failed: {e}")


def test_vision_processor_init_with_valid_config(vision_processor_instance, dummy_vision_processor_config):
    """Tests that VisionProcessor initializes successfully with a valid config."""
    test_logger.info("test_vision_processor_init_with_valid_config test started.")
    # The fixture itself ensures successful initialization.
    # Additional assertions can check if configuration values were assigned correctly.
    assert vision_processor_instance.output_width == 64
    assert vision_processor_instance.output_height == 64
    assert vision_processor_instance.canny_low_threshold == 50
    assert vision_processor_instance.canny_high_threshold == 150
    assert vision_processor_instance.brightness_threshold_high == 200.0
    assert vision_processor_instance.brightness_threshold_low == 50.0
    assert vision_processor_instance.visual_edges_threshold == 50.0 # Check that visual_edges_threshold was read


    test_logger.info("test_vision_processor_init_with_valid_config test completed successfully.")


def test_vision_processor_process_basic(vision_processor_instance, dummy_vision_processor_config):
    """
    Tests that the VisionProcessor's process method produces the correct output format
    with a basic dummy image input.
    """
    test_logger.info("test_vision_processor_process_basic test started.")

    # Dummy input data for VisionProcessor.process method (BGR numpy array).
    # Use dummy sensor dimensions from the config fixture.
    dummy_input_height = get_config_value(dummy_vision_processor_config, 'vision', 'dummy_height', default=480, expected_type=int)
    dummy_input_width = get_config_value(dummy_vision_processor_config, 'vision', 'dummy_width', default=640, expected_type=int)

    # Use the retrieved dummy input size for the dummy frame.
    input_height = dummy_input_height
    input_width = dummy_input_width
    dummy_frame = np.random.randint(0, 256, size=(input_height, input_width, 3), dtype=np.uint8)
    test_logger.debug(f"Created dummy input image: {dummy_frame.shape}, {dummy_frame.dtype}")


    # Call the process method
    try:
        processed_output = vision_processor_instance.process(dummy_frame)
        test_logger.debug(f"VisionProcessor.process called. Output type: {type(processed_output)}")

    except Exception as e:
        test_logger.error(f"Unexpected error while executing VisionProcessor.process: {e}", exc_info=True)
        pytest.fail(f"Unexpected error while executing VisionProcessor.process: {e}")


    # --- Assert the Output ---
    # VisionProcessor.process is expected to return a dictionary with 'grayscale' and 'edges' keys.
    # The arrays inside should have the configured output dimensions and uint8 dtype.

    # 1. Is the output a dictionary?
    assert isinstance(processed_output, dict), f"Process output should be a dict, received type: {type(processed_output)}"
    test_logger.debug("Assertion passed: Output type is dict.")

    # 2. Does it contain the expected keys?
    expected_keys = ['grayscale', 'edges']
    for key in expected_keys:
        assert key in processed_output, f"Output dictionary should contain key '{key}'."
        assert isinstance(processed_output[key], np.ndarray), f"Value for key '{key}' should be a numpy array."
        # Check the shape and dtype of the output arrays based on processor config.
        expected_output_shape = (
             vision_processor_instance.output_height,
             vision_processor_instance.output_width
        )
        assert processed_output[key].shape == expected_output_shape, f"Output for '{key}' should have expected shape. Expected: {expected_output_shape}, Received: {processed_output[key].shape}"
        assert processed_output[key].dtype == np.uint8, f"Output for '{key}' should have expected dtype. Expected: np.uint8, Received: {processed_output[key].dtype}"
        test_logger.debug(f"Assertion passed: Output dictionary contains key '{key}', is numpy array, and has expected shape/dtype.")


    # 3. Is the output not None?
    assert processed_output is not None, "Process output should not be None."
    test_logger.debug("Assertion passed: Output is not None.")

    test_logger.info("test_vision_processor_basic_processing test completed successfully.")


# TODO: More test scenarios for VisionProcessor:
# - Test with grayscale input image.
# - Test with None or empty input.
# - Test with invalid input (wrong dtype, wrong dimensions).
# - Test that thresholds affect the output content (requires more advanced assertions).