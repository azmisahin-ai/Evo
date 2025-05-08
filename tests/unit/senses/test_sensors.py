# tests/unit/senses/test_sensors.py

import pytest
import numpy as np
import os
import sys
import logging
# We use pytest-mock for mocking. The 'mocker' fixture is provided automatically.
import cv2 # VisionSensor uses cv2, must be imported before mocking cv2.VideoCapture.
import pyaudio # AudioSensor uses pyaudio, must be imported before mocking pyaudio.PyAudio.
from unittest.mock import MagicMock, call # Import MagicMock and call

# conftest.py should set up sys.path, these imports should work directly now.
from src.senses.vision import VisionSensor
from src.senses.audio import AudioSensor
from src.core.config_utils import get_config_value
# setup_logging will not be called here, it's handled by conftest.
# from src.core.logging_utils import setup_logging # If conftest is used, this import is okay but not used directly here.
from src.core.utils import cleanup_safely # For fixture cleanup


# Create a logger for this test file. Level will be configured by conftest.
test_logger = logging.getLogger(__name__)
test_logger.info("src.senses modules and necessary helpers imported successfully.")


# VisionSensor Tests
@pytest.fixture(scope="function")
def dummy_vision_sensor_config():
    """Provides a dummy configuration dictionary for VisionSensor testing."""
    config = {
        'vision': {
            'camera_index': 0,       # Dummy camera index for the test
            'dummy_width': 640,      # Simulated frame width
            'dummy_height': 480,     # Simulated frame height
            'is_dummy': False,       # Simulate real hardware mode (will use mocking)
        },
        # ... other general config sections ...
    }
    test_logger.debug("Dummy vision sensor config fixture created.")
    return config

@pytest.fixture(scope="function")
def vision_sensor_instance(dummy_vision_sensor_config, mocker):
    """Provides a VisionSensor instance with dummy configuration and mocks."""
    test_logger.debug("Creating VisionSensor instance (mocking cv2.VideoCapture)...")

    # Mock the cv2.VideoCapture class.
    # The cv2.VideoCapture(self.camera_index) call inside VisionSensor.__init__ will now use this mock class.
    mock_cv2_videocapture_class = mocker.patch('cv2.VideoCapture')

    # Get the instance that the Mock VideoCapture class will return when called (the mock cap object).
    # VisionSensor's self.cap attribute will be this mock instance.
    mock_cv2_videocapture_instance = mock_cv2_videocapture_class.return_value

    # Configure the methods of the mock VideoCapture instance used in VisionSensor init.
    mock_cv2_videocapture_instance.isOpened.return_value = True # Simulate: Camera opened successfully

    # The get() method is used in VisionSensor's init to get frame dimensions.
    # Use side_effect to return different mock values for different CAP_PROP values.
    # Retrieve dummy dimensions from the config fixture.
    dummy_height = get_config_value(dummy_vision_sensor_config, 'vision', 'dummy_height', default=480, expected_type=int)
    dummy_width = get_config_value(dummy_vision_sensor_config, 'vision', 'dummy_width', default=640, expected_type=int)
    mock_cv2_videocapture_instance.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: dummy_width, # Use dimensions from dummy config
        cv2.CAP_PROP_FRAME_HEIGHT: dummy_height, # Use dimensions from dummy config
        # Add other CAP_PROP values if VisionSensor init retrieves them (e.g., fps)
        cv2.CAP_PROP_FPS: 30.0, # Add a default FPS value
    }.get(prop, 0) # Return a default of 0 if an unknown property is requested


    # Mock the release() method called in VisionSensor's cleanup.
    # Can be asserted later in the teardown.
    mock_cv2_videocapture_instance.release.return_value = None

    # Mock the read() method called within VisionSensor.capture_frame.
    # Test functions will configure the return value of this mock based on their scenario.
    # Default to a failure read (will be overridden in test scenarios).
    mock_cv2_videocapture_instance.read.return_value = (False, None)


    try:
        # Initialize the VisionSensor. This will use the mocked cv2.VideoCapture.
        sensor = VisionSensor(dummy_vision_sensor_config)

        # Verify that the mock classes and methods were called correctly during VisionSensor init.
        mock_cv2_videocapture_class.assert_called_once_with(sensor.camera_index)
        mock_cv2_videocapture_instance.isOpened.assert_called_once() # Called once during init
        # Check that get was called for expected properties
        mock_cv2_videocapture_instance.get.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH)
        mock_cv2_videocapture_instance.get.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT)
        # release should NOT have been called during init (as fixture simulates successful opening).
        mock_cv2_videocapture_instance.release.assert_not_called()


        test_logger.debug("VisionSensor instance created (with cv2.VideoCapture mock).")
        yield sensor # Provide the instance to the test function (self.cap is now the mock instance)

        # --- Fixture Teardown (Runs after the test function completes) ---
        # Call the cleanup method of the sensor instance (if it exists and ran).
        # VisionSensor's cleanup method should call stop_stream(), which should call self.cap.release().
        if hasattr(sensor, 'cleanup'):
             test_logger.debug("VisionSensor fixture teardown: calling cleanup.")
             # Use cleanup_safely to log errors during cleanup but not stop pytest.
             cleanup_safely(sensor.cleanup, logger_instance=test_logger, error_message="Error during VisionSensor instance cleanup (teardown)")
             test_logger.debug("VisionSensor cleanup called.")
             # Verify that the mock release method was called by cleanup.
             # It should be called exactly ONCE if it wasn't called during init.
             mock_cv2_videocapture_instance.release.assert_called_once()
             test_logger.debug("VisionSensor fixture teardown: Mock release call verified.")


    except Exception as e:
        # Log any errors during fixture setup or teardown.
        test_logger.error(f"Error during VisionSensor fixture setup or cleanup: {e}", exc_info=True)
        pytest.fail(f"VisionSensor fixture error: {e}")


def test_vision_sensor_capture_frame_success(vision_sensor_instance, mocker, dummy_vision_sensor_config):
    """
    Tests that the VisionSensor's capture_frame method returns a frame in the expected format
    from the mocked hardware in a successful read scenario.
    """
    test_logger.info("test_vision_sensor_capture_frame_success test started.")

    # Get the mock cap object from the VisionSensor instance created by the fixture.
    mock_cap_instance = vision_sensor_instance.cap
    # Configure the return value of the mock cap instance's read method: (success flag, frame data)
    # Retrieve dummy dimensions from the config fixture.
    dummy_height = get_config_value(dummy_vision_sensor_config, 'vision', 'dummy_height', default=480, expected_type=int)
    dummy_width = get_config_value(dummy_vision_sensor_config, 'vision', 'dummy_width', default=640, expected_type=int)
    # Create realistic dummy frame data (uint8).
    mock_frame_data = np.random.randint(0, 256, size=(dummy_height, dummy_width, 3), dtype=np.uint8)
    mock_cap_instance.read.return_value = (True, mock_frame_data) # Simulate successful read


    # Call the capture_frame method
    try:
        captured_frame = vision_sensor_instance.capture_frame()
        test_logger.debug(f"VisionSensor.capture_frame called. Output type: {type(captured_frame)}")

    except Exception as e:
        # Log and fail the test if an unexpected error occurs during capture_frame execution.
        test_logger.error(f"Unexpected error while executing VisionSensor.capture_frame: {e}", exc_info=True)
        pytest.fail(f"Unexpected error while executing VisionSensor.capture_frame: {e}")


    # --- Assert the Output ---
    # The capture_frame method should return a numpy array.
    assert isinstance(captured_frame, np.ndarray), f"Capture output should be a numpy array, received type: {type(captured_frame)}"
    test_logger.debug("Assertion passed: Output type is numpy array.")

    # The numpy array should have the expected shape and dtype (BGR uint8).
    expected_shape = (dummy_height, dummy_width, 3) # BGR format
    expected_dtype = np.uint8
    assert captured_frame.shape == expected_shape, f"Capture output should have the expected shape. Expected: {expected_shape}, Received: {captured_frame.shape}"
    assert captured_frame.dtype == expected_dtype, f"Capture output should have the expected dtype. Expected: {expected_dtype}, Received: {captured_frame.dtype}"
    test_logger.debug("Assertion passed: Output has expected shape and dtype.")

    # Verify that the mock read method was called once within capture_frame.
    mock_cap_instance.read.assert_called_once()
    test_logger.debug("Assertion passed: Mock read method called once.")

    # Verify that the returned array is equal to the mock data (assuming VisionSensor doesn't process the frame itself).
    assert np.array_equal(captured_frame, mock_frame_data)
    test_logger.debug("Assertion passed: Captured frame is equal to mock data.")

    # Verify that is_camera_available remains True (as the read was successful).
    assert vision_sensor_instance.is_camera_available, "is_camera_available should remain True."
    test_logger.debug("Assertion passed: is_camera_available is True.")


    test_logger.info("test_vision_sensor_capture_frame_success test completed successfully.")


def test_vision_sensor_capture_frame_failure(vision_sensor_instance, mocker):
    """
    Tests that the VisionSensor's capture_frame method returns the expected output (a dummy frame)
    and transitions to simulated mode when frame reading fails.
    """
    test_logger.info("test_vision_sensor_capture_frame_failure test started.")

    # Get the mock cap object from the VisionSensor instance created by the fixture.
    mock_cap_instance = vision_sensor_instance.cap
    # Configure the return value of the mock cap instance's read method: (failure flag, None)
    mock_cap_instance.read.return_value = (False, None) # Simulate read failure


    # Call the capture_frame method
    try:
        captured_frame = vision_sensor_instance.capture_frame()
        test_logger.debug(f"VisionSensor.capture_frame called. Output type: {type(captured_frame)}")

    except Exception as e:
        # Log and fail the test if an unexpected error occurs during capture_frame execution.
        test_logger.error(f"Unexpected error while executing VisionSensor.capture_frame: {e}", exc_info=True)
        pytest.fail(f"Unexpected error while executing VisionSensor.capture_frame: {e}")


    # --- Assert the Output ---
    # When reading fails, capture_frame should return a dummy frame.
    assert isinstance(captured_frame, np.ndarray), f"Capture output should be a numpy array (dummy frame), received type: {type(captured_frame)}"
    test_logger.debug("Assertion passed: Output type is numpy array (dummy frame).")

    # The dummy frame should have the expected shape and dtype (from config dummy dimensions).
    dummy_height = get_config_value(vision_sensor_instance.config, 'vision', 'dummy_height', default=480, expected_type=int)
    dummy_width = get_config_value(vision_sensor_instance.config, 'vision', 'dummy_width', default=640, expected_type=int)
    expected_shape = (dummy_height, dummy_width, 3) # BGR dummy format
    expected_dtype = np.uint8
    assert captured_frame.shape == expected_shape, f"Capture output should have the expected (dummy) shape. Expected: {expected_shape}, Received: {captured_frame.shape}"
    assert captured_frame.dtype == expected_dtype, f"Capture output should have the expected (dummy) dtype. Expected: {expected_dtype}, Received: {captured_frame.dtype}"
    test_logger.debug("Assertion passed: Output has expected (dummy) shape and dtype.")

    # Verify that the mock read method was called once within capture_frame.
    mock_cap_instance.read.assert_called_once()
    test_logger.debug("Assertion passed: Mock read method called once.")

    # Verify that the is_camera_available flag was updated to False.
    assert not vision_sensor_instance.is_camera_available, "is_camera_available should be updated to False."
    test_logger.debug("Assertion passed: is_camera_available is False.")

    test_logger.info("test_vision_sensor_capture_frame_failure test completed successfully.")

# TODO: Tests for VisionSensor initialization/capture when Exception is raised (simulate using mock's side_effect).
# TODO: Test VisionSensor when is_dummy=True in config (should not call real hardware mocks, should produce dummy data).
# TODO: Test capture_chunk method (if it exists for visual - usually not).
# TODO: Test cleanup method (verify mock release is called) -> Already implicitly tested in fixture teardown.


# AudioSensor Tests
@pytest.fixture(scope="function")
def dummy_audio_sensor_config():
    """Provides a dummy configuration dictionary for AudioSensor testing."""
    config = {
        'audio': {
            'audio_rate': 44100,        # Audio sample rate
            'audio_chunk_size': 1024,   # Number of audio samples to read per chunk
            'audio_input_device_index': None, # Default device (will be mocked)
            'is_dummy': False,         # Simulate real hardware mode (will use mocking)
        },
        # ... other general config sections ...
    }
    test_logger.debug("Dummy audio sensor config fixture created.")
    return config

@pytest.fixture(scope="function")
def audio_sensor_instance(dummy_audio_sensor_config, mocker):
    """Provides an AudioSensor instance with dummy configuration and mocks."""
    test_logger.debug("Creating AudioSensor instance (mocking pyaudio.PyAudio)...")

    # Mock the pyaudio.PyAudio class.
    # The pyaudio.PyAudio() call inside AudioSensor.__init__ will now use this mock class.
    mock_pyaudio_class = mocker.patch('pyaudio.PyAudio')

    # Get the instance that the Mock PyAudio class will return when called (the mock p object).
    # AudioSensor's self.p attribute will be this mock instance.
    mock_pyaudio_instance = mock_pyaudio_class.return_value

    # Mock the getDefaultInputDeviceInfo() method of the mock PyAudio instance (simulating finding the default device).
    # AudioSensor.__init__ calls this method if audio_input_device_index=None in config.
    # The mock return value simulates the device index that will be used in the open() method.
    mock_default_device_info = {'index': 0, 'name': 'Mock Default Audio Device'} # Simulate default device index as 0.
    mock_pyaudio_instance.getDefaultInputDeviceInfo.return_value = mock_default_device_info

    # Mock the open() method of the mock PyAudio instance.
    # This open() method should return a Stream object, so mock that object too.
    mock_stream_instance = mocker.Mock() # Create a separate mock Stream object.
    mock_pyaudio_instance.open.return_value = mock_stream_instance # The mock open() should return this mock stream object.

    # Mock the terminate() method of the PyAudio instance (called in cleanup).
    mock_pyaudio_instance.terminate.return_value = None


    # Mock the methods of the Stream object used by AudioSensor.
    mock_stream_instance.stop_stream.return_value = None
    mock_stream_instance.close.return_value = None
    # Mock the read() method called within AudioSensor.capture_chunk.
    # Test functions will configure the return value of this mock based on their scenario.
    # Default to empty bytes (will be overridden in test scenarios).
    chunk_size = get_config_value(dummy_audio_sensor_config, 'audio', 'audio_chunk_size', default=1024, expected_type=int)
    bytes_per_sample = 2 # For int16
    mock_stream_instance.read.return_value = b'\x00' * chunk_size * bytes_per_sample


    try:
        # Initialize the AudioSensor. This will use the mocked pyaudio.PyAudio.
        sensor = AudioSensor(dummy_audio_sensor_config)

        # Verify that mock classes and methods were called correctly during AudioSensor init.
        mock_pyaudio_class.assert_called_once() # pyaudio.PyAudio() should be called once.

        # If audio_input_device_index is None in config, getDefaultInputDeviceInfo should be called, and open called with that index.
        # If it's an int in config, getDefaultInputDeviceInfo should NOT be called, and open called with the config int index.
        device_index_used_by_sensor = sensor.audio_input_device_index # The value (None or int) from config.
        if device_index_used_by_sensor is None:
            mock_pyaudio_instance.getDefaultInputDeviceInfo.assert_called_once() # Verify default device info is obtained.
            # The index used for the open() call should be the one from the mock getDefaultInputDeviceInfo's return.
            device_index_for_open_call = mock_default_device_info['index'] # The index from mock default device info
        else:
            mock_pyaudio_instance.getDefaultInputDeviceInfo.assert_not_called() # If specified in config, default info is not obtained.
            device_index_for_open_call = device_index_used_by_sensor # The int index from config is used.


        # Verify that the AudioSensor init called the PyAudio instance's open method with correct arguments.
        expected_open_args = {
            'rate': sensor.audio_rate,
            'channels': 1, # AudioSensor assumes mono audio
            'format': pyaudio.paInt16, # AudioSensor assumes int16 format
            'input': True,
            'frames_per_buffer': sensor.audio_chunk_size,
            'input_device_index': device_index_for_open_call # The actual index used in the open call
        }
        mock_pyaudio_instance.open.assert_called_once_with(**expected_open_args)

        # terminate, stop_stream, close should NOT be called during init.
        mock_pyaudio_instance.terminate.assert_not_called()
        mock_stream_instance.stop_stream.assert_not_called()
        mock_stream_instance.close.assert_not_called()


        test_logger.debug("AudioSensor instance created (with pyaudio mock).")
        yield sensor # Provide the instance to the test function (self.p, self.stream are now mock instances)

        # --- Fixture Teardown (Runs after the test function completes) ---
        # Call the cleanup method of the sensor instance (if it exists and ran).
        # AudioSensor's cleanup method should call stop_stream() and terminate_pyaudio().
        if hasattr(sensor, 'cleanup'):
             test_logger.debug("AudioSensor fixture teardown: calling cleanup.")
             # Use cleanup_safely to log errors during cleanup but not stop pytest.
             cleanup_safely(sensor.cleanup, logger_instance=test_logger, error_message="Error during AudioSensor instance cleanup (teardown)")
             test_logger.debug("AudioSensor cleanup called.")
             # Verify that the mock methods were called by cleanup.
             # Should be called once if they weren't called during init.
             mock_pyaudio_instance.terminate.assert_called_once()
             mock_stream_instance.stop_stream.assert_called_once() # stop_stream is called once in AudioSensor code.
             mock_stream_instance.close.assert_called_once() # close is called once in AudioSensor code.
             test_logger.debug("AudioSensor fixture teardown: Mock calls verified.")


    except Exception as e:
        # Log any errors during fixture setup or teardown.
        test_logger.error(f"Error during AudioSensor fixture setup or cleanup: {e}", exc_info=True)
        pytest.fail(f"AudioSensor fixture error: {e}")


def test_audio_sensor_capture_chunk_success(audio_sensor_instance, mocker, dummy_audio_sensor_config):
    """
    Tests that the AudioSensor's capture_chunk method returns an audio block (chunk) in the expected format
    from the mocked hardware in a successful read scenario.
    """
    test_logger.info("test_audio_sensor_capture_chunk_success test started.")

    # Get the mock stream object from the AudioSensor instance created by the fixture.
    mock_stream_instance = audio_sensor_instance.stream # Stream object is stored as self.stream in AudioSensor.
    # Configure the return value of the mock stream instance's read method (bytes data).
    # Retrieve chunk size from the config fixture.
    chunk_size = get_config_value(dummy_audio_sensor_config, 'audio', 'audio_chunk_size', default=1024, expected_type=int)
    bytes_per_sample = 2 # For int16
    # Create dummy raw audio data (bytes) - create a numpy array and convert to bytes for more realistic test data.
    dummy_audio_np = (np.random.rand(chunk_size) * 32767).astype(np.int16) # Simulate int16 audio data
    mock_audio_bytes = dummy_audio_np.tobytes()

    mock_stream_instance.read.return_value = mock_audio_bytes # Set the bytes data to be returned by the mock read.


    # Call the capture_chunk method
    try:
        captured_chunk = audio_sensor_instance.capture_chunk()
        test_logger.debug(f"AudioSensor.capture_chunk called. Output type: {type(captured_chunk)}")

    except Exception as e:
        # Log and fail the test if an unexpected error occurs during capture_chunk execution.
        test_logger.error(f"Unexpected error while executing AudioSensor.capture_chunk: {e}", exc_info=True)
        pytest.fail(f"Unexpected error while executing AudioSensor.capture_chunk: {e}")


    # --- Assert the Output ---
    # The capture_chunk method should convert the read bytes data into a numpy array (int16).
    assert isinstance(captured_chunk, np.ndarray), f"Capture chunk output should be a numpy array, received type: {type(captured_chunk)}"
    test_logger.debug("Assertion passed: Output type is numpy array.")

    # The numpy array should have the expected shape and dtype (processed raw audio, int16).
    expected_shape = (chunk_size,)
    expected_dtype = np.int16
    assert captured_chunk.shape == expected_shape, f"Capture chunk output should have the expected shape. Expected: {expected_shape}, Received: {captured_chunk.shape}"
    assert captured_chunk.dtype == expected_dtype, f"Capture chunk output should have the expected dtype. Expected: {expected_dtype}, Received: {captured_chunk.dtype}"
    test_logger.debug("Assertion passed: Output has expected shape and dtype.")

    # Verify that the mock stream read method was called once with the correct arguments (chunk_size and exception_on_overflow=False).
    # PyAudio's read method typically expects frames and exception_on_overflow=False.
    mock_stream_instance.read.assert_called_once_with(chunk_size, exception_on_overflow=False)
    test_logger.debug("Assertion passed: Mock stream read method called once (with expected arguments).")

    # Verify that the returned array was correctly converted from the mock data.
    # np.frombuffer converts bytes to a numpy array.
    assert np.array_equal(captured_chunk, dummy_audio_np) # Was the bytes data correctly converted to an int16 numpy array?
    test_logger.debug("Assertion passed: Captured chunk is equal to mock data (numpy).")

    # Verify that is_audio_available remains True (as the read was successful).
    assert audio_sensor_instance.is_audio_available, "is_audio_available should remain True."
    test_logger.debug("Assertion passed: is_audio_available is True.")


    test_logger.info("test_audio_sensor_capture_chunk_success test completed successfully.")

# TODO: Tests for AudioSensor initialization/capture when Exception (IOError or other) is raised (simulate by setting mock read's side_effect).
# TODO: Tests for AudioSensor initialization when real pyaudio.PyAudio() or open() raises an error (simulate by setting mocks' side_effect).
# TODO: Test initialization with different audio_input_device_index values (asserting the argument to the mock open call).
# TODO: Test that it produces simulated data when is_dummy=True in config.
# TODO: Test cleanup method (verify mock stop_stream, close, terminate are called) -> Already implicitly tested in fixture teardown.