# src/processing/vision.py
#
# Processes visual sensory data.
# Extracts basic visual features (e.g., resizing, grayscale, edges) from raw pixel data.
# Part of Evo's Phase 1 processing capabilities.

import cv2 # OpenCV library, for camera capture and basic image processing. Should be in requirements.txt.
import time # For timing or simulation if needed. Not directly used currently.
import numpy as np # For numerical operations and arrays.
import logging # For logging.

# Import utility functions (input checks from src/core/utils, config from src/core/config_utils)
from src.core.config_utils import get_config_value
from src.core.utils import check_input_not_none, check_numpy_input # <<< Utils imports


# Create a logger for this module
# Returns a logger named 'src.processing.vision'.
logger = logging.getLogger(__name__)

class VisionProcessor:
    """
    Evo's visual data processing class (Phase 1 implementation).

    Receives raw visual input (frame) from the VisionSensor.
    Performs basic operations (resizing, grayscale conversion, edge detection)
    to prepare it for the RepresentationLearner.
    Manages potential errors during processing and ensures flow continuity.
    Returns a dictionary containing different processed features.
    """
    def __init__(self, config):
        """
        Initializes the VisionProcessor.

        Args:
            config (dict): Processor configuration settings (full config dict).
                           Settings for this module are read from the 'processors.vision' section
                           and also brightness thresholds from the 'cognition' section.
        """
        self.config = config # VisionProcessor receives the full config
        logger.info("VisionProcessor initializing...")

        # Get output dimensions and Canny thresholds from config using get_config_value.
        # These settings are under the 'processors.vision' key.
        self.output_width = get_config_value(config, 'processors', 'vision', 'output_width', default=64, expected_type=int, logger_instance=logger)
        self.output_height = get_config_value(config, 'processors', 'vision', 'output_height', default=64, expected_type=int, logger_instance=logger)
        self.canny_low_threshold = get_config_value(config, 'processors', 'vision', 'canny_low_threshold', default=50, expected_type=int, logger_instance=logger)
        self.canny_high_threshold = get_config_value(config, 'processors', 'vision', 'canny_high_threshold', default=150, expected_type=int, logger_instance=logger)

        # Get brightness thresholds from config (These are under the 'cognition' key)
        # Corrected: Add these attributes and read them from config.
        self.brightness_threshold_high = get_config_value(config, 'cognition', 'brightness_threshold_high', default=200.0, expected_type=(float, int), logger_instance=logger)
        self.brightness_threshold_low = get_config_value(config, 'cognition', 'brightness_threshold_low', default=50.0, expected_type=(float, int), logger_instance=logger)

        # Ensure valid output dimensions (must be positive)
        if self.output_width <= 0 or self.output_height <= 0:
             logger.error(f"VisionProcessor: Invalid output dimensions in config: {self.output_width}x{self.output_height}. Using defaults (64x64).")
             self.output_width = 64
             self.output_height = 64
             # Policy: Don't make initialization critical in this case, just log and use defaults.

        logger.info(f"VisionProcessor initialized. Output dimensions: {self.output_width}x{self.output_height}, Canny Thresholds: [{self.canny_low_threshold}, {self.canny_high_threshold}], Brightness Thresholds: [{self.brightness_threshold_low}, {self.brightness_threshold_high}]")


    def process(self, visual_input):
        """
        Processes raw visual input and extracts basic features.
        ... (Docstring same) ...
        """
        # Error handling: If input is None or not of expected type
        # Use check_input_not_none function (logs and returns False if None)
        if not check_input_not_none(visual_input, input_name="visual_input for VisionProcessor", logger_instance=logger):
             # If input is None, skip processing and return an empty dictionary (Graceful failure).
             logger.debug("VisionProcessor.process: Input is None. Returning empty dictionary.")
             return {} # Return empty dictionary instead of None.


        # Check if the input is a numpy array and has the correct dtype (uint8).
        # Use check_numpy_input function. This function also checks for np.ndarray type.
        # Expected dimensions can be 2D (gray) or 3D (color). uint8 dtype is expected.
        # check_numpy_input logs an ERROR and returns False on failure.
        if not check_numpy_input(visual_input, expected_dtype=np.uint8, expected_ndim=(2, 3), input_name="visual_input for VisionProcessor", logger_instance=logger):
             # If type or dtype/dimensions are invalid, stop processing and return an empty dictionary.
             logger.error("VisionProcessor.process: Input is not a numpy array or has wrong dtype/dimensions. Returning empty dictionary.") # check_numpy_input already logs internally.
             return {} # Return empty dictionary instead of None.


        # DEBUG log: Log input details (dimensions and type). Similar log exists in check_numpy_input, but kept here too.
        logger.debug(f"VisionProcessor.process: Visual data received. Shape: {visual_input.shape}, Dtype: {visual_input.dtype}. Processing...")

        processed_features = {} # Dictionary to hold processed results. Starts empty.

        try:
            # 1. Convert to grayscale (If input is color).
            # Assuming input shape is (Height, Width, Channels) and channel count is 3 (BGR).
            # Checking len(visual_input.shape) == 3 and visual_input.shape[2] == 3 is sufficient.
            if len(visual_input.shape) == 3 and visual_input.shape[2] == 3:
                gray_frame = cv2.cvtColor(visual_input, cv2.COLOR_BGR2GRAY)
                logger.debug("VisionProcessor.process: Visual data converted from BGR to grayscale.")
            elif len(visual_input.shape) == 2:
                # If the input is already 2D (like grayscale), use it directly.
                gray_frame = visual_input.copy() # Make a copy to avoid modifying the original input.
                logger.debug("VisionProcessor.process: Visual input appears to be already grayscale. Skipping conversion.")
            else:
                 # The ndim=(2,3) check in check_numpy_input should prevent reaching here, but included for robustness.
                 logger.warning(f"VisionProcessor.process: Unexpected visual input dimensions (not ndim 2 or 3): {visual_input.shape}. Could not process.")
                 return {} # If dimensions are unexpected, cannot process, return empty dictionary.

            # Ensure the grayscale frame is still uint8 dtype after conversion/copy.
            if gray_frame.dtype != np.uint8:
                 gray_frame = gray_frame.astype(np.uint8)
                 logger.debug(f"VisionProcessor.process: Ensured grayscale frame dtype is uint8.")


            # 2. Yeniden boyutlandır (Yapılandırmada belirtilen output_width x output_height boyutuna).
            # Hedef boyut tuple olarak verilir: (genişlik, yükseklik).
            # Interpolation metodu belirtilebilir, INTER_AREA küçültme için iyidir.
            # Hedef boyutların pozitif olduğu init'te kontrol edildi veya varsayılan atandı.
            resized_frame = cv2.resize(gray_frame, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)
            logger.debug(f"VisionProcessor.process: Görsel veri ({self.output_width}, {self.output_height}) boyutuna yeniden boyutlandırıldı. Shape: {resized_frame.shape}, Dtype: {resized_frame.dtype}")

            # Add the grayscale, resized frame to the processed features dictionary.
            processed_features['grayscale'] = resized_frame


            # 3. Kenar tespiti uygula.
            # Canny kenar dedektörü gri tonlamalı 8-bit (uint8) resimler üzerinde çalışır.
            # resized_frame uint8 ve gri olduğu için doğrudan kullanabiliriz.
            # Eşikler init'te yapılandırmadan alındı ve int olduğu kontrol edildi.
            edges = cv2.Canny(resized_frame, self.canny_low_threshold, self.canny_high_threshold)
            logger.debug(f"VisionProcessor.process: Canny kenar tespiti uygulandı. Shape: {edges.shape}, Dtype: {edges.dtype}")

            # Add the edge map to the processed features dictionary.
            processed_features['edges'] = edges


            # TODO: In the future: Add more low-level features (e.g., color histograms - from original color image, simple texture features).

            # DEBUG Log: Average brightness and edge density comparison with thresholds.
            # Only calculate means if the arrays are not empty.
            if 'grayscale' in processed_features and processed_features['grayscale'].size > 0:
                 avg_brightness = np.mean(processed_features['grayscale'])
                 # Log comparing the mean brightness to config thresholds.
                 # Use self.brightness_threshold_high/low which are now assigned in __init__
                 logger.debug(f"VisionProcessor.process: Avg Brightness: {avg_brightness:.2f} (High: {self.brightness_threshold_high:.2f}, Low: {self.brightness_threshold_low:.2f})")
            if 'edges' in processed_features and processed_features['edges'].size > 0:
                 avg_edges = np.mean(processed_features['edges'])
                 # Log comparing the mean edges to config threshold.
                 logger.debug(f"VisionProcessor.process: Avg Edges: {avg_edges:.2f} (Threshold: {self.visual_edges_threshold:.2f})")


        # Corrected: Use a more general Exception catch as cv2.Error might not inherit BaseException in all environments.
        except Exception as e:
            # Catch any unexpected error during processing steps (e.g., opencv, numpy, or other errors).
            # These errors are logged, and an empty dictionary is returned.
            logger.error(f"VisionProcessor: Unexpected error during processing: {e}", exc_info=True)
            # Even if the processed_features dictionary is partially filled in case of error,
            # processing is considered failed for this frame, and an empty dictionary is returned.
            return {}

        # Return the processed features dictionary on success.
        # DEBUG log: Information that the processing was successful.
        logger.debug(f"VisionProcessor.process: Visual data processed successfully. Output Features: {list(processed_features.keys())}")
        # Logging the processed_features dictionary content itself can be very verbose,
        # basic info like keys is logged above.
        return processed_features

    def cleanup(self):
        """
        Cleans up VisionProcessor resources.

        Currently, this processor does not use specific resources (files, connections, etc.)
        and does not require a cleanup step beyond basic object deletion.
        Includes an informational log.
        Called by module_loader.py when the program terminates (if it exists).
        """
        logger.info("VisionProcessor object cleaning up.")
        # Processor typically does not require explicit cleanup.
        pass