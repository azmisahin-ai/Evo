# src/processing/audio.py

# Processes audio sensory data.
# Extracts basic auditory features (e.g., energy, frequency) from raw audio data (chunk).
# Part of Evo's Phase 1 processing capabilities.

import numpy as np # For numerical operations and arrays.
import logging # For logging.

# Import utility functions (especially input checks and config)
from src.core.config_utils import get_config_value
from src.core.utils import check_input_not_none, check_numpy_input # <<< Utils imports


# Create a logger for this module
# Returns a logger named 'src.processing.audio'.
logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Evo's auditory data processing class (Phase 1 implementation).

    Receives raw audio input (chunk) from the AudioSensor,
    performs basic operations (calculating energy, Spectral Centroid)
    to prepare it for the RepresentationLearner.
    Manages potential errors during processing and ensures flow continuity.
    Returns a numpy array containing basic auditory features as output.
    """
    def __init__(self, config):
        """
        Initializes the AudioProcessor.

        Args:
            config (dict): Processor configuration settings (full config dict).
                           Settings for this module are read from the 'processors.audio' section.
                           'audio_rate': Audio sample rate (int, default 44100 Hz). Required for Spectral Centroid calculation.
                           'output_dim': Dimension of the processed audio output (int, default 2).
                                         Currently 2 is expected for energy and Spectral Centroid.
                                         This number might increase for other features in the future.
        """
        self.config = config # AudioProcessor receives the full config
        logger.info("AudioProcessor initializing...")

        # Get sample rate and output dimension from config using get_config_value.
        # These settings are under the 'processors.audio' key.
        # Note: audio_rate also exists under 'audio' key. Using 'processors.audio' for consistency with VisionProcessor.
        # Corrected: Path should use 'processors', then 'audio', then the key name.
        self.audio_rate = get_config_value(config, 'processors', 'audio', 'audio_rate', default=44100, expected_type=int, logger_instance=logger)
        self.output_dim = get_config_value(config, 'processors', 'audio', 'output_dim', default=2, expected_type=int, logger_instance=logger)

        # output_dim check (will be relevant when more features are returned)
        # Currently returning 2 features: energy and Spectral Centroid.
        if self.output_dim != 2:
             logger.warning(f"AudioProcessor: Config output_dim ({self.output_dim}) does not match implemented feature count (2). Please check config and implementation consistency. RepresentationLearner's input_dim for audio should match the actual output dimension.")
             # This is a warning, not an error. The implementation returns 2 features, config should ideally match.


        logger.info(f"AudioProcessor initialized. Sample Rate: {self.audio_rate} Hz, Implemented Output Dimension: {self.output_dim}")


    def process(self, audio_input):
        """
        Processes raw auditory input and extracts basic auditory features.

        Receives the input (typically an int16 numpy array), calculates basic
        features like energy and Spectral Centroid. Returns a numpy array
        containing these features.
        Returns None if the input is None or an error occurs during processing.

        Args:
            audio_input (numpy.ndarray or None): Raw auditory data (chunk) or None.
                                                  Typically comes from AudioSensor.
                                                  Expected format: shape (N,), dtype int16.

        Returns:
            numpy.ndarray or None: The calculated feature vector (shape (output_dim,), dtype float32)
                                   or None on error or if input is None.
        """
        # Error handling: If input is None or not of expected type
        # Use check_input_not_none function (logs and returns False if None)
        if not check_input_not_none(audio_input, input_name="audio_input for AudioProcessor", logger_instance=logger):
             logger.debug("AudioProcessor.process: Input is None. Returning None.")
             return None # If input is None, skip processing and return None.

        # Check if the input is a numpy array and has the correct dtype (int16).
        # Use check_numpy_input function. This also checks for np.ndarray type.
        # Expected_ndim=1 because a chunk is expected to be a 1D array. dtype int16 is expected.
        # check_numpy_input logs an ERROR and returns False on failure.
        if not check_numpy_input(audio_input, expected_dtype=np.int16, expected_ndim=1, input_name="audio_input for AudioProcessor", logger_instance=logger):
             logger.error("AudioProcessor.process: Input is not a numpy array or has wrong dtype/dimensions. Returning None.") # check_numpy_input already logs internally.
             return None # If type, dtype, or dimensions are invalid, stop processing and return None.

        # If an empty chunk (audio_input.size == 0) is received, return None without processing.
        if audio_input.size == 0:
             logger.debug("AudioProcessor.process: Received empty audio chunk. Skipping processing, returning None.")
             return None


        # DEBUG log: Input details (dimensions and type). Similar log exists in check_numpy_input, but kept here.
        logger.debug(f"AudioProcessor.process: Audio data received. Shape: {audio_input.shape}, Dtype: {audio_input.dtype}. Processing...")

        energy = 0.0 # Variable to hold the energy value. Starts at 0.
        spectral_centroid = 0.0 # Variable to hold the Spectral Centroid value. Starts at 0.
        processed_features_vector = None # The feature vector to be returned.

        try:
            # 1. Convert data to float (for calculations).
            # int16 values are in the range [-32768, 32767]. Converting to float32.
            # Normalization (e.g., to -1.0 to 1.0 range) might be better (Future TODO).
            audio_float = audio_input.astype(np.float32)
            # logger.debug(f"AudioProcessor.process: Audio data converted to float32. Shape: {audio_float.shape}")

            # 2. Calculate audio energy (e.g., RMS - Root Mean Square).
            # np.mean might raise an error for an empty chunk, but size check is done above.
            energy = np.sqrt(np.mean(audio_float**2)) if audio_float.size > 0 else 0.0
            # logger.debug(f"AudioProcessor.process: Audio energy calculated: {energy:.4f}") # Logging now part of vector log


            # 3. Calculate Spectral Centroid.
            # Spectral Centroid = sum(frequencies * magnitudes) / sum(magnitudes)
            # a) Apply windowing (e.g., Hanning window)
            window = np.hanning(len(audio_float))
            audio_windowed = audio_float * window
            # logger.debug("AudioProcessor.process: Hanning window applied.")

            # b) Apply FFT (Fast Fourier Transform)
            fft_result = np.fft.fft(audio_windowed)
            # logger.debug(f"AudioProcessor.process: FFT applied. Output Shape: {fft_result.shape}")

            # c) Get the magnitude spectrum (absolute value from complex numbers)
            magnitude_spectrum = np.abs(fft_result)
            # logger.debug(f"AudioProcessor.process: Magnitude spectrum calculated. Shape: {magnitude_spectrum.shape}")

            # d) Get the single-sided spectrum (the part up to Nyquist frequency).
            # For real signals, the spectrum is symmetric. The first half is sufficient.
            # If chunk size is N, the single-sided spectrum will have N/2 + 1 elements (including DC and Nyquist).
            single_sided_spectrum = magnitude_spectrum[:len(magnitude_spectrum)//2 + 1]
            # logger.debug(f"AudioProcessor.process: Single-sided spectrum obtained. Shape: {single_sided_spectrum.shape}")

            # e) Create the frequency axis (from 0 to Nyquist frequency).
            # Nyquist frequency = audio_rate / 2.
            # Number of frequency bins = len(single_sided_spectrum).
            # endpoint=True to include the Nyquist frequency.
            frequencies = np.linspace(0, self.audio_rate / 2, len(single_sided_spectrum))
            # logger.debug(f"AudioProcessor.process: Frequency axis created. Shape: {frequencies.shape}")

            # f) Calculate the Spectral Centroid.
            # Check if the denominator (sum of magnitudes) is zero to avoid division by zero errors.
            # This typically happens with entirely silent chunks.
            sum_magnitudes = np.sum(single_sided_spectrum)
            if sum_magnitudes > 1e-6: # Use a small threshold to account for floating point inaccuracies
                 spectral_centroid = np.sum(frequencies * single_sided_spectrum) / sum_magnitudes
                 # logger.debug(f"AudioProcessor.process: Spectral Centroid calculated: {spectral_centroid:.4f}") # Logging now part of vector log
            else:
                 # For silent chunks or near-zero total magnitude, set centroid to 0.
                 spectral_centroid = 0.0
                 logger.debug("AudioProcessor.process: Total magnitude near zero, Spectral Centroid set to 0.")

            # TODO: In the future: Add more features (e.g., Spectral Spread, Spectral Flux, MFCC).
            # In this case, new elements would be added to processed_features_vector, and output_dim would be increased.

            # 4. Combine the extracted features into a numpy array.
            # Currently combining energy and Spectral Centroid.
            # This is the 1D feature vector expected by the RepresentationLearner.
            processed_features_vector = np.array([energy, spectral_centroid], dtype=np.float32)

            # Check: Does the resulting vector dimension match the expected output_dim from config?
            if processed_features_vector.shape[0] != self.output_dim:
                 logger.warning(f"AudioProcessor.process: Generated feature vector dimension ({processed_features_vector.shape[0]}) does not match output_dim in config ({self.output_dim}). Please check the config file and the implementation. RepresentationLearner's input_dim for audio should match the actual output dimension.")
                 # This is a warning, not an error. The implementation returns 2 features, config should ideally match.
                 # Optional: Could adjust processed_features_vector here if needed (e.g., pad with zeros or truncate).
                 # For now, just logging the warning.


        except Exception as e:
            # Catch unexpected errors that might occur during processing steps (e.g., numpy, fft errors).
            logger.error(f"AudioProcessor: Unexpected error during processing: {e}", exc_info=True)
            return None # Return None in case of error.

        # Return the processed feature vector on success.
        # DEBUG log: Information that the processing was successful, including shape, dtype, and values.
        logger.debug(f"AudioProcessor.process: Processed audio data successfully. Output Shape: {processed_features_vector.shape}, Dtype: {processed_features_vector.dtype}. Values (Energy, Centroid): {processed_features_vector}")
        return processed_features_vector

    def cleanup(self):
        """
        Cleans up AudioProcessor resources.

        Currently, this processor does not use specific resources (files, connections, etc.)
        and does not require a cleanup step beyond basic object deletion.
        Includes an informational log.
        Called by module_loader.py when the program terminates (if it exists).
        """
        logger.info("AudioProcessor object cleaning up.")
        # Processor typically does not require explicit cleanup.
        pass