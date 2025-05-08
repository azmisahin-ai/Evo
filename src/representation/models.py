# src/representation/models.py
#
# Learns or extracts internal representations (latent vectors) from processed sensory data.
# Implements basic neural network layers.
# Part of Evo's Phase 1 representation learning capabilities.

import numpy as np # For numerical operations and arrays.
import logging # For logging.

# Import utility functions
from src.core.config_utils import get_config_value
from src.core.utils import check_input_not_none, check_numpy_input # <<< Utils imports


# Create a logger for this module
# Returns a logger named 'src.representation.models'.
logger = logging.getLogger(__name__)

# A simple Dense (Fully Connected) Layer class
# TODO: In the future, using the central version from src/core/nn_components.py would be cleaner.
class Dense:
    """
    A basic implementation of a dense (fully connected) layer.

    Contains weight and bias parameters.
    Supports ReLU activation function.
    Performs the forward pass calculation.
    """
    def __init__(self, input_dim, output_dim, activation='relu'):
        """
        Initializes the Dense layer.

        Initializes weights and biases randomly.

        Args:
            input_dim (int): The dimension of the input feature.
            output_dim (int): The dimension of the output feature.
            activation (str, optional): The name of the activation function to use ('relu'). Defaults to 'relu'.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        # Create a logger for this specific class instance or use module logger.
        # Currently using the module logger, which is fine.
        logger.info(f"Dense layer initializing: Input={input_dim}, Output={output_dim}, Activation={activation}")

        # Initialize weights and biases.
        # Better initialization methods (He, Xavier) could be used (Future TODO).
        # Using np.random.uniform for a simple initialization.
        # Scaling by sqrt(1. / input_dim) (fan-in) is a common heuristic.
        # If using PyTorch, weights would be PyTorch tensors and could be moved to GPU.
        # For now, staying on CPU with NumPy.
        limit = np.sqrt(1. / input_dim) # Simple initialization scale (fan-in)
        self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.bias = np.zeros(output_dim) # Biases are commonly initialized to zero.

        logger.info("Dense layer initialized.")

    def forward(self, inputs):
        """
        Computes the forward pass of the layer.

        Multiplies the input vector or matrix by the weight matrix, adds the bias, and
        applies the activation function.
        Returns None if the input is None or has the wrong type/dimensions.
        Returns None if an error occurs during calculation.

        Args:
            inputs (numpy.ndarray or None): The input data for the layer.
                                            Expected format: shape (..., input_dim), numerical dtype.
                                            ... can be a batch dimension.

        Returns:
            numpy.ndarray or None: The output data of the layer, or None on error.
                                   shape (..., output_dim), numerical dtype.
        """
        # Error handling: Is input None? Use check_input_not_none.
        if not check_input_not_none(inputs, input_name="dense_inputs", logger_instance=logger):
             logger.debug("Dense.forward: Input is None. Returning None.") # None input is not an error, just informational.
             return None # Return None if input is None.

        # Error handling: Check if the input is a numpy array and has a numerical dtype.
        # expected_ndim=None because input can be a single vector or a batch (ndim >= 1).
        # Shape check will be done separately.
        if not check_numpy_input(inputs, expected_dtype=np.number, expected_ndim=None, input_name="dense_inputs", logger_instance=logger):
             logger.error("Dense.forward: Input is not a numpy array or is not numerical. Returning None.") # check_numpy_input already logs internally.
             return None # If type or dtype is invalid, return None.

        # Error handling: Check if the input dimension (last dimension) matches the input_dim.
        # inputs.shape[-1] gives the last dimension.
        if inputs.shape[-1] != self.input_dim:
             logger.error(f"Dense.forward: Unexpected input dimension: {inputs.shape}. The last dimension ({inputs.shape[-1]}) does not match the expected input_dim ({self.input_dim}). Returning None.")
             return None # If dimensions don't match, return None.


        # DEBUG log: Input details.
        logger.debug(f"Dense.forward: Input received. Shape: {inputs.shape}, Dtype: {inputs.dtype}. Performing calculation.")


        output = None # Variable to hold the output.

        try:
            # Linear transformation: output = input_data @ weights + bias
            # For a single example (input_dim,) * weights (input_dim, output_dim) -> (output_dim,)
            # For a batch ((batch_size, input_dim)) * weights (input_dim, output_dim) -> (batch_size, output_dim)
            # np.dot handles both cases correctly.
            linear_output = np.dot(inputs, self.weights) + self.bias

            # Apply activation function (if any)
            if self.activation == 'relu':
                # ReLU: output = max(0, output)
                output = np.maximum(0, linear_output) # ReLU activation
            # TODO: Other activation functions will be added (sigmoid, tanh, etc.)
            # elif self.activation == 'sigmoid':
            #      output = 1 / (1 + np.exp(-linear_output)) # Be careful with potential over/underflow in exp.
            # elif self.activation == 'tanh':
            #      output = np.tanh(linear_output)
            # Add a warning/error log for unknown activation names.
            elif self.activation is None: # None activation
                 output = linear_output
            else:
                 logger.warning(f"Dense.forward: Unknown activation function: '{self.activation}'. Using linear activation.")
                 output = linear_output


        except Exception as e:
             # Catch any unexpected error during the forward pass calculations (e.g., np.dot, activation function error).
             logger.error(f"Dense.forward: Unexpected error during forward pass: {e}", exc_info=True)
             return None # Return None in case of error.


        # Return the calculated output on success.
        logger.debug(f"Dense.forward: Forward pass completed. Output Shape: {output.shape}, Dtype: {output.dtype}.")
        return output

    def cleanup(self):
        """
        Cleans up Dense layer resources.

        Currently, this layer does not use specific resources (files, connections, etc.)
        and does not require a cleanup step beyond basic object deletion.
        Includes an informational log.
        NumPy arrays generally do not require explicit cleanup.
        """
        # Informational log.
        logger.info(f"Dense layer object cleaning up: Input={self.input_dim}, Output={self.output_dim}")
        pass


# A simple Representation Learner class
class RepresentationLearner:
    """
    Learns or extracts internal representations (latent vectors) from processed sensory data.

    Receives processed sensory data (in a dictionary format) from Processing modules.
    Combines this data into a unified input vector.
    Passes the input vector through an Encoder layer (Dense) to create a low-dimensional,
    meaningful representation vector (latent).
    Also includes a Decoder layer (Dense) for Autoencoder principle, which produces a
    reconstruction in the original input dimensions from the latent vector.
    More complex models (CNN, RNN, Transformer-based) will be added here in the future.
    Returns None if the module fails to initialize or if an error occurs during
    representation learning/extraction.
    """
    def __init__(self, config):
        """
        Initializes the RepresentationLearner module.

        Sets up the structure of the representation model (currently Encoder and Decoder Dense layers).
        input_dim and representation_dim are obtained from config.

        Args:
            config (dict): RepresentationLearner configuration settings (full config dict).
                           Settings for this module are read from the 'representation' section,
                           and processor dimensions from the 'processors' section.
        """
        self.config = config # RepresentationLearner receives the full config
        logger.info("RepresentationLearner initializing...")

        # Get input and representation dimensions from config using get_config_value.
        # Based on config, these settings are under the 'representation' key.
        self.input_dim = get_config_value(config, 'representation', 'input_dim', default=8194, expected_type=int, logger_instance=logger)
        self.representation_dim = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)

        self.encoder = None # Encoder layer (currently Dense).
        self.decoder = None # Decoder layer (currently Dense).
        self.is_initialized = False # Tracks if the module was initialized successfully.

        # TODO: In the future: Get input dimensions from different modalities from config,
        # TODO: and verify that the input_dim value matches the sum of these dimensions.
        # Processor output dimensions are used by RepresentationLearner for calculating the expected input_dim.
        # Get processor output dimensions using get_config_value for consistency.
        # Get the processors section, then vision and audio sub-sections.
        processors_config = config.get('processors', {})
        visual_config = processors_config.get('vision', {})
        audio_config = processors_config.get('audio', {})

        # Use get_config_value to retrieve values from the nested config structures.
        # Paths here are relative to the visual_config or audio_config dicts obtained above.
        visual_out_width = get_config_value(visual_config, 'output_width', default=64, expected_type=int, logger_instance=logger) # Use visual_config here
        visual_out_height = get_config_value(visual_config, 'output_height', default=64, expected_type=int, logger_instance=logger) # Use visual_config here
        audio_features_dim = get_config_value(audio_config, 'output_dim', default=2, expected_type=int, logger_instance=logger) # Use audio_config here

        visual_gray_size = visual_out_width * visual_out_height
        visual_edges_size = visual_out_width * visual_out_height # Assume same dimensions for simplicity
        expected_input_dim_calc = visual_gray_size + visual_edges_size + audio_features_dim

        if self.input_dim != expected_input_dim_calc:
             logger.warning(f"RepresentationLearner: Configured input_dim ({self.input_dim}) does not match calculated expected value ({expected_input_dim_calc}). Please check the 'representation.input_dim' setting in the config file and the output dimensions of the Processors ('processors.vision.output_width/height', 'processors.audio.output_dim'). Calculated dimension is based on Processor output dimensions.")
             # Optional: Could set self.input_dim to the calculated value in this case.
             # self.input_dim = expected_input_dim_calc


        try:
            # Create the Encoder layer (Input dimension: self.input_dim, Output dimension: self.representation_dim).
            # The last layer of an encoder typically has no activation or linear activation (latent space).
            # Added None activation option to Dense class. Using None activation here.
            self.encoder = Dense(self.input_dim, self.representation_dim, activation=None) # No activation in latent space

            # Create the Decoder layer (Input dimension: self.representation_dim, Output dimension: self.input_dim).
            # The decoder output should be a reconstruction of the original input dimensions.
            # The last layer of the decoder might have an activation suitable for the output data type (e.g., sigmoid/tanh for images, linear for numerical).
            # For now, using None activation again.
            self.decoder = Dense(self.representation_dim, self.input_dim, activation=None) # Reconstruction output


            # Initialization successful flag: Set to True if Encoder and Decoder objects were created successfully.
            self.is_initialized = (self.encoder is not None and self.decoder is not None)

        except Exception as e:
            # Catch any unexpected error during model layer initialization.
            # This error can be considered critical during initialization (Represent module cannot function without a model).
            logger.critical(f"RepresentationLearner initialization failed critically: {e}", exc_info=True)
            self.is_initialized = False # Mark as not initialized in case of error.
            self.encoder = None
            self.decoder = None


        logger.info(f"RepresentationLearner initialized. Input dimension: {self.input_dim}, Representation dimension: {self.representation_dim}. Initialization Successful: {self.is_initialized}")


    def learn(self, processed_inputs):
        """
        Extracts a representation (latent vector) from processed sensory input (Encoder forward pass).

        Receives processed sensory data (in a dictionary format) from Processing modules.
        Combines this data into a unified input vector.
        Passes the input vector through the Encoder layer to compute the representation vector.
        Returns None if the module is not initialized, input is empty, or an error occurs
        during processing/forward pass.

        Args:
            processed_inputs (dict): Dictionary of processed sensory data.
                                     E.g., {'visual': dict or None, 'audio': numpy.ndarray or None}.
                                     Typically comes from Process modules.

        Returns:
            numpy.ndarray or None: The learned representation vector (shape (representation_dim,), numerical dtype)
                                   or None on error or if representation could not be extracted.
        """
        # Error handling: If module is not initialized, do not proceed.
        if not self.is_initialized or self.encoder is None: # Even if decoder is None, encoder might still work (policy).
             logger.error("RepresentationLearner.learn: Module is not initialized or Encoder layer is missing. Cannot learn representation.")
             return None # Return None if not initialized.

        # Error handling: If the processed input dictionary is None or empty, do not proceed.
        # Use check_input_not_none function for the processed_inputs dictionary.
        if not check_input_not_none(processed_inputs, input_name="processed_inputs for RepresentationLearner", logger_instance=logger):
             logger.debug("RepresentationLearner.learn: Processed input dictionary is None. Cannot learn representation.")
             return None # Return None if input is None.

        if not processed_inputs: # If the dictionary is empty
            logger.debug("RepresentationLearner.learn: Processed input dictionary is empty. Cannot learn representation.")
            return None # Return None if there is no input.


        input_vector_parts = [] # List to hold parts of the input vector to be concatenated.

        try:
            # Take the processed sensory data and combine it into a single input vector as expected by the model.
            # This part determines how we combine processed data from different modalities (visual, auditory)
            # into a single input format for the model.
            # Expected format:
            # visual: dict from VisionProcessor {'grayscale': np.ndarray (64x64 uint8), 'edges': np.ndarray (64x64 uint8)}
            # audio: np.ndarray from AudioProcessor (shape (output_dim,), dtype float32) - [energy, spectral_centroid]
            # Goal is to create a vector matching RepresentationLearner's input_dim (8194).
            # 8194 = (64*64 grayscale) + (64*64 edges) + (AudioProcessor output_dim)

            # Process visual data and add to the list of parts to be concatenated
            visual_processed = processed_inputs.get('visual')
            # Check if the processed visual data is a dictionary.
            if isinstance(visual_processed, dict):
                 # Check for 'grayscale' and 'edges' keys within the dictionary and ensure their values are numpy arrays.
                 grayscale_data = visual_processed.get('grayscale')
                 edges_data = visual_processed.get('edges')

                 # Process grayscale data
                 if grayscale_data is not None and check_numpy_input(grayscale_data, expected_dtype=np.uint8, expected_ndim=2, input_name="processed_inputs['visual']['grayscale']", logger_instance=logger):
                      # Flatten the visual data and convert to float32. Normalization is optional (Future TODO).
                      flattened_grayscale = grayscale_data.flatten().astype(np.float32)
                      # TODO: Normalize flattened_grayscale to 0-1 range? flattened_grayscale = flattened_grayscale / 255.0
                      input_vector_parts.append(flattened_grayscale)
                      logger.debug(f"RepresentationLearner.learn: Visual data (grayscale) flattened. Shape: {flattened_grayscale.shape}")
                 else:
                      # Log a warning if 'grayscale' is not a valid numpy array or has wrong dimensions/type.
                      logger.warning("RepresentationLearner.learn: 'grayscale' in processed visual input dictionary is not a valid numpy array or has wrong dimensions/type. Skipping.")

                 # Process edges data
                 if edges_data is not None and check_numpy_input(edges_data, expected_dtype=np.uint8, expected_ndim=2, input_name="processed_inputs['visual']['edges']", logger_instance=logger):
                      # Flatten the edge map and convert to float32. Values are typically 0 or 255.
                      flattened_edges = edges_data.flatten().astype(np.float32)
                      # TODO: Normalize flattened_edges? flattened_edges = flattened_edges / 255.0 # Or keep as 0/1 binary values?
                      input_vector_parts.append(flattened_edges)
                      logger.debug(f"RepresentationLearner.learn: Visual data (edges) flattened. Shape: {flattened_edges.shape}")
                 else:
                      # Log a warning if 'edges' is not a valid numpy array or has wrong dimensions/type.
                      logger.warning("RepresentationLearner.learn: 'edges' in processed visual input dictionary is not a valid numpy array or has wrong dimensions/type. Skipping.")

            # elif visual_processed is not None: # If not a dict but also not None (unexpected format)
            #      logger.warning(f"RepresentationLearner.learn: Processed visual input has unexpected type ({type(visual_processed)}). Dictionary expected. Skipping.")
            # else: # visual_processed is None
            #      logger.debug("RepresentationLearner.learn: Processed visual input is None. Skipping.")


            # Process auditory data and add to the list of parts to be concatenated
            audio_processed = processed_inputs.get('audio')
            # AudioProcessor is expected to return a numpy array (shape (output_dim,), dtype float32) or None.
            # Check if the incoming data is a numpy array and has the correct dimensions/dtype.
            # Expected dtype is np.number (or np.float32). Expected ndim is 1. Expected shape[0] is the output_dim from AudioProcessor config (2).
            # Get the expected audio output dimension from config.
            # Corrected: Path should use 'processors', 'audio', then 'output_dim' relative to the full config.
            expected_audio_dim = get_config_value(self.config, 'processors', 'audio', 'output_dim', default=2, expected_type=int, logger_instance=logger)
            if audio_processed is not None and check_numpy_input(audio_processed, expected_dtype=np.number, expected_ndim=1, input_name="processed_inputs['audio']", logger_instance=logger):
                 # check_numpy_input performed basic array/dtype/ndim checks. Now check the specific shape[0] dimension.
                 if audio_processed.shape[0] == expected_audio_dim:
                      # Add the audio features array directly (it's already a 1D array). Should be float32 from AudioProcessor.
                      audio_features_array = audio_processed.astype(np.float32) # Ensure float32
                      # TODO: Normalize audio_features_array?
                      input_vector_parts.append(audio_features_array)
                      logger.debug(f"RepresentationLearner.learn: Audio data array added. Shape: {audio_features_array.shape}")
                 else:
                      # Log a warning if the expected first dimension doesn't match, even if other checks pass.
                      logger.warning(f"RepresentationLearner.learn: Processed audio input has unexpected dimension. Expected shape ({expected_audio_dim},), received shape {audio_processed.shape}. Skipping.")


            # elif audio_processed is not None: # Invalid format (not numpy array or wrong dtype)
            #      logger.warning(f"RepresentationLearner.learn: Processed audio input has unexpected format ({type(audio_processed)}, shape {getattr(audio_processed, 'shape', 'N/A')}). Expected 1D numpy array. Skipping.")
            # else: # audio_processed is None
            #      logger.debug("RepresentationLearner.learn: Processed audio input is None. Skipping.")


            # Concatenate all valid input parts into a single numpy vector.
            # Continue only if at least one valid input part (grayscale, edges, audio_features) was added.
            if not input_vector_parts: # If no valid input parts were added
                 logger.debug("RepresentationLearner.learn: No valid data extracted from processed inputs (grayscale, edges, audio). Cannot learn representation.")
                 return None # Cannot learn representation, return None.

            # Concatenate the arrays in the list into a single array. axis=0 concatenates along the first dimension by default.
            combined_input = np.concatenate(input_vector_parts, axis=0)

            # Input dimension check: The dimension of the combined input (shape[0]) should match the input_dim from config.
            # This check is also performed inside the Dense layer (forward method), but doing it here too
            # makes it easier to understand data preparation issues at the representation learning step.
            # self.input_dim is obtained from config (default 8194). The dimension of the combined data must match.
            if combined_input.shape[0] != self.input_dim:
                 # Make the error message more informative.
                 logger.error(f"RepresentationLearner.learn: Combined input dimension ({combined_input.shape[0]}) does not match input_dim in config ({self.input_dim}). Please check the 'representation.input_dim' setting in the config file and the output format/dimensions of the Processors (currently expects 64x64 grayscale, 64x64 edges, and {expected_audio_dim} audio features).")
                 # Dimension mismatch is a serious configuration/data flow error. Return None.
                 return None # Cannot learn representation if dimensions don't match, return None.

            # DEBUG log: Combined input details.
            logger.debug(f"RepresentationLearner.learn: Visual and audio data combined. Shape: {combined_input.shape}, Dtype: {combined_input.dtype}.")


            # Run the representation model (encoder forward pass).
            # Our representation model's encoder is self.encoder (a Dense layer).
            # The encoder layer's forward method returns None if input is None or an error occurs.
            representation = self.encoder.forward(combined_input)

            # If the model returned None (an error occurred inside the encoder)
            if representation is None:
                 logger.error("RepresentationLearner.learn: Encoder model returned None, forward pass failed. Cannot learn representation.")
                 return None # Return None in case of error.

            # DEBUG log: Representation output details.
            # if representation is not None: # We are in this block only if representation is not None.
            logger.debug(f"RepresentationLearner.learn: Representation successfully learned. Output Shape: {representation.shape}, Dtype: {representation.dtype}.")


            # Return the learned representation vector on success.
            return representation

        except Exception as e:
            # General error handling: Catch any unexpected error during the process (data preparation, concatenation, model execution).
            logger.error(f"RepresentationLearner.learn: Unexpected error during representation learning: {e}", exc_info=True)
            return None # Return None in case of error.


    def decode(self, latent_vector):
        """
        Produces a reconstruction in the original input dimensions from a latent vector (Decoder forward pass).

        This method simulates the decoder part of an Autoencoder principle.
        It will not be used in run_evo.py's main loop currently.

        Args:
            latent_vector (numpy.ndarray or None): The latent vector input for the Decoder (of dimension representation_dim)
                                                  or None.

        Returns:
            numpy.ndarray or None: The reconstruction output (shape (input_dim,), numerical dtype)
                                   or None on error or if input is None.
        """
        # Error handling: If the module is not initialized or Decoder is missing, do not proceed.
        if not self.is_initialized or self.decoder is None:
             logger.error("RepresentationLearner.decode: Module is not initialized or Decoder layer is missing. Cannot perform reconstruction.")
             return None

        # Error handling: Is the latent vector input None? Use check_input_not_none.
        if not check_input_not_none(latent_vector, input_name="latent_vector for RepresentationLearner.decode", logger_instance=logger):
             logger.debug("RepresentationLearner.decode: Input latent_vector is None. Returning None.")
             return None

        # Error handling: Is the latent vector a numpy array and has the correct dimensions/dtype?
        # A 1D numpy array of dimension representation_dim is expected.
        if not check_numpy_input(latent_vector, expected_dtype=np.number, expected_ndim=1, input_name="latent_vector for RepresentationLearner.decode", logger_instance=logger):
             # check_numpy_input performed basic array/dtype/ndim checks. Now check the specific shape[0] dimension.
             if not (isinstance(latent_vector, np.ndarray) and latent_vector.shape[0] == self.representation_dim):
                   logger.error(f"RepresentationLearner.decode: Input latent_vector has wrong format. Expected shape ({self.representation_dim},), ndim 1, numerical dtype. Received shape {getattr(latent_vector, 'shape', 'N/A')}, ndim {getattr(latent_vector, 'ndim', 'N/A')}, dtype {getattr(latent_vector, 'dtype', 'N/A')}. Returning None.")
                   return None
             # If check_numpy_input already logged an error, just return None.
             return None


        logger.debug(f"RepresentationLearner.decode: Latent vector received. Shape: {latent_vector.shape}, Dtype: {latent_vector.dtype}. Performing reconstruction.")

        reconstruction = None

        try:
            # Run the Decoder layer (forward pass).
            # The Decoder layer's forward method returns None if input is None or an error occurs.
            reconstruction = self.decoder.forward(latent_vector)

            # If the model returned None (an error occurred inside the decoder)
            if reconstruction is None:
                 logger.error("RepresentationLearner.decode: Decoder model returned None, forward pass failed. Cannot perform reconstruction.")
                 return None # Return None in case of error.

            # DEBUG log: Reconstruction output details.
            logger.debug(f"RepresentationLearner.decode: Reconstruction completed. Output Shape: {reconstruction.shape}, Dtype: {reconstruction.dtype}.")


            # Return the reconstruction output on success.
            return reconstruction

        except Exception as e:
             # General error handling: Catch any unexpected error during the process.
             logger.error(f"RepresentationLearner.decode: Unexpected error during reconstruction: {e}", exc_info=True)
             return None # Return None in case of error.


    def cleanup(self):
        """
        Cleans up RepresentationLearner module resources (model layers, etc.).

        Calls the cleanup methods of its internal model layers (if they exist).
        Called by module_loader.py when the program terminates (if it exists).
        """
        logger.info("RepresentationLearner object cleaning up.")
        # Call the cleanup methods of Encoder and Decoder layers (if they exist)
        if hasattr(self.encoder, 'cleanup'):
             self.encoder.cleanup()
        if hasattr(self.decoder, 'cleanup'):
             self.decoder.cleanup()

        logger.info("RepresentationLearner object cleaning up completed.")