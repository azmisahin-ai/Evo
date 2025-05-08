# scripts/test_module.py
#
# Script to test Evo modules in isolation with input/output.
# Initializes a specific module, provides dummy input, and logs its output.
# This script is used during development to quickly verify that modules
# behave as expected with sample data.

# DEBUG: Add a print statement right at the very beginning to guarantee some output
print("--- Evo Test Script Started (Diagnostic Print) ---")

import argparse
import logging
import sys
import importlib
import inspect
import numpy as np
import time
import random
import os # For creating dummy data files if needed
import json # For logging dict/json outputs

# Import Evo's logging and config utilities
# Assumes: script is run from the Evo root directory.
from src.core.logging_utils import setup_logging
from src.core.config_utils import load_config_from_yaml, get_config_value # Import get_config_value
from src.core.utils import cleanup_safely # Import cleanup_safely for robust cleanup


# Create a logger for this script itself
logger = logging.getLogger(__name__)

def load_module_class(module_path, class_name):
    """
    Dynamically loads a class (e.g., VisionProcessor) from a specified module path (e.g., src.processing.vision).

    Args:
        module_path (str): The Python path of the module (e.g., 'src.processing.vision').
        class_name (str): The name of the class within the module (e.g., 'VisionProcessor').

    Returns:
        class or None: The loaded class object, or None on error.
    """
    try:
        # Import the module
        module = importlib.import_module(module_path)
        logger.debug(f"Module loaded: {module_path}")
        # Get the class from the module
        class_obj = getattr(module, class_name)
        logger.debug(f"Class loaded: {class_name} from {module_path}")

        # Ensure the loaded object is indeed a class
        if not inspect.isclass(class_obj):
             logger.error(f"Loaded '{class_name}' from '{module_path}' is not a class.")
             return None

        return class_obj

    except ModuleNotFoundError:
        logger.error(f"Module not found: {module_path}. Please check the path.")
        return None
    except AttributeError:
        logger.error(f"Class not found: {class_name} in {module_path}. Please check the class name.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while loading module or class: {e}", exc_info=True)
        return None

# Creates simple dummy input data for the main processing method of a given class.
# This function should be specialized based on the expected input format of the tested class's main method.
def create_dummy_method_inputs(class_name, config):
    """
    Creates simple dummy input data for the main processing method of the specified class.

    Args:
        class_name (str): The name of the class being tested (e.g., 'VisionProcessor', 'AudioProcessor').
                          Case-insensitive.
        config (dict): The global configuration dictionary.

    Returns:
        tuple: A tuple containing the positional arguments for the module's main method call,
               or None if not supported.
               Single-argument methods should still return a tuple (e.g., (input_data,)).
               If the method call should not be tested or requires no arguments, an empty tuple () is returned.
    """
    # Convert class_name to lowercase for consistent internal logic
    class_name_lower = class_name.lower()

    logger.debug(f"Creating dummy method inputs for '{class_name}'...")

    # Create dummy input based on the module name.
    # This part must be customized for the main method of each module you want to test.
    # The returned value should be a tuple of the positional arguments expected by the method.

    if class_name_lower in ['visionsensor', 'audiosensor']:
        # VisionSensor.capture_frame() and AudioSensor.capture_chunk() take no arguments.
        # This function is not strictly for "creating input" here, but for defining method args.
        # If sensor capture methods are tested, they will be called with no args.
        logger.debug(f"'{class_name}' capture method takes no arguments. Returning empty input tuple.")
        return () # Return an empty tuple, the call will be with no args.

    elif class_name_lower == 'visionprocessor':
        # VisionProcessor.process(visual_input) expects a numpy array.
        # Dummy data can be created in different sizes and channels (BGR or Gray).
        # Using VisionSensor dummy dimensions from config for test data size.
        dummy_width = get_config_value(config, 'vision', 'dummy_width', default=640, expected_type=int, logger_instance=logger)
        dummy_height = get_config_value(config, 'vision', 'dummy_height', default=480, expected_type=int, logger_instance=logger)
        # Create a dummy color BGR image (uint8)
        dummy_frame = np.random.randint(0, 256, size=(dummy_height, dummy_width, 3), dtype=np.uint8)
        logger.debug(f"Created dummy process input frame for VisionProcessor ({dummy_frame.shape}, {dummy_frame.dtype}).")
        return (dummy_frame,) # Return inside a tuple.


    elif class_name_lower == 'audioprocessor':
        # AudioProcessor.process(audio_input) expects an int16 numpy array.
        # Using AudioSensor chunk_size from config for test data size.
        chunk_size = get_config_value(config, 'audio', 'audio_chunk_size', default=1024, expected_type=int, logger_instance=logger)
        sample_rate = get_config_value(config, 'audio', 'audio_rate', default=44100, expected_type=int, logger_instance=logger) # Get from config
        # Create dummy int16 audio data (like a tone)
        frequency = 880
        amplitude = np.iinfo(np.int16).max * 0.1
        t = np.linspace(0., chunk_size / sample_rate, chunk_size)
        dummy_chunk = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)

        logger.debug(f"Created dummy process input chunk for AudioProcessor ({dummy_chunk.shape}, {dummy_chunk.dtype}).")
        return (dummy_chunk,) # Return inside a tuple.


    elif class_name_lower == 'representationlearner':
        # RepresentationLearner.learn(processed_inputs) expects a processed_inputs dictionary: {'visual': dict, 'audio': np.ndarray}.
        # This should be in the format of Processor module outputs.

        # Dummy VisionProcessor output dictionary
        vis_out_w = get_config_value(config, 'processors', 'vision', 'output_width', default=64, expected_type=int, logger_instance=logger)
        vis_out_h = get_config_value(config, 'processors', 'vision', 'output_height', default=64, expected_type=int, logger_instance=logger)
        # Dummy grayscale and edges arrays
        dummy_processed_visual_gray = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
        dummy_processed_visual_edges = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
        dummy_processed_visual_dict = {'grayscale': dummy_processed_visual_gray, 'edges': dummy_processed_visual_edges}

        # Dummy AudioProcessor output array
        audio_out_dim = get_config_value(config, 'processors', 'audio', 'output_dim', default=2, expected_type=int, logger_instance=logger)
        dummy_processed_audio_features = np.random.rand(audio_out_dim).astype(np.float32) # Random floats between 0 and 1

        dummy_processed_inputs = {
            'visual': dummy_processed_visual_dict,
            'audio': dummy_processed_audio_features
        }
        logger.debug(f"Created dummy learn input processed_inputs for RepresentationLearner ({list(dummy_processed_inputs.keys())}).")
        return (dummy_processed_inputs,) # Return inside a tuple.


    elif class_name_lower == 'memory':
        # Memory.store(representation, metadata=None) expects a Representation vector.
        # Memory.retrieve(query_representation, num_results=None) expects a Representation vector and an int.
        # By default, let's create input for the store method.
        # Input for store: A Representation vector
        repr_dim = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)
        dummy_representation = np.random.rand(repr_dim).astype(np.float64) # Assuming RL returns float64
        # Optional metadata.
        dummy_metadata = {"source": "test_script", "timestamp": time.time()}

        logger.debug(f"Created dummy store input Representation ({dummy_representation.shape}, {dummy_representation.dtype}) and Metadata for Memory.")
        # Arguments for the store method (representation, metadata=None)
        return (dummy_representation, dummy_metadata)


    elif class_name_lower == 'cognitioncore':
         # CognitionCore.decide(processed_inputs, learned_representation, relevant_memory_entries, current_concepts) expects these arguments.
         # These should be in the format of Processor, RepresentationLearner, Memory, and LearningModule outputs.

         # Dummy processed_inputs (Processor output) - Same logic as RepresentationLearner input
         vis_out_w = get_config_value(config, 'processors', 'vision', 'output_width', default=64, expected_type=int, logger_instance=logger)
         vis_out_h = get_config_value(config, 'processors', 'vision', 'output_height', default=64, expected_type=int, logger_instance=logger)
         dummy_processed_visual_gray = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
         dummy_processed_visual_edges = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
         dummy_processed_visual_dict = {'grayscale': dummy_processed_visual_gray, 'edges': dummy_processed_visual_edges}

         audio_out_dim = get_config_value(config, 'processors', 'audio', 'output_dim', default=2, expected_type=int, logger_instance=logger)
         dummy_processed_audio_features = np.random.rand(audio_out_dim).astype(np.float32)

         dummy_processed_inputs = {
             'visual': dummy_processed_visual_dict,
             'audio': dummy_processed_audio_features
         }

         # Dummy learned_representation (RepresentationLearner output)
         repr_dim = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)
         dummy_representation = np.random.rand(repr_dim).astype(np.float64)

         # Dummy relevant_memory_entries (Memory.retrieve output)
         # Memory.retrieve returns a list of dicts: [{'representation': array, 'metadata': {}, 'timestamp': float}, ...]
         num_mem = get_config_value(config, 'memory', 'num_retrieved_memories', default=5, expected_type=int, logger_instance=logger)
         dummy_memory_entries = []
         for i in range(num_mem):
              dummy_mem_rep = np.random.rand(repr_dim).astype(np.float64)
              # Make the first memory entry very similar to the query to simulate 'familiarity' sometimes
              if i == 0: dummy_mem_rep = dummy_representation.copy() + np.random.randn(repr_dim).astype(np.float64) * 0.01
              dummy_memory_entries.append({
                  'representation': dummy_mem_rep,
                  'metadata': {'source': 'test', 'index': i},
                  'timestamp': time.time() - i # Simulate different timestamps
              })

         # Dummy current_concepts (LearningModule.get_concepts() output)
         # LearningModule returns a list of arrays.
         num_concepts = 3 # Number of dummy concepts for testing
         dummy_concepts = []
         for i in range(num_concepts):
              dummy_concepts.append(np.random.rand(repr_dim).astype(np.float64))


         # Return arguments for the CognitionCore.decide method as a tuple.
         # decide(self, processed_inputs, learned_representation, relevant_memory_entries, current_concepts)
         logger.debug("Created dummy decide input tuple for CognitionCore.")
         return (dummy_processed_inputs, dummy_representation, dummy_memory_entries, dummy_concepts)


    elif class_name_lower == 'decisionmodule':
         # DecisionModule.decide(understanding_signals, relevant_memory_entries, current_concepts=None) expects these arguments.
         # These should be in the format of UnderstandingModule and Memory outputs.

         # Dummy understanding_signals dictionary. Values can be changed to test different scenarios.
         dummy_understanding_signals = {
             'similarity_score': random.random(), # Random memory similarity between 0.0 - 1.0
             'high_audio_energy': random.choice([True, False]), # Random audio detection
             'high_visual_edges': random.choice([True, False]), # Random edge detection
             'is_bright': random.choice([True, False]), # Random brightness
             'is_dark': random.choice([True, False]),   # Random darkness
             'max_concept_similarity': random.random(), # Random concept similarity between 0.0 - 1.0
             'most_similar_concept_id': random.choice([None, 0, 1, 2]), # Random concept ID or None
         }
         # Ensure consistency: if is_bright is True, is_dark must be False
         if dummy_understanding_signals.get('is_bright', False) and dummy_understanding_signals.get('is_dark', False):
             dummy_understanding_signals['is_dark'] = False

         # Dummy relevant_memory_entries (Memory.retrieve output format, although not directly used in DecisionModule's current logic)
         # An empty list is sufficient.
         dummy_memory_entries = []

         # Dummy current_concepts (LearningModule.get_concepts() output format)
         # DecisionModule expects this list as the third positional argument (after understanding_signals, relevant_memory_entries).
         # Let's create a simple dummy list, matching the format expected by the `decide` method signature.
         repr_dim = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)
         num_concepts = 3 # Number of dummy concepts
         dummy_concepts = [np.random.rand(repr_dim).astype(np.float64) for _ in range(num_concepts)] # List of dummy concept vectors


         # Return arguments for the DecisionModule.decide method as a tuple.
         # decide(self, understanding_signals, relevant_memory_entries, current_concepts)
         logger.debug("Created dummy decide input tuple for DecisionModule.")
         return (dummy_understanding_signals, dummy_memory_entries, dummy_concepts)


    elif class_name_lower == 'motorcontrolcore':
         # MotorControlCore.generate_response(decision) expects a string or any decision input.
         # This should be in the format of DecisionModule output (a decision string).

         # Create a dummy decision string. Can be changed to test different scenarios.
         # Example: test all possible decision strings.
         possible_decisions = [
             "explore_randomly", "make_noise",
             "sound_detected", "complex_visual_detected",
             "bright_light_detected", "dark_environment_detected",
             "recognized_concept_0", "recognized_concept_1",
             "familiar_input_detected", "new_input_detected",
             "unknown_decision", # Test unknown decision
             None, # Test None decision
         ]
         dummy_decision = random.choice(possible_decisions)

         logger.debug(f"Created dummy generate_response input tuple for MotorControlCore: '{dummy_decision}'.")
         return (dummy_decision,) # Return inside a tuple.


    elif class_name_lower == 'expressiongenerator':
         # ExpressionGenerator.generate(command) expects a string or any command input.
         # This should be in the format of MotorControlCore output (an ExpressionGenerator command string).

         # Create a dummy command string. Can be changed to test different scenarios.
         possible_commands = [
             "explore_randomly_response", "make_noise_response",
             "sound_detected_response", "complex_visual_response",
             "bright_light_response", "dark_environment_response",
             "recognized_concept_response_0", "recognized_concept_response_1",
             "familiar_response", "new_response",
             "default_response", # Test default command
             "unknown_command", # Test unknown command
             None, # Test None command
         ]
         dummy_command = random.choice(possible_commands)

         logger.debug(f"Created dummy generate input tuple for ExpressionGenerator: '{dummy_command}'.")
         return (dummy_command,) # Return inside a tuple.


    elif class_name_lower == 'interactionapi':
         # InteractionAPI.send_output(output_data) expects any output data.
         # This should be in the format of MotorControlCore output (typically a text string from ExpressionGenerator, or None).

         # Create dummy output data. Can be changed to test different scenarios.
         possible_outputs = [
             "This feels familiar.",
             "I sense something new.",
             "I hear a sound.",
             "I think this is concept 0.",
             None, # Test None output
             {"status": "ok", "message": "API response"}, # Test other data types (WebAPI channel)
         ]
         dummy_output_data = random.choice(possible_outputs)

         logger.debug(f"Created dummy send_output input tuple for InteractionAPI: '{dummy_output_data}'.")
         return (dummy_output_data,) # Return inside a tuple.

    elif class_name_lower == 'learningmodule':
         # LearningModule.learn_concepts(representation_list) expects a list of Representation vectors.
         # This should be in the format of a Memory sample (list of RepresentationLearner outputs).
         # Representations stored in Memory are in the format of RepresentationLearner output (shape (repr_dim,), numerical dtype).

         repr_dim = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)
         num_samples = get_config_value(config, 'cognition', 'learning', 'learning_memory_sample_size', default=50, expected_type=int, logger_instance=logger) # Learning sample size from config

         # Create a list of dummy Representations.
         dummy_rep_list = []
         for _ in range(num_samples):
              dummy_rep_list.append(np.random.rand(repr_dim).astype(np.float64)) # Assuming float64 dtype

         # Make some vectors similar to simulate concept clustering potential.
         if num_samples > 5:
              # Very similar (sim ~1.0)
              dummy_rep_list[1] = dummy_rep_list[0].copy() + np.random.randn(repr_dim).astype(np.float64) * 0.01
              # Somewhat similar (sim < 1.0, >= threshold)
              # To get a similarity of ~0.7 with a random vector, requires careful construction or many trials.
              # Let's add noise to an existing vector, but with larger std_dev than "very similar".
              # This should result in similarity values spread between 0 and 1.
              dummy_rep_list[2] = dummy_rep_list[0].copy() + np.random.randn(repr_dim).astype(np.float64) * 0.2

              dummy_rep_list[4] = dummy_rep_list[3].copy() + np.random.randn(repr_dim).astype(np.float64) * 0.01 # Similar to another vector (another cluster)


         logger.debug(f"Created dummy learn_concepts input list for LearningModule ({len(dummy_rep_list)} elements).")
         return (dummy_rep_list,) # Return inside a tuple.

    elif class_name_lower in ['episodicmemory', 'semanticmemory', 'manipulator', 'locomotioncontroller']:
         # Placeholder or unimplemented modules.
         # Main methods for these modules are not yet defined or implemented.
         # Indicate that creating dummy input for these is not currently supported.
         logger.warning(f"'{class_name}' module is a placeholder or not fully implemented. Dummy input creation is not supported.")
         return None # Return None for unsupported modules.


    # If we reach here, the name didn't match any known module.
    logger.error(f"Dummy input creation is not implemented or known for module '{class_name}'.")
    return None # Return None for unknown module names.

# Runs a single module test
def run_module_test(module_path, class_name, config):
    """
    Initializes the specified module, creates dummy input, and runs the module's main processing method.

    Args:
        module_path (str): The Python path of the module (e.g., src.processing.vision).
        class_name (str): The name of the class within the module (e.g., VisionProcessor).
        config (dict): The global configuration dictionary.

    Returns:
        tuple: (success, output_data)
               success (bool): Was the test successful? (Module initialized and method ran without error?)
               output_data (any): The output returned by the module's processing method, or None if the method was not called or an error occurred.
    """
    logger.info(f"--- Module Test Starting: {class_name} ({module_path}) ---")
    # Load the module class dynamically.
    module_class = load_module_class(module_path, class_name)

    if module_class is None:
        logger.error("Module test failed to start: Class could not be loaded.")
        return False, None

    module_instance = None
    output_data = None
    test_success = True # Tracks initialization success. Assumes success initially.
    method_call_success = None # Tracks method call success. None if method not called, True/False if called.

    try:
        # --- Initialize the Module ---
        # Most modules initialize with just the config. CognitionCore requires module_objects.
        # Prepare initialization arguments.
        # The config object passed here is the full config dict loaded by main.
        init_args = [config] # The first argument is always config

        # If the module is CognitionCore, also pass a dummy module_objects dictionary as a required argument.
        # This dummy structure prevents null reference errors during CognitionCore's init when it tries to get references.
        # This mock-like structure is only for initialization purposes in the test script.
        if class_name.lower() == 'cognitioncore':
            # Create a dummy module_objects dictionary.
            # It should contain keys for modules that CognitionCore init expects references to (Memory, etc.),
            # with None values as we don't need actual instances for CognitionCore's init (only references).
            dummy_module_objects = {
                'memories': {'core_memory': None}, # Memory might be needed by CognitionCore init
                'cognition': {}, # Refers to itself, not needed for init args
                'motor_control': {'core_motor_control': None}, # MotorControl might be needed (though likely its sub-modules)
                'interaction': {'core_interaction': None}, # Interaction might be needed
                'representers': {'main_learner': None}, # RepresentationLearner might be needed
                # Sensor and Processor objects are typically not needed by CognitionCore init.
            }
            init_args.append(dummy_module_objects) # Add the dummy module_objects dictionary as a positional argument.


        logger.debug(f"Initializing module '{class_name}'...")
        # Initialize the module object. If an error occurs during init, an exception should be raised.
        # Unpack init_args tuple/list into positional arguments for the __init__ method.
        module_instance = module_class(*init_args)

        # If initialization completed successfully, module_instance is not None (except potentially for Sensor init on failure).
        # If a Sensor init handles failure by returning None or False, initialization is considered failed.
        if module_instance is None or (hasattr(module_instance, 'is_camera_available') and not module_instance.is_camera_available) or (hasattr(module_instance, 'is_audio_available') and not module_instance.is_audio_available):
             logger.error(f"Module '{class_name}' failed to initialize (init returned None/False or indicated unavailability).")
             test_success = False # Initialization failed, so the overall test fails.
             return False, None # Return False success and None output.


        # --- Find and Call the Module's Main Processing Method ---
        # The method to be called should be determined based on the class being tested.
        # The create_dummy_method_inputs function should return a tuple of the arguments expected by this method.

        # Get the dummy input arguments for the method.
        # If create_dummy_method_inputs returns None, it means testing the main method is not supported for this module in the script.
        dummy_method_inputs = create_dummy_method_inputs(class_name, config)

        if dummy_method_inputs is None:
             logger.warning(f"Dummy input creation or main method call not implemented for module '{class_name}'. Only initialization was tested.")
             # The method call was skipped, but initialization was successful (checked above).
             # test_success holds the initialization success status.
             final_success_status = test_success # Initialization success determines overall success if method is skipped.
             logger.debug(f"'{class_name}': Main method test skipped. Initialization Successful: {final_success_status}")
             # Cleanup will happen in the finally block.
             return final_success_status, None # No output since the method was not called.

        else:
             # DEBUG: Log the shape and content of the input arguments.
             logger.debug(f"Created dummy input for method call for '{class_name}'. Arguments: {dummy_method_inputs} (Length: {len(dummy_method_inputs)})")
             # DEBUG: Log the method signature before calling.
             method_to_test = None # Object representing the method to call
             method_name = None

             # --- Method Selection Logic (This is somewhat repetitive and could be refactored) ---
             # Select the main method to test for each module.
             if class_name.lower() in ['visionprocessor', 'audioprocessor', 'understandingmodule']:
                  method_to_test = getattr(module_instance, 'process', None)
                  method_name = 'process'
             elif class_name.lower() in ['visionsensor']:
                  method_to_test = getattr(module_instance, 'capture_frame', None)
                  method_name = 'capture_frame'
                  # capture methods take no arguments, create_dummy_method_inputs should return an empty tuple ().
                  if dummy_method_inputs: # Ensure it's empty tuple if create_dummy_method_inputs returned non-empty
                       logger.warning(f"'{class_name}.capture_frame' method should take no arguments, but dummy input {dummy_method_inputs} was created. Using empty tuple.")
                       dummy_method_inputs = ()
             elif class_name.lower() in ['audiosensor']:
                  method_to_test = getattr(module_instance, 'capture_chunk', None)
                  method_name = 'capture_chunk'
                  # capture methods take no arguments
                  if dummy_method_inputs: # Ensure it's empty tuple
                        logger.warning(f"'{class_name}.capture_chunk' method should take no arguments, but dummy input {dummy_method_inputs} was created. Using empty tuple.")
                        dummy_method_inputs = ()
             elif class_name.lower() == 'representationlearner':
                  method_to_test = getattr(module_instance, 'learn', None)
                  method_name = 'learn'
             elif class_name.lower() == 'memory':
                  # For Memory, we are currently testing the 'store' method by default.
                  # If testing 'retrieve', this selection and dummy input creation would need to change.
                  method_to_test = getattr(module_instance, 'store', None)
                  method_name = 'store'
                  # If retrieve were tested, dummy_method_inputs should return its required args (query_representation, num_results).

             elif class_name.lower() == 'cognitioncore':
                  method_to_test = getattr(module_instance, 'decide', None)
                  method_name = 'decide'
                  # DecisionModule expects understanding_signals, relevant_memory_entries, current_concepts=None
                  # Test script's create_dummy_method_inputs for DecisionModule should match this.

             elif class_name.lower() == 'decisionmodule':
                  method_to_test = getattr(module_instance, 'decide', None)
                  method_name = 'decide'
                  # DecisionModule's decide signature is decide(self, understanding_signals, relevant_memory_entries, current_concepts).
                  # Dummy input creation for DecisionModule should match this.

             elif class_name.lower() == 'motorcontrolcore':
                  method_to_test = getattr(module_instance, 'generate_response', None)
                  method_name = 'generate_response'
             elif class_name.lower() == 'expressiongenerator':
                  method_to_test = getattr(module_instance, 'generate', None)
                  method_name = 'generate'
             elif class_name.lower() == 'interactionapi':
                  method_to_test = getattr(module_instance, 'send_output', None)
                  method_name = 'send_output'
             elif class_name.lower() == 'learningmodule':
                  method_to_test = getattr(module_instance, 'learn_concepts', None)
                  method_name = 'learn_concepts'
             # --- End of Method Selection ---


             if method_to_test is None:
                  logger.error(f"Main processing method ('{method_name}') not found for module '{class_name}'.")
                  method_call_success = False # Method not found, call failed.
                  output_data = None # No output from method call.
             else:
                  # DEBUG: Log the method signature before calling.
                  try:
                      method_signature = inspect.signature(method_to_test)
                      logger.debug(f"'{class_name}.{method_name}' signature: {method_signature}") # DEBUG: İmza bilgisi
                  except Exception as sig_e:
                      logger.warning(f"Could not get signature for '{class_name}.{method_name}': {sig_e}")

                  logger.debug(f"Calling method '{class_name}.{method_name}'...")
                  try:
                      # Call the method with dummy input arguments.
                      # Unpack the tuple into positional arguments using *.
                      output_data = method_to_test(*dummy_method_inputs)

                      # If we reached here without an exception, the method call was successful.
                      method_call_success = True
                      logger.debug(f"Method '{class_name}.{method_name}' executed successfully.")

                      # Log the output if it's not None.
                      if output_data is not None:
                           logger.debug(f"Output from '{class_name}.{method_name}':")
                           log_output_data(output_data) # Use the helper function
                      else:
                          logger.debug(f"Output from '{class_name}.{method_name}': None")


                  except Exception as e:
                      # Catch any unexpected error during the method execution.
                      logger.error(f"Unexpected error while executing method '{class_name}.{method_name}': {e}", exc_info=True)
                      method_call_success = False # Error during execution, call failed.
                      output_data = None # No valid output in case of error.


    except Exception as e:
        # Catch any unexpected error during module initialization (module_instance remains None).
        logger.error(f"Unexpected error while initializing module '{class_name}': {e}", exc_info=True)
        test_success = False # Initialization failed.
        method_call_success = False # Method was not even called.
        output_data = None # No output.

    finally:
        # Clean up resources (if a cleanup method exists).
        # This will be called regardless of whether initialization or method execution failed.
        # Handle cases where module_instance is None due to initialization failure.
        if module_instance and hasattr(module_instance, 'cleanup'):
            logger.debug(f"Calling cleanup for module '{class_name}'.")
            # Use cleanup_safely to catch errors during cleanup itself.
            cleanup_safely(module_instance.cleanup, logger_instance=logger, error_message=f"Error during cleanup for module '{class_name}'")
            logger.debug(f"Cleanup completed for module '{class_name}'.")

    # Determine the final success status:
    # Success is true if Initialization was successful AND (If the method was tested, the method call was successful).
    # Initialization success is tracked by test_success (True unless init raised an exception).
    # Method was tested if dummy_method_inputs is not None.
    # Method call success is tracked by method_call_success (True if it ran without error).

    final_success_status = test_success # Start with initialization success.

    if dummy_method_inputs is not None: # If the method test was attempted...
         # ... then the overall success depends on BOTH initialization success AND method call success.
         # If init failed (test_success is False), final_success_status will be False regardless of method_call_success.
         # If init succeeded (test_success is True), final_success_status will be True only if method_call_success is True.
         final_success_status = final_success_status and (method_call_success is True) # Combine init success with method call success.
    # else: If method test was NOT attempted (dummy_method_inputs is None), final_success_status remains just the initialization success.

    logger.info(f"'{class_name}': Final success status calculated: {final_success_status}")

    return final_success_status, output_data


def log_output_data(data):
    """
    Logs output data in a meaningful way based on its type.
    """
    if data is None:
        logger.debug("  Output: None")
    elif isinstance(data, np.ndarray):
        logger.debug(f"  Output: numpy.ndarray, Shape: {data.shape}, Dtype: {data.dtype}")
        # Log values for small arrays
        if data.size > 0 and data.size < 20: # Arbitrary threshold, don't print values for very large arrays
             # Catch errors when converting NumPy array to string representation (e.g., very large/small numbers)
             try:
                 logger.debug(f"  Output Values: {data}")
             except Exception as e:
                 logger.debug(f"  Output Values (error converting to string): {e}")

        elif data.size > 0 and (data.ndim == 1 or data.ndim == 2): # Log min/max/mean for 1D/2D arrays
             # Use isfinite check to prevent ValueError when converting float NaN to integer in stats calculation.
             finite_data = data[np.isfinite(data)]
             if finite_data.size > 0:
                 logger.debug(f"  Output Stats (min/max/mean): {finite_data.min():.4f}/{finite_data.max():.4f}/{finite_data.mean():.4f}")
             else:
                 logger.debug("  Output Stats: (All values are NaN or Inf)")
        elif data.size == 0:
             logger.debug("  Output: Empty numpy array.")

    elif isinstance(data, dict):
        logger.debug(f"  Output: dict, Keys: {list(data.keys())}")
        # Log dict content in JSON format for better readability
        try:
             # Catch errors when converting NumPy arrays or other non-serializable types to JSON.
             # Use a custom default encoder for non-serializable types.
             def default_json_encoder(obj):
                 if isinstance(obj, np.ndarray):
                     return obj.tolist() # Convert NumPy arrays to list for JSON serialization
                 # Add handling for other specific non-serializable types here if needed
                 return f"<not serializable: {type(obj).__name__}>"

             logger.debug(f"  Output Content: {json.dumps(data, indent=2, sort_keys=True, default=default_json_encoder)}")
        except Exception as e:
             # If JSON serialization fails, log the raw dict and the error.
             logger.debug(f"  Output Content (raw): {data}")
             logger.error(f"Error serializing dict to JSON for logging: {e}", exc_info=True)

    elif isinstance(data, list):
        logger.debug(f"  Output: list, Length: {len(data)}")
        if len(data) > 0:
            logger.debug(f"  First element type: {type(data[0]).__name__}") # Log type name
            # Log the first few elements of the list (if they are numpy arrays or dicts)
            for i, item in enumerate(data[:3]): # Log the first 3 elements
                 logger.debug(f"  Element {i}:")
                 # Recursive call for nested structures (be careful with deep recursion)
                 # For now, just log type and a short representation of the item.
                 try:
                     item_repr = repr(item)
                     if len(item_repr) > 100: item_repr = item_repr[:97] + "..." # Truncate very long representations
                     logger.debug(f"    Type: {type(item).__name__}, Value (repr): {item_repr}")
                 except Exception as e:
                      logger.debug(f"    Type: {type(item).__name__}, Error getting repr: {e}")

        # No specific action needed for an empty list, the length 0 is already logged.
    else:
        # For other basic types (str, int, float, bool, None)
        logger.debug(f"  Output: type {type(data).__name__}, Value: {data}")


def main():
    """
    Main execution function of the script. Parses arguments and starts the module tests.
    """
    # Argument parsing is removed. The script is configured to test specific modules when run directly.

    # Configure the logging system (using default settings if config file is not loaded or missing logging section)
    # Optional: For basic logging before config is loaded. setup_logging can handle None config.
    # setup_logging(config=None)

    # Load the configuration file.
    config_path = "config/main_config.yaml"
    config = load_config_from_yaml(config_path) # Returns empty dict on error.

    # If config fails to load, terminate the script.
    if not config:
        # If logging was already set up, the error will be logged inside load_config_from_yaml.
        # Just exit here.
        sys.exit(1)

    # Re-configure the logging system with the loaded config (logging settings from config will be used).
    setup_logging(config=config)

    # Get the logger for this script itself (after logging setup).
    global logger # Declare that we will use the global logger variable
    logger = logging.getLogger(__name__)
    logger.info("Test script started (Specific modules are being tested).")


    # --- Define Modules to Test ---
    # Target the main modules defined in ROADMAP and STRUCTURE.md.
    modules_to_test = [
        # Senses
        ('src.senses.vision', 'VisionSensor'),
        ('src.senses.audio', 'AudioSensor'),
        # Processing
        ('src.processing.vision', 'VisionProcessor'),
        ('src.processing.audio', 'AudioProcessor'),
        # Representation
        ('src.representation.models', 'RepresentationLearner'),
        # Memory
        ('src.memory.core', 'Memory'), # Assuming testing the store method by default
        # Cognition
        ('src.cognition.understanding', 'UnderstandingModule'),
        ('src.cognition.decision', 'DecisionModule'),
        ('src.cognition.learning', 'LearningModule'),
        ('src.cognition.core', 'CognitionCore'), # Assuming testing the decide method by default
        # Motor Control
        ('src.motor_control.expression', 'ExpressionGenerator'),
        ('src.motor_control.core', 'MotorControlCore'), # Assuming testing the generate_response method by default
        # Interaction
        ('src.interaction.api', 'InteractionAPI'), # Assuming testing the send_output method by default
        # Placeholder modules are not tested for now
        # ('src.memory.episodic', 'EpisodicMemory'),
        # ('src.memory.semantic', 'SemanticMemory'),
        # ('src.motor_control.manipulation', 'Manipulator'),
        # ('src.motor_control.locomotion', 'LocomotionController'),
    ]

    # List to collect all test results
    overall_success = True
    test_results = {} # Dictionary to store results: {'ModuleName': True/False}

    # Run the test for each module
    for module_path, class_name in modules_to_test:
        logger.info(f"\n>>> TESTING: {class_name} ({module_path}) <<<")
        # run_module_test returns boolean success and the output.
        success, _ = run_module_test(module_path, class_name, config) # We log the output inside run_module_test, so don't need it here.
        test_results[class_name] = success
        if not success:
            overall_success = False # If even one test fails, the overall result is False.
        logger.info(f">>> TEST RESULT: {class_name}: {'SUCCESS' if success else 'FAILED'} <<<")


    # Report the overall test results.
    logger.info("\n--- ALL MODULE TESTS COMPLETED ---")
    for class_name, success in test_results.items():
        logger.info(f"Test Result '{class_name}': {'SUCCESS' if success else 'FAILED'}")

    logger.info(f"\nOVERALL TEST RESULT: { 'ALL TESTS SUCCESSFUL' if overall_success else 'SOME TESTS FAILED' }")

    # If any test failed, exit with a non-zero status code.
    if not overall_success:
        sys.exit(1) # Exit with error code

    logger.info("Test script finished.")


if __name__ == '__main__':
    # Call the main function when the script is executed directly.
    main()