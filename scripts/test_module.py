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
        logger.debug(f"'{class_name}' capture method takes no arguments. Returning empty input tuple.")
        return ()

    elif class_name_lower == 'visionprocessor':
        # VisionProcessor.process(visual_input) expects a numpy array.
        dummy_width = get_config_value(config, 'vision', 'dummy_width', default=640, expected_type=int, logger_instance=logger)
        dummy_height = get_config_value(config, 'vision', 'dummy_height', default=480, expected_type=int, logger_instance=logger)
        dummy_frame = np.random.randint(0, 256, size=(dummy_height, dummy_width, 3), dtype=np.uint8)
        logger.debug(f"Created dummy process input frame for VisionProcessor ({dummy_frame.shape}, {dummy_frame.dtype}).")
        return (dummy_frame,)


    elif class_name_lower == 'audioprocessor':
        # AudioProcessor.process(audio_input) expects an int16 numpy array.
        chunk_size = get_config_value(config, 'audio', 'audio_chunk_size', default=1024, expected_type=int, logger_instance=logger)
        sample_rate = get_config_value(config, 'audio', 'audio_rate', default=44100, expected_type=int, logger_instance=logger)
        frequency = 440 # A4 note
        amplitude = np.iinfo(np.int16).max * 0.5 # Half of max amplitude
        t = np.linspace(0., float(chunk_size) / sample_rate, chunk_size, endpoint=False)
        dummy_chunk = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)
        logger.debug(f"Created dummy process input chunk for AudioProcessor ({dummy_chunk.shape}, {dummy_chunk.dtype}).")
        return (dummy_chunk,)


    elif class_name_lower == 'representationlearner':
        # RepresentationLearner.learn(processed_inputs) expects a processed_inputs dictionary.
        vis_out_w = get_config_value(config, 'processors', 'vision', 'output_width', default=64, expected_type=int, logger_instance=logger)
        vis_out_h = get_config_value(config, 'processors', 'vision', 'output_height', default=64, expected_type=int, logger_instance=logger)
        dummy_processed_visual_gray = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
        dummy_processed_visual_edges = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
        dummy_processed_visual_dict = {'grayscale': dummy_processed_visual_gray, 'edges': dummy_processed_visual_edges}

        audio_out_dim = get_config_value(config, 'processors', 'audio', 'output_dim', default=2, expected_type=int, logger_instance=logger)
        dummy_processed_audio_features = np.random.rand(audio_out_dim).astype(np.float32) # AudioProcessor outputs float32

        dummy_processed_inputs = {
            'visual': dummy_processed_visual_dict,
            'audio': dummy_processed_audio_features
        }
        logger.debug(f"Created dummy learn input processed_inputs for RepresentationLearner ({list(dummy_processed_inputs.keys())}).")
        return (dummy_processed_inputs,)


    elif class_name_lower == 'memory':
        # Memory.store(representation, metadata=None)
        repr_dim = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)
        dummy_representation = np.random.rand(repr_dim).astype(np.float64) # RL outputs float64
        dummy_metadata = {"source": "test_script_memory", "timestamp": time.time(), "random_val": random.randint(1,100)}
        logger.debug(f"Created dummy store input Representation ({dummy_representation.shape}, {dummy_representation.dtype}) and Metadata for Memory.")
        return (dummy_representation, dummy_metadata) # Tuple of arguments for 'store'


    elif class_name_lower == 'understandingmodule':
        # UnderstandingModule.process(processed_inputs, learned_representation, relevant_memory_entries, current_concepts)
        logger.debug("Creating dummy inputs for UnderstandingModule.process...")

        # Dummy processed_inputs (similar to RL input)
        vis_out_w_um = get_config_value(config, 'processors', 'vision', 'output_width', default=64, expected_type=int, logger_instance=logger)
        vis_out_h_um = get_config_value(config, 'processors', 'vision', 'output_height', default=64, expected_type=int, logger_instance=logger)
        dummy_um_visual_gray = np.random.randint(0, 256, size=(vis_out_h_um, vis_out_w_um), dtype=np.uint8)
        dummy_um_visual_edges = np.random.randint(0, 256, size=(vis_out_h_um, vis_out_w_um), dtype=np.uint8)
        dummy_um_visual_dict = {'grayscale': dummy_um_visual_gray, 'edges': dummy_um_visual_edges}

        audio_out_dim_um = get_config_value(config, 'processors', 'audio', 'output_dim', default=2, expected_type=int, logger_instance=logger)
        dummy_um_audio_features = np.random.rand(audio_out_dim_um).astype(np.float32) # AudioProcessor outputs float32
        # Ensure audio features can trigger threshold for testing
        if random.choice([True, False]): # Randomly make audio energy high
             audio_energy_threshold_um = get_config_value(config, 'cognition', 'audio_energy_threshold', default=1000.0)
             dummy_um_audio_features[0] = float(audio_energy_threshold_um) + random.uniform(1,100) # Index 0 is energy
        else:
             dummy_um_audio_features[0] = random.uniform(0, get_config_value(config, 'cognition', 'audio_energy_threshold', default=1000.0) -1)


        dummy_um_processed_inputs = {
            'visual': dummy_um_visual_dict,
            'audio': dummy_um_audio_features
        }

        # Dummy learned_representation
        repr_dim_um = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)
        dummy_um_representation = np.random.rand(repr_dim_um).astype(np.float64) # RL outputs float64

        # Dummy relevant_memory_entries
        num_mem_um = get_config_value(config, 'memory', 'num_retrieved_memories', default=5, expected_type=int, logger_instance=logger)
        num_retrieved = random.randint(0, num_mem_um) # Simulate varying number of retrieved memories
        dummy_um_memory_entries = []
        for i in range(num_retrieved):
             dummy_mem_rep_um = np.random.rand(repr_dim_um).astype(np.float64)
             if i == 0 and num_retrieved > 0: # Make first retrieved memory similar to current representation
                 dummy_mem_rep_um = dummy_um_representation.copy() + np.random.randn(repr_dim_um).astype(np.float64) * 0.01
             dummy_um_memory_entries.append({
                 'representation': dummy_mem_rep_um,
                 'metadata': {'source': 'test_script_um_mem', 'index': i},
                 'timestamp': time.time() - random.uniform(0, 3600)
             })

        # Dummy current_concepts
        num_concepts_um = random.randint(0, 3) # Simulate 0 to 3 learned concepts
        dummy_um_concepts = []
        for i in range(num_concepts_um):
             concept_rep_um = np.random.rand(repr_dim_um).astype(np.float64)
             if i == 0 and num_concepts_um > 0 and random.choice([True, False]): # Make first concept similar to current representation
                 concept_rep_um = dummy_um_representation.copy() + np.random.randn(repr_dim_um).astype(np.float64) * 0.05
             dummy_um_concepts.append(concept_rep_um)

        logger.debug("Created dummy process input tuple for UnderstandingModule.")
        return (dummy_um_processed_inputs, dummy_um_representation, dummy_um_memory_entries, dummy_um_concepts)


    elif class_name_lower == 'cognitioncore':
         # CognitionCore.decide(processed_inputs, learned_representation, relevant_memory_entries, current_concepts)
         # This will use the same logic as UnderstandingModule's dummy input generation for its arguments.
         # For simplicity, let's call the UnderstandingModule's dummy input generation helper
         # as CognitionCore expects similar inputs.
         # Note: This creates a slight coupling but is pragmatic for this test script.
         logger.debug("Creating dummy inputs for CognitionCore.decide (reusing UM input logic)...")
         # Arguments for decide(self, processed_inputs, learned_representation, relevant_memory_entries, current_concepts)
         # The create_dummy_method_inputs for 'understandingmodule' returns exactly this tuple.
         # To avoid direct call, we can replicate the logic or make it more generic.
         # For now, let's replicate the essential parts.

         vis_out_w_cc = get_config_value(config, 'processors', 'vision', 'output_width', default=64, expected_type=int, logger_instance=logger)
         vis_out_h_cc = get_config_value(config, 'processors', 'vision', 'output_height', default=64, expected_type=int, logger_instance=logger)
         dummy_cc_visual_gray = np.random.randint(0, 256, size=(vis_out_h_cc, vis_out_w_cc), dtype=np.uint8)
         dummy_cc_visual_edges = np.random.randint(0, 256, size=(vis_out_h_cc, vis_out_w_cc), dtype=np.uint8)
         dummy_cc_visual_dict = {'grayscale': dummy_cc_visual_gray, 'edges': dummy_cc_visual_edges}
         audio_out_dim_cc = get_config_value(config, 'processors', 'audio', 'output_dim', default=2, expected_type=int, logger_instance=logger)
         dummy_cc_audio_features = np.random.rand(audio_out_dim_cc).astype(np.float32)
         dummy_cc_processed_inputs = {'visual': dummy_cc_visual_dict, 'audio': dummy_cc_audio_features}

         repr_dim_cc = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)
         dummy_cc_representation = np.random.rand(repr_dim_cc).astype(np.float64)

         num_mem_cc = get_config_value(config, 'memory', 'num_retrieved_memories', default=5, expected_type=int, logger_instance=logger)
         dummy_cc_memory_entries = [{'representation': np.random.rand(repr_dim_cc).astype(np.float64), 'metadata': {}, 'timestamp': time.time()} for _ in range(random.randint(0, num_mem_cc))]

         num_concepts_cc = random.randint(0,3)
         dummy_cc_concepts = [np.random.rand(repr_dim_cc).astype(np.float64) for _ in range(num_concepts_cc)]

         logger.debug("Created dummy decide input tuple for CognitionCore.")
         return (dummy_cc_processed_inputs, dummy_cc_representation, dummy_cc_memory_entries, dummy_cc_concepts)


    elif class_name_lower == 'decisionmodule':
         # DecisionModule.decide(understanding_signals, relevant_memory_entries, current_concepts)
         dummy_understanding_signals = {
             'similarity_score': random.uniform(0.0, 1.0),
             'high_audio_energy': random.choice([True, False]),
             'high_visual_edges': random.choice([True, False]),
             'is_bright': False, # Start with False
             'is_dark': False,   # Start with False
             'max_concept_similarity': random.uniform(0.0, 1.0),
             'most_similar_concept_id': random.choice([None] + list(range(random.randint(0,3)))),
         }
         # Ensure is_bright and is_dark are not both True
         if random.choice([True, False]): dummy_understanding_signals['is_bright'] = True
         else: dummy_understanding_signals['is_dark'] = True
         if dummy_understanding_signals['is_bright'] and dummy_understanding_signals['is_dark']: # Should not happen with above logic, but safety
             if random.choice([True,False]): dummy_understanding_signals['is_dark'] = False
             else: dummy_understanding_signals['is_bright'] = False


         repr_dim_dm = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)
         dummy_dm_memory_entries = [] # DecisionModule does not use content of memory entries currently
         dummy_dm_concepts = [np.random.rand(repr_dim_dm).astype(np.float64) for _ in range(random.randint(0,3))]

         logger.debug("Created dummy decide input tuple for DecisionModule.")
         return (dummy_understanding_signals, dummy_dm_memory_entries, dummy_dm_concepts)


    elif class_name_lower == 'motorcontrolcore':
         # MotorControlCore.generate_response(decision)
         possible_decisions = [
             "explore_randomly", "make_noise", "sound_detected", "complex_visual_detected",
             "bright_light_detected", "dark_environment_detected", "recognized_concept_0",
             "familiar_input_detected", "new_input_detected", None, "some_other_decision"
         ]
         dummy_decision = random.choice(possible_decisions)
         logger.debug(f"Created dummy generate_response input tuple for MotorControlCore: '{dummy_decision}'.")
         return (dummy_decision,)


    elif class_name_lower == 'expressiongenerator':
         # ExpressionGenerator.generate(command)
         possible_commands = [
             "explore_randomly_response", "make_noise_response", "sound_detected_response",
             "recognized_concept_response_0", "default_response", None, "unknown_command"
         ]
         dummy_command = random.choice(possible_commands)
         logger.debug(f"Created dummy generate input tuple for ExpressionGenerator: '{dummy_command}'.")
         return (dummy_command,)


    elif class_name_lower == 'interactionapi':
         # InteractionAPI.send_output(output_data)
         possible_outputs = [
             "Test output string.", None, {"type": "status", "content": "OK"},
             np.array([1,2,3]) # Test non-string/dict output
         ]
         dummy_output_data = random.choice(possible_outputs)
         logger.debug(f"Created dummy send_output input tuple for InteractionAPI: type {type(dummy_output_data)}.")
         return (dummy_output_data,)

    elif class_name_lower == 'learningmodule':
         # LearningModule.learn_concepts(representation_list)
         repr_dim_lm = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)
         num_samples_lm = get_config_value(config, 'cognition', 'learning_memory_sample_size', default=50, expected_type=int, logger_instance=logger)
         num_to_create = random.randint(1, num_samples_lm if num_samples_lm > 0 else 1)

         dummy_rep_list = []
         if num_to_create > 0:
             base_rep = np.random.rand(repr_dim_lm).astype(np.float64)
             dummy_rep_list.append(base_rep) # First representation
             for i in range(1, num_to_create):
                 if i % 3 == 0: # Create a similar vector
                     dummy_rep_list.append(base_rep.copy() + np.random.randn(repr_dim_lm).astype(np.float64) * 0.01)
                 elif i % 3 == 1 and num_to_create > 2: # Create another cluster
                     another_base = np.random.rand(repr_dim_lm).astype(np.float64)
                     dummy_rep_list.append(another_base)
                 else: # Create a random vector
                     dummy_rep_list.append(np.random.rand(repr_dim_lm).astype(np.float64))
         logger.debug(f"Created dummy learn_concepts input list for LearningModule ({len(dummy_rep_list)} elements).")
         return (dummy_rep_list,)

    elif class_name_lower in ['episodicmemory', 'semanticmemory', 'manipulator', 'locomotioncontroller']:
         logger.warning(f"'{class_name}' module is a placeholder or not fully implemented. Dummy input creation is not supported.")
         return None


    logger.error(f"Dummy input creation is not implemented or known for module '{class_name}'.")
    return None

# Runs a single module test
def run_module_test(module_path, class_name, config):
    """
    Initializes the specified module, creates dummy input, and runs the module's main processing method.
    """
    logger.info(f"--- Module Test Starting: {class_name} ({module_path}) ---")
    module_class = load_module_class(module_path, class_name)

    if module_class is None:
        logger.error("Module test failed to start: Class could not be loaded.")
        return False, None

    module_instance = None
    output_data = None
    initialization_success = False
    method_call_success = None # None: not called, True: success, False: failed

    try:
        init_args = [config]
        if class_name.lower() == 'cognitioncore':
            dummy_module_objects = {
                'memories': {'core_memory': None},
                'cognition': {},
                'motor_control': {'core_motor_control': None},
                'interaction': {'core_interaction': None},
                'representers': {'main_learner': None},
            }
            init_args.append(dummy_module_objects)

        logger.debug(f"Initializing module '{class_name}'...")
        module_instance = module_class(*init_args)

        # Check for specific failure indicators from sensor initializations
        init_failed_sensor = False
        if class_name.lower() == 'visionsensor' and (module_instance is None or not getattr(module_instance, 'is_camera_available', True)):
            init_failed_sensor = True
        if class_name.lower() == 'audiosensor' and (module_instance is None or not getattr(module_instance, 'is_audio_available', True)):
            init_failed_sensor = True

        if module_instance is None or init_failed_sensor:
             logger.error(f"Module '{class_name}' failed to initialize properly.")
             initialization_success = False
        else:
            logger.info(f"Module '{class_name}' initialized successfully.")
            initialization_success = True

        if not initialization_success:
            return False, None # Skip method call if init failed

        # --- Find and Call the Module's Main Processing Method ---
        dummy_method_inputs = create_dummy_method_inputs(class_name, config)

        if dummy_method_inputs is None:
             logger.warning(f"Dummy input creation or main method call not implemented for module '{class_name}'. Only initialization was tested.")
             # Method call skipped, overall success depends only on initialization_success
        else:
             method_name_map = {
                 'visionsensor': 'capture_frame', 'audiosensor': 'capture_chunk',
                 'visionprocessor': 'process', 'audioprocessor': 'process',
                 'representationlearner': 'learn', 'memory': 'store', # Defaulting to 'store' for Memory
                 'understandingmodule': 'process', 'decisionmodule': 'decide',
                 'learningmodule': 'learn_concepts', 'cognitioncore': 'decide',
                 'expressiongenerator': 'generate', 'motorcontrolcore': 'generate_response',
                 'interactionapi': 'send_output',
             }
             method_name = method_name_map.get(class_name.lower())

             if not method_name:
                 logger.error(f"Main processing method name not defined in script for module '{class_name}'.")
                 method_call_success = False
             else:
                 method_to_test = getattr(module_instance, method_name, None)
                 if method_to_test is None:
                      logger.error(f"Method '{method_name}' not found in module '{class_name}'.")
                      method_call_success = False
                 else:
                      # Special handling for sensor capture methods (no args)
                      if class_name.lower() in ['visionsensor', 'audiosensor'] and dummy_method_inputs:
                           logger.warning(f"'{class_name}.{method_name}' takes no arguments, but dummy input {dummy_method_inputs} was created. Using empty tuple.")
                           dummy_method_inputs = ()

                      logger.debug(f"Calling method '{class_name}.{method_name}' with args: {dummy_method_inputs if len(dummy_method_inputs) < 2 else str(type(dummy_method_inputs[0])) + '...'}")
                      try:
                          output_data = method_to_test(*dummy_method_inputs)
                          method_call_success = True
                          logger.debug(f"Method '{class_name}.{method_name}' executed successfully.")
                          if output_data is not None:
                               logger.debug(f"Output from '{class_name}.{method_name}':")
                               log_output_data(output_data)
                          else:
                              logger.debug(f"Output from '{class_name}.{method_name}': None")
                      except Exception as e:
                          logger.error(f"Error executing method '{class_name}.{method_name}': {e}", exc_info=True)
                          method_call_success = False
                          output_data = None
    except Exception as e:
        logger.error(f"Error during test for module '{class_name}': {e}", exc_info=True)
        initialization_success = False # Assume init failed if error before method_call_success is set
        # method_call_success will remain None or False depending on where the error occurred

    finally:
        if module_instance and hasattr(module_instance, 'cleanup'):
            logger.debug(f"Calling cleanup for module '{class_name}'.")
            cleanup_safely(module_instance.cleanup, logger_instance=logger, error_message=f"Error during cleanup for module '{class_name}'")
            logger.debug(f"Cleanup completed for module '{class_name}'.")

    # Determine final success
    final_success_status = initialization_success
    if dummy_method_inputs is not None and initialization_success: # If method was attempted and init was successful
        final_success_status = method_call_success is True

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
        if data.size > 0 and data.size < 20:
             try:
                 logger.debug(f"  Output Values: {np.array2string(data, precision=4, suppress_small=True, max_line_width=120)}")
             except Exception as e:
                 logger.debug(f"  Output Values (error converting to string): {e}")
        elif data.size > 0 and (data.ndim == 1 or data.ndim == 2):
             finite_data = data[np.isfinite(data)]
             if finite_data.size > 0:
                 logger.debug(f"  Output Stats (min/max/mean): {finite_data.min():.4f}/{finite_data.max():.4f}/{finite_data.mean():.4f}")
             else:
                 logger.debug("  Output Stats: (All values are NaN or Inf, or array is empty after filtering)")
        elif data.size == 0:
             logger.debug("  Output: Empty numpy array.")
    elif isinstance(data, dict):
        logger.debug(f"  Output: dict, Keys: {list(data.keys())}")
        try:
             def default_json_encoder(obj):
                 if isinstance(obj, np.ndarray):
                     return obj.tolist()
                 if isinstance(obj, (np.generic, np.number)): # Handle numpy scalar types
                    return obj.item()
                 try: # Try to get a string representation for other unhandled types
                     return repr(obj)
                 except Exception:
                    return f"<not serializable: {type(obj).__name__}>"

             # Log only a few items if dict is too large
             if len(data) > 10:
                 logger.debug(f"  Output Content (first 10 items): {json.dumps(dict(list(data.items())[:10]), indent=2, sort_keys=True, default=default_json_encoder)}")
                 logger.debug(f"  ... and {len(data)-10} more items.")
             else:
                logger.debug(f"  Output Content: {json.dumps(data, indent=2, sort_keys=True, default=default_json_encoder)}")
        except Exception as e:
             logger.debug(f"  Output Content (raw): {data}")
             logger.error(f"Error serializing dict to JSON for logging: {e}", exc_info=False) # exc_info=False for brevity
    elif isinstance(data, list):
        logger.debug(f"  Output: list, Length: {len(data)}")
        if len(data) > 0:
            logger.debug(f"  First element type: {type(data[0]).__name__}")
            # Log first few elements more concisely
            if len(data) > 5:
                logger.debug(f"  First 5 elements (repr): {[repr(item)[:80] + '...' if len(repr(item)) > 80 else repr(item) for item in data[:5]]}")
                logger.debug(f"  ... and {len(data)-5} more elements.")
            else:
                logger.debug(f"  Elements (repr): {[repr(item)[:80] + '...' if len(repr(item)) > 80 else repr(item) for item in data]}")
    else:
        logger.debug(f"  Output: type {type(data).__name__}, Value: {str(data)[:200]}") # Truncate long strings

def main():
    config_path = "config/main_config.yaml"
    config = load_config_from_yaml(config_path)

    if not config:
        print("CRITICAL: Configuration file could not be loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    setup_logging(config=config)

    global logger # Use global logger
    logger = logging.getLogger(__name__) # Re-initialize after setup_logging
    logger.info("Test script started (Specific modules are being tested).")

    modules_to_test = [
        ('src.senses.vision', 'VisionSensor'),
        ('src.senses.audio', 'AudioSensor'),
        ('src.processing.vision', 'VisionProcessor'),
        ('src.processing.audio', 'AudioProcessor'),
        ('src.representation.models', 'RepresentationLearner'),
        ('src.memory.core', 'Memory'),
        ('src.cognition.understanding', 'UnderstandingModule'), # Added
        ('src.cognition.decision', 'DecisionModule'),
        ('src.cognition.learning', 'LearningModule'),
        ('src.cognition.core', 'CognitionCore'),
        ('src.motor_control.expression', 'ExpressionGenerator'),
        ('src.motor_control.core', 'MotorControlCore'),
        ('src.interaction.api', 'InteractionAPI'),
    ]

    overall_success = True
    test_results = {}

    for module_path, class_name in modules_to_test:
        logger.info(f"\n>>> TESTING: {class_name} ({module_path}) <<<")
        success, _ = run_module_test(module_path, class_name, config)
        test_results[class_name] = success
        if not success:
            overall_success = False
        logger.info(f">>> TEST RESULT: {class_name}: {'SUCCESS' if success else 'FAILED'} <<<")

    logger.info("\n--- ALL MODULE TESTS COMPLETED ---")
    for class_name, success in test_results.items():
        logger.info(f"Test Result '{class_name}': {'SUCCESS' if success else 'FAILED'}")

    logger.info(f"\nOVERALL TEST RESULT: { 'ALL TESTS SUCCESSFUL' if overall_success else 'SOME TESTS FAILED' }")

    if not overall_success:
        sys.exit(1)

    logger.info("Test script finished.")

if __name__ == '__main__':
    main()