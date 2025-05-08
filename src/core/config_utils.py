# src/core/config_utils.py
#
# Evo projesi için merkezi yapılandırma (config) yükleme yardımcı fonksiyonlarını içerir (Konsolide Edilmiş).
# YAML dosyasından config yükler ve iç içe geçmiş değerlere güvenli erişim sağlar.

import yaml
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

def load_config_from_yaml(filepath="config/main_config.yaml"):
    """
    Belirtilen YAML dosyasından yapılandırmayı yükler.

    Args:
        filepath (str): Yüklemek için YAML dosyasının yolu.

    Returns:
        dict: Yüklenen yapılandırma sözlüğü veya hata durumunda boş sözlük.
    """
    if not isinstance(filepath, str) or not filepath:
        logger.error("Invalid file path specified when loading configuration.")
        return {}

    # Debug: Log current working directory
    # logger.debug(f"Current working directory: {os.getcwd()}")

    if not os.path.exists(filepath):
        logger.error(f"Configuration file not found: {filepath}. Please ensure the path is correct.")
        return {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully from: {filepath}")
            # yaml.safe_load can return None for empty files
            return config if config is not None else {}
    except yaml.YAMLError as e:
        logger.error(f"YAML error while reading configuration file: {filepath}", exc_info=True)
        return {}
    except Exception as e:
        logger.error(f"Unexpected error while loading configuration file: {filepath}", exc_info=True)
        return {}


# --- get_config_value function (supports nested key paths) ---
# Signature changed: Only config and keys are positional, default and expected_type are keyword only.
# Workaround removed.
def get_config_value(config: dict, *keys: str, default=None, expected_type=None, logger_instance=None):
    """
    Retrieves a value from a nested dictionary using a chain of keys (*keys).
    If the key is not found, the path is invalid, or the type doesn't match,
    returns the specified default value.
    Optionally checks the expected type and accepts a logger instance for logging.

    Args:
        config (dict): The configuration dictionary to look into.
        *keys (str): Nested key steps. E.g., 'logging', 'level' or ('logging', 'level').
                     This parameter MUST contain only strings (or tuples of strings).
        default (any, optional): The default value to return if the key is not found,
                                 the path is invalid, or the type doesn't match. Defaults to None.
                                 This parameter SHOULD ONLY be provided as a keyword argument default=....
        expected_type (type or tuple of types, optional): Expected type or tuple of types for the value.
                                                       If None, no type check is performed.
                                                       Special types like np.number should be imported from numpy.
        logger_instance (logging.Logger, optional): The logger instance to use for logging.
                                                   If None, the module logger is used.

    Returns:
        any: The value found or the default value.
    """
    log = logger_instance if logger_instance is not None else logger

    # Workaround for positional defaults is removed.
    # Calls must now conform to the new signature (keys are strings, default is keyword).

    path_str = ' -> '.join(map(str, keys)) # keys should always be strings

    current_value = config
    final_value = default # The default value is the one provided via default=

    # Ensure the initial config is a dictionary
    if not isinstance(current_value, dict):
        log.debug(f"get_config_value: Initial config is not a valid dictionary (type: {type(current_value)}). Returning default ({default}) for path '{path_str}'.")
        return default

    # If the key path is empty, return the config dict itself (and check type).
    # This case is handled if no keys are provided, e.g., get_config_value(config, default=X).
    if not keys:
         log.debug("get_config_value: No keys specified. Returning the config dict itself.")
         final_value = config # The found value is the config dict itself


    try:
        # Traverse the key path
        for i, key in enumerate(keys):
            # If the key is not a string, it's likely an old positional default format error.
            if not isinstance(key, str):
                 log.error(f"get_config_value: Key step in path '{path_str}' is not a string: type {type(key)} (step {i+1}/{len(keys)}, key '{key}'). Returning default ({default}).", exc_info=True)
                 return default # Return default if key is not a string

            # If the current value is not a dict and we are still in the middle of the path, the path is invalid.
            # The loop is entered only if keys is not empty.
            if not isinstance(current_value, dict):
                 # If we are here, current_value is not a dict, and i < len(keys) is true by loop logic.
                 # So the path is indeed invalid at this step.
                 log.debug(f"get_config_value: Path '{path_str}' traversal failed: intermediate value is not a dictionary (step {i+1}/{len(keys)}, key '{key}', type: {type(current_value)}). Returning default ({default}).")
                 return default # Expected a dict in the middle of the path, but got something else.


            try:
                 # We know current_value is a dict (from the check above), we can safely access the key.
                 current_value = current_value[key]

            except (KeyError, TypeError):
                 # Key not found (at the end or in the middle of the path) or current_value was not a dict (TypeError).
                 # The TypeError case should be caught by the isinstance check above, but catching here adds robustness.
                 # Make the log message clearer.
                 log.debug(f"get_config_value: Key '{key}' not found or intermediate value was not a dictionary along path '{path_str}' (step {i+1}/{len(keys)}, current type: {type(current_value)}). Returning default ({default}).")
                 return default # Return default if key is missing or intermediate value is not a dict.

            except Exception as e:
                 # Other unexpected errors (e.g., if key type is not valid for dict access, although handled by isinstance(key, str))
                 log.error(f"get_config_value: Unexpected error during path traversal for '{path_str}' (step {i+1}/{len(keys)}, key '{key}'): {e}", exc_info=True)
                 log.debug(f"get_config_value: Returning default value ({default}) after error.")
                 return default


        # The loop completed successfully (all keys were in the path).
        # current_value is now the found value.
        # If keys were empty (handled before the try block), current_value is still the initial config.
        final_value = current_value # Assign the found value to final_value


    except Exception as e:
        # Catch any general unexpected errors during the process (this block should theoretically not be hit often)
        log.error(f"get_config_value: General error caught while processing path '{path_str}': {e}", exc_info=True)
        log.debug(f"get_config_value: Returning default value ({default}) after general error.")
        return default # Return default in case of a general error.


    # --- Perform type check ---
    # Don't perform type check for None value (None is always None)
    if final_value is not None and expected_type is not None:
        # isinstance accepts a tuple of types. Add special handling for numpy types.
        is_correct_type = False

        # If expected_type is a tuple, check against each type in the tuple
        if isinstance(expected_type, tuple):
            for t in expected_type:
                 if t == np.number:
                      # Check for numpy number: int, float, or numpy scalar/array with number dtype.
                      if isinstance(final_value, (int, float)) or (isinstance(final_value, np.ndarray) and np.issubdtype(final_value.dtype, np.number)) or (np.isscalar(final_value) and np.issubdtype(type(final_value), np.number)):
                           is_correct_type = True
                           break # Found correct type in tuple, break the loop.
                 elif t == np.ndarray:
                      if isinstance(final_value, np.ndarray):
                           is_correct_type = True
                           break # Found correct type in tuple, break the loop.
                 # Normal type check (int, float, str, list, dict, etc.)
                 elif isinstance(final_value, t):
                    is_correct_type = True
                    break # Found correct type in tuple, break the loop.
        # If expected_type is not a tuple, check against the single type
        else:
             if expected_type == np.number:
                  if isinstance(final_value, (int, float)) or (isinstance(final_value, np.ndarray) and np.issubdtype(final_value.dtype, np.number)) or (np.isscalar(final_value) and np.issubdtype(type(final_value), np.number)):
                       is_correct_type = True
             elif expected_type == np.ndarray:
                  if isinstance(final_value, np.ndarray):
                       is_correct_type = True
             # Normal type check
             elif isinstance(final_value, expected_type):
                is_correct_type = True


        if not is_correct_type:
             log.warning(f"get_config_value: Value found for config path '{path_str}' is not of the expected type (expected: {expected_type}, got: {type(final_value)}). Returning default value ({default}).")
             # The default value MUST be returned in case of a type mismatch.
             return default # Return default if type mismatch


    # Value found and passed type check (or type check was not requested/value is None). Return the found value.
    # log.debug(f"get_config_value: Value successfully retrieved for '{path_str}' (type: {type(final_value).__name__}).") # Can be noisy
    return final_value