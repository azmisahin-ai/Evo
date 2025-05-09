# src/core/compute_utils.py
import logging
import torch
import numpy as np
from src.core.config_utils import get_config_value

logger = logging.getLogger(__name__)

_DEVICE = None
_BACKEND = "numpy" 

def initialize_compute_backend(config):
    global _DEVICE, _BACKEND

    preferred_backend = get_config_value(config, 'compute', 'backend', default="numpy", expected_type=str, logger_instance=logger).lower()
    preferred_device_str = get_config_value(config, 'compute', 'device', default="cpu", expected_type=str, logger_instance=logger).lower()

    if preferred_backend == "pytorch":
        _BACKEND = "pytorch"
        if preferred_device_str.startswith("cuda") and torch.cuda.is_available():
            try:
                _DEVICE = torch.device(preferred_device_str)
                torch.randn(1, device=_DEVICE) # Test device access
                logger.info(f"Compute backend: PyTorch. Using device: {_DEVICE}")
            except Exception as e:
                logger.warning(f"PyTorch CUDA device '{preferred_device_str}' requested but failed: {e}. Falling back to CPU.")
                _DEVICE = torch.device("cpu")
                logger.info(f"Compute backend: PyTorch. Using device: {_DEVICE}")
        elif preferred_device_str == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            try:
                _DEVICE = torch.device("mps")
                torch.randn(1, device=_DEVICE) # Test device access
                logger.info(f"Compute backend: PyTorch. Using device: {_DEVICE} (Apple Silicon MPS)")
            except Exception as e:
                logger.warning(f"PyTorch MPS device requested but failed: {e}. Falling back to CPU.")
                _DEVICE = torch.device("cpu")
                logger.info(f"Compute backend: PyTorch. Using device: {_DEVICE}")
        else:
            if preferred_device_str not in ["cpu", "cuda", "mps"]: # Bilinmeyen veya desteklenmeyen bir cihazsa
                 logger.warning(f"Requested PyTorch device '{preferred_device_str}' is not 'cpu', 'cuda', or 'mps', or not available. Falling back to CPU.")
            _DEVICE = torch.device("cpu")
            logger.info(f"Compute backend: PyTorch. Using device: {_DEVICE}")
    
    # TODO: TensorFlow backend desteği
    # elif preferred_backend == "tensorflow":
    #     _BACKEND = "tensorflow"
    #     # ... TensorFlow GPU/CPU ayarları ...
    #     _DEVICE = "cpu" # Placeholder
    #     logger.info(f"Compute backend: TensorFlow. Using device: {_DEVICE} (Placeholder)")
            
    else: # Varsayılan veya 'numpy'
        _BACKEND = "numpy"
        _DEVICE = "cpu" # NumPy için cihaz her zaman CPU
        logger.info(f"Compute backend: NumPy. Using device: {_DEVICE}")

def get_device():
    if _DEVICE is None:
        logger.critical("Compute device not initialized! Call initialize_compute_backend() first. THIS IS A BUG.")
        # Acil durum fallback
        if _BACKEND == "pytorch": return torch.device("cpu")
        return "cpu"
    return _DEVICE

def get_backend():
    return _BACKEND

def to_tensor(numpy_array, target_device=None):
    """Converts a NumPy array to the backend's tensor format and moves it to the target device."""
    if not isinstance(numpy_array, np.ndarray):
        logger.warning(f"Input to to_tensor is not a NumPy array (type: {type(numpy_array)}). Returning as is.")
        return numpy_array
        
    if _BACKEND == "pytorch":
        try:
            # .float() ile çoğu durumda uyumlu hale getiriyoruz.
            tensor = torch.from_numpy(numpy_array).float() 
            return tensor.to(target_device if target_device else get_device())
        except Exception as e:
            logger.error(f"Error converting NumPy array to PyTorch tensor: {e}", exc_info=True)
            return None # Hata durumunda None dön
    # elif _BACKEND == "tensorflow":
    #     # return tf.convert_to_tensor(numpy_array, dtype=...)
    #     pass
    return numpy_array # NumPy backend'inde veya bilinmeyen backend'de olduğu gibi döndür

def to_numpy(tensor_object):
    """Converts a backend's tensor object back to a NumPy array (on CPU)."""
    if _BACKEND == "pytorch" and isinstance(tensor_object, torch.Tensor):
        try:
            return tensor_object.detach().cpu().numpy() # detach() gradient takibini keser
        except Exception as e:
            logger.error(f"Error converting PyTorch tensor to NumPy array: {e}", exc_info=True)
            return None
    # elif _BACKEND == "tensorflow" and tf.is_tensor(tensor_object):
    #     # return tensor_object.numpy()
    #     pass
    if isinstance(tensor_object, np.ndarray):
        return tensor_object
    logger.warning(f"Input to to_numpy (type: {type(tensor_object)}) is not a recognized tensor or NumPy array. Returning as is.")
    return tensor_object