# src/representation/models.py
import logging
import torch
import torch.nn as nn
import numpy as np # Sadece NumPy backend modunda geçici çözüm için

from src.core.config_utils import get_config_value
from src.core.compute_utils import get_device, get_backend 

logger = logging.getLogger(__name__)

class RepresentationLearner(nn.Module):
    # >>>>>>>>>> DEĞİŞİKLİK BURADA <<<<<<<<<<
    def __init__(self, full_config): # Sadece full_config al
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        super().__init__() 
        
        # Kendi config bölümünü full_config'ten al
        self.learner_config = full_config.get('representation', {})
        if not self.learner_config: # Eğer 'representation' bölümü yoksa veya boşsa
            logger.warning("RepresentationLearner: 'representation' section not found in config or is empty. Using defaults.")
            self.learner_config = {} # Hata vermemesi için boş dict ata

        logger.info("RepresentationLearner (PyTorch Module) initializing...")

        self.input_dim = get_config_value(self.learner_config, 'input_dim', expected_type=int, logger_instance=logger)
        self.representation_dim = get_config_value(self.learner_config, 'representation_dim', default=128, expected_type=int, logger_instance=logger)
        
        self._expected_input_order = self.learner_config.get('_expected_input_order_', [])
        logger.info(f"RepresentationLearner: Expected input order: {self._expected_input_order}")

        if self.input_dim is None or self.input_dim <= 0:
            logger.critical(f"RepresentationLearner: Invalid or missing input_dim ({self.input_dim}).")
            # ValueError fırlatmak, _initialize_single_module'ün None döndürmesini sağlar
            raise ValueError(f"RepresentationLearner: Invalid input_dim: {self.input_dim}") 
        
        self.current_backend = get_backend()
        self.device = get_device() 
        logger.info(f"RepresentationLearner will use PyTorch models on device: {self.device}")

        if self.current_backend == "numpy":
            logger.warning("RepresentationLearner is PyTorch-based, but a NumPy compute backend is active. "
                           "Data conversion will occur, which is inefficient.")

        try:
            self.encoder = nn.Linear(self.input_dim, self.representation_dim)
            self.decoder = nn.Linear(self.representation_dim, self.input_dim)
            
            self.to(self.device)
            
            self.is_initialized = True
            logger.info(f"RepresentationLearner (PyTorch) initialized. Input: {self.input_dim}, Representation: {self.representation_dim}. Device: {self.device}")

        except Exception as e:
            logger.critical(f"RepresentationLearner (PyTorch) initialization failed: {e}", exc_info=True)
            self.is_initialized = False
            raise e
        

    def forward(self, combined_input_tensor):
        if not self.is_initialized:
            logger.error("RepresentationLearner.forward: Module not initialized.")
            return None

        if not isinstance(combined_input_tensor, torch.Tensor):
            logger.error(f"RepresentationLearner.forward: Input must be a PyTorch Tensor. Got {type(combined_input_tensor)}")
            return None

        if combined_input_tensor.device != self.device:
            combined_input_tensor = combined_input_tensor.to(self.device)
        
        expected_shape_part = combined_input_tensor.shape[-1]
        if expected_shape_part != self.input_dim:
            logger.error(f"RepresentationLearner.forward: Input tensor last dimension ({expected_shape_part}) "
                           f"does not match expected input_dim ({self.input_dim}). Full shape: {combined_input_tensor.shape}")
            return None
        
        try:
            representation = self.encoder(combined_input_tensor.float()) 
            logger.debug(f"RepresentationLearner.forward: Representation extracted. Shape: {representation.shape}")
            return representation

        except Exception as e:
            logger.error(f"RepresentationLearner.forward: Error during encoder pass: {e}", exc_info=True)
            return None


    def learn(self, combined_input): 
        if not self.is_initialized: return None

        if self.current_backend == "pytorch":
            if not isinstance(combined_input, torch.Tensor):
                logger.error(f"RepresentationLearner.learn (PyTorch backend): Expected torch.Tensor, got {type(combined_input)}")
                return None
            with torch.no_grad(): 
                return self.forward(combined_input)
        
        elif self.current_backend == "numpy":
            if not isinstance(combined_input, np.ndarray):
                logger.error(f"RepresentationLearner.learn (NumPy backend mode): Expected np.ndarray, got {type(combined_input)}")
                return None
            try:
                temp_tensor = torch.from_numpy(combined_input).float().to(self.device)
                with torch.no_grad():
                    representation_tensor = self.forward(temp_tensor)
                if representation_tensor is not None:
                    return representation_tensor.cpu().numpy()
                return None
            except Exception as e:
                logger.error(f"RepresentationLearner.learn (NumPy backend mode): Error in conversion/processing: {e}", exc_info=True)
                return None
        else:
            logger.error(f"RepresentationLearner.learn: Unsupported backend '{self.current_backend}'")
            return None
        

    def decode_representation(self, latent_tensor):
        if not self.is_initialized: return None
        if not isinstance(latent_tensor, torch.Tensor):
            logger.error(f"RepresentationLearner.decode: Input must be a PyTorch Tensor. Got {type(latent_tensor)}")
            return None
        if latent_tensor.device != self.device:
            latent_tensor = latent_tensor.to(self.device)
        
        if latent_tensor.shape[-1] != self.representation_dim:
            logger.error(f"RepresentationLearner.decode: Latent tensor dim ({latent_tensor.shape[-1]}) mismatches representation_dim ({self.representation_dim}).")
            return None
            
        try:
            with torch.no_grad():
                reconstruction = self.decoder(latent_tensor.float())
            logger.debug(f"RepresentationLearner.decode: Reconstruction complete. Shape: {reconstruction.shape}")
            return reconstruction
        except Exception as e:
            logger.error(f"RepresentationLearner.decode: Error during decoder pass: {e}", exc_info=True)
            return None



    def cleanup(self):
        logger.info(f"RepresentationLearner (PyTorch Module) cleaning up. Model was on {self.device}.")
        pass