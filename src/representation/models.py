# src/representation/models.py
# (Bir önceki tam içerikli mesajdaki RepresentationLearner sınıfı buraya gelecek)
# Önemli olan __init__ içinde self.hidden_dim_ae'yi okuması ve
# encoder/decoder katmanlarını ona göre oluşturması.
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 

from src.core.config_utils import get_config_value
from src.core.compute_utils import get_device, get_backend

logger = logging.getLogger(__name__)

class RepresentationLearner(nn.Module):
    def __init__(self, full_config):
        super().__init__() 
        
        self.learner_config = full_config.get('representation', {})
        if not self.learner_config:
            logger.warning("RepresentationLearner: 'representation' section not found/empty. Using defaults.")
            self.learner_config = {}

        logger.info("RepresentationLearner (Autoencoder - PyTorch Module) initializing...")

        self.input_dim = get_config_value(self.learner_config, 'input_dim', expected_type=int, logger_instance=logger)
        self.representation_dim = get_config_value(self.learner_config, 'representation_dim', default=128, expected_type=int, logger_instance=logger)
        self.hidden_dim_ae = get_config_value(self.learner_config, 'hidden_dim_ae', default=None, expected_type=(int, type(None)), logger_instance=logger)
        self.learning_rate = float(get_config_value(self.learner_config, 'learning_rate', default=1e-4, expected_type=(float, int, np.floating), logger_instance=logger))

        self._expected_input_order = self.learner_config.get('_expected_input_order_', [])
        if self._expected_input_order:
            logger.info(f"RepresentationLearner: Expected input feature order: {self._expected_input_order}")

        if self.input_dim is None or self.input_dim <= 0:
            raise ValueError(f"RepresentationLearner: Invalid input_dim: {self.input_dim}")
        if self.representation_dim <= 0:
            raise ValueError(f"RepresentationLearner: Invalid representation_dim: {self.representation_dim}")
        if self.hidden_dim_ae is not None:
            if not isinstance(self.hidden_dim_ae, int) or self.hidden_dim_ae <= 0:
                logger.warning(f"RepresentationLearner: Invalid 'hidden_dim_ae' ({self.hidden_dim_ae}). Disabling hidden layer.")
                self.hidden_dim_ae = None
            elif self.hidden_dim_ae <= self.representation_dim :
                 logger.warning(f"RepresentationLearner: 'hidden_dim_ae' ({self.hidden_dim_ae}) not > 'representation_dim' ({self.representation_dim}).")
            elif self.hidden_dim_ae >= self.input_dim:
                logger.warning(f"RepresentationLearner: 'hidden_dim_ae' ({self.hidden_dim_ae}) not < 'input_dim' ({self.input_dim}).")
        
        self.current_backend = get_backend()
        self.device = get_device() 
        logger.info(f"RepresentationLearner using PyTorch models on device: {self.device}, LR: {self.learning_rate}, HiddenDimAE: {self.hidden_dim_ae if self.hidden_dim_ae else 'N/A'}")

        try:
            encoder_layers = []
            decoder_layers = []

            if self.hidden_dim_ae:
                encoder_layers.append(nn.Linear(self.input_dim, self.hidden_dim_ae))
                encoder_layers.append(nn.ReLU(True)) 
                encoder_layers.append(nn.Linear(self.hidden_dim_ae, self.representation_dim))
                
                decoder_layers.append(nn.Linear(self.representation_dim, self.hidden_dim_ae))
                decoder_layers.append(nn.ReLU(True))
                decoder_layers.append(nn.Linear(self.hidden_dim_ae, self.input_dim))
            else: 
                encoder_layers.append(nn.Linear(self.input_dim, self.representation_dim))
                decoder_layers.append(nn.Linear(self.representation_dim, self.input_dim))
            
            decoder_layers.append(nn.Sigmoid()) 

            self.encoder = nn.Sequential(*encoder_layers)
            self.decoder = nn.Sequential(*decoder_layers)
            
            self.to(self.device)
            
            self.loss_function = nn.MSELoss()
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
            
            self.is_initialized = True
            self.total_loss_accumulated = 0.0
            self.train_step_count = 0
            logger.info(f"RepresentationLearner (Autoencoder) initialized. Input: {self.input_dim}, Latent: {self.representation_dim}, Device: {self.device}")

        except Exception as e:
            logger.critical(f"RepresentationLearner (Autoencoder) initialization failed: {e}", exc_info=True)
            self.is_initialized = False
            raise e

    def _train_step(self, input_tensor_batch):
        self.train() 
        if input_tensor_batch.device != self.device:
            input_tensor_batch = input_tensor_batch.to(self.device)
        input_tensor_batch = input_tensor_batch.float()

        self.optimizer.zero_grad()
        latent_representation_batch = self.encoder(input_tensor_batch)
        reconstruction_batch = self.decoder(latent_representation_batch)
        loss = self.loss_function(reconstruction_batch, input_tensor_batch)
        loss.backward()
        self.optimizer.step()
        
        current_loss_item = loss.item()
        self.total_loss_accumulated += current_loss_item
        self.train_step_count += 1
        
        if self.train_step_count > 0 and self.train_step_count % 50 == 0: # Her 50 adımda bir ortalama kaybı logla
            avg_loss_so_far = self.total_loss_accumulated / self.train_step_count
            logger.info(f"RL Train Step {self.train_step_count}: Batch Loss: {current_loss_item:.6f}, Avg Total Loss: {avg_loss_so_far:.6f}")
        
        return latent_representation_batch.detach()

    def learn(self, combined_input): 
        if not self.is_initialized: 
            logger.error("RL.learn: Module not initialized.")
            return None

        if not isinstance(combined_input, torch.Tensor):
            if isinstance(combined_input, np.ndarray) and get_backend() == "pytorch":
                combined_input = torch.from_numpy(combined_input).float()
            else:
                logger.error(f"RL.learn: Expected PyTorch Tensor or compatible NumPy array, got {type(combined_input)}.")
                return None
        
        if combined_input.ndim == 1: 
            if combined_input.shape[0] != self.input_dim:
                logger.error(f"RL.learn: Single input dim ({combined_input.shape[0]}) !~ expected ({self.input_dim}).")
                return None
            input_batch = combined_input.unsqueeze(0)
        elif combined_input.ndim == 2: 
            if combined_input.shape[1] != self.input_dim:
                logger.error(f"RL.learn: Batched input dim ({combined_input.shape[1]}) !~ expected ({self.input_dim}).")
                return None
            input_batch = combined_input
        else:
            logger.error(f"RL.learn: Input tensor unexpected ndim: {combined_input.ndim}.")
            return None
            
        try:
            latent_tensor_batch = self._train_step(input_batch)
            if combined_input.ndim == 1 and latent_tensor_batch.ndim == 2 and latent_tensor_batch.shape[0] == 1:
                return latent_tensor_batch.squeeze(0)
            return latent_tensor_batch
        except Exception as e:
            logger.error(f"RL.learn: Error during training step: {e}", exc_info=True)
            return None

    def get_representation(self, combined_input): # Sadece inferans için
        # ... (Bir önceki mesajdaki get_representation kodu buraya gelecek, değişiklik yok) ...
        if not self.is_initialized:
            logger.error("RL.get_representation: Module not initialized.")
            return None
        
        self.eval() 
        with torch.no_grad(): 
            if not isinstance(combined_input, torch.Tensor):
                if isinstance(combined_input, np.ndarray) and get_backend() == "pytorch":
                    combined_input = torch.from_numpy(combined_input).float()
                else:
                    logger.error(f"RL.get_representation: Expected PyTorch Tensor, got {type(combined_input)}.")
                    return None

            if combined_input.device != self.device:
                combined_input = combined_input.to(self.device)
            
            input_tensor = combined_input.float()

            if input_tensor.ndim == 1:
                if input_tensor.shape[0] != self.input_dim:
                    logger.error(f"RL.get_representation: Single input tensor dim mismatch.")
                    return None
                input_batch = input_tensor.unsqueeze(0)
            elif input_tensor.ndim == 2:
                if input_tensor.shape[1] != self.input_dim:
                    logger.error(f"RL.get_representation: Batched input tensor dim mismatch.")
                    return None
                input_batch = input_tensor
            else:
                logger.error(f"RL.get_representation: Input tensor unexpected ndim: {input_tensor.ndim}.")
                return None
            
            try:
                latent_batch = self.encoder(input_batch)
                if input_tensor.ndim == 1: 
                    return latent_batch.squeeze(0)
                return latent_batch
            except Exception as e:
                logger.error(f"RL.get_representation: Error during encoder forward pass: {e}", exc_info=True)
                return None

    def decode_representation(self, latent_tensor_or_array):
        # ... (Bir önceki mesajdaki decode_representation kodu buraya gelecek, değişiklik yok) ...
        if not self.is_initialized: 
            logger.error("RL.decode_representation: Module not initialized.")
            return None
        
        self.eval() 
        with torch.no_grad():
            input_data = latent_tensor_or_array
            if isinstance(latent_tensor_or_array, np.ndarray):
                if latent_tensor_or_array.ndim == 1 and latent_tensor_or_array.shape[0] == self.representation_dim:
                    input_data = torch.from_numpy(latent_tensor_or_array).float().unsqueeze(0)
                elif latent_tensor_or_array.ndim == 2 and latent_tensor_or_array.shape[1] == self.representation_dim:
                    input_data = torch.from_numpy(latent_tensor_or_array).float()
                else:
                    logger.error(f"RL.decode_representation: NumPy input dim/shape mismatch.")
                    return None
            elif not isinstance(latent_tensor_or_array, torch.Tensor):
                logger.error(f"RL.decode_representation: Input must be PyTorch Tensor or NumPy array. Got {type(latent_tensor_or_array)}")
                return None

            if input_data.device != self.device:
                input_data = input_data.to(self.device)
            
            if input_data.ndim == 1:
                input_data = input_data.unsqueeze(0)
            
            if input_data.shape[-1] != self.representation_dim:
                logger.error(f"RL.decode_representation: Latent data last dim mismatch.")
                return None
            try:
                reconstruction_batch = self.decoder(input_data.float())
                if isinstance(latent_tensor_or_array, np.ndarray) and latent_tensor_or_array.ndim == 1:
                    return reconstruction_batch.squeeze(0)
                if isinstance(latent_tensor_or_array, torch.Tensor) and latent_tensor_or_array.ndim == 1:
                     return reconstruction_batch.squeeze(0)
                return reconstruction_batch
            except Exception as e:
                logger.error(f"RL.decode_representation: Error during decoder pass: {e}", exc_info=True)
                return None

    def cleanup(self):
        avg_loss = (self.total_loss_accumulated / self.train_step_count) if self.train_step_count > 0 else float('nan')
        logger.info(f"RepresentationLearner (Autoencoder) cleaning up. Avg training loss: {avg_loss:.6f} over {self.train_step_count} total steps.")