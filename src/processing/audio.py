# src/processing/audio.py
import numpy as np
import logging
import torch
from src.core.compute_utils import get_device, get_backend, to_tensor, to_numpy
from src.core.config_utils import get_config_value
from src.core.utils import check_input_not_none

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, config):
        self.full_config = config
        self.ap_config = config.get('processors', {}).get('audio', {})
        logger.info("AudioProcessor initializing...")

        # AudioSensor'dan gelen rate'i kullanmak daha iyi olabilir, eğer AudioSensor'da rate ayarı varsa
        # Şimdilik processor config'den alalım
        self.audio_rate = get_config_value(self.ap_config, 'audio_rate', default=44100, expected_type=int, logger_instance=logger)
        
        self._implemented_feature_count = 2 # Energy, Spectral Centroid
        self.output_dim_from_config = get_config_value(self.ap_config, 'output_dim', default=self._implemented_feature_count, expected_type=int, logger_instance=logger)

        if self.output_dim_from_config != self._implemented_feature_count:
             logger.warning(f"AudioProcessor: Config 'processors.audio.output_dim' ({self.output_dim_from_config}) "
                            f"does not match implemented feature count ({self._implemented_feature_count}). "
                            f"Using implemented count. Please update config if necessary.")
        
        self.actual_output_dim = self._implemented_feature_count
        
        self.current_backend = get_backend()
        # Ses işlemleri genellikle CPU'da kalır. PyTorch backend'inde bile CPU tensörü olarak tutulabilir.
        if self.current_backend == "pytorch":
            self.device_for_output_tensor = torch.device("cpu") # Ses için CPU'yu zorla
            logger.info(f"AudioProcessor (PyTorch backend): Output tensors will be on CPU.")
        else:
            self.device_for_output_tensor = "cpu" # NumPy için sembolik
            logger.info(f"AudioProcessor (NumPy backend): Output will be NumPy arrays on CPU.")

        logger.info(f"AudioProcessor initialized. Sample Rate: {self.audio_rate} Hz, Effective Output Dimension: {self.actual_output_dim}.")

    def get_output_shape_info(self):
        shapes = {}
        shapes['audio_features'] = (self.actual_output_dim,)
        return shapes

    def process(self, audio_input_int16):
        if not check_input_not_none(audio_input_int16, "audio_input_int16 for AudioProcessor", logger):
            return None 

        if not isinstance(audio_input_int16, np.ndarray) or audio_input_int16.dtype != np.int16:
            logger.error(f"AudioProcessor: Input must be a NumPy array with dtype int16. Got {type(audio_input_int16)}")
            return None

        if audio_input_int16.size == 0:
            logger.debug("AudioProcessor: Received empty audio chunk. Returning None.")
            return None
            
        logger.debug(f"AudioProcessor: Audio data received. Shape: {audio_input_int16.shape}")

        processed_features_np = None
        try:
            # Normalizasyon: int16'yı [-1, 1] aralığında float32'ye çevir
            audio_float = audio_input_int16.astype(np.float32) / 32768.0 
            
            energy = np.sqrt(np.mean(audio_float**2)) if audio_float.size > 0 else 0.0
            
            spectral_centroid = 0.0
            if audio_float.size > 1 : # FFT için en az 2 örnek (pratikte daha fazla)
                window = np.hanning(len(audio_float))
                audio_windowed = audio_float * window
                # FFT için chunk size'ın 2'nin kuvveti olması gerekmez ama bazı implementasyonlarda daha hızlı olabilir.
                fft_result = np.fft.rfft(audio_windowed) # rfft reel girdiler için daha verimli
                magnitude_spectrum = np.abs(fft_result)
                # rfft frekansları np.fft.rfftfreq ile alınır
                frequencies = np.fft.rfftfreq(len(audio_windowed), d=1.0/self.audio_rate)
                
                sum_magnitudes = np.sum(magnitude_spectrum)
                if sum_magnitudes > 1e-9: 
                    spectral_centroid = np.sum(frequencies * magnitude_spectrum) / sum_magnitudes
            
            feature_list = [energy, spectral_centroid] 
            processed_features_np = np.array(feature_list[:self._implemented_feature_count], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"AudioProcessor: Error during NumPy processing: {e}", exc_info=True)
            return None

        if processed_features_np is None: return None

        if self.current_backend == "pytorch":
            # to_tensor zaten float'a çeviriyor ve hedef cihaza yolluyor
            output_tensor = to_tensor(processed_features_np, target_device=self.device_for_output_tensor)
            if output_tensor is not None:
                logger.debug(f"AudioProcessor: Processed (PyTorch Tensor on {output_tensor.device}). Shape: {output_tensor.shape}, Values: {output_tensor.numpy()}")
            return output_tensor
        else: # numpy backend
            logger.debug(f"AudioProcessor: Processed (NumPy Array). Shape: {processed_features_np.shape}, Values: {processed_features_np}")
            return processed_features_np

    def cleanup(self):
        logger.info("AudioProcessor cleaning up.")