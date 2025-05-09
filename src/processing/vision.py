# src/processing/vision.py
import cv2
import numpy as np
import logging
import torch 
from src.core.compute_utils import get_device, get_backend, to_tensor 
from src.core.config_utils import get_config_value
from src.core.utils import check_input_not_none

logger = logging.getLogger(__name__)

class VisionProcessor:
    def __init__(self, config):
        self.full_config = config # Tam config'i sakla (diğer bölümlere erişim için)
        self.vp_config = config.get('processors', {}).get('vision', {})
        self.sensor_vision_config = config.get('vision', {})
        
        logger.info("VisionProcessor initializing...")

        self.output_width = get_config_value(self.vp_config, 'output_width', default=64, expected_type=int, logger_instance=logger)
        self.output_height = get_config_value(self.vp_config, 'output_height', default=64, expected_type=int, logger_instance=logger)
        
        self.process_color = get_config_value(self.sensor_vision_config, 'process_color', default=True, expected_type=bool, logger_instance=logger)
        self.output_main_channels_config = get_config_value(self.vp_config, 'output_channels', default=3, expected_type=int, logger_instance=logger)
        
        # output_main_channels'ı process_color'a göre ayarla/doğrula
        if self.process_color:
            self.actual_output_main_channels = 3
            if self.output_main_channels_config != 3:
                logger.warning(f"VisionProcessor: 'vision.process_color' is True, but 'processors.vision.output_channels' is {self.output_main_channels_config}. "
                               f"Output will be 3 channels (RGB). Consider updating config.")
        else: # Grayscale
            self.actual_output_main_channels = 1
            if self.output_main_channels_config != 1:
                logger.warning(f"VisionProcessor: 'vision.process_color' is False, but 'processors.vision.output_channels' is {self.output_main_channels_config}. "
                               f"Output will be 1 channel (Grayscale). Consider updating config.")

        self.use_gpu_tensor = get_config_value(self.vp_config, 'use_gpu_if_available', default=True, expected_type=bool, logger_instance=logger)
        
        self.canny_low_threshold = get_config_value(self.vp_config, 'canny_low_threshold', default=50, expected_type=int, logger_instance=logger)
        self.canny_high_threshold = get_config_value(self.vp_config, 'canny_high_threshold', default=150, expected_type=int, logger_instance=logger)

        self.device_for_output_tensor = None
        self.current_backend = get_backend()

        if self.current_backend == "pytorch":
            if self.use_gpu_tensor:
                self.device_for_output_tensor = get_device()
            else:
                self.device_for_output_tensor = torch.device("cpu")
            logger.info(f"VisionProcessor (PyTorch backend): Output tensors will be on device: {self.device_for_output_tensor}")
        else: # numpy backend
            self.device_for_output_tensor = "cpu" # NumPy için sembolik
            logger.info("VisionProcessor (NumPy backend): Output will be NumPy arrays on CPU.")


        if self.output_width <= 0 or self.output_height <= 0:
             logger.error(f"VisionProcessor: Invalid output dimensions. Using defaults (64x64).")
             self.output_width = 64
             self.output_height = 64
        
        logger.info(f"VisionProcessor initialized. Output: {self.output_width}x{self.output_height}x{self.actual_output_main_channels} (main), Edges: ...x1. Color processing: {self.process_color}")

    def get_output_shape_info(self):
        shapes = {}
        shapes['main_image'] = (self.actual_output_main_channels, self.output_height, self.output_width)
        shapes['edges'] = (1, self.output_height, self.output_width) # Kenarlar her zaman tek kanal (C,H,W)
        return shapes

    def process(self, visual_input_bgr):
        if not check_input_not_none(visual_input_bgr, "visual_input_bgr for VisionProcessor", logger):
            return {}

        if not isinstance(visual_input_bgr, np.ndarray) or visual_input_bgr.dtype != np.uint8:
            logger.error(f"VisionProcessor: Input must be a NumPy array with dtype uint8. Got {type(visual_input_bgr)}")
            return {}
        
        processed_features_np = {}
        
        try:
            # 1. Ana Görüntü (Renkli veya Gri)
            main_image_for_resize_np = None
            if self.process_color: # Renkli işleme (BGR -> RGB)
                if visual_input_bgr.ndim == 3 and visual_input_bgr.shape[2] == 3:
                    main_image_for_resize_np = cv2.cvtColor(visual_input_bgr, cv2.COLOR_BGR2RGB)
                else: # Beklenmedik format, griye çevirip 3 kanal yap
                    logger.warning(f"VisionProcessor: process_color=True but input is not BGR. Shape: {visual_input_bgr.shape}. Converting to 3-channel gray.")
                    gray = visual_input_bgr
                    if visual_input_bgr.ndim == 3 and visual_input_bgr.shape[2] == 1: gray = visual_input_bgr[:,:,0]
                    elif visual_input_bgr.ndim != 2: gray = cv2.cvtColor(visual_input_bgr, cv2.COLOR_BGR2GRAY) # Garantiye al
                    main_image_for_resize_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else: # Gri tonlamalı işleme
                if visual_input_bgr.ndim == 3 and visual_input_bgr.shape[2] == 3:
                    main_image_for_resize_np = cv2.cvtColor(visual_input_bgr, cv2.COLOR_BGR2GRAY)
                elif visual_input_bgr.ndim == 2:
                    main_image_for_resize_np = visual_input_bgr.copy()
                elif visual_input_bgr.ndim == 3 and visual_input_bgr.shape[2] == 1: # Zaten tek kanal gri
                     main_image_for_resize_np = visual_input_bgr[:,:,0]
                else: # Bilinmeyen format, griye zorla
                    logger.warning(f"VisionProcessor: process_color=False, input unusual. Shape: {visual_input_bgr.shape}. Forcing to grayscale.")
                    main_image_for_resize_np = cv2.cvtColor(visual_input_bgr, cv2.COLOR_BGR2GRAY)

            # Yeniden Boyutlandır (Ana Görüntü)
            resized_main_np = cv2.resize(main_image_for_resize_np, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)
            if resized_main_np.ndim == 2: # Eğer sonuç gri ise kanal boyutu ekle (H,W) -> (H,W,1)
                resized_main_np = np.expand_dims(resized_main_np, axis=-1)
            
            # Kanal sayısının self.actual_output_main_channels ile eşleştiğinden emin ol
            if resized_main_np.shape[2] != self.actual_output_main_channels:
                 logger.error(f"VisionProcessor: Internal channel mismatch for main_image. Expected {self.actual_output_main_channels}, got {resized_main_np.shape[2]}. This is a bug.")
                 # Hata durumunda bu özelliği atla
            else:
                processed_features_np['main_image'] = resized_main_np


            # 2. Kenar Tespiti (Her zaman orijinal BGR'nin gri tonlamalısı üzerinden)
            gray_for_edges_np = None
            if visual_input_bgr.ndim == 3 and visual_input_bgr.shape[2] == 3:
                gray_for_edges_np = cv2.cvtColor(visual_input_bgr, cv2.COLOR_BGR2GRAY)
            elif visual_input_bgr.ndim == 2: # Zaten gri ise
                gray_for_edges_np = visual_input_bgr.copy()
            elif visual_input_bgr.ndim == 3 and visual_input_bgr.shape[2] == 1: # Zaten tek kanal gri
                gray_for_edges_np = visual_input_bgr[:,:,0].copy()
            else: # Garantiye almak için
                logger.warning(f"VisionProcessor: Unusual input for edge detection. Shape: {visual_input_bgr.shape}. Forcing to grayscale.")
                gray_for_edges_np = cv2.cvtColor(visual_input_bgr, cv2.COLOR_BGR2GRAY)

            if gray_for_edges_np is not None:
                resized_gray_for_edges_np = cv2.resize(gray_for_edges_np, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)
                edges_np = cv2.Canny(resized_gray_for_edges_np, self.canny_low_threshold, self.canny_high_threshold)
                processed_features_np['edges'] = np.expand_dims(edges_np, axis=-1) # (H, W) -> (H, W, 1)

        except cv2.error as e:
            logger.error(f"VisionProcessor: OpenCV error: {e}", exc_info=True)
            return {}
        except Exception as e:
            logger.error(f"VisionProcessor: Unexpected error during NumPy processing: {e}", exc_info=True)
            return {}

        # NumPy array'lerini backend tensor'lerine/array'lerine çevir
        processed_features_backend = {}
        for key, np_array in processed_features_np.items():
            if np_array is None: continue

            # Normalizasyon (0-1 arasına, tensöre çevirmeden önce)
            # Kenarlar zaten 0 veya 255, ana görüntü 0-255 uint8
            # PyTorch float tensör bekler, bu yüzden /255.0 iyi bir pratiktir.
            np_array_float = np_array.astype(np.float32) / 255.0
            
            if self.current_backend == "pytorch":
                # (H, W, C) -> (C, H, W) PyTorch formatına çevir
                chw_array = np_array_float.transpose((2, 0, 1))
                # to_tensor zaten float'a çeviriyor ve hedef cihaza yolluyor
                tensor_chw = to_tensor(chw_array, target_device=self.device_for_output_tensor)
                if tensor_chw is not None:
                    processed_features_backend[key] = tensor_chw
            else: # numpy backend
                processed_features_backend[key] = np_array_float # Zaten NumPy array, normalize edilmiş float

        if not processed_features_backend: # Eğer hiçbir özellik işlenemediyse
            logger.warning("VisionProcessor: No features could be processed.")
            return {}
            
        logger.debug(f"VisionProcessor: Processing complete. Output keys: {list(processed_features_backend.keys())}")
        return processed_features_backend

    def cleanup(self):
        logger.info("VisionProcessor cleaning up.")