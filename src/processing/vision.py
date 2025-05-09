# src/processing/vision.py
import cv2
import numpy as np
import logging
import torch 
import os

from src.core.compute_utils import get_device, get_backend, to_tensor 
from src.core.config_utils import get_config_value
from src.core.utils import check_input_not_none

logger = logging.getLogger(__name__)

class VisionProcessor:
    def __init__(self, full_config):
        self.full_config = full_config
        self.vp_config = full_config.get('processors', {}).get('vision', {})
        self.sensor_vision_config = full_config.get('vision', {})
        
        logger.info("VisionProcessor initializing...")

        self.output_width = get_config_value(self.vp_config, 'output_width', default=64, expected_type=int)
        self.output_height = get_config_value(self.vp_config, 'output_height', default=64, expected_type=int)
        
        self.process_color = get_config_value(self.sensor_vision_config, 'process_color', default=True, expected_type=bool)
        # output_channels_main_image config'den okunur ve process_color ile tutarlılığı kontrol edilir.
        default_main_channels = 3 if self.process_color else 1
        self.actual_output_main_channels = get_config_value(self.vp_config, 'output_channels_main_image', default=default_main_channels, expected_type=int)
        
        if self.process_color and self.actual_output_main_channels != 3:
            logger.warning(f"VisionProcessor: process_color is True, but output_channels_main_image is {self.actual_output_main_channels}. Forcing to 3 (RGB).")
            self.actual_output_main_channels = 3
        elif not self.process_color and self.actual_output_main_channels != 1:
            logger.warning(f"VisionProcessor: process_color is False, but output_channels_main_image is {self.actual_output_main_channels}. Forcing to 1 (Grayscale).")
            self.actual_output_main_channels = 1

        self.enable_edge_detection = get_config_value(self.vp_config, 'enable_edge_detection', default=True, expected_type=bool)
        if self.enable_edge_detection:
            self.canny_low_threshold = get_config_value(self.vp_config, 'canny_low_threshold', default=50, expected_type=int)
            self.canny_high_threshold = get_config_value(self.vp_config, 'canny_high_threshold', default=150, expected_type=int)
            blur_kernel_size_cfg = get_config_value(self.vp_config, 'gaussian_blur_kernel_size', default="5,5", expected_type=str)
            try:
                k_w, k_h = map(int, blur_kernel_size_cfg.split(','))
                if k_w > 0 and k_w % 2 == 1 and k_h > 0 and k_h % 2 == 1:
                    self.gaussian_blur_kernel = (k_w, k_h)
                else:
                    logger.warning(f"Invalid gaussian_blur_kernel_size '{blur_kernel_size_cfg}'. Using (5,5).")
                    self.gaussian_blur_kernel = (5, 5)
            except ValueError:
                logger.warning(f"Format error for gaussian_blur_kernel_size '{blur_kernel_size_cfg}'. Using (5,5).")
                self.gaussian_blur_kernel = (5, 5)
            logger.info(f"Edge detection enabled. Blur Kernel: {self.gaussian_blur_kernel}, Canny Thresh: [{self.canny_low_threshold},{self.canny_high_threshold}]")
        else:
            logger.info("Edge detection is disabled.")

        self.use_gpu_tensor = get_config_value(self.vp_config, 'use_gpu_if_available', default=True, expected_type=bool)
        self.current_backend = get_backend()
        self.device_for_output_tensor = get_device() if (self.current_backend == "pytorch" and self.use_gpu_tensor) else torch.device("cpu") if self.current_backend == "pytorch" else "cpu"
        
        logger.info(f"VisionProcessor initialized. Main Image: {self.output_width}x{self.output_height}x{self.actual_output_main_channels}, Color: {self.process_color}. Output Backend: {self.current_backend}, Device: {self.device_for_output_tensor}")

    def get_output_shape_info(self):
        shapes = {}
        shapes['main_image'] = (self.actual_output_main_channels, self.output_height, self.output_width)
        if self.enable_edge_detection:
            shapes['edges'] = (1, self.output_height, self.output_width) 
        if not shapes:
            logger.error("VisionProcessor.get_output_shape_info: No features are configured to be outputted!")
            shapes['dummy_feature'] = (1,) # Hata durumunda en az bir şey
        return shapes

    def process(self, visual_input_bgr):
        if not check_input_not_none(visual_input_bgr, "visual_input_bgr", logger) or \
           not (isinstance(visual_input_bgr, np.ndarray) and visual_input_bgr.dtype == np.uint8):
            logger.error(f"VisionProcessor: Invalid input. Must be non-None uint8 NumPy array. Got {type(visual_input_bgr)}")
            return {}
        
        processed_features_np = {}
        try:
            # 1. Ana Görüntü (Renkli veya Gri)
            main_image_for_resize_np = None
            if self.process_color: 
                if visual_input_bgr.ndim == 3 and visual_input_bgr.shape[2] == 3:
                    main_image_for_resize_np = cv2.cvtColor(visual_input_bgr, cv2.COLOR_BGR2RGB)
                else: 
                    gray = visual_input_bgr
                    if visual_input_bgr.ndim == 3 and visual_input_bgr.shape[2] == 1: gray = visual_input_bgr[:,:,0]
                    elif visual_input_bgr.ndim != 2: gray = cv2.cvtColor(visual_input_bgr, cv2.COLOR_BGR2GRAY)
                    main_image_for_resize_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else: 
                if visual_input_bgr.ndim == 3 and visual_input_bgr.shape[2] == 3:
                    main_image_for_resize_np = cv2.cvtColor(visual_input_bgr, cv2.COLOR_BGR2GRAY)
                elif visual_input_bgr.ndim == 2: main_image_for_resize_np = visual_input_bgr.copy()
                elif visual_input_bgr.ndim == 3 and visual_input_bgr.shape[2] == 1: main_image_for_resize_np = visual_input_bgr[:,:,0]
                else: main_image_for_resize_np = cv2.cvtColor(visual_input_bgr, cv2.COLOR_BGR2GRAY)

            resized_main_np = cv2.resize(main_image_for_resize_np, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)
            if resized_main_np.ndim == 2: 
                resized_main_np = np.expand_dims(resized_main_np, axis=-1)
            
            if resized_main_np.shape[2] == self.actual_output_main_channels:
                processed_features_np['main_image'] = resized_main_np
            else:
                 logger.error(f"VisionProcessor: Channel mismatch for main_image. Exp {self.actual_output_main_channels}, got {resized_main_np.shape[2]}. Skipping.")

            # 2. Kenar Tespiti (eğer aktifse)
            if self.enable_edge_detection:
                gray_for_edges_np = None
                if visual_input_bgr.ndim == 3 and visual_input_bgr.shape[2] == 3:
                    gray_for_edges_np = cv2.cvtColor(visual_input_bgr, cv2.COLOR_BGR2GRAY)
                elif visual_input_bgr.ndim == 2: gray_for_edges_np = visual_input_bgr.copy()
                elif visual_input_bgr.ndim == 3 and visual_input_bgr.shape[2] == 1: gray_for_edges_np = visual_input_bgr[:,:,0].copy()
                else: gray_for_edges_np = cv2.cvtColor(visual_input_bgr, cv2.COLOR_BGR2GRAY)

                if gray_for_edges_np is not None:
                    resized_gray_for_edges_np = cv2.resize(gray_for_edges_np, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)
                    blurred_gray_for_edges = cv2.GaussianBlur(resized_gray_for_edges_np, self.gaussian_blur_kernel, 0)
                    edges_np = cv2.Canny(blurred_gray_for_edges, self.canny_low_threshold, self.canny_high_threshold)
                    processed_features_np['edges'] = np.expand_dims(edges_np, axis=-1) # (H, W) -> (H, W, 1)

        except cv2.error as e:
            logger.error(f"VisionProcessor: OpenCV error: {e}", exc_info=True)
            return {} # Hata durumunda boş dict
        except Exception as e:
            logger.error(f"VisionProcessor: Unexpected error during NumPy processing: {e}", exc_info=True)
            return {}

        processed_features_backend = {}
        for key, np_array_uint8 in processed_features_np.items(): # np_array'ler uint8 olmalı
            if np_array_uint8 is None: continue
            
            # Normalizasyon (0-1 arasına)
            np_array_float = np_array_uint8.astype(np.float32) / 255.0
            
            if self.current_backend == "pytorch":
                # (H, W, C) -> (C, H, W) PyTorch formatına çevir
                chw_array = np_array_float.transpose((2, 0, 1))
                tensor_chw = to_tensor(chw_array, target_device=self.device_for_output_tensor)
                if tensor_chw is not None:
                    processed_features_backend[key] = tensor_chw
            else: # numpy backend
                processed_features_backend[key] = np_array_float
        
        if not processed_features_backend: # Eğer hiçbir özellik işlenemediyse veya üretilemediyse
            logger.warning("VisionProcessor: No features were generated from visual input.")
            # module_loader'ın input_dim hesaplamasında sorun olmaması için,
            # get_output_shape_info'da tanımlanan her özellik için boş/sıfır bir tensör/array döndür
            # Bu, RepresentationLearner'a tutarlı sayıda girdi gitmesini sağlar.
            shape_info = self.get_output_shape_info()
            for key, shape_chw in shape_info.items():
                if key == 'dummy_feature' and len(shape_info) > 1: continue # dummy_feature'ı atla eğer başka özellik varsa
                
                # shape_chw (C,H,W) formatında. NumPy için (H,W,C) veya (H,W) olmalı.
                # PyTorch için (C,H,W) kalabilir.
                if self.current_backend == "pytorch":
                    processed_features_backend[key] = torch.zeros(shape_chw, dtype=torch.float32, device=self.device_for_output_tensor)
                else: # numpy
                    if len(shape_chw) == 3: # C,H,W -> H,W,C
                        shape_hwc = (shape_chw[1], shape_chw[2], shape_chw[0])
                        processed_features_backend[key] = np.zeros(shape_hwc, dtype=np.float32)
                    elif len(shape_chw) == 2: # H,W (tek kanallıydı, C=1 olarak geldi)
                         processed_features_backend[key] = np.zeros(shape_chw, dtype=np.float32) # H,W
                    else: # Tek boyutlu ise (örn: (D,))
                         processed_features_backend[key] = np.zeros(shape_chw, dtype=np.float32)


        return processed_features_backend

    def cleanup(self):
        logger.info("VisionProcessor cleaning up.")