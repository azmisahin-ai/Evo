# src/senses/vision.py
import cv2
import time
import numpy as np
import logging
import os
import platform # Platformu kontrol etmek için

from src.core.config_utils import get_config_value

logger = logging.getLogger(__name__)

class VisionSensor:
    def __init__(self, full_config): # Artık tam config alıyor
        self.config = full_config.get('vision', {}) 
        logger.info("VisionSensor initializing...")

        self.source_type = get_config_value(self.config, 'source_type', default="usb_camera", expected_type=str).lower()
        self.is_dummy = get_config_value(self.config, 'is_dummy', default=False, expected_type=bool)
        
        self.cap = None
        self.is_stream_active = False # Gerçek bir akışın aktif olup olmadığını belirtir
        self.frame_width = 0
        self.frame_height = 0
        self.source_description = "Simulated"

        self.dummy_width = get_config_value(self.config, 'dummy_width', default=640, expected_type=int)
        self.dummy_height = get_config_value(self.config, 'dummy_height', default=480, expected_type=int)

        if self.is_dummy:
            logger.info(f"VisionSensor: Dummy mode enabled. Source type '{self.source_type}' will be simulated.")
            self.frame_width = self.dummy_width
            self.frame_height = self.dummy_height
            # is_stream_active False kalır
            return

        logger.info(f"VisionSensor: Initializing for source type: '{self.source_type}'")
        init_success = False
        try:
            if self.source_type == "usb_camera":
                camera_index = get_config_value(self.config, 'camera_index', default=0, expected_type=int)
                capture_api = cv2.CAP_ANY # Varsayılan
                api_desc = ""
                if platform.system() == "Windows":
                    capture_api = cv2.CAP_DSHOW
                    api_desc = " using CAP_DSHOW"
                self.cap = cv2.VideoCapture(camera_index, capture_api)
                self.source_description = f"USB Camera (Index: {camera_index}{api_desc})"
            elif self.source_type == "ip_camera":
                ip_camera_url = get_config_value(self.config, 'ip_camera_url', expected_type=str)
                if not ip_camera_url: raise ValueError("ip_camera_url not defined for ip_camera source.")
                self.cap = cv2.VideoCapture(ip_camera_url)
                self.source_description = f"IP Camera ({ip_camera_url})"
            elif self.source_type == "video_file":
                video_file_path = get_config_value(self.config, 'video_file_path', expected_type=str)
                if not video_file_path or not os.path.exists(video_file_path):
                    raise FileNotFoundError(f"Video file not found: {video_file_path}")
                self.cap = cv2.VideoCapture(video_file_path)
                self.loop_video = get_config_value(self.config, 'loop_video', default=True, expected_type=bool)
                self.source_description = f"Video File ({video_file_path})"
            else:
                raise ValueError(f"Unsupported vision source_type: {self.source_type}")

            if self.cap and self.cap.isOpened():
                self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if self.frame_width == 0 or self.frame_height == 0 :
                    ret_init, frame_init = self.cap.read()
                    if ret_init and frame_init is not None:
                        self.frame_height, self.frame_width = frame_init.shape[:2]
                        # Kareyi geri sarmak video dosyaları için önemli olabilir, kameralar için sorun değil
                        if self.source_type == "video_file":
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
                    else:
                        logger.warning(f"Could not determine frame dimensions for {self.source_description}.")
                
                if self.frame_width > 0 and self.frame_height > 0:
                    logger.info(f"VisionSensor: Source '{self.source_description}' opened. Dimensions: {self.frame_width}x{self.frame_height}")
                    init_success = True
                else: # Boyut hala sıfırsa
                    logger.warning(f"Failed to get valid frame dimensions for {self.source_description}.")

            if not init_success: # Eğer cap açılamadıysa veya boyut alınamadıysa
                logger.warning(f"VisionSensor: Could not open or get dimensions from source: {self.source_description}.")
                if self.cap: self.cap.release()
                self.cap = None
        
        except Exception as e:
            logger.error(f"VisionSensor initialization for '{self.source_type}' failed: {e}", exc_info=True)
            if self.cap: 
                try: self.cap.release()
                except: pass
            self.cap = None
        
        self.is_stream_active = init_success
        if not self.is_stream_active and not self.is_dummy:
            logger.warning(f"VisionSensor: Falling back to simulated input as stream is not active.")
            self.frame_width = self.dummy_width
            self.frame_height = self.dummy_height
        
        logger.info(f"VisionSensor final state - Active Stream: {self.is_stream_active}, Simulating: {self.is_dummy or not self.is_stream_active}")

    @property # is_camera_available'ı dinamik bir özellik yapalım
    def is_camera_available(self): # module_loader bunu kullanıyor
        return self.is_stream_active and not self.is_dummy

    def capture_frame(self):
        if self.is_dummy or not self.is_stream_active:
            return np.full((self.dummy_height, self.dummy_width, 3), 128, dtype=np.uint8)

        if self.source_type in ["usb_camera", "ip_camera", "video_file"]:
            if not self.cap or not self.cap.isOpened(): # Ekstra güvenlik kontrolü
                logger.error(f"VisionSensor: cap is not open/None in capture_frame. Source: {self.source_description}. Switching to simulated.")
                self.is_stream_active = False
                return self.capture_frame() # Rekürsif çağrı ile dummy frame al

            ret, frame = self.cap.read()
            if not ret:
                if self.source_type == "video_file" and self.loop_video:
                    logger.info(f"Video '{self.source_description}' ended. Looping.")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.warning(f"Failed to read after looping video '{self.source_description}'. Switching to simulated.")
                        self.is_stream_active = False
                        return self.capture_frame()
                else:
                    logger.warning(f"Failed to read frame from {self.source_description}. Switching to simulated.")
                    self.is_stream_active = False
                    return self.capture_frame()
            return frame
        else: # Desteklenmeyen veya başlatılamayan kaynak tipi için
            logger.error(f"VisionSensor: capture_frame called for unhandled source_type '{self.source_type}'.")
            self.is_stream_active = False
            return self.capture_frame()

    def stop_stream(self):
        logger.info(f"VisionSensor: Stopping stream for source '{self.source_description}'...")
        if self.cap is not None:
            try:
                self.cap.release()
                logger.info(f"VisionSensor: VideoCapture released for {self.source_description}.")
            except Exception as e:
                 logger.error(f"VisionSensor: Error releasing VideoCapture: {e}", exc_info=False)
        self.cap = None
        self.is_stream_active = False

    def cleanup(self):
        logger.info("VisionSensor cleaning up...")
        self.stop_stream()
        logger.info("VisionSensor cleaned up.")