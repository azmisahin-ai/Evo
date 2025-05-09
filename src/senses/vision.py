# src/senses/vision.py
# (Verdiğin orijinal içerik buraya gelecek - değişiklik önermemiştik)
# ... (Orijinal VisionSensor kodun) ...
import cv2 
import time 
import numpy as np 
import logging 
from src.core.config_utils import get_config_value

logger = logging.getLogger(__name__)

class VisionSensor:
    def __init__(self, config):
        self.config = config
        logger.info("VisionSensor initializing...")
        self.camera_index = get_config_value(config, 'vision', 'camera_index', default=0, expected_type=int, logger_instance=logger)
        self.dummy_width = get_config_value(config, 'vision', 'dummy_width', default=640, expected_type=int, logger_instance=logger)
        self.dummy_height = get_config_value(config, 'vision', 'dummy_height', default=480, expected_type=int, logger_instance=logger)
        self.is_dummy = get_config_value(config, 'vision', 'is_dummy', default=False, expected_type=bool, logger_instance=logger)
        
        self.cap = None 
        self.is_camera_available = False 

        if self.is_dummy:
             logger.info("VisionSensor: is_dummy=True. Using simulated visual input.")
             self.is_camera_available = False # Dummy modda kamera aktif değil
             self.cap = None 
        else:
            try:
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap or not self.cap.isOpened():
                    logger.warning(f"VisionSensor: Could not open camera {self.camera_index}. Using simulated visual input.")
                    self.is_camera_available = False
                    if self.cap:
                        try: self.cap.release()
                        except: pass
                    self.cap = None
                else:
                    width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    logger.info(f"VisionSensor: Camera {self.camera_index} initialized successfully. Dimensions: {width}x{height}")
                    self.is_camera_available = True
            except Exception as e:
                logger.error(f"VisionSensor initialization failed: {e}", exc_info=True)
                self.is_camera_available = False
                if self.cap:
                     try: self.cap.release()
                     except: pass
                self.cap = None
        logger.info(f"VisionSensor initialized. Camera active: {self.is_camera_available}, Simulated Mode: {self.is_dummy}")

    def capture_frame(self):
        if self.is_dummy or not self.is_camera_available or self.cap is None:
            dummy_frame = np.full((self.dummy_height, self.dummy_width, 3), 128, dtype=np.uint8) # Orta gri
            return dummy_frame
        else:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("VisionSensor: Camera frame read failed. Switching to simulated mode.")
                    self.is_camera_available = False 
                    return self.capture_frame() 
                return frame
            except Exception as e:
                logger.error(f"VisionSensor: Error capturing frame: {e}. Switching to simulated mode.", exc_info=True)
                self.is_camera_available = False
                return self.capture_frame()

    def stop_stream(self):
        if self.cap is not None and self.cap.isOpened():
            logger.info("VisionSensor: Stopping visual stream...")
            try:
                self.cap.release()
                logger.info("VisionSensor: Camera released.")
            except Exception as e:
                 logger.error(f"VisionSensor: Error releasing camera: {e}", exc_info=True)
            self.cap = None
        self.is_camera_available = False

    def cleanup(self):
        logger.info("VisionSensor cleaning up...")
        self.stop_stream()
        logger.info("VisionSensor cleaned up.")