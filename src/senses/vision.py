# src/senses/vision.py
import cv2
import time
import numpy as np
import logging # Loglama için

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

class VisionSensor:
    """
    Evo'nun görsel duyu organı. Kamera akışından kareleri yakalar.
    """
    def __init__(self, config):
        self.config = config
        self.camera_index = config.get('camera_index', 0)
        self.dummy_width = config.get('dummy_width', 640)
        self.dummy_height = config.get('dummy_height', 480)
        self.cap = None # Kamera yakalama objesi
        self.is_camera_available = False # Kameranın aktif olup olmadığını tutar

        logger.info("VisionSensor başlatılıyor...")
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.warning(f"Kamera {self.camera_index} açılamadı. Simüle edilmiş görsel girdi kullanılacak.")
                self.is_camera_available = False
                self.cap = None # Açılmadıysa None olarak bırak
            else:
                # Kameradan gerçek boyutları almaya çalış
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"Kamera {self.camera_index} başarıyla başlatıldı. Boyut: {width}x{height}")
                self.is_camera_available = True

        except Exception as e:
            logger.error(f"VisionSensor başlatılırken hata oluştu: {e}", exc_info=True)
            self.is_camera_available = False
            self.cap = None # Hata durumunda None olarak bırak

        logger.info(f"VisionSensor başlatıldı. Kamera aktif: {self.is_camera_available}")

    def capture_frame(self):
        """
        Kamera akışından bir kare yakalar veya simüle edilmiş kare döndürür.

        Returns:
            numpy.ndarray or None: Yakalanan kamera karesi (BGR formatında) veya
                                    hata durumunda veya kamera aktif değilse None.
        """
        if self.is_camera_available and self.cap is not None:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Kamera akışından kare okunamadı. Bağlantı kopmuş olabilir.")
                    # Bağlantı kopmuşsa is_camera_available False yapılabilir veya yeniden bağlanma denenebilir
                    self.is_camera_available = False # Geçici olarak kapat
                    return None # Geçersiz kare döndür
                # logger.debug(f"VisionSensor: Gerçek kare yakalandı. Shape: {frame.shape}, Dtype: {frame.dtype}")
                return frame # Başarılı durumda kareyi döndür
            except Exception as e:
                # Yakalama sırasında beklenmedik hata
                logger.error(f"VisionSensor: Kare yakalama sırasında beklenmedik hata: {e}", exc_info=True)
                self.is_camera_available = False # Hata durumunda kapat
                return None # Hata durumunda None döndür
        else:
            # Kamera aktif değilse veya cap None ise simüle edilmiş kare döndür
            # logger.debug("VisionSensor: Simüle edilmiş kare üretiliyor.")
            # Simüle edilmiş BGR renkli (uint8) kare döndür. Değerler 0-255 arası.
            # Rastgele kare yerine sabit bir renk veya desen döndürmek de düşünülebilir.
            dummy_frame = np.zeros((self.dummy_height, self.dummy_width, 3), dtype=np.uint8)
            # Örneğin, gri bir kare yapalım
            dummy_frame[:, :, :] = 128 # Orta gri renk
            # logger.debug(f"VisionSensor: Simüle edilmiş kare üretildi. Shape: {dummy_frame.shape}, Dtype: {dummy_frame.dtype}")
            return dummy_frame

    def stop_stream(self):
        """
        Kamera akışını durdurur ve kaynakları serbest bırakır.
        """
        if self.cap is not None:
            logger.info("Görsel akış durduruluyor...")
            try:
                self.cap.release() # Kamera kaynağını serbest bırak
                logger.info("Kamera serbest bırakıldı.")
            except Exception as e:
                 logger.error(f"VisionSensor: Kamera serbest bırakılırken hata oluştu: {e}", exc_info=True)
            self.cap = None
            self.is_camera_available = False
        else:
             logger.info("Görsel akış zaten durdurulmuş veya hiç açılamamış.")


# Örnek Kullanım (run_evo.py'ye taşındı)
# if __name__ == '__main__':
#     # Basit test (setup_logging henüz burada çağrılmadığı için loglar tam görünmeyebilir)
#     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     test_config = {'camera_index': 0} # Veya 1, 2...
#     vision_sensor = VisionSensor(test_config)
#     frame = vision_sensor.capture_frame()
#     if frame is not None:
#         print(f"Captured frame with shape: {frame.shape}")
#         # frame'i göstermek için cv2.imshow('Frame', frame) kullanılabilir
#     vision_sensor.stop_stream()
#     print("VisionSensor testi tamamlandı.")