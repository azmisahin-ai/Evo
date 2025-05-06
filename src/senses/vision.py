# src/senses/vision.py
#
# Evo'nın görsel duyu organını temsil eder.
# Kamera akışından ham görüntü karelerini yakalar.
# Kamera kullanılamadığında simüle edilmiş girdi sağlar.

import cv2 # OpenCV kütüphanesi, kamera yakalama ve temel görsel işlemler için. requirements.txt'e eklenmeli.
import time # Gerekirse zamanlama veya simülasyon için. Şu an doğrudan kullanılmıyor.
import numpy as np # Görüntü verisi (piksel matrisi) için.
import logging # Loglama için.

# Yardımcı fonksiyonları import et
from src.core.config_utils import get_config_value # <<< get_config_value import edildi


# Bu modül için bir logger oluştur
# 'src.senses.vision' adında bir logger döndürür.
logger = logging.getLogger(__name__)

class VisionSensor:
    """
    Evo'nın görsel duyu organı sınıfı.

    Belirtilen kamera indeksinden sürekli görüntü akışı yakalamayı dener.
    Eğer kamera başlatılamazsa veya yakalama sırasında hata oluşursa,
    simüle edilmiş (dummy) görüntü kareleri döndürür.
    Kamera durumu (aktif olup olmadığı) takip edilir.
    """
    def __init__(self, config):
        """
        VisionSensor'ı başlatır.

        Args:
            config (dict): Sensör yapılandırma ayarları.
                           'camera_index': Kullanılacak kamera indeksi (int, varsayılan 0).
                           'dummy_width': Simüle karenin genişliği (int, varsayılan 640).
                           'dummy_height': Simüle karenin yüksekliği (int, varsayılan 480).
                           'is_dummy': Simüle mod etkin mi (bool, varsayılan False).
        """
        self.config = config
        logger.info("VisionSensor başlatılıyor...")

        # Yapılandırmadan ayarları alırken get_config_value kullan
        # audio config anahtarı altındaki değerleri okuyoruz
        self.camera_index = get_config_value(config, 'vision', 'camera_index', default=0, expected_type=int, logger_instance=logger)
        self.dummy_width = get_config_value(config, 'vision', 'dummy_width', default=640, expected_type=int, logger_instance=logger)
        self.dummy_height = get_config_value(config, 'vision', 'dummy_height', default=480, expected_type=int, logger_instance=logger)
        self.is_dummy = get_config_value(config, 'vision', 'is_dummy', default=False, expected_type=bool, logger_instance=logger)


        self.cap = None # cv2.VideoCapture objesi. Kamera başarılı başlatılırsa atanır.
        self.is_camera_available = False # Gerçek kameranın şu an aktif olup olmadığını tutar. Başlangıçta False.

        # Eğer is_dummy True ise gerçek kamerayı başlatmaya çalışma
        if self.is_dummy:
             logger.info("VisionSensor: is_dummy=True. Simüle edilmiş görsel girdi kullanılacak.")
             self.is_camera_available = False
             self.cap = None # Cap objesi None kalmalı
        else:
            # Gerçek kamerayı başlatmayı dene
            try:
                self.cap = cv2.VideoCapture(self.camera_index)

                # isOpened() metodu, VideoCapture objesinin video kaynağına başarılı bir şekilde bağlanıp bağlanmadığını kontrol eder.
                if not self.cap or not self.cap.isOpened(): # cap'in None olup olmadığını da kontrol et
                    # Kamera açılamazsa uyarı logu ver.
                    logger.warning(f"VisionSensor: Kamera {self.camera_index} açılamadı. Simüle edilmiş görsel girdi kullanılacak.")
                    self.is_camera_available = False # Kamera aktif değil bayrağını False yap.
                    # Başarısız olursa self.cap objesini temizle
                    if self.cap:
                        try:
                            self.cap.release()
                        except Exception: pass
                    self.cap = None
                else:
                    # Kamera başarıyla açıldıysa
                    # Kameradan gerçek kare boyutlarını almayı dene (opsiyonel ama bilgilendirici).
                    width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    logger.info(f"VisionSensor: Kamera {self.camera_index} başarıyla başlatıldı. Boyut: {width}x{height}")
                    self.is_camera_available = True # Kamera aktif bayrağını True yap.

            except Exception as e:
                logger.error(f"VisionSensor başlatılırken hata oluştu: {e}", exc_info=True)
                self.is_camera_available = False
                if self.cap:
                     try:
                          self.cap.release()
                     except Exception: pass
                self.cap = None # Hata durumunda cap objesini None yap.


        logger.info(f"VisionSensor başlatıldı. Kamera aktif: {self.is_camera_available}, Simüle Mod: {self.is_dummy}")


    def capture_frame(self):
        """
        Kamera akışından bir kare yakalar.

        Eğer gerçek kamera aktifse, VideoCapture objesinden bir kare okur.
        Eğer gerçek kamera aktif değilse (başlatılamadı veya hata oluştu),
        yapılandırmada belirtilen boyutta simüle edilmiş (dummy) bir kare döndürür.
        Kare yakalama sırasında hata oluşursa (gerçek kameradan), hatayı loglar
        ve None döndürmek yerine simüle moda geçer ve simüle kare döndürür.

        Returns:
            numpy.ndarray: Başarıyla yakalanan (gerçek veya simüle) kamera karesi
                                    (BGR formatında, dtype uint8). Hata durumunda None dönmez.
        """
        # Eğer simüle mod aktifse veya gerçek kamera kullanılamıyorsa dummy kare döndür.
        if self.is_dummy or not self.is_camera_available or self.cap is None:
            # logger.debug("VisionSensor: Simüle edilmiş kare üretiliyor.")
            # dummy_width ve dummy_height'ın int olduğundan emin olmak için get_config_value init'te kullanıldı.
            dummy_frame = np.zeros((self.dummy_height, self.dummy_width, 3), dtype=np.uint8)
            dummy_frame[:, :, :] = 128 # Orta gri
            # logger.debug(f"VisionSensor: Simüle edilmiş kare üretildi. Shape: {dummy_frame.shape}, Dtype: {dummy_frame.dtype}")
            return dummy_frame
        else:
            # Gerçek kamera aktifse kare oku
            try:
                ret, frame = self.cap.read()

                # Eğer kare başarıyla okunamadıysa (örn: akış kesildi)
                if not ret:
                    logger.warning("VisionSensor: Kamera akışından kare okunamadı. Bağlantı kopmuş olabilir. Simüle moda geçiliyor.")
                    self.is_camera_available = False # Simüle moda geçişi tetikle.
                    # Rekürsif çağrı ile self.capture_frame() metodunu çağırarak dummy kare al.
                    return self.capture_frame() # Rekürsif çağrı ile dummy kare al.

                # logger.debug(f"VisionSensor: Gerçek kare yakalandı. Shape: {frame.shape}, Dtype: {frame.dtype}")
                # Başarılı durumda yakalanan gerçek kareyi döndür.
                return frame

            except Exception as e:
                # Kare yakalama sırasında beklenmedik bir istisna oluşursa.
                logger.error(f"VisionSensor: Kare yakalama sırasında beklenmedik hata: {e}. Simüle moda geçiliyor.", exc_info=True)
                self.is_camera_available = False # Hata durumunda kamerayı aktif değil yap.
                 # Rekürsif çağrı ile self.capture_frame() metodunu çağırarak dummy kare al.
                return self.capture_frame() # Rekürsif çağrı ile dummy kare al.


    def stop_stream(self):
        """
        Kamera akışını durdurur ve VideoCapture objesi tarafından kullanılan kaynakları serbest bırakır.
        Program sonlanırken cleanup metodu tarafından çağrılır.
        """
        # Eğer VideoCapture objesi mevcutsa ve hala açıksa
        # isOpened() kontrolü release sonrası false döner.
        if self.cap is not None and self.cap.isOpened(): # cap'in varlığı ve açık olması kontrol ediliyor.
            logger.info("VisionSensor: Görsel akış durduruluyor...")
            try:
                # Kamera kaynağını serbest bırak.
                self.cap.release()
                logger.info("VisionSensor: Kamera serbest bırakıldı.")
            except Exception as e:
                 # Serbest bırakma sırasında hata oluşursa logla.
                 logger.error(f"VisionSensor: Kamera serbest bırakılırken hata oluştu: {e}", exc_info=True)
            # Serbest bırakma işlemi sonrası cap objesini None yap.
            self.cap = None # Temizleme sonrası objeyi None yapmak iyi pratiktir.
        else:
             # cap objesi yoksa, None ise veya zaten kapalıysa (isOpened false) logla.
             logger.info("VisionSensor: Görsel akış zaten durdurulmuş, hiç açılamamış veya kaynak zaten serbest bırakılmış.")

        # is_camera_available bayrağını False yap (zaten stop sonrası aktif olmayacaktır).
        self.is_camera_available = False

    # --- cleanup metodunu düzeltiyoruz ---
    def cleanup(self):
        """
        Nesne temizlendiğinde çağrılır. stop_stream metodunu çağırarak kaynakları serbest bırakır.
        module_loader.py bu metodu program sonlanırken çağırır (varsa).
        """
        logger.info("VisionSensor objesi temizleniyor.")
        # Kaynak temizliği için stop_stream metodunu çağır.
        # stop_stream içindeki None ve isOpened kontrolü sayesinde çift çağrı sorun olmaz.
        self.stop_stream()
        # cleanup_safely ile çağrıldığı için stop_stream içindeki hata yönetimi yeterli.
        logger.info("VisionSensor objesi silindi.")