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
from src.core.utils import get_config_value # <<< get_config_value import edildi


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
        """
        self.config = config
        logger.info("VisionSensor başlatılıyor...")

        # Yapılandırmadan ayarları alırken get_config_value kullan
        # Bu, tip kontrolü ve varsayılan değer atamayı sadeleştirir.
        # logger_instance parametresi ile kendi logger'ımızı gönderiyoruz.
        self.camera_index = get_config_value(config, 'camera_index', 0, expected_type=int, logger_instance=logger)
        self.dummy_width = get_config_value(config, 'dummy_width', 640, expected_type=int, logger_instance=logger)
        self.dummy_height = get_config_value(config, 'dummy_height', 480, expected_type=int, logger_instance=logger)


        self.cap = None # cv2.VideoCapture objesi. Kamera başarılı başlatılırsa atanır.
        self.is_camera_available = False # Gerçek kameranın şu an aktif olup olmadığını tutar. Başlangıçta False.

        try:
            # OpenCV VideoCapture objesini oluşturarak kamerayı açmayı dene.
            # camera_index artık config'ten get_config_value ile int olarak alındı.
            self.cap = cv2.VideoCapture(self.camera_index)

            # isOpened() metodu, VideoCapture objesinin video kaynağına başarılı bir şekilde bağlanıp bağlanmadığını kontrol eder.
            if not self.cap or not self.cap.isOpened(): # cap'in None olup olmadığını da kontrol et
                # Kamera açılamazsa uyarı logu ver.
                logger.warning(f"VisionSensor: Kamera {self.camera_index} açılamadı. Simüle edilmiş görsel girdi kullanılacak.")
                self.is_camera_available = False # Kamera aktif değil bayrağını False yap.
                self.cap = None # Başarısız olursa cap objesini None yap.
            else:
                # Kamera başarıyla açıldıysa
                # Kameradan gerçek kare boyutlarını almayı dene (opsiyonel ama bilgilendirici).
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"VisionSensor: Kamera {self.camera_index} başarıyla başlatıldı. Boyut: {width}x{height}")
                self.is_camera_available = True # Kamera aktif bayrağını True yap.

        except Exception as e:
            # Kamera başlatma sırasında beklenmedik bir istisna oluşursa (örn: geçersiz index, donanım hatası).
            # Bu hata init sırasında kritik değildir (simüle moda geçiyoruz).
            logger.error(f"VisionSensor başlatılırken hata oluştu: {e}", exc_info=True)
            self.is_camera_available = False # Hata durumunda kamera aktif değil.
            # Hata olsa bile self.cap objesi oluşmuş olabilir, serbest bırakmak gerekebilir.
            if self.cap:
                 try:
                      self.cap.release()
                      # logger.debug("VisionSensor: Başlatma hatası sonrası kamera kaynağı serbest bırakıldı.") # Zaten hata logu var
                 except Exception:
                      pass # Serbest bırakma hatasını yoksay
            self.cap = None # Hata durumında cap objesini None yap.

        # Başlatma işleminin sonucunu logla.
        logger.info(f"VisionSensor başlatıldı. Kamera aktif: {self.is_camera_available}")

    def capture_frame(self):
        """
        Kamera akışından bir kare yakalar.

        Eğer gerçek kamera aktifse, VideoCapture objesinden bir kare okur.
        Eğer gerçek kamera aktif değilse (başlatılamadı veya hata oluştu),
        yapılandırmada belirtilen boyutta simüle edilmiş (dummy) bir kare döndürür.
        Kare yakalama sırasında hata oluşursa (gerçek kameradan), hatayı loglar
        ve None döndürerek main loop'un çökmesini engeller.

        Returns:
            numpy.ndarray or None: Başarıyla yakalanan (gerçek veya simüle) kamera karesi
                                    (BGR formatında, dtype uint8) veya
                                    gerçek kameradan yakalama sırasında hata oluştuysa None.
        """
        # Bu metot girdi almadığı için check_input_not_none veya check_numpy_input kullanılmaz.
        # Mantık aynı kalır.

        # Eğer gerçek kamera aktifse ve VideoCapture objesi mevcutsa
        if self.is_camera_available and self.cap is not None:
            try:
                # Kameradan bir kare oku. ret: bool (başarı), frame: numpy array (kare verisi)
                ret, frame = self.cap.read()

                # Eğer kare başarıyla okunamadıysa (örn: akış kesildi)
                if not ret:
                    logger.warning("VisionSensor: Kamera akışından kare okunamadı. Bağlantı kopmuş olabilir.")
                    # Bağlantı koptuysa, is_camera_available bayrağını False yaparak simüle moda geçişi tetikleyebiliriz.
                    self.is_camera_available = False # Geçici olarak kamerayı aktif değil yap.
                    # Bu durumda None döndürerek main loop'un bu kareyi atlamasını sağla.
                    return None

                # logger.debug(f"VisionSensor: Gerçek kare yakalandı. Shape: {frame.shape}, Dtype: {frame.dtype}")
                # Başarılı durumda yakalanan gerçek kareyi döndür.
                return frame

            except Exception as e:
                # Kare yakalama sırasında beklenmedik bir istisna oluşursa.
                # Bu, modül içindeki try-except bloklarını atlayan bir hata olabilir.
                logger.error(f"VisionSensor: Kare yakalama sırasında beklenmedik hata: {e}", exc_info=True)
                # Hata durumunda is_camera_available bayrağını False yaparak simüle moda geçişi tetikleyebiliriz.
                self.is_camera_available = False # Hata durumunda kamerayı aktif değil yap.
                # Hata durumunda None döndürerek main loop'un çökmesini engelle.
                return None
        else:
            # Eğer gerçek kamera aktif değilse (başlatılamadı veya hata oluştu)
            # Yapılandırmada belirtilen boyutta simüle edilmiş (dummy) bir kare döndür.
            # logger.debug("VisionSensor: Simüle edilmiş kare üretiliyor.")
            # dummy_width ve dummy_height'ın int olduğundan emin olmak için get_config_value init'te kullanıldı.
            dummy_frame = np.zeros((self.dummy_height, self.dummy_width, 3), dtype=np.uint8)
            # Örnek olarak, simüle kareyi orta gri renk yapalım.
            dummy_frame[:, :, :] = 128 # Tüm kanallara 128 değerini atar.
            # logger.debug(f"VisionSensor: Simüle edilmiş kare üretildi. Shape: {dummy_frame.shape}, Dtype: {dummy_frame.dtype}")
            # Simüle edilmiş kareyi döndür.
            return dummy_frame

    def stop_stream(self):
        """
        Kamera akışını durdurur ve VideoCapture objesi tarafından kullanılan kaynakları serbest bırakır.
        Program sonlanırken module_loader.py tarafından çağrılır.
        """
        # Eğer VideoCapture objesi mevcutsa ve hala açıksa
        if self.cap is not None: # self.cap'in init sırasında None yapılmasını sağlıyoruz.
            logger.info("VisionSensor: Görsel akış durduruluyor...")
            try:
                # Kamera kaynağını serbest bırak.
                self.cap.release()
                logger.info("VisionSensor: Kamera serbest bırakıldı.")
            except Exception as e:
                 # Serbest bırakma sırasında hata oluşursa logla.
                 logger.error(f"VisionSensor: Kamera serbest bırakılırken hata oluştu: {e}", exc_info=True)
            # Serbest bırakma işlemi sonrası cap objesini None yap.
            self.cap = None
        else:
             # cap objesi yoksa veya zaten None ise logla.
             logger.info("VisionSensor: Görsel akış zaten durdurulmuş veya hiç açılamamış.")

        # is_camera_available bayrağını False yap (zaten stop sonrası aktif olmayacaktır).
        self.is_camera_available = False

    def cleanup(self):
        """
        Nesne temizlendiğinde çağrılır.

        Bu metot genellikle stop_stream metodunu çağırmak için bir placeholder'dır.
        Kaynak temizliği stop_stream metodunda yapılır.
        module_loader.py bu metodu program sonlanırken çağırır (varsa).
        """
        logger.info("VisionSensor objesi temizleniyor.")
        # Kaynak temizliği stop_stream metodunda yapıldığı için burada ekstra bir şey yapmaya gerek yok.
        # self.stop_stream() # cleanup içinde stop_stream çağırmak çift çağrıya neden olabilir,
                             # o yüzden bu genellikle cleanup'ın görevi değildir.
        pass # Genellikle sadece loglamak yeterli.