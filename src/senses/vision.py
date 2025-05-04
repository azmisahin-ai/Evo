# src/senses/vision.py

import logging
import cv2
import time
import numpy as np # Sahte veri için NumPy

class VisionSensor:
    """
    Evo'nun görsel duyu organını temsil eder.
    Kameradan görüntü akışını yakalamaktan sorumludur.
    Kamera bulunamazsa veya başlatılamazsa simüle edilmiş (dummy) kareler döndürebilir.
    """
    def __init__(self, config=None):
        logging.info("VisionSensor başlatılıyor...")
        self.config = config if config is not None else {} # None kontrolü ekle

        camera_index = self.config.get('camera_index', 0)
        self.cap = None # Başlangıçta None olarak ayarla
        self.is_camera_available = False # Kameranın başarılı başlatılıp başlatılmadığı

        try:
            self.cap = cv2.VideoCapture(camera_index)

            # Kameranın açılmasını bekle ve kontrol et
            if not self.cap.isOpened():
                logging.warning(f"Kamera {camera_index} açılamadı. Simüle edilmiş görsel girdi kullanılacak.")
                self.is_camera_available = False
                self.cap = None # Kameranın açılamadığını netleştir
            else:
                # Kameranın bazı özelliklerini ayarlayabiliriz (isteğe bağlı)
                # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600) # Örnek boyutlar
                # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400) # Örnek boyutlar
                # Gerçek boyutları al (veya varsayılan sahte boyutu kullan)
                self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.is_camera_available = True
                logging.info(f"Kamera {camera_index} başarıyla başlatıldı. Boyut: {self.frame_width}x{self.frame_height}")

        except Exception as e:
            logging.error(f"VisionSensor başlatılırken hata oluştu: {e}. Simüle edilmiş görsel girdi kullanılacak.")
            self.is_camera_available = False
            self.cap = None # Hata durumunda da None yap


        # Kamera kullanılamıyorsa sahte veri boyutlarını belirle
        if not self.is_camera_available:
            self.frame_width = self.config.get('dummy_width', 640) # Varsayılan sahte boyut
            self.frame_height = self.config.get('dummy_height', 480) # Varsayılan sahte boyut
            logging.info(f"VisionSensor sahte kare boyutu ayarlandı: {self.frame_width}x{self.frame_height}")


        logging.info(f"VisionSensor başlatıldı. Kamera aktif: {self.is_camera_available}")


    def capture_frame(self):
        """
        Kameradan tek bir kare yakalar veya kamera kullanılamıyorsa sahte bir kare döndürür.
        """
        if self.is_camera_available and self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                logging.warning("Kameradan kare yakalanamadı (Akış durmuş olabilir?). Simüle edilmiş kare döndürülüyor.")
                return self._generate_dummy_frame() # Hata durumunda sahte kare döndür
            else:
                # logging.debug(f"Gerçek kare başarıyla yakalandı. Boyut: {frame.shape}")
                # İsteğe bağlı: Kare üzerinde temel işlemler veya yeniden boyutlandırma burada yapılabilir
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Renk formatını değiştirme örneği
                return frame # OpenCV kare (NumPy array) döndürür
        else:
            # logging.debug("Kamera mevcut değil. Simüle edilmiş kare döndürülüyor.")
            return self._generate_dummy_frame() # Kamera mevcut değilse sahte kare döndür

    def _generate_dummy_frame(self):
        """Belirlenen boyutta siyah bir sahte kare oluşturur."""
        # Siyah (0 değeri) bir 3 kanallı (BGR veya RGB) 8-bit tam sayı kare
        dummy_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        # İsteğe bağlı: Ortasına bir metin ekle
        # cv2.putText(dummy_frame, "NO CAMERA", (self.frame_width // 4, self.frame_height // 2),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return dummy_frame


    def start_stream(self):
        """
        Görsel akışı başlatır (Gerçek implementasyon bekleniyor).
        Sürekli akış iş parçacığı veya döngüsü burada yönetilecek.
        """
        logging.info("Görsel akış başlatılıyor (Gerçek implementasyon bekleniyor)...")
        pass

    def stop_stream(self):
        """
        Görsel akışı durdurur ve kamera kaynağını serbest bırakır.
        """
        logging.info("Görsel akış durduruluyor...")
        if self.is_camera_available and self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logging.info("Kamera serbest bırakıldı.")
            self.is_camera_available = False # Durduruldu olarak işaretle
        elif self.cap is not None: # is_camera_available False ama cap None değilse
             logging.info("Kamera zaten serbest bırakılmış veya hiç açılamamış.")
        else: # cap None ise
             logging.info("VisionSensor tam başlatılamamıştı, serbest bırakılacak kaynak yok.")


    def __del__(self):
        """
        Nesne silindiğinde kaynakları serbest bırakır.
        """
        self.stop_stream()
        # logging.info("VisionSensor objesi silindi.") # __del__ içinde loglama bazen sorunlu olabilir


# Modülü bağımsız test etmek için örnek kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print("VisionSensor test ediliyor...")

    # Basit bir config objesi (gerçek uygulamada config dosyasından okunacak)
    # camera_index = 0 genellikle varsayılanı dener.
    # dummy_width/height sahte kare boyutunu ayarlar.
    test_config = {'camera_index': 0, 'dummy_width': 320, 'dummy_height': 240}

    sensor = None
    try:
        sensor = VisionSensor(test_config)

        # Kameranın hazır olması için kısa bir bekleme (gerçek kamera deneniyorsa)
        if sensor.is_camera_available:
             time.sleep(1)

        print("Kare yakalama denemesi (gerçek veya simüle)...")
        frame = sensor.capture_frame()

        if frame is not None:
            print(f"Yakalanan Kare Verisi (NumPy Array Shape): {frame.shape}, Data Type: {frame.dtype}")

            # Sahte olup olmadığını kontrol etmek için basit bir test
            if not sensor.is_camera_available and np.all(frame == 0):
                print("Simüle edilmiş (siyah) kare başarıyla alındı.")
            elif sensor.is_camera_available:
                 print("Gerçek veya yakalanamayan kare başarıyla alındı/oluşturuldu.")


            # İsteğe bağlı: Yakalanan kareyi bir dosyaya kaydet
            # cv2.imwrite("captured_frame.png", frame)
            # print("Kare 'captured_frame.png' olarak kaydedildi.")

            # İsteğe bağlı: Kareyi ekranda göster (GUI ortamı gerektirir, Codespace'de çalışmaz)
            # cv2.imshow("Yakalanan Kare", frame)
            # print("Kare görüntüleniyor. Kapatmak için bir tuşa basın.")
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        else:
             # Bu kısma normalde düşmemeli çünkü hata durumunda sahte kare dönüyor.
             print("Hata: capture_frame None döndürdü (beklenmeyen durum).")

    except Exception as e:
        logging.exception("VisionSensor test sırasında hata oluştu:")

    finally:
        if sensor:
            sensor.stop_stream() # Kaynakları temizle
        print("VisionSensor testi bitti.")