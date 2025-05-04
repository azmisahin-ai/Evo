# src/senses/vision.py

import logging
import cv2
import time # Kamera başlatılırken gecikme eklemek için

class VisionSensor:
    """
    Evo'nun görsel duyu organını temsil eder.
    Kameradan görüntü akışını yakalamaktan sorumludur.
    """
    def __init__(self, config=None):
        logging.info("VisionSensor başlatılıyor...")
        self.config = config # Konfigürasyon ayarları burada kullanılabilir

        # Kamerayı başlatma
        # Genellikle 0 varsayılan kameradır. Farklı kameralar için indeks değişebilir (1, 2, ...)
        # Yapılandırma (config) üzerinden kamera indeksi ayarlanabilir.
        camera_index = config.get('camera_index', 0) if config else 0
        self.cap = cv2.VideoCapture(camera_index)

        # Kameranın açılmasını bekle (Bazı kameralar hemen hazır olmayabilir)
        if not self.cap.isOpened():
            logging.error(f"Kamera {camera_index} açılamadı. Lütfen bağlı olduğundan emin olun.")
            self.cap = None # Kameranın açılamadığını belirt
            # Hata yönetimi: Uygulama burada durdurulabilir veya kamerasız devam edilebilir.
            # Şimdilik hata loglayıp devam edeceğiz, capture_frame None döndürecek.
        else:
            # Kameranın bazı özelliklerini ayarlayabiliriz (isteğe bağlı)
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            logging.info(f"Kamera {camera_index} başarıyla başlatıldı.")

    def capture_frame(self):
        """
        Kameradan tek bir kare yakalar.
        Gelecekte sürekli akış mantığı buraya eklenecek.
        """
        if self.cap is None or not self.cap.isOpened():
            logging.warning("Kamera hazır değil veya açılamadı. Kare yakalanamadı.")
            return None

        ret, frame = self.cap.read()

        if not ret:
            logging.error("Kare yakalanamadı (Akış durmuş olabilir?).")
            return None # Kare yakalanamazsa None döndür

        # İsteğe bağlı: Kare üzerinde temel işlemler veya yeniden boyutlandırma burada yapılabilir
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Renk formatını değiştirme örneği

        logging.debug(f"Kare başarıyla yakalandı. Boyut: {frame.shape}")
        return frame # OpenCV kare (NumPy array) döndürür

    def start_stream(self):
        """
        Görsel akışı başlatır (gelecekte, sürekli akış iş parçacığı veya döngüsü burada yönetilecek).
        Şimdilik capture_frame tek kare alıyor. Gerçek akış için bu metot genişletilecek.
        """
        logging.info("Görsel akış başlatılıyor (Gerçek implementasyon bekleniyor)...")
        # Gerçek sürekli akış mantığı (örn: bir thread içinde capture_frame'i çağırmak) buraya gelecek
        pass

    def stop_stream(self):
        """
        Görsel akışı durdurur ve kamera kaynağını serbest bırakır.
        """
        logging.info("Görsel akış durduruluyor...")
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logging.info("Kamera serbest bırakıldı.")
        elif hasattr(self, 'cap') and self.cap is not None:
             logging.info("Kamera zaten serbest bırakılmış veya açılamamış.")
        else:
             logging.info("VisionSensor henüz tam başlatılmamıştı.")


    def __del__(self):
        """
        Nesne silindiğinde kaynakları serbest bırakır.
        """
        self.stop_stream()
        logging.info("VisionSensor objesi silindi.")

# Modülü bağımsız test etmek için örnek kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print("VisionSensor test ediliyor...")

    # Basit bir config objesi (gerçek uygulamada config dosyasından okunacak)
    test_config = {'camera_index': 0} # Varsa farklı bir kamera indeksi deneyin

    sensor = None
    try:
        sensor = VisionSensor(test_config)
        # Kameranın hazır olması için kısa bir bekleme
        time.sleep(1)

        if sensor.cap and sensor.cap.isOpened():
            frame = sensor.capture_frame()

            if frame is not None:
                print(f"Yakalanan Kare Verisi (NumPy Array Shape): {frame.shape}, Data Type: {frame.dtype}")
                # İsteğe bağlı: Yakalanan kareyi bir dosyaya kaydet
                # cv2.imwrite("captured_frame.png", frame)
                # print("Kare 'captured_frame.png' olarak kaydedildi.")

                # İsteğe bağlı: Kareyi ekranda göster (GUI ortamı gerektirir)
                # cv2.imshow("Yakalanan Kare", frame)
                # print("Kare görüntüleniyor. Kapatmak için bir tuşa basın.")
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            else:
                print("Kare yakalama testi BAŞARISIZ oldu.")
        else:
             print("VisionSensor başlatma testi BAŞARISIZ oldu (Kamera açılamadı).")

    except Exception as e:
        logging.exception("VisionSensor test sırasında hata oluştu:")

    finally:
        if sensor:
            sensor.stop_stream() # Kaynakları temizle
        print("VisionSensor testi bitti.")