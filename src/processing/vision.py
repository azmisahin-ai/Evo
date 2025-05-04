# src/processing/vision.py

import logging
import numpy as np # Girdi verisi NumPy array olacak

class VisionProcessor:
    """
    Evo'nun görsel işlem birimini temsil eder.
    Ham görsel veriden temel özellikleri çıkarmaktan sorumludur.
    """
    def __init__(self, config=None):
        logging.info("VisionProcessor başlatılıyor...")
        self.config = config if config is not None else {}

        # Gerekirse burada işlem için model veya ayarlar yüklenebilir
        logging.info("VisionProcessor başlatıldı.")

    def process(self, visual_data):
        """
        Ham görsel veriyi (NumPy array) alır ve işlenmiş özellikleri döndürür.
        """
        if visual_data is None:
            # logging.debug("VisionProcessor: İşlenecek görsel veri yok.")
            return None # Veri yoksa None döndür

        # logging.debug(f"VisionProcessor: Görsel veri alindi. Shape: {visual_data.shape}, Dtype: {visual_data.dtype}")

        # --- Gerçek Görsel İşleme Mantığı Buraya Gelecek (Faz 1 ve sonrası) ---
        # Örnek: Gri tonlamaya çevirme, kenar tespiti, boyutlandırma vb.
        # processed_data = cv2.cvtColor(visual_data, cv2.COLOR_BGR2GRAY) # Örnek: Gri tonlama
        # processed_data = cv2.resize(processed_data, (64, 64)) # Örnek: Yeniden boyutlandırma

        # Şimdilik sadece alınan veriyi logla ve olduğu gibi döndür
        processed_data = visual_data

        # logging.debug(f"VisionProcessor: Görsel veri işlendi (placeholder). Output Shape: {processed_data.shape if isinstance(processed_data, np.ndarray) else 'None'}")

        return processed_data # İşlenmiş veriyi (NumPy array veya None) döndür

    def __del__(self):
        """
        Nesne silindiğinde kaynakları temizler.
        """
        logging.info("VisionProcessor objesi silindi.")

# Modülü bağımsız test etmek için örnek kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print("VisionProcessor test ediliyor...")

    processor = VisionProcessor()

    # Sahte bir görsel veri oluştur (örneğin 640x480 siyah kare)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    print(f"Sahte girdi verisi oluşturuldu. Shape: {dummy_frame.shape}")

    processed_output = processor.process(dummy_frame)

    if processed_output is not None:
        print(f"İşlenmiş çıktı alındı. Shape: {processed_output.shape}, Dtype: {processed_output.dtype}")
        # Temel bir kontrol yap
        if np.array_equal(processed_output, dummy_frame):
            print("Processor placeholder olarak çalışıyor (girdiyi aynen döndürdü).")
    else:
        print("İşleme sonucu None döndü (beklenmeyen durum).")

    # None girdi ile dene
    print("\nNone girdi ile VisionProcessor testi:")
    processed_output_none = processor.process(None)
    if processed_output_none is None:
        print("None girdi ile işlem sonucu doğru şekilde None döndü.")
    else:
         print("None girdi ile işlem sonucu None dönmedi (beklenmeyen durum).")

    print("VisionProcessor testi bitti.")