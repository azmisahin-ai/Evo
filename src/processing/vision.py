# src/processing/vision.py

import logging
import numpy as np
import cv2 # OpenCV for image processing

class VisionProcessor:
    """
    Evo'nun görsel işlem birimini temsil eder.
    Ham görsel veriden temel özellikleri çıkarmaktan sorumludur.
    """
    def __init__(self, config=None):
        logging.info("VisionProcessor başlatılıyor...")
        self.config = config if config is not None else {}

        # İşleme çıktı boyutu veya diğer ayarlar buradan alınabilir
        self.output_width = self.config.get('output_width', 64) # Varsayılan çıktı boyutu
        self.output_height = self.config.get('output_height', 64) # Varsayılan çıktı boyutu
        logging.info(f"VisionProcessor çıktı boyutu ayarlandı: {self.output_width}x{self.output_height}")

        logging.info("VisionProcessor başlatıldı.")

    def process(self, visual_data):
        """
        Ham görsel veriyi (NumPy array - BGR formatında olması beklenir) alır
        ve işlenmiş özellikleri (örneğin, yeniden boyutlandırılmış gri tonlama) döndürür.
        """
        if visual_data is None:
            # logging.debug("VisionProcessor: İşlenecek görsel veri yok.")
            return None # Veri yoksa None döndür

        # logging.debug(f"VisionProcessor: Görsel veri alindi. Shape: {visual_data.shape}, Dtype: {visual_data.dtype}")

        try:
            # --- Gerçek Görsel İşleme Mantığı (Faz 1) ---

            # 1. Gri tonlamaya çevir
            # OpenCV renk formatı BGR'dir. Eğer girdi RGB ise cv2.COLOR_RGB2GRAY kullanın.
            if len(visual_data.shape) == 3 and visual_data.shape[2] == 3:
                gray_image = cv2.cvtColor(visual_data, cv2.COLOR_BGR2GRAY)
            elif len(visual_data.shape) == 2: # Zaten gri tonlama ise
                 gray_image = visual_data
            else:
                 logging.warning(f"Beklenmeyen görsel veri formatı (shape: {visual_data.shape}). Gri tonlama işlemi atlandı.")
                 # Gri tonlama yapılamazsa sadece yeniden boyutlandırmayı dene
                 gray_image = visual_data # Veya hata döndür None gibi


            # 2. Belirlenen çıktı boyutuna yeniden boyutlandır
            # Eğer gri tonlama yapılamadıysa (visual_data hala 3 kanallı veya farklı)
            # yeniden boyutlandırma renkli olarak yapılır.
            if gray_image is not None:
                 processed_data = cv2.resize(gray_image, (self.output_width, self.output_height))
            else:
                 logging.error("Görsel veri işlenemedi, yeniden boyutlandırma atlandı.")
                 return None


            # logging.debug(f"VisionProcessor: Görsel veri işlendi. Output Shape: {processed_data.shape}, Dtype: {processed_data.dtype}")

            return processed_data # İşlenmiş veriyi (NumPy array) döndür

        except Exception as e:
            logging.error(f"VisionProcessor sırasında hata oluştu: {e}", exc_info=True)
            return None # İşleme hatasında None döndür


    def __del__(self):
        """
        Nesne silindiğinde kaynakları temizler.
        """
        logging.info("VisionProcessor objesi silindi.")

# Modülü bağımsız test etmek için örnek kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print("VisionProcessor test ediliyor...")

    processor = VisionProcessor({'output_width': 32, 'output_height': 32}) # Daha küçük çıktı boyutu test et

    # Sahte bir görsel veri oluştur (örneğin 640x480 renkli kare)
    # cv2 default BGR kullanır.
    dummy_frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    # Test için bir kaç pikseli farklı renk yapalım
    dummy_frame_bgr[100:110, 100:110, :] = [0, 0, 255] # Mavi kare
    dummy_frame_bgr[200:210, 200:210, :] = [0, 255, 0] # Yeşil kare

    print(f"Sahte girdi verisi oluşturuldu. Shape: {dummy_frame_bgr.shape}")

    processed_output = processor.process(dummy_frame_bgr)

    if processed_output is not None:
        print(f"İşlenmiş çıktı alındı. Shape: {processed_output.shape}, Dtype: {processed_output.dtype}")
        # İşlenmiş çıktı gri tonlama ve yeniden boyutlandırılmış olmalı
        if len(processed_output.shape) == 2:
             print("Çıktı gri tonlama görünüyor.")
        if processed_output.shape == (32, 32):
             print("Çıktı doğru boyuta yeniden boyutlandırılmış.")
        # Sahte girdiye rağmen çıktı başarılı olduysa
        if processed_output.shape == (processor.output_height, processor.output_width):
             print("Processor başarıyla çalıştı.")

        # İsteğe bağlı: İşlenmiş kareyi kaydet veya göster (GUI gerektirir)
        # cv2.imwrite("processed_frame.png", processed_output)
        # cv2.imshow("İşlenmiş Kare", processed_output)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    else:
        print("İşleme sonucu None döndü (hata oluştu).")

    # None girdi ile dene
    print("\nNone girdi ile VisionProcessor testi:")
    processed_output_none = processor.process(None)
    if processed_output_none is None:
        print("None girdi ile işlem sonucu doğru şekilde None döndü.")
    else:
         print("None girdi ile işlem sonucu None dönmedi (beklenmeyen durum).")


    print("VisionProcessor testi bitti.")