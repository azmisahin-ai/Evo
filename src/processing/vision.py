# src/processing/vision.py
import cv2
import numpy as np
import logging

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

class VisionProcessor:
    """
    Görsel veriyi işler (örn. yeniden boyutlandırma, gri tonlama).
    """
    def __init__(self, config):
        self.config = config
        self.output_width = config.get('output_width', 64)
        self.output_height = config.get('output_height', 64)
        logger.info("VisionProcessor başlatılıyor...")
        # Ek yapılandırma veya model yükleme buraya gelebilir
        logger.info(f"VisionProcessor çıktı boyutu ayarlandı: {self.output_width}x{self.output_height}")
        logger.info("VisionProcessor başlatıldı.")

    def process(self, visual_input):
        """
        Ham görsel girdiyi işler (yeniden boyutlandırma, gri tonlama vb.).

        Args:
            visual_input (numpy.ndarray or None): Ham görsel veri (kare) veya None.

        Returns:
            numpy.ndarray or None: İşlenmiş görsel veri (yeniden boyutlandırılmış, gri tonlama)
                                    veya hata durumunda veya girdi None ise None.
        """
        # Temel hata yönetimi: Girdi None ise veya beklenen tipte değilse
        if visual_input is None:
            logger.debug("VisionProcessor: Girdi None. İşleme atlanıyor.")
            return None

        if not isinstance(visual_input, np.ndarray):
             logger.error(f"VisionProcessor: Beklenmeyen girdi tipi: {type(visual_input)}. numpy.ndarray bekleniyordu.")
             return None

        # DEBUG logu: Girdi detayları
        # logger.debug(f"VisionProcessor: Görsel veri alindi. Shape: {visual_input.shape}, Dtype: {visual_input.dtype}")

        processed_frame = None # İşlem sonucunu tutacak değişken

        try:
            # 1. Gri tonlamaya çevir (Eğer renkli ise)
            # Shape'in (H, W, C) ve C'nin 3 (BGR) olduğunu varsayıyoruz. Gri zaten (H, W) olur.
            if len(visual_input.shape) == 3 and visual_input.shape[2] == 3:
                processed_frame = cv2.cvtColor(visual_input, cv2.COLOR_BGR2GRAY)
                # logger.debug("VisionProcessor: Görsel veri gri tonlamaya çevrildi.")
            elif len(visual_input.shape) == 2:
                processed_frame = visual_input # Zaten gri tonlama gibi görünüyor
                # logger.debug("VisionProcessor: Görsel veri zaten gri tonlama gibi. Çevirme atlandi.")
            else:
                 logger.warning(f"VisionProcessor: Beklenmeyen görsel girdi boyutu: {visual_input.shape}. Gri tonlamaya çevrilemedi.")
                 return None # Beklenmeyen boyutsa işleyemeyiz

            # 2. Yeniden boyutlandır
            processed_frame = cv2.resize(processed_frame, (self.output_width, self.output_height))
            # logger.debug(f"VisionProcessor: Görsel veri ({self.output_width}, {self.output_height}) boyutuna yeniden boyutlandırıldı.")


            # Çıktının doğru dtype (uint8) olduğundan emin olalım
            if processed_frame.dtype != np.uint8:
                 processed_frame = processed_frame.astype(np.uint8)
                 # logger.debug(f"VisionProcessor: Yeniden boyutlandırılmış veri dtype uint8 yapildi.")


        except cv2.Error as e:
            # OpenCV'den kaynaklanan spesifik hatalar (örn. geçersiz boyut)
            logger.error(f"VisionProcessor: OpenCV hatasi olustu: {e}", exc_info=True)
            return None
        except Exception as e:
            # Diğer beklenmedik hatalar
            logger.error(f"VisionProcessor: İşleme sırasında beklenmedik hata: {e}", exc_info=True)
            return None

        # Başarılı durumda işlenmiş kareyi döndür
        # logger.debug(f"VisionProcessor: Görsel veri işlendi. Output Shape: {processed_frame.shape}, Dtype: {processed_frame.dtype}")
        return processed_frame

    def cleanup(self):
        """Kaynakları temizler (şimdilik gerek yok, placeholder)."""
        logger.info("VisionProcessor objesi silindi.")
        pass # İşlemci genellikle temizlik gerektirmez