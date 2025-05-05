# src/processing/vision.py
#
# Görsel duyu verisini işler.
# Ham piksel verisinden temel görsel özellikleri (örn. yeniden boyutlandırma, gri tonlama) çıkarır.

import cv2 # OpenCV kütüphanesi. requirements.txt'e eklenmeli.
import numpy as np # Sayısal işlemler ve arrayler için.
import logging # Loglama için.

# Bu modül için bir logger oluştur
# 'src.processing.vision' adında bir logger döndürür.
logger = logging.getLogger(__name__)

class VisionProcessor:
    """
    Görsel veriyi işleyen sınıf.

    VisionSensor'dan gelen ham görsel girdiyi (kare) alır,
    üzerinde temel işlemler yaparak (şimdilik yeniden boyutlandırma ve gri tonlama)
    RepresentationLearner için uygun hale getirir.
    İşleme sırasında oluşabilecek hataları yönetir.
    """
    def __init__(self, config):
        """
        VisionProcessor'ı başlatır.

        Args:
            config (dict): İşlemci yapılandırma ayarları.
                           'output_width': İşlenmiş görsel çıktının genişliği (int, varsayılan 64).
                           'output_height': İşlenmiş görsel çıktının yüksekliği (int, varsayılan 64).
        """
        self.config = config
        # Yapılandırmadan çıktı boyutlarını al, yoksa varsayılanları kullan.
        self.output_width = config.get('output_width', 64)
        self.output_height = config.get('output_height', 64)

        logger.info("VisionProcessor başlatılıyor...")
        # Ek yapılandırma veya model yükleme buraya gelebilir (ileride).
        logger.info(f"VisionProcessor çıktı boyutu ayarlandı: {self.output_width}x{self.output_height}")
        logger.info("VisionProcessor başlatıldı.")

    def process(self, visual_input):
        """
        Ham görsel girdiyi işler.

        Girdiyi alır (genellikle BGR renkli numpy array), gri tonlamaya çevirir,
        belirtilen çıktı boyutuna yeniden boyutlandırır ve RepresentationLearner'a
        gönderilmek üzere işlenmiş kareyi döndürür.
        Girdi None ise veya işleme sırasında hata oluşursa None döndürür.

        Args:
            visual_input (numpy.ndarray or None): Ham görsel veri (kare) veya None.
                                                  Genellikle VisionSensor'dan gelir.
                                                  Beklenen format: shape (Y, X, C) veya (Y, X), dtype uint8.

        Returns:
            numpy.ndarray or None: İşlenmiş görsel veri (yeniden boyutlandırılmış, gri tonlama,
                                    shape (output_height, output_width), dtype uint8)
                                    veya hata durumunda veya girdi None ise None.
        """
        # Hata yönetimi: Girdi None ise veya beklenen tipte değilse
        if visual_input is None:
            # Girdi None ise, bu bir hata değil, sadece o döngüde görsel veri yok demektir.
            # DEBUG seviyesinde logla ve None döndürerek işlemeyi atla.
            logger.debug("VisionProcessor: Girdi None. İşleme atlanıyor.")
            return None

        # Girdinin numpy array ve doğru dtype (uint8) olup olmadığını kontrol et.
        # VisionSensor uint8 döndürdüğü için bu kontrol önemlidir.
        if not isinstance(visual_input, np.ndarray) or visual_input.dtype != np.uint8:
             logger.error(f"VisionProcessor: Beklenmeyen girdi tipi veya dtype: {type(visual_input)}, {visual_input.dtype}. numpy.ndarray (dtype uint8) bekleniyordu.")
             return None # Geçersiz tip veya dtype ise işlemeyi durdur ve None döndür.

        # DEBUG logu: Girdi detayları (boyutları ve tipi).
        # logger.debug(f"VisionProcessor: Görsel veri alindi. Shape: {visual_input.shape}, Dtype: {visual_input.dtype}")

        processed_frame = None # İşlem sonucunu tutacak değişken.

        try:
            # 1. Gri tonlamaya çevir (Eğer girdi renkli ise).
            # Girdinin shape'i (Yükseklik, Genişlik, Kanal) ve kanal sayısının 3 (BGR) olduğunu varsayıyoruz.
            # Gri tonlama görüntünün shape'i (Yükseklik, Genişlik) olur.
            if len(visual_input.shape) == 3 and visual_input.shape[2] == 3:
                processed_frame = cv2.cvtColor(visual_input, cv2.COLOR_BGR2GRAY)
                # logger.debug("VisionProcessor: Görsel veri BGR'den gri tonlamaya çevrildi.")
            elif len(visual_input.shape) == 2:
                # Eğer girdi zaten 2 boyutlu ise (gri gibi), doğrudan kullan.
                processed_frame = visual_input.copy() # Orijinal girdiyi değiştirmemek için kopya al.
                # logger.debug("VisionProcessor: Görsel girdi zaten gri tonlama gibi. Çevirme atlandi.")
            else:
                 # Beklenmeyen girdi boyutu (ne 2D gri ne de 3D renkli).
                 logger.warning(f"VisionProcessor: Beklenmeyen görsel girdi boyutu: {visual_input.shape}. Gri tonlamaya çevrilemedi veya işlenemedi.")
                 return None # Beklenmeyen boyutsa işleyemeyiz, None döndür.

            # 2. Yeniden boyutlandır (Yapılandırmada belirtilen output_width x output_height boyutuna).
            # Interpolation metodu belirtilebilir, INTER_AREA küçültme için iyidir.
            processed_frame = cv2.resize(processed_frame, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)
            # logger.debug(f"VisionProcessor: Görsel veri ({self.output_width}, {self.output_height}) boyutuna yeniden boyutlandırıldı.")


            # İşlem sonrası çıktının hala doğru dtype (uint8) olduğundan emin olalım.
            # cv2.resize genellikle dtype'ı korur ama yine de kontrol etmek sağlamlık katar.
            if processed_frame.dtype != np.uint8:
                 processed_frame = processed_frame.astype(np.uint8)
                 # logger.debug(f"VisionProcessor: Yeniden boyutlandırılmış veri dtype uint8 yapildi.")


        except cv2.Error as e:
            # OpenCV kütüphanesinden kaynaklanan spesifik hatalar (örn. yeniden boyutlandırma için geçersiz boyutlar).
            logger.error(f"VisionProcessor: OpenCV hatasi olustu isleme sirasinda: {e}", exc_info=True)
            return None # Hata durumunda None döndür.
        except Exception as e:
            # İşleme adımları sırasında oluşabilecek diğer beklenmedik hatalar.
            logger.error(f"VisionProcessor: İşleme sırasında beklenmedik hata: {e}", exc_info=True)
            return None # Hata durumunda None döndür.

        # Başarılı durumda işlenmiş kareyi döndür.
        # logger.debug(f"VisionProcessor: Görsel veri başarıyla işlendi. Output Shape: {processed_frame.shape}, Dtype: {processed_frame.dtype}")
        return processed_frame

    def cleanup(self):
        """
        VisionProcessor kaynaklarını temizler.

        Şimdilik bu işlemci özel bir kaynak (dosya, bağlantı vb.) kullanmadığı için
        temizleme adımı içermez, sadece bilgilendirme logu içerir.
        Gelecekte gerekirse kaynak temizleme mantığı buraya eklenebilir.
        module_loader.py bu metodu program sonlanırken çağırır (varsa).
        """
        logger.info("VisionProcessor objesi temizleniyor.")
        pass # İşlemci genellikle explicit temizlik gerektirmez.