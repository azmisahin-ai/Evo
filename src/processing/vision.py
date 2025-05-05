# src/processing/vision.py
#
# Görsel duyu verisini işler.
# Ham piksel verisinden temel görsel özellikleri (örn. yeniden boyutlandırma, gri tonlama, kenarlar) çıkarır.
# Evo'nın Faz 1'deki işleme yeteneklerinin bir parçasıdır.

import cv2 # OpenCV kütüphanesi, kamera yakalama ve temel görsel işlemler için. requirements.txt'e eklenmeli.
import numpy as np # Sayısal işlemler ve arrayler için.
import logging # Loglama için.

# Yardımcı fonksiyonları import et (özellikle girdi kontrolleri ve config için)
from src.core.utils import check_input_not_none, check_numpy_input, get_config_value # <<< Utils importları


# Bu modül için bir logger oluştur
# 'src.processing.vision' adında bir logger döndürür.
logger = logging.getLogger(__name__)

class VisionProcessor:
    """
    Evo'nın görsel veriyi işleyen sınıfı (Faz 1 implementasyonu).

    VisionSensor'dan gelen ham görsel girdiyi (kare) alır,
    üzerinde temel işlemler yaparak (yeniden boyutlandırma, gri tonlama, kenar tespiti)
    RepresentationLearner için uygun hale getirir.
    İşleme sırasında oluşabilecek hataları yönetir ve akışın devamlılığını sağlar.
    Çıktı olarak işlenmiş farklı özellikleri içeren bir sözlük döndürür.
    """
    def __init__(self, config):
        """
        VisionProcessor'ı başlatır.

        Args:
            config (dict): İşlemci yapılandırma ayarları.
                           'output_width': İşlenmiş görsel çıktının genişliği (int, varsayılan 64).
                           'output_height': İşlenmiş görsel çıktının yüksekliği (int, varsayılan 64).
                           'canny_low_threshold': Canny kenar tespiti düşük eşiği (int, varsayılan 50).
                           'canny_high_threshold': Canny kenar tespiti yüksek eşiği (int, varsayılan 150).
        """
        self.config = config
        logger.info("VisionProcessor başlatılıyor...")

        # Yapılandırmadan çıktı boyutlarını ve Canny eşiklerini alırken get_config_value kullan.
        # Bu, tip kontrolü ve varsayılan değer atamayı sadeleştirir ve loglar.
        self.output_width = get_config_value(config, 'output_width', 64, expected_type=int, logger_instance=logger)
        self.output_height = get_config_value(config, 'output_height', 64, expected_type=int, logger_instance=logger)
        self.canny_low_threshold = get_config_value(config, 'canny_low_threshold', 50, expected_type=int, logger_instance=logger)
        self.canny_high_threshold = get_config_value(config, 'canny_high_threshold', 150, expected_type=int, logger_instance=logger)


        # Geçerli çıktı boyutları olduğundan emin ol (negatif veya sıfır olmamalı)
        if self.output_width <= 0 or self.output_height <= 0:
             logger.error(f"VisionProcessor: Konfigurasyonda geçersiz çıktı boyutları: {self.output_width}x{self.output_height}. Varsayılan (64x64) kullanılıyor.")
             self.output_width = 64
             self.output_height = 64
             # Bu durumda başlatmayı kritik yapmıyoruz, sadece loglayıp varsayılanlarla devam ediyoruz (Policy).

        logger.info(f"VisionProcessor başlatıldı. Çıktı boyutu: {self.output_width}x{self.output_height}, Canny Eşikleri: [{self.canny_low_threshold}, {self.canny_high_threshold}]")


    def process(self, visual_input):
        """
        Ham görsel girdiyi işler, temel özellikleri çıkarır.

        Girdiyi alır (genellikle BGR renkli numpy array), gri tonlamaya çevirir,
        belirtilen çıktı boyutuna yeniden boyutlandırır ve kenar tespiti uygular.
        İşlenmiş kareyi (gri tonlama ve kenar haritası) RepresentationLearner'a
        gönderilmek üzere bir sözlük içinde döndürür.
        Girdi None ise veya işleme sırasında hata oluşursa boş sözlük `{}` döndürür.

        Args:
            visual_input (numpy.ndarray or None): Ham görsel veri (kare) veya None.
                                                  Genellikle VisionSensor'dan gelir.
                                                  Beklenen format: shape (Y, X, C) veya (Y, X), dtype uint8.

        Returns:
            dict: İşlenmiş görsel özellikleri içeren bir sözlük.
                  Anahtarlar: 'grayscale' (numpy.ndarray, (output_height, output_width), uint8),
                              'edges' (numpy.ndarray, (output_height, output_width), uint8).
                  Hata durumunda veya girdi None ise boş sözlük `{}` döner.
        """
        # Hata yönetimi: Girdi None ise veya beklenen tipte değilse
        # check_input_not_none fonksiyonunu kullan (None ise loglar ve False döner)
        if not check_input_not_none(visual_input, input_name="visual_input for VisionProcessor", logger_instance=logger):
             # Girdi None ise işlemeyi atla ve boş sözlük döndür (Graceful failure).
             logger.debug("VisionProcessor.process: Girdi None. Boş sözlük döndürülüyor.")
             return {} # Boş sözlük döndür, None yerine.


        # Girdinin numpy array ve doğru dtype (uint8) olup olmadığını kontrol et.
        # check_numpy_input fonksiyonunu kullan. Bu fonksiyon aynı zamanda np.ndarray kontrolü de yapar.
        # Beklenen boyut 2D (gri) veya 3D (renkli) olabilir. dtype uint8 bekleniyor.
        # check_numpy_input, hata durumunda ERROR loglar ve False döner.
        if not check_numpy_input(visual_input, expected_dtype=np.uint8, expected_ndim=(2, 3), input_name="visual_input for VisionProcessor", logger_instance=logger):
             # Geçersiz tip veya dtype/boyut ise işlemeyi durdur ve boş sözlük döndür.
             logger.error("VisionProcessor.process: Girdi numpy array değil veya yanlış dtype/boyut. Boş sözlük döndürülüyor.") # check_numpy_input zaten kendi içinde loglar.
             return {} # Boş sözlük döndür, None yerine.


        # DEBUG logu: Girdi detayları (boyutları ve tipi). Artık check_numpy_input içinde de benzer log var ama burada da kalabilir.
        logger.debug(f"VisionProcessor.process: Görsel veri alindi. Shape: {visual_input.shape}, Dtype: {visual_input.dtype}. İşleme yapılıyor.")

        processed_features = {} # İşlem sonucunu tutacak sözlük. Başlangıçta boş.

        try:
            # 1. Gri tonlamaya çevir (Eğer girdi renkli ise).
            # Girdinin shape'i (Yükseklik, Genişlik, Kanal) ve kanal sayısının 3 (BGR) olduğunu varsayıyoruz.
            # len(visual_input.shape) == 3 ve visual_input.shape[2] == 3 kontrolü yeterli.
            if len(visual_input.shape) == 3 and visual_input.shape[2] == 3:
                gray_frame = cv2.cvtColor(visual_input, cv2.COLOR_BGR2GRAY)
                logger.debug("VisionProcessor.process: Görsel veri BGR'den gri tonlamaya çevrildi.")
            elif len(visual_input.shape) == 2:
                # Eğer girdi zaten 2 boyutlu ise (gri gibi), doğrudan kullan.
                gray_frame = visual_input.copy() # Orijinal girdiyi değiştirmemek için kopya al.
                logger.debug("VisionProcessor.process: Görsel girdi zaten gri tonlama gibi. Çevirme atlandi.")
            else:
                 # check_numpy_input'ta ndim=(2,3) kontrolü yapıldı, buraya gelinmemeli ama sağlamlık için kalsın.
                 logger.warning(f"VisionProcessor.process: Beklenmeyen görsel girdi boyutu (ndim değil): {visual_input.shape}. İşlenemedi.")
                 return {} # Beklenmeyen boyutsa işleyemeyiz, boş sözlük döndür.

            # İşlem sonrası gri karenin hala doğru dtype (uint8) olduğundan emin olalım.
            if gray_frame.dtype != np.uint8:
                 gray_frame = gray_frame.astype(np.uint8)
                 logger.debug(f"VisionProcessor.process: Gri kare dtype uint8 yapildi.")


            # 2. Yeniden boyutlandır (Yapılandırmada belirtilen output_width x output_height boyutuna).
            # Hedef boyut tuple olarak verilir: (genişlik, yükseklik).
            # Interpolation metodu belirtilebilir, INTER_AREA küçültme için iyidir.
            # Hedef boyutların pozitif olduğu init'te kontrol edildi veya varsayılan atandı.
            resized_frame = cv2.resize(gray_frame, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)
            logger.debug(f"VisionProcessor.process: Görsel veri ({self.output_width}, {self.output_height}) boyutuna yeniden boyutlandırıldı. Shape: {resized_frame.shape}, Dtype: {resized_frame.dtype}")

            # İşlenmiş özellikler sözlüğüne gri tonlamalı, yeniden boyutlandırılmış kareyi ekle.
            processed_features['grayscale'] = resized_frame


            # 3. Kenar tespiti uygula.
            # Canny kenar dedektörü gri tonlamalı 8-bit (uint8) resimler üzerinde çalışır.
            # resized_frame uint8 ve gri olduğu için doğrudan kullanabiliriz.
            # Eşikler init'te yapılandırmadan alındı ve int olduğu kontrol edildi.
            edges = cv2.Canny(resized_frame, self.canny_low_threshold, self.canny_high_threshold)
            logger.debug(f"VisionProcessor.process: Canny kenar tespiti uygulandı. Shape: {edges.shape}, Dtype: {edges.dtype}")

            # Kenar haritasını işlenmiş özellikler sözlüğüne ekle.
            processed_features['edges'] = edges


            # TODO: Gelecekte: Daha fazla düşük seviye özellik ekle (örn: renk histogramları - orijinal renkli görüntüden, basit doku özellikleri).


        except cv2.Error as e:
            # OpenCV kütüphanesinden kaynaklanan spesifik hatalar (örn. yeniden boyutlandırma için geçersiz boyutlar).
            # Bu hatalar loglanır ve boş sözlük döndürülür.
            logger.error(f"VisionProcessor: OpenCV hatasi olustu isleme sirasinda: {e}", exc_info=True)
            # Hata durumunda processed_features sözlüğü kısmen dolu olsa bile,
            # bu kare için işlem başarısız kabul edilir ve boş sözlük döndürülür.
            return {}
        except Exception as e:
            # İşleme adımları sırasında oluşabilecek diğer beklenmedik hatalar (örn: numpy veya başka kütüphane hataları).
            # Bu hatalar loglanır ve boş sözlük döndürülür.
            logger.error(f"VisionProcessor: İşleme sırasında beklenmedik hata: {e}", exc_info=True)
            # Hata durumunda processed_features sözlüğü kısmen dolu olsa bile,
            # bu kare için işlem başarısız kabul edilir ve boş sözlük döndürülür.
            return {}

        # Başarılı durumda işlenmiş özellikler sözlüğünü döndür.
        # DEBUG logu: İşlem sonucunun başarıyla döndürüldüğü bilgisi.
        logger.debug(f"VisionProcessor.process: Görsel veri başarıyla işlendi. Çıktı Özellikleri: {list(processed_features.keys())}")
        return processed_features

    def cleanup(self):
        """
        VisionProcessor kaynaklarını temizler.

        Şimdilik bu işlemci özel bir kaynak (dosya, bağlantı vb.) kullanmadığı için
        temizleme adımı içermez, sadece bilgilendirme logu içerir.
        Gelecekte gerekirse kaynak temizleme mantığı buraya eklenebilir.
        module_loader.py bu metotu program sonlanırken çağırır (varsa).
        """
        logger.info("VisionProcessor objesi temizleniyor.")
        # İşlemci genellikle explicit temizlik gerektirmez.
        pass