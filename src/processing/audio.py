# src/processing/audio.py
#
# İşitsel duyu verisini işler.
# Ham ses verisinden (chunk) temel işitsel özellikleri (örn. enerji) çıkarır.

import numpy as np # Sayısal işlemler ve arrayler için.
import logging # Loglama için.
# import sys # Hata yönetimi veya stream handler için kullanılabilir. Şu an doğrudan kullanılmıyor.

# Yardımcı fonksiyonları import et (özellikle girdi kontrolleri için)
from src.core.utils import check_input_not_none, check_numpy_input # <<< Yeni import


# Bu modül için bir logger oluştur
# 'src.processing.audio' adında bir logger döndürür.
logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    İşitsel veriyi işleyen sınıf.

    AudioSensor'dan gelen ham ses girdisini (chunk) alır,
    üzerinde temel işlemler yaparak (şimdilik enerji hesaplama)
    RepresentationLearner için uygun hale getirir.
    İşleme sırasında oluşabilecek hataları yönetir.
    """
    def __init__(self, config):
        """
        AudioProcessor'ı başlatır.

        Args:
            config (dict): İşlemci yapılandırma ayarları.
                           'output_dim': İşlenmiş ses çıktısının boyutu (int, varsayılan 1).
                                         Şimdilik sadece enerji hesaplandığı için 1 beklenir.
                           Gelecekte 'output_features' gibi ayarlar eklenebilir.
        """
        self.config = config
        # Yapılandırmadan çıktı boyutunu al, yoksa varsayılanı kullan.
        self.output_dim = config.get('output_dim', 1) # Şu an sadece enerji için 1 bekleniyor

        logger.info("AudioProcessor başlatılıyor...")
        # Ek yapılandırma veya model yükleme buraya gelebilir (ileride).
        # Örneğin, bir ses özellik çıkarma modeli yüklenebilir.
        # logger.info(f"AudioProcessor çıktı boyutu ayarlandı: {self.output_dim}") # output_dim şu an sadece 1
        logger.info("AudioProcessor başlatıldı.")

    def process(self, audio_input):
        """
        Ham işitsel girdiyi işler (örn. enerji hesaplama).

        Girdiyi alır (genellikle int16 numpy array), enerji gibi temel bir özellik hesaplar
        ve RepresentationLearner'a gönderilmek üzere işlenmiş özelliği (veya özellikleri) döndürür.
        Girdi None ise veya işleme sırasında hata oluşursa None döndürür.

        Args:
            audio_input (numpy.ndarray or None): Ham işitsel veri (chunk) veya None.
                                                  Genellikle AudioSensor'dan gelir.
                                                  Beklenen format: shape (N,), dtype int16.

        Returns:
            float or None: Hesaplanan ses enerjisi (şimdilik tek bir float değer)
                           veya hata durumunda veya girdi None ise None.
                           (Gelecekte: numpy array of features)
        """
        # Hata yönetimi: Girdi None ise veya beklenen tipte değilse
        # check_input_not_none fonksiyonunu kullan (None ise loglar ve False döner)
        if not check_input_not_none(audio_input, input_name="audio_input", logger_instance=logger):
             return None # Girdi None ise işlemeyi atla ve None döndür.

        # Girdinin numpy array ve sayısal bir dtype olup olmadığını kontrol et.
        # check_numpy_input fonksiyonunu kullan. int16 bekleniyor, np.number daha genel.
        # expected_ndim=1 çünkü chunk 1D array bekleniyor.
        # check_numpy_input, hata durumunda ERROR loglar ve False döner.
        if not check_numpy_input(audio_input, expected_dtype=np.number, expected_ndim=1, input_name="audio_input", logger_instance=logger):
             return None # Geçersiz tip, dtype veya boyut ise işlemeyi durdur ve None döndür.

        # DEBUG logu: Girdi detayları (boyutları ve tipi). Artık check_numpy_input içinde de benzer log var ama burada da kalabilir.
        # logger.debug(f"AudioProcessor: Ses verisi alindi. Shape: {audio_input.shape}, Dtype: {audio_input.dtype}")

        processed_feature = None # İşlem sonucunu (şimdilik enerji) tutacak değişken.

        try:
            # 1. Veriyi float'a çevir (Hesaplamalar için).
            # int16 değerleri [-32768, 32767] aralığındadır. float32'ye çeviriyoruz.
            audio_float = audio_input.astype(np.float32)
            # logger.debug(f"AudioProcessor: Ses verisi float32'ye çevrildi. Shape: {audio_float.shape}")

            # 2. Ses enerjisini hesapla (Örnek: RMS - Kök Ortalama Kare).
            # Enerji = sqrt(mean(samples^2))
            # Boş chunk gelirse np.mean hata verebilir, o yüzden kontrol et.
            if audio_float.size > 0:
                 # RMS hesaplama: Karelerin ortalamasının karekökü.
                 energy = np.sqrt(np.mean(audio_float**2))

                 # Hesaplanan özelliği çıktı boyutu ile uyumlu hale getir.
                 # Şu an output_dim 1 olduğu için doğrudan enerji değerini döndürüyoruz.
                 processed_feature = energy
                 # logger.debug(f"AudioProcessor: Ses enerjisi hesaplandı: {processed_feature:.4f}")

            else:
                 # Boş chunk gelirse enerji 0 olarak kabul edilebilir.
                 processed_feature = 0.0
                 logger.debug("AudioProcessor: Boş ses chunk'i alindi. Enerji 0 olarak ayarlandi.")


        except Exception as e:
            # İşleme adımları sırasında oluşabilecek beklenmedik hatalar (örn. np.mean veya np.sqrt gibi matematiksel hatalar).
            logger.error(f"AudioProcessor: İşleme sırasında beklenmedik hata: {e}", exc_info=True)
            return None # Hata durumunda None döndür.

        # Başarılı durumda işlenmiş özelliği döndür.
        # logger.debug(f"AudioProcessor: Ses verisi başarıyla işlendi. Output (Energy): {processed_feature:.4f}")
        return processed_feature

    def cleanup(self):
        """
        AudioProcessor kaynaklarını temizler.

        Şimdilik bu işlemci özel bir kaynak (dosya, bağlantı vb.) kullanmadığı için
        temizleme adımı içermez, sadece bilgilendirme logu içerir.
        Gelecekte gerekirse kaynak temizleme mantığı buraya eklenebilir.
        module_loader.py bu metodu program sonlanırken çağırır (varsa).
        """
        logger.info("AudioProcessor objesi temizleniyor.")
        pass # İşlemci genellikle explicit temizlik gerektirmez.