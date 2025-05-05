# src/processing/audio.py
import numpy as np
import logging
import sys # Ses enerjisi hesaplama için kullanabiliriz (isteğe bağlı)

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    İşitsel veriyi işler (örn. enerji hesaplama, özellik çıkarma).
    """
    def __init__(self, config):
        self.config = config
        self.output_dim = config.get('output_dim', 1) # Şu an sadece enerji
        # Gelecekte diğer özellikler için output_features gibi ayarlar kullanılabilir.
        logger.info("AudioProcessor başlatılıyor...")
        # Ek yapılandırma veya model yükleme buraya gelebilir
        logger.info("AudioProcessor başlatıldı.")

    def process(self, audio_input):
        """
        Ham işitsel girdiyi işler (örn. enerji hesaplama).

        Args:
            audio_input (numpy.ndarray or None): Ham işitsel veri (chunk) veya None.

        Returns:
            float or None: Hesaplanan ses enerjisi veya hata durumunda veya girdi None ise None.
                           (Gelecekte numpy array of features)
        """
        # Temel hata yönetimi: Girdi None ise veya beklenen tipte değilse
        if audio_input is None:
            logger.debug("AudioProcessor: Girdi None. İşleme atlanıyor.")
            return None # Hata durumunda None döndür

        # Girdinin numpy array ve sayısal tipte olduğunu kontrol et (int16 bekleniyor)
        if not isinstance(audio_input, np.ndarray) or not np.issubdtype(audio_input.dtype, np.number):
             logger.error(f"AudioProcessor: Beklenmeyen girdi tipi veya dtype: {type(audio_input)}, {audio_input.dtype}. numpy.ndarray (sayısal) bekleniyordu.")
             return None

        # DEBUG logu: Girdi detayları
        # logger.debug(f"AudioProcessor: Ses verisi alindi. Shape: {audio_input.shape}, Dtype: {audio_input.dtype}")

        processed_feature = None # İşlem sonucunu tutacak değişken (şimdilik float enerji)

        try:
            # 1. Veriyi float'a çevir (hesaplamalar için)
            # int16 değerleri -32768 ile 32767 arasındadır. Normalizasyon veya float'a çevirme önemlidir.
            # Maksimum olası değere bölerek -1.0 ile 1.0 arasına normalleştirebiliriz.
            # Şu anki energy hesaplaması için doğrudan float'a çevirme yeterli.
            audio_float = audio_input.astype(np.float32)
            # logger.debug(f"AudioProcessor: Ses verisi float32'ye çevrildi. Shape: {audio_float.shape}")


            # 2. Ses enerjisini hesapla (Örnek: RMS - Kök Ortalama Kare)
            # Enerji = sqrt(mean(samples^2))
            if audio_float.size > 0: # Boş chunk gelirse hata vermemesi için kontrol
                 # RMS hesaplama
                 energy = np.sqrt(np.mean(audio_float**2))
                 # Alternatif: Sadece mutlak değerlerin ortalaması
                 # energy = np.mean(np.abs(audio_float))

                 # Hesaplanan enerjiyi çıktı boyutu ile uyumlu hale getir
                 # Şu an output_dim 1 olduğu için doğrudan enerji değerini döndürüyoruz.
                 processed_feature = energy
                 # logger.debug(f"AudioProcessor: Ses enerjisi hesaplandı: {processed_feature:.4f}")

            else:
                 # Boş chunk gelirse enerji 0 olarak kabul edilebilir.
                 processed_feature = 0.0
                 logger.debug("AudioProcessor: Boş ses chunk'i alindi. Enerji 0.")


        except Exception as e:
            # İşleme sırasında beklenmedik hatalar (örn. matematiksel hata)
            logger.error(f"AudioProcessor: İşleme sırasında beklenmedik hata: {e}", exc_info=True)
            return None # Hata durumunda None döndür

        # Başarılı durumda işlenmiş özelliği döndür
        # logger.debug(f"AudioProcessor: Ses verisi işlendi. Output (Energy): {processed_feature:.4f}")
        return processed_feature

    def cleanup(self):
        """Kaynakları temizler (şimdilik gerek yok, placeholder)."""
        logger.info("AudioProcessor objesi silindi.")
        pass # İşlemci genellikle temizlik gerektirmez