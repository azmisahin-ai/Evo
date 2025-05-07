# src/processing/audio.py
#
# İşitsel duyu verisini işler.
# Ham ses verisinden (chunk) temel işitsel özellikleri (örn. enerji, frekans) çıkarır.
# Evo'nın Faz 1'deki işleme yeteneklerinin bir parçasıdır.

import numpy as np # Sayısal işlemler ve arrayler için.
import logging # Loglama için.

# Yardımcı fonksiyonları import et (özellikle girdi kontrolleri ve config için)
from src.core.config_utils import get_config_value
from src.core.utils import check_input_not_none, check_numpy_input # <<< Utils importları


# Bu modül için bir logger oluştur
# 'src.processing.audio' adında bir logger döndürür.
logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Evo'nın işitsel veriyi işleyen sınıfı (Faz 1 implementasyonu).

    AudioSensor'dan gelen ham ses girdisini (chunk) alır,
    üzerinde temel işlemler yaparak (enerji, Spectral Centroid hesaplama)
    RepresentationLearner için uygun hale getirir.
    İşleme sırasında oluşabilecek hataları yönetir ve akışın devamlılığını sağlar.
    Çıktı olarak temel işitsel özellikleri içeren bir numpy array döndürür.
    """
    def __init__(self, config):
        """
        AudioProcessor'ı başlatır.

        Args:
            config (dict): İş işlemci yapılandırma ayarları.
                           'audio_rate': Ses örnekleme oranı (int, varsayılan 44100 Hz).
                                         Spectral Centroid hesaplaması için gereklidir.
                           'output_dim': İşlenmiş ses çıktısının boyutu (int, varsayılan 2).
                                         Şimdilik enerji ve Spectral Centroid için 2 beklenir.
                                         Gelecekte farklı özellikler için bu sayı artabilir.
        """
        self.config = config
        logger.info("AudioProcessor başlatılıyor...")

        # Yapılandırmadan örnekleme oranını ve çıktı boyutunu alırken get_config_value kullan.
        self.audio_rate = get_config_value(config, 'audio_rate', 44100, expected_type=int, logger_instance=logger)
        # Artık enerji ve Spectral Centroid döndüreceğimiz için varsayılan output_dim 2.
        self.output_dim = get_config_value(config, 'output_dim', 2, expected_type=int, logger_instance=logger)

        # output_dim kontrolü (gelecekte birden fazla özellik dönerse anlamlı olacak)
        # Şu an enerji ve Spectral Centroid olmak üzere 2 özellik döndürüyoruz.
        if self.output_dim != 2:
             logger.warning(f"AudioProcessor: Konfigurasyonda output_dim beklenenden farklı ({self.output_dim}). Implementasyon 2 özellik döndürüyor (Enerji, Spectral Centroid).")
             # Bu uyarı, config'in implementasyonla uyumlu olması gerektiğini hatırlatır.
             # RepresentationLearner config'i bu output boyutuna göre ayarlanmalıdır.


        logger.info(f"AudioProcessor başlatıldı. Örnekleme Oranı: {self.audio_rate} Hz, Çıktı Boyutu (implemente edilen): {self.output_dim}")


    def process(self, audio_input):
        """
        Ham işitsel girdiyi işler, temel işitsel özellikleri çıkarır.

        Girdiyi alır (genellikle int16 numpy array), enerji ve Spectral Centroid gibi
        temel özellikleri hesaplar. Bu özellikleri içeren bir numpy array döndürür.
        Girdi None ise veya işleme sırasında hata oluşursa None döndürür.

        Args:
            audio_input (numpy.ndarray or None): Ham işitsel veri (chunk) veya None.
                                                  Genellikle AudioSensor'dan gelir.
                                                  Beklenen format: shape (N,), dtype int16.

        Returns:
            numpy.ndarray or None: Hesaplanan özellik vektörü (shape (output_dim,), dtype float32)
                                   veya hata durumunda veya girdi None ise None.
        """
        # Hata yönetimi: Girdi None ise veya beklenen tipte değilse
        # check_input_not_none fonksiyonunu kullan (None ise loglar ve False döner)
        if not check_input_not_none(audio_input, input_name="audio_input for AudioProcessor", logger_instance=logger):
             logger.debug("AudioProcessor.process: Girdi None. None döndürülüyor.")
             return None # Girdi None ise işlemeyi atla ve None döndür.

        # Girdinin numpy array ve doğru dtype (int16) olup olmadığını kontrol et.
        # check_numpy_input fonksiyonunu kullan. Bu fonksiyon aynı zamanda np.ndarray kontrolü de yapar.
        # Expected_ndim=1 çünkü chunk 1D array bekleniyor. dtype int16 bekleniyor.
        # check_numpy_input, hata durumunda ERROR loglar ve False döner.
        if not check_numpy_input(audio_input, expected_dtype=np.int16, expected_ndim=1, input_name="audio_input for AudioProcessor", logger_instance=logger):
             logger.error("AudioProcessor.process: Girdi numpy array değil veya yanlış dtype/boyut. None döndürülüyor.") # check_numpy_input zaten kendi içinde loglar.
             return None # Geçersiz tip, dtype veya boyut ise işlemeyi durdur ve None döndür.

        # Boş chunk (audio_input.size == 0) gelirse işleme yapmadan None döndür.
        if audio_input.size == 0:
             logger.debug("AudioProcessor.process: Boş ses chunk'i alindi. İşleme atlandi, None döndürülüyor.")
             return None


        # DEBUG logu: Girdi detayları (boyutları ve tipi). Artık check_numpy_input içinde de benzer log var.
        logger.debug(f"AudioProcessor.process: Ses verisi alindi. Shape: {audio_input.shape}, Dtype: {audio_input.dtype}. İşleme yapılıyor.")

        energy = 0.0 # Enerji değerini tutacak değişken. Başlangıçta 0.
        spectral_centroid = 0.0 # Spectral Centroid değerini tutacak değişken. Başlangıçta 0.
        processed_features_vector = None # Döndürülecek özellik vektörü.

        try:
            # 1. Veriyi float'a çevir (Hesaplamalar için).
            # int16 değerleri [-32768, 32767] aralığındadır. float32'ye çeviriyoruz.
            # Normalizasyon (-1.0 ile 1.0 arasına) daha iyi olabilir (Gelecek TODO)
            audio_float = audio_input.astype(np.float32)
            # logger.debug(f"AudioProcessor.process: Ses verisi float32'ye çevrildi. Shape: {audio_float.shape}")

            # 2. Ses enerjisini hesapla (Örnek: RMS).
            # Boş chunk gelirse np.mean hata verebilir, ancak size kontrolü yukarıda yapıldı.
            energy = np.sqrt(np.mean(audio_float**2)) if audio_float.size > 0 else 0.0
            #logger.debug(f"AudioProcessor.process: Ses enerjisi hesaplandı: {energy:.4f}") # Loglama artık vektör logunda yapılacak


            # 3. Spectral Centroid hesapla.
            # Spectral Centroid = sum(frequencies * magnitudes) / sum(magnitudes)
            # a) Pencereleme uygula (örn: Hanning penceresi)
            window = np.hanning(len(audio_float))
            audio_windowed = audio_float * window
            # logger.debug("AudioProcessor.process: Hanning penceresi uygulandı.")

            # b) FFT (Hızlı Fourier Dönüşümü) uygula
            fft_result = np.fft.fft(audio_windowed)
            # logger.debug(f"AudioProcessor.process: FFT uygulandı. Çıktı Shape: {fft_result.shape}")

            # c) Genlik spektrumunu al (karmaşık sayılardan mutlak değeri)
            magnitude_spectrum = np.abs(fft_result)
            # logger.debug(f"AudioProcessor.process: Genlik spektrumu hesaplandı. Shape: {magnitude_spectrum.shape}")

            # d) Tek taraflı spektrumu al (Nyquist'e kadar olan kısım)
            # Gerçek sinyal
            # ler için spektrum simetriktir. İlk yarısı yeterlidir.
            # Eğer chunk boyutu N ise, N/2+1 boyutunda olur (DC ve Nyquist dahil).
            single_sided_spectrum = magnitude_spectrum[:len(magnitude_spectrum)//2 + 1]
            # logger.debug(f"AudioProcessor.process: Tek taraflı spektrum alındı. Shape: {single_sided_spectrum.shape}")

            # e) Frekans eksenini oluştur (0'dan Nyquist frekansına kadar)
            # Nyquist frekansı = audio_rate / 2.
            # Freq bins sayısı = len(single_sided_spectrum).
            # Endpoint=True, Nyquist frekansını dahil etmek için.
            frequencies = np.linspace(0, self.audio_rate / 2, len(single_sided_spectrum))
            # logger.debug(f"AudioProcessor.process: Frekans ekseni oluşturuldu. Shape: {frequencies.shape}")

            # f) Spectral Centroid'i hesapla
            # Payda (genliklerin toplamı) sıfırsa bölme hatası olmaması için kontrol et.
            # Bu durum genellikle tamamen sessiz bir chunk geldiğinde olur.
            sum_magnitudes = np.sum(single_sided_spectrum)
            if sum_magnitudes > 1e-6: # Küçük bir eşik kullanmak float hatalarını önler
                 spectral_centroid = np.sum(frequencies * single_sided_spectrum) / sum_magnitudes
                 #logger.debug(f"AudioProcessor.process: Spectral Centroid hesaplandı: {spectral_centroid:.4f}") # Loglama artık vektör logunda yapılacak
            else:
                 # Sessiz chunk veya sıfır toplam genlik durumunda centroid'i 0 olarak ayarla.
                 spectral_centroid = 0.0
                 logger.debug("AudioProcessor.process: Toplam genlik sıfıra yakın, Spectral Centroid 0 olarak ayarlandı.")

            # TODO: Gelecekte: Daha fazla özellik ekle (örn: Spectral Spread, Spectral Flux, MFCC).
            # Bu durumda processed_features_vector'a yeni elemanlar eklenecek ve output_dim artırılacak.

            # 4. Çıkarılan özellikleri bir numpy array'de topla.
            # Şu an enerji ve Spectral Centroid'i topluyoruz.
            # Bu, RepresentationLearner'ın bekleyeceği 1D özellik vektörüdür.
            processed_features_vector = np.array([energy, spectral_centroid], dtype=np.float32)

            # Kontrol: Üretilen vektör boyutu beklenene eşit mi?
            if processed_features_vector.shape[0] != self.output_dim:
                 logger.warning(f"AudioProcessor.process: Üretilen özellik vektör boyutu config'teki output_dim ile eşleşmiyor: {processed_features_vector.shape[0]} != {self.output_dim}. Lütfen config dosyasını ve implementasyonu kontrol edin. RepresentationLearner'ın input_dim'i bu boyuta göre ayarlanmalıdır.")
                 # Bu bir hata değil, sadece bir uyarı. Implementasyonumuz (2 özellik) config'ten (output_dim) farklıysa logluyoruz.
                 # RepresentationLearner config'i (input_dim) bu boyuta göre ayarlanmalıdır.


        except Exception as e:
            # İşleme adımları sırasında oluşabilecek beklenmedik hatalar (örn. numpy, fft hataları).
            logger.error(f"AudioProcessor: İşleme sırasında beklenmedik hata: {e}", exc_info=True)
            return None # Hata durumunda None döndür.

        # Başarılı durumda işlenmiş özellik vektörünü döndür.
        logger.debug(f"AudioProcessor.process: Output Shape: {processed_features_vector.shape}, Dtype: {processed_features_vector.dtype}. Değerler (Enerji, Centroid): {processed_features_vector}")
        return processed_features_vector

    def cleanup(self):
        """
        AudioProcessor kaynaklarını temizler.

        Şimdilik bu işlemci özel bir kaynak (dosya, bağlantı vb.) kullanmadığı için
        temizleme adımı içermez, sadece bilgilendirme logu içerir.
        Gelecekte gerekirse kaynak temizleme mantığı buraya eklenebilir.
        module_loader.py bu metotu program sonlanırken çağırır (varsa).
        """
        logger.info("AudioProcessor objesi temizleniyor.")
        pass # İşlemci genellikle explicit temizlik gerektirmez.