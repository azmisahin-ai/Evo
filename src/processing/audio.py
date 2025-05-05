# src/processing/audio.py

import logging
import numpy as np
# Ses işleme için ek kütüphaneler gelecekte eklenebilir (SciPy, Librosa vb.)
# import scipy.signal

class AudioProcessor:
    """
    Evo'nın işitsel işlem birimini temsil eder.
    Ham ses veriden temel özellikleri çıkarmaktan sorumludur.
    """
    def __init__(self, config=None):
        logging.info("AudioProcessor başlatılıyor...")
        self.config = config if config is not None else {}

        # İşleme ayarları veya modelleri buradan yüklenebilir
        # Örneğin: self.sample_rate = self.config.get('sample_rate', 44100)

        logging.info("AudioProcessor başlatıldı.")

    def process(self, audio_data):
        """
        Ham ses veriyi (NumPy array - int16 olması beklenir) alır
        ve işlenmiş özellikleri (örneğin, enerji) döndürür.
        """
        if audio_data is None:
            logging.debug("AudioProcessor: İşlenecek ses verisi yok.") # DEBUG logu
            return None # Veri yoksa None döndür

        logging.debug(f"AudioProcessor: Ses verisi alindi. Shape: {audio_data.shape}, Dtype: {audio_data.dtype}") # DEBUG logu

        try:
            # --- Gerçek İşitsel İşleme Mantığı (Faz 1) ---

            # Ses chunk'ının enerji seviyesini hesapla (RMS - Root Mean Square)
            # Veri int16 olduğu için, önce float'a çevirmek daha iyi olabilir.
            if audio_data.dtype == np.int16:
                # int16 değerleri -32768 ile 32767 arasındadır. Normalize etmek gerekebilir.
                normalized_audio = audio_data.astype(np.float32) / 32768.0
            else:
                 normalized_audio = audio_data.astype(np.float32) # Zaten float ise

            logging.debug(f"AudioProcessor: Ses verisi float32'ye çevrildi. Shape: {normalized_audio.shape}") # DEBUG logu

            if normalized_audio.size == 0: # Boş array kontrolü
                 energy = 0.0
                 logging.debug("AudioProcessor: Boş ses chunk'ı, enerji 0.") # DEBUG logu
            else:
                 # RMS = sqrt(mean(square(samples)))
                 energy = np.sqrt(np.mean(np.square(normalized_audio)))
                 logging.debug(f"AudioProcessor: Ses enerjisi hesaplandı: {energy:.4f}") # DEBUG logu


            # --- Çıktı Formatı ---
            # Processorlar genellikle bir özellik vektörü veya daha karmaşık bir yapı döndürür.
            # Şimdilik sadece enerjiyi veya daha fazla özelliği içeren bir NumPy array döndürelim.
            # Gelecekte MFCC gibi daha karmaşık özellikler çıkarılacak.
            # processed_features = self._extract_mfcc(audio_data) # Örnek: MFCC

            # Şimdilik sadece enerjiyi döndür (float değeri)
            processed_data = energy # float değeri döndür


            logging.debug(f"AudioProcessor: Ses verisi işlendi. Output (Energy): {processed_data:.4f}") # DEBUG logu

            return processed_data # İşlenmiş veriyi (float veya NumPy array) döndür

        except Exception as e:
            logging.error(f"AudioProcessor sırasında hata oluştu: {e}", exc_info=True)
            return None # İşleme hatasında None döndür


    # Gelecekte MFCC gibi daha karmaşık özellik çıkarma metodu
    # def _extract_mfcc(self, audio_data):
    #     # librosa gibi kütüphaneler MFCC çıkarmak için kullanılabilir
    #     # mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=20)
    #     # return mfccs.T # Zaman serisi olarak döndürmek genellikle faydalıdır
    #     pass


    def __del__(self):
        """
        Nesne silindiğinde kaynakları temizler.
        """
        logging.info("AudioProcessor objesi silindi.")

# Modülü bağımsız test etmek için örnek kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print("AudioProcessor test ediliyor...")

    processor = AudioProcessor()

    # Sahte bir ses verisi oluştur (örneğin 1024 frame int16 değerleri)
    # Sessizlik (sıfırlar)
    dummy_chunk_silent = np.zeros(1024, dtype=np.int16)
    # Gürültülü (rastgele değerler)
    dummy_chunk_noisy = np.random.randint(-32768, 32767, size=1024, dtype=np.int16)

    print(f"Sahte sessiz girdi verisi oluşturuldu. Shape: {dummy_chunk_silent.shape}")
    processed_output_silent = processor.process(dummy_chunk_silent)
    if processed_output_silent is not None:
        print(f"İşlenmiş çıktı (sessiz): {processed_output_silent:.4f}")
        if processed_output_silent < 0.001: # Sıfıra yakın olmalı
            print("Processor sessiz girdiyi doğru işledi (düşük enerji).")
    else:
        print("Sessiz girdi işleme sonucu None döndü (hata oluştu).")

    print(f"\nSahte gürültülü girdi verisi oluşturuldu. Shape: {dummy_chunk_noisy.shape}")
    processed_output_noisy = processor.process(dummy_chunk_noisy)
    if processed_output_noisy is not None:
        print(f"İşlenmiş çıktı (gürültülü): {processed_output_noisy:.4f}")
        if processed_output_noisy > 0.1: # Yüksek bir enerji olmalı
             print("Processor gürültülü girdiyi doğru işledi (yüksek enerji).")
    else:
        print("Gürültülü girdi işleme sonucu None döndü (hata oluştu).")


    # None girdi ile dene
    print("\nNone girdi ile AudioProcessor testi:")
    processed_output_none = processor.process(None)
    if processed_output_none is None:
        print("None girdi ile işlem sonucu doğru şekilde None döndü.")
    else:
         print("None girdi ile işlem sonucu None dönmedi (beklenmeyen durum).")


    print("AudioProcessor testi bitti.")