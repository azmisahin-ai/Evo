# src/senses/audio.py
import pyaudio
import numpy as np
import time # Simülasyon için
import logging # Loglama için

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

class AudioSensor:
    """
    Evo'nun işitsel duyu organı. Mikrofon akışından ses chunk'ları yakalar.
    """
    def __init__(self, config):
        self.config = config
        self.audio_rate = config.get('audio_rate', 44100)
        self.audio_chunk_size = config.get('audio_chunk_size', 1024)
        self.audio_format = pyaudio.paInt16 # 16-bit integer formatı yaygın ve basit
        self.audio_channels = 1 # Mono ses
        self.audio_input_device_index = config.get('audio_input_device_index') # None varsayılanı kullanır

        self.p = None # PyAudio objesi
        self.stream = None # PyAudio stream objesi
        self.is_audio_available = False # Ses akışının aktif olup olmadığını tutar

        logger.info("AudioSensor başlatılıyor...")
        try:
            self.p = pyaudio.PyAudio()

            # Varsayılan input cihazını bulmaya çalış eğer config'te belirtilmemişse
            if self.audio_input_device_index is None:
                 try:
                      default_device = self.p.getDefaultInputDevice()
                      self.audio_input_device_index = default_device['index']
                      logger.info(f"Varsayılan ses input cihazı bulundu: {default_device['name']} (Index: {self.audio_input_device_index})")
                 except Exception:
                      # Varsayılan cihaz bulunamazsa, cihaz indeksi None kalır ve PyAudio kendi varsayılanını kullanır veya hata verir
                      logger.warning("Varsayılan ses input cihazı bulunamadı. PyAudio varsayılanını kullanmaya çalışılacak.")
                      # Index None kalırsa open stream'de exception verebilir.
                      # Bu durumda PyAudio'nun davranışı sistem bağımlıdır.
                      # En güvenlisi: Default bulunamazsa None bırakmak ve open stream'in hatasını yakalamak.


            # Ses akışını başlat
            self.stream = self.p.open(format=self.audio_format,
                                      channels=self.audio_channels,
                                      rate=self.audio_rate,
                                      input=True,
                                      input_device_index=self.audio_input_device_index, # Config'ten veya bulunan index
                                      frames_per_buffer=self.audio_chunk_size)

            self.is_audio_available = True
            logger.info(f"AudioSensor başarıyla başlatıldı. Cihaz: {self.audio_input_device_index}, Rate: {self.audio_rate}, Chunk: {self.audio_chunk_size}")

        except Exception as e:
            logger.error(f"AudioSensor başlatılırken hata oluştu: {e}", exc_info=True)
            self.is_audio_available = False
            # Hata durumunda objeleri None yap
            if self.stream: self.stream.stop_stream(); self.stream.close()
            if self.p: self.p.terminate()
            self.stream = None
            self.p = None


        logger.info(f"AudioSensor başlatıldı. Ses aktif: {self.is_audio_available}")

    def capture_chunk(self):
        """
        Ses akışından bir chunk (parça) yakalar veya simüle edilmiş veri döndürür.

        Returns:
            numpy.ndarray or None: Yakalanan ses verisi (numpy array) veya
                                    hata durumunda veya akış aktif değilse None.
        """
        if self.is_audio_available and self.stream is not None:
            try:
                # Akıştan veri oku (non-blocking veya blocking olabilir, read blocking)
                # read(self.audio_chunk_size) blocking'tir ve chunk boyutu kadar veri geldiğinde döner.
                # Eğer timeout olursa IOError verir.
                # Timeout'u kısa tutmak gerekebilir veya non-blocking read kullanmak.
                # Şimdilik basit blocking read kullanalım ve timeout'u exception'da yakalayalım.
                data = self.stream.read(self.audio_chunk_size)

                # Okunan byte verisini numpy array'e dönüştür
                # np.frombuffer, byte'ları belirtilen dtype'a çevirir.
                audio_chunk = np.frombuffer(data, dtype=np.int16)

                # logger.debug(f"AudioSensor: Gerçek chunk yakalandı. Shape: {audio_chunk.shape}, Dtype: {audio_chunk.dtype}")
                return audio_chunk # Başarılı durumda chunk'ı döndür

            except IOError as e:
                 # Timeout veya akış hatası gibi PyAudio okuma hataları
                 logger.error(f"AudioSensor: Ses chunk okuma hatası (IOError): {e}", exc_info=True)
                 # Bu tür bir hata akışın durduğu anlamına gelebilir.
                 self.is_audio_available = False # Geçici olarak kapat
                 return None # Hata durumunda None döndür
            except Exception as e:
                # Okuma sırasında beklenmedik hata
                logger.error(f"AudioSensor: Ses chunk yakalama sırasında beklenmedik hata: {e}", exc_info=True)
                self.is_audio_available = False # Hata durumunda kapat
                return None # Hata durumunda None döndür
        else:
            # Ses akışı aktif değilse veya stream None ise simüle edilmiş veri döndür
            # logger.debug("AudioSensor: Simüle edilmiş chunk üretiliyor.")
            # Simüle edilmiş sessiz (sıfır) int16 formatında chunk döndür
            dummy_chunk = np.zeros(self.audio_chunk_size, dtype=np.int16)
            # logger.debug(f"AudioSensor: Simüle edilmiş chunk üretildi. Shape: {dummy_chunk.shape}, Dtype: {dummy_chunk.dtype}")
            return dummy_chunk

    def stop_stream(self):
        """
        Ses akışını durdurur ve kaynakları serbest bırakır.
        """
        if self.stream is not None:
            logger.info("İşitsel akış durduruluyor...")
            try:
                 self.stream.stop_stream()
                 logger.info("Ses akışı durduruldu.")
                 self.stream.close()
                 logger.info("Ses akışı kapatıldı.")
            except Exception as e:
                 logger.error(f"AudioSensor: Ses akışı durdurulurken hata oluştu: {e}", exc_info=True)
            self.stream = None
        else:
            logger.info("İşitsel akış zaten durdurulmuş veya hiç açılamamış.")

        if self.p is not None:
            try:
                 self.p.terminate()
                 logger.info("PyAudio sonlandırıldı.")
            except Exception as e:
                 logger.error(f"AudioSensor: PyAudio sonlandırılırken hata oluştu: {e}", exc_info=True)
            self.p = None

        self.is_audio_available = False

# Örnek Kullanım (run_evo.py'ye taşındı)
# if __name__ == '__main__':
#     # Basit test (setup_logging henüz burada çağrılmadığı için loglar tam görünmeyebilir)
#     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     test_config = {'audio_rate': 16000, 'audio_chunk_size': 512}
#     audio_sensor = AudioSensor(test_config)
#     print("Ses yakalamaya başlanıyor (5 chunk)...")
#     for _ in range(5):
#         chunk = audio_sensor.capture_chunk()
#         if chunk is not None:
#             print(f"Captured audio chunk with shape: {chunk.shape}, dtype: {chunk.dtype}")
#         else:
#             print("Failed to capture audio chunk.")
#         time.sleep(0.1) # Küçük bir bekleme

#     audio_sensor.stop_stream()
#     print("AudioSensor testi tamamlandı.")