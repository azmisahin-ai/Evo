# src/senses/audio.py
#
# Evo'nın işitsel duyu organını temsil eder.
# Mikrofon akışından ham ses parçalarını (chunk) yakalar.
# Mikrofon kullanılamadığında simüle edilmiş girdi sağlar.

import pyaudio # PyAudio kütüphanesi, ses akışı yönetimi için. requirements.txt'e eklenmeli.
import numpy as np # Ses verisi (örnekler dizisi) için.
import time # Simülasyon veya zamanlama için. Şu an doğrudan kullanılmıyor.
import logging # Loglama için.
# import sys # Hata yönetimi veya stream handler için kullanılabilir. Şu an doğrudan kullanılmıyor.

# Yardımcı fonksiyonları import et
from src.core.config_utils import get_config_value # <<< get_config_value import edildi
from src.core.utils import run_safely # run_safely import edildi


# Bu modül için bir logger oluştur
# 'src.senses.audio' adında bir logger döndürür.
logger = logging.getLogger(__name__)


class AudioSensor:
    """
    Evo'nın işitsel duyu organı sınıfı.

    Mikrofon akışından belirli boyutlarda (chunk) ses verisi yakalamayı dener.
    Eğer mikrofon başlatılamazsa veya yakalama sırasında hata oluşursa,
    simüle edilmiş (dummy) ses verisi döndürür.
    Ses akışı durumu (aktif olup olmadığı) takip edilir.
    """
    def __init__(self, config):
        """
        AudioSensor'ı başlatır.

        PyAudio'yu başlatır, belirtilen cihazdan (veya varsayılandan) ses akışını açar.
        Başlatma sırasında oluşabilecek hataları yönetir.

        Args:
            config (dict): Sensör yapılandırma ayarları.
                           'audio_rate': Ses örnekleme oranı (int, varsayılan 44100 Hz).
                           'audio_chunk_size': Her yakalamada alınacak ses örneği sayısı (int, varsayılan 1024).
                           'audio_input_device_index': Kullanılacak ses cihazının indeksi (int veya None, varsayılan None).
                                                       None ise varsayılan cihaz kullanılır.
                           'is_dummy': Simüle mod etkin mi (bool, varsayılan False).
        """
        self.config = config
        logger.info("AudioSensor initializing...")

        # Get configuration settings using get_config_value with keyword arguments
        # These settings are under the 'audio' key in the config.
        # Corrected: Use default= keyword format for all calls.
        self.audio_rate = get_config_value(config, 'audio', 'audio_rate', default=44100, expected_type=int, logger_instance=logger)
        self.audio_chunk_size = get_config_value(config, 'audio', 'audio_chunk_size', default=1024, expected_type=int, logger_instance=logger)
        self.audio_input_device_index = get_config_value(config, 'audio', 'audio_input_device_index', default=None, expected_type=(int, type(None)), logger_instance=logger)
        # Corrected: is_dummy config is under 'audio' key.
        self.is_dummy = get_config_value(config, 'audio', 'is_dummy', default=False, expected_type=bool, logger_instance=logger)


        self.audio_format = pyaudio.paInt16 # 16-bit integer format is common and simple
        self.audio_channels = 1 # Mono audio


        self.p = None # PyAudio object. Assigned if PyAudio is initialized.
        self.stream = None # PyAudio Stream object. Assigned if audio stream is opened.
        self.is_audio_available = False # Tracks if the real audio stream is currently active. Starts as False.

        # If is_dummy is True, don't try to initialize the real microphone
        if self.is_dummy:
             logger.info("AudioSensor: is_dummy=True. Using simulated audio input.")
             self.is_audio_available = False
             self.p = None # PyAudio object should remain None
             self.stream = None # Stream object should remain None
        else:
            # Try to initialize the real microphone
            try:
                # Initialize the PyAudio object.
                self.p = pyaudio.PyAudio()

                # If a specific device index is not specified in config (is None)
                device_index_to_open = self.audio_input_device_index # Initial value comes from config
                if device_index_to_open is None:
                     try:
                          # Get PyAudio's default input device.
                          # This line might fail if PyAudio setup is incomplete or no device exists.
                          default_device_info = self.p.getDefaultInputDeviceInfo()
                          # Use the default device's index.
                          device_index_to_open = default_device_info['index']
                          # Log the default device's name and index.
                          logger.info(f"AudioSensor: Found default audio input device: {default_device_info['name']} (Index: {device_index_to_open})")
                     except Exception as e:
                          # Log a warning if the default device cannot be found.
                          logger.warning(f"AudioSensor: Default audio input device not found: {e}. Attempting to open stream with device_index=None.")
                          # If index remains None, PyAudio's open stream method might still attempt to use the default.
                          # Any error during open will be caught in the try-except block below.


                # Use PyAudio's open method to start the audio stream.
                # format=pyaudio.paInt16, channels=1 are common settings.
                # frames_per_buffer should match chunk_size.
                # input=True for input stream.
                # The ValueError: input_device_index must be integer (or None) error seen in logs suggests
                # that device_index_to_open ended up being something other than an integer or None.
                # Since get_config_value is now fixed and checks expected_type=(int, type(None)),
                # this error is likely due to the PyAudio environment setup or a transient issue,
                # not the get_config_value call itself anymore.
                self.stream = self.p.open(format=self.audio_format,
                                          channels=self.audio_channels,
                                          rate=self.audio_rate, # Retrieved from config as int
                                          input=True,
                                          input_device_index=device_index_to_open, # The determined (from config or default) device index (int or None)
                                          frames_per_buffer=self.audio_chunk_size) # Retrieved from config as int


                # If the stream object was created successfully (is not None)
                if self.stream is not None:
                     logger.info(f"AudioSensor: Stream started successfully. Device: {device_index_to_open}, Rate: {self.audio_rate}, Chunk: {self.audio_chunk_size}")
                     self.is_audio_available = True # Stream is active.
                     # Can also check self.stream.is_active() here if needed.
                     # if self.stream.is_active(): logger.info("Audio stream is active.")

                else:
                     # If the stream object could not be created (e.g., if open returned None, which is unlikely for PyAudio on failure)
                     # Exceptions from open() are caught below. This else block might be redundant.
                     logger.warning(f"AudioSensor: Audio stream could not be started (open returned None). Using simulated audio input.")
                     self.is_audio_available = False
                     # Clean up the PyAudio instance on failure.
                     if self.p:
                         try:
                              self.p.terminate()
                         except Exception: pass # Ignore errors during terminate
                     self.p = None # Set PyAudio instance to None.
                     # Note: __init__ cannot return None. Initialization failure should be handled by setting state (is_audio_available = False)
                     # and potentially logging, letting the caller (run_module_test or module_loader) handle the failure based on state/exceptions.


            except Exception as e:
                # Catch any unexpected exceptions during PyAudio initialization or opening the stream.
                # This exception will be caught by run_module_test's try block, leading to test failure reporting.
                logger.error(f"AudioSensor initialization failed: {e}", exc_info=True)
                self.is_audio_available = False # Set the flag to False in case of error.
                # Try to clean up any resources that might have been opened.
                if self.stream:
                     try:
                          self.stream.stop_stream()
                          self.stream.close()
                     except Exception: pass # Ignore errors during cleanup
                self.stream = None
                if self.p:
                     try:
                          self.p.terminate()
                     except Exception: pass # Ignore errors during terminate
                self.p = None
                # __init__ method itself does not raise exceptions, it logs and sets state.
                # The exception is raised by the PyAudio calls within the try block.


        # Log the result of the initialization process.
        # The status (active or not based on is_audio_available state) is logged.
        logger.info(f"AudioSensor initialized. Audio active: {self.is_audio_available}, Simulated Mode: {self.is_dummy}")


    # ... (capture_chunk, stop_stream, terminate_pyaudio, cleanup methods - same as before) ...

    def capture_chunk(self):
        """
        Mikrofon akışından bir chunk (parça) ses verisi yakalar.

        Eğer gerçek ses akışı aktifse, PyAudio akış objesinden bir blok okur
        ve bu bayt veriyi numpy array'e çevirir.
        Eğer gerçek ses akışı aktif değilse (başlatılamadı veya hata oluştu),
        yapılandırmada belirtilen boyutta simüle edilmiş (dummy) bir ses bloğu döndürür.
        Blok yakalama sırasında hata oluşursa (gerçek mikrofondan), hatayı loglar
        ve simüle moda geçerek simüle ses bloğu döndürür.

        Returns:
            numpy.ndarray: Başarıyla yakalanan (gerçek veya simüle) ses chunk'ı
                                    (dtype int16). Hata durumunda None dönmez.
        """
        # Eğer simüle mod aktifse veya gerçek ses akışı kullanılamıyorsa dummy blok döndür.
        if self.is_dummy or not self.is_audio_available or self.stream is None:
            # logger.debug("AudioSensor: Simüle edilmiş ses bloğu üretiliyor.")
            # dummy_audio_chunk np.int16 dtype'ında ve chunk_size boyutunda olmalı.
            # Rastgele gürültü veya sessizlik simüle edilebilir.
            dummy_chunk = np.zeros(self.audio_chunk_size, dtype=np.int16) # Sessizlik simülasyonu
            # np.random.randint(-32768, 32767, size=self.audio_chunk_size, dtype=np.int16) # Rastgele gürültü
            # logger.debug(f"AudioSensor: Simüle edilmiş ses bloğu üretildi. Shape: {dummy_chunk.shape}, Dtype: {dummy_chunk.dtype}")
            return dummy_chunk
        else:
            # Gerçek mikrofon aktifse blok oku
            try:
                # Akıştan belirtilen chunk boyutu kadar veri oku.
                # read() metodu blocking'tir. IOError'ı yakalıyoruz.
                # exception_on_overflow=False argümanı PyAudio okuma hatası durumunda
                # istisna fırlatmak yerine veriyi kırpmasını sağlar.
                # Bu argümanı teste eklemeliyiz.
                data = self.stream.read(self.audio_chunk_size, exception_on_overflow=False) # <<< exception_on_overflow=False eklendi


                # Okunan bayt verisini numpy array'e dönüştür (int16 formatında).
                # np.frombuffer kullanılır. dtype=np.int16 olmalı.
                # Eğer okunan data boş gelirse np.frombuffer hata vermez, boş array döndürür.
                audio_chunk = np.frombuffer(data, dtype=np.int16)

                # Okunan veri boyutu (sample sayısı) beklenenden farklıysa (örn: akış kesildi, overflow oldu)
                # Bu durumda np.frombuffer daha az eleman döndürebilir.
                # Boyut uyuşmazlığı durumunda None döndürmek yerine simüle moda geçelim.
                if audio_chunk.shape[0] != self.audio_chunk_size:
                    logger.warning(f"AudioSensor: Ses akışından okunan blok boyutu beklenenden farklı ({audio_chunk.shape[0]} yerine {self.audio_chunk_size}). Akışta sorun olabilir. Simüle moda geçiliyor.")
                    self.is_audio_available = False # Simüle moda geçişi tetikleyebiliriz.
                    # Rekürsif çağrı ile self.capture_chunk() metodunu çağırarak dummy blok al.
                    return self.capture_chunk() # Rekürsif çağrı ile dummy blok al.


                # logger.debug(f"AudioSensor: Gerçek chunk yakalandı. Shape: {audio_chunk.shape}, Dtype: {audio_chunk.dtype}")
                # Başarılı durumda yakalanan gerçek ses chunk'ını döndür.
                return audio_chunk

            except IOError as e:
                 # PyAudio okuma sırasında oluşabilecek spesifik hatalar (örn: -9981 Stream is stopped, -9988 Stream closed).
                 # Bu hatalar genellikle akışın durdurulması veya kapatılması anlamına gelir.
                 logger.error(f"AudioSensor: Ses chunk okuma hatası (IOError): {e}. Simüle moda geçiliyor.", exc_info=True)
                 self.is_audio_available = False # Simüle moda geçişi tetikleyebiliriz.
                 # Rekürsif çağrı ile self.capture_chunk() metodunu çağırarak dummy blok al.
                 return self.capture_chunk() # Rekürsif çağrı ile dummy blok al.
            except Exception as e:
                # Okuma sırasında beklenmedik bir istisna oluşursa.
                logger.error(f"AudioSensor: Ses chunk yakalama sırasında beklenmedik hata: {e}. Simüle moda geçiliyor.", exc_info=True)
                self.is_audio_available = False # Hata durumunda akışı aktif değil yap.
                # Rekürsif çağrı ile self.capture_chunk() metodunu çağırarak dummy blok al.
                return self.capture_chunk() # Rekürsif çağrı ile dummy blok al.


    def stop_stream(self):
        """
        Ses akışını durdurur ve kapatır. PyAudio instance'ını sonlandırmaz!
        Program sonlanırken cleanup metodu tarafından çağrılır.
        """
        logger.info("AudioSensor: İşitsel akış durduruluyor...")
        # Stream objesi mevcutsa ve açıksa
        if self.stream is not None and self.stream.is_active(): # is_active() ile akışın hala çalışıp çalışmadığını kontrol et.
            try:
                 self.stream.stop_stream()
                 logger.info("AudioSensor: Ses akışı durduruldu.")
                 self.stream.close()
                 logger.info("AudioSensor: Ses akışı kapatıldı.")
            except Exception as e:
                 # Durdurma veya kapatma sırasında hata oluşursa logla.
                 logger.error(f"AudioSensor: Ses akışı durdurulurken veya kapatılırken hata oluştu: {e}", exc_info=True)
            # İşlem sonrası stream objesini None yap.
            self.stream = None
        else:
             logger.info("AudioSensor: İşitsel akış zaten durdurulmuş, hiç açılamamış veya kaynak zaten serbest bırakılmış.")

        # is_audio_available bayrağını False yap (zaten stop sonrası aktif olmayacaktır).
        self.is_audio_available = False


    def terminate_pyaudio(self):
        """
        PyAudio instance'ını sonlandırır. PyAudio objesi PyAudio.PyAudio() ile başlatıldıysa bu metot çağrılmalıdır.
        Program sonlanırken cleanup metodu tarafından çağrılır.
        """
        if self.p is not None:
             logger.info("AudioSensor: PyAudio instance'ı sonlandırılıyor...")
             try:
                  self.p.terminate()
                  logger.info("AudioSensor: PyAudio instance'ı sonlandırıldı.")
             except Exception as e:
                  # Sonlandırma sırasında hata oluşursa logla.
                  logger.error(f"AudioSensor: PyAudio instance'ı sonlandırılırken hata oluştu: {e}", exc_info=True)
             self.p = None # İşlem sonrası objeyi None yap.
        else:
             logger.info("AudioSensor: PyAudio instance'ı zaten None.")


    # --- cleanup metodunu düzeltiyoruz ---
    def cleanup(self):
        """
        Nesne temizlendiğinde çağrılır. stop_stream metodunu ve terminate_pyaudio metodunu çağırır.
        module_loader.py bu metodu program sonlanırken çağırır (varsa).
        """
        logger.info("AudioSensor objesi temizleniyor.")
        # Akışı durdur ve kapat. run_safely ile hata yönetimini sağla.
        run_safely(self.stop_stream, logger_instance=logger, error_message="AudioSensor: stop_stream cleanup sırasında hata")

        # PyAudio instance'ını sonlandır. run_safely ile hata yönetimini sağla.
        run_safely(self.terminate_pyaudio, logger_instance=logger, error_message="AudioSensor: terminate_pyaudio cleanup sırasında hata")

        # is_audio_available bayrağını False yap (zaten stop_stream içinde yapılıyor).
        # self.is_audio_available = False # Zaten stop_stream içinde yapılıyor.

        logger.info("AudioSensor objesi silindi.")