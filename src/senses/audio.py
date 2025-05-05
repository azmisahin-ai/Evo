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
from src.core.utils import get_config_value # <<< get_config_value import edildi
# check_input_not_none, check_numpy_input bu modülde girdi alan metot olmadığı için gerekli değil şimdilik.


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
        """
        self.config = config
        logger.info("AudioSensor başlatılıyor...")

        # Yapılandırmadan ayarları alırken get_config_value kullan
        # audio_input_device_index hem int hem de None olabildiği için expected_type olarak (int, type(None)) tuple verilebilir.
        self.audio_rate = get_config_value(config, 'audio_rate', 44100, expected_type=int, logger_instance=logger)
        self.audio_chunk_size = get_config_value(config, 'audio_chunk_size', 1024, expected_type=int, logger_instance=logger)
        self.audio_input_device_index = get_config_value(config, 'audio_input_device_index', None, expected_type=(int, type(None)), logger_instance=logger)


        self.audio_format = pyaudio.paInt16 # 16-bit integer formatı yaygın ve basit
        self.audio_channels = 1 # Mono ses


        self.p = None # PyAudio objesi. PyAudio başlatılırsa atanır.
        self.stream = None # PyAudio Stream objesi. Ses akışı açılırsa atanır.
        self.is_audio_available = False # Gerçek ses akışının şu an aktif olup olmadığını tutar. Başlangıçta False.

        try:
            # PyAudio objesini başlat.
            self.p = pyaudio.PyAudio()

            # Eğer config'te belirli bir cihaz indeksi belirtilmemişse (None ise) ve PyAudio varsayılan bulabiliyorsa, onu kullan.
            # get_config_value None olarak döndürdüyse veya config'te None ise buraya girilir.
            if self.audio_input_device_index is None:
                 try:
                      # PyAudio'nın varsayılan input cihazını getir.
                      default_device_info = self.p.getDefaultInputDevice()
                      self.audio_input_device_index = default_device_info['index']
                      # Varsayılan cihazın adını ve indeksini logla.
                      logger.info(f"AudioSensor: Varsayılan ses input cihazı bulundu: {default_device_info['name']} (Index: {self.audio_input_device_index})")
                 except Exception:
                      # Varsayılan cihaz bulunamazsa uyarı logla.
                      logger.warning("AudioSensor: Varsayılan ses input cihazı bulunamadı. PyAudio varsayılanını kullanmaya çalışılacak.")
                      # İndeks None kaldığı için PyAudio'nun open stream metodunun hata verme ihtimali yüksektir.
                      # Bu durum open stream try-except bloğunda yakalanacaktır.

            # Ses akışını başlatmak için PyAudio'nun open methodunu kullan.
            self.stream = self.p.open(format=self.audio_format,
                                      channels=self.audio_channels,
                                      rate=self.audio_rate, # config'ten int olarak alındı
                                      input=True,
                                      input_device_index=self.audio_input_device_index, # config'ten int veya None olarak alındı
                                      frames_per_buffer=self.audio_chunk_size) # config'ten int olarak alındı

            # Akış başarıyla açıldıysa
            self.is_audio_available = True # Ses aktif bayrağını True yap.
            logger.info(f"AudioSensor: Akış başarıyla başlatıldı. Cihaz: {self.audio_input_device_index}, Rate: {self.audio_rate}, Chunk: {self.audio_chunk_size}")

        except Exception as e:
            # PyAudio başlatma veya akış açma sırasında beklenmedik bir istisna oluşursa.
            # Bu hata init sırasında kritik değildir (simüle moda geçiyoruz).
            logger.error(f"AudioSensor başlatılırken hata oluştu: {e}", exc_info=True)
            self.is_audio_available = False # Hata durumunda ses aktif değil.
            # Hata durumunda açılmış olabilecek kaynakları temizlemeyi dene.
            if self.stream:
                 try:
                      self.stream.stop_stream()
                      self.stream.close()
                      # logger.debug("AudioSensor: Başlatma hatası sonrası stream temizlendi.") # Zaten hata logu var
                 except Exception:
                      pass # Temizleme hatasını yoksay
            if self.p:
                 try:
                      self.p.terminate()
                      # logger.debug("AudioSensor: Başlatma hatası sonrası PyAudio sonlandırıldı.") # Zaten hata logu var
                 except Exception:
                      pass # Terminate hatasını yoksay
            self.stream = None # Objeleri None yap.
            self.p = None


        # Başlatma işleminin sonucunu logla.
        logger.info(f"AudioSensor başlatıldı. Ses aktif: {self.is_audio_available}")


    def capture_chunk(self):
        """
        Ses akışından bir chunk (parça) ses verisi yakalar.

        Bu metot girdi almadığı için check_input_* fonksiyonları kullanılmaz.
        Mantık aynı kalır.

        Returns:
            numpy.ndarray or None: Başarıyla yakalanan (gerçek veya simüle) ses chunk'ı
                                    (dtype int16) veya
                                    gerçek akıştan yakalama sırasında hata oluştuysa None.
        """
        # Eğer gerçek ses akışı aktifse ve stream objesi mevcutsa
        if self.is_audio_available and self.stream is not None:
            try:
                # Akıştan belirtilen chunk boyutu kadar veri oku.
                # read() metodu blocking'tir. IOError'ı yakalıyoruz.
                data = self.stream.read(self.audio_chunk_size)

                # Okunan byte verisini numpy array'e dönüştür.
                audio_chunk = np.frombuffer(data, dtype=np.int16)

                # logger.debug(f"AudioSensor: Gerçek chunk yakalandı. Shape: {audio_chunk.shape}, Dtype: {audio_chunk.dtype}")
                # Başarılı durumda yakalanan gerçek ses chunk'ını döndür.
                return audio_chunk

            except IOError as e:
                 # PyAudio okuma sırasında oluşabilecek spesifik hatalar.
                 logger.error(f"AudioSensor: Ses chunk okuma hatası (IOError): {e}", exc_info=True)
                 self.is_audio_available = False # Geçici olarak akışı aktif değil yap.
                 return None
            except Exception as e:
                # Okuma sırasında beklenmedik bir istisna oluşursa.
                logger.error(f"AudioSensor: Ses chunk yakalama sırasında beklenmedik hata: {e}", exc_info=True)
                self.is_audio_available = False # Hata durumunda akışı aktif değil yap.
                return None
        else:
            # Eğer gerçek ses akışı aktif değilse
            # Yapılandırmada belirtilen chunk boyutunda simüle edilmiş (sessiz) bir ses verisi döndür.
            # logger.debug("AudioSensor: Simüle edilmiş chunk üretiliyor.")
            # audio_chunk_size'ın int olduğundan emin olmak için get_config_value init'te kullanıldı.
            dummy_chunk = np.zeros(self.audio_chunk_size, dtype=np.int16)
            # logger.debug(f"AudioSensor: Simüle edilmiş chunk üretildi. Shape: {dummy_chunk.shape}, Dtype: {dummy_chunk.dtype}")
            return dummy_chunk

    def stop_stream(self):
        """
        Ses akışını durdurur ve PyAudio tarafından kullanılan kaynakları serbest bırakır.
        Program sonlanırken module_loader.py tarafından çağrılır.
        """
        logger.info("AudioSensor: İşitsel akış durduruluyor...")
        # Stream objesi mevcutsa
        if self.stream is not None:
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

        # PyAudio objesi mevcutsa sonlandır.
        if self.p is not None:
            try:
                 self.p.terminate()
                 logger.info("AudioSensor: PyAudio sonlandırıldı.")
            except Exception as e:
                 # Sonlandırma sırasında hata oluşursa logla.
                 logger.error(f"AudioSensor: PyAudio sonlandırılırken hata oluştu: {e}", exc_info=True)
            # İşlem sonrası PyAudio objesini None yap.
            self.p = None

        # is_audio_available bayrağını False yap (zaten stop sonrası aktif olmayacaktır).
        self.is_audio_available = False

    def cleanup(self):
        """
        Nesne temizlendiğinde çağrılır.

        Bu metot genellikle stop_stream metodunu çağırmak için bir placeholder'dır.
        Kaynak temizliği stop_stream metodunda yapılır.
        module_loader.py bu metodu program sonlanırken çağırır (varsa).
        """
        logger.info("AudioSensor objesi temizleniyor.")
        # Kaynak temizliği stop_stream metodunda yapıldığı için burada ekstra bir şey yapmaya gerek yok,
        # ancak stop_stream metodunun çağrıldığından emin olmalıyız (module_loader.py bunu yapıyor).
        # self.stop_stream() # cleanup içinde stop_stream çağırmak çift çağrıya neden olabilir.
        pass # Genellikle sadece loglamak yeterli.