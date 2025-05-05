# src/senses/audio.py
#
# Evo'nun işitsel duyu organını temsil eder.
# Mikrofon akışından ham ses parçalarını (chunk) yakalar.
# Mikrofon kullanılamadığında simüle edilmiş girdi sağlar.

import pyaudio # PyAudio kütüphanesi, ses akışı yönetimi için. requirements.txt'e eklenmeli.
import numpy as np # Ses verisi (örnekler dizisi) için.
import time # Simülasyon veya zamanlama için. Şu an doğrudan kullanılmıyor.
import logging # Loglama için.
import sys # Hata yönetimi veya stream handler için kullanılabilir. Şu an doğrudan kullanılmıyor.

# Bu modül için bir logger oluştur
# 'src.senses.audio' adında bir logger döndürür.
logger = logging.getLogger(__name__)

class AudioSensor:
    """
    Evo'nun işitsel duyu organı sınıfı.

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
        # Yapılandırmadan ayarları al, yoksa varsayılanları kullan.
        self.audio_rate = config.get('audio_rate', 44100)
        self.audio_chunk_size = config.get('audio_chunk_size', 1024)
        # PyAudio formatı (16-bit integer yaygın ve çoğu donanımla uyumludur).
        self.audio_format = pyaudio.paInt16
        self.audio_channels = 1 # Şimdilik Mono ses. Stereo için 2 yapılabilir.
        # Kullanılacak input cihazı indeksi. Config'ten alınır. None ise PyAudio varsayılanı bulmaya çalışır.
        self.audio_input_device_index = config.get('audio_input_device_index')

        self.p = None # PyAudio objesi. PyAudio başlatılırsa atanır.
        self.stream = None # PyAudio Stream objesi. Ses akışı açılırsa atanır.
        self.is_audio_available = False # Gerçek ses akışının şu an aktif olup olmadığını tutar. Başlangıçta False.

        logger.info("AudioSensor başlatılıyor...")
        try:
            # PyAudio objesini başlat. Bu, sistemdeki ses donanımlarıyla iletişim kurmayı sağlar.
            self.p = pyaudio.PyAudio()

            # Eğer config'te belirli bir cihaz indeksi belirtilmemişse, varsayılan input cihazını bulmaya çalış.
            if self.audio_input_device_index is None:
                 try:
                      # PyAudio'nun varsayılan input cihazını getir.
                      default_device_info = self.p.getDefaultInputDevice()
                      self.audio_input_device_index = default_device_info['index']
                      # Varsayılan cihazın adını ve indeksini logla.
                      logger.info(f"AudioSensor: Varsayılan ses input cihazı bulundu: {default_device_info['name']} (Index: {self.audio_input_device_index})")
                 except Exception:
                      # Varsayılan cihaz bulunamazsa (örn: hiç mikrofon bağlı değilse veya driver sorunu varsa).
                      # Cihaz indeksi None kalır. PyAudio open stream çağrısında hata verecektir.
                      logger.warning("AudioSensor: Varsayılan ses input cihazı bulunamadı. PyAudio varsayılanını kullanmaya çalışılacak.")
                      # İndeks None kaldığı için PyAudio'nun open stream metodunun hata verme ihtimali yüksektir.
                      # Bu durum open stream try-except bloğunda yakalanacaktır.


            # Ses akışını başlatmak için PyAudio'nun open methodunu kullan.
            # format: ses örneği tipi (örn: 16-bit int)
            # channels: kanal sayısı (mono=1, stereo=2)
            # rate: örnekleme oranı (Hz)
            # input=True: Bu bir input (yakalama) akışı
            # input_device_index: kullanılacak cihaz indeksi (None ise PyAudio varsayılanı dener)
            # frames_per_buffer: capture_chunk metodunda okunacak veri boyutu (chunk size)
            self.stream = self.p.open(format=self.audio_format,
                                      channels=self.audio_channels,
                                      rate=self.audio_rate,
                                      input=True,
                                      input_device_index=self.audio_input_device_index,
                                      frames_per_buffer=self.audio_chunk_size)

            # Akış başarıyla açıldıysa
            self.is_audio_available = True # Ses aktif bayrağını True yap.
            logger.info(f"AudioSensor: Akış başarıyla başlatıldı. Cihaz: {self.audio_input_device_index}, Rate: {self.audio_rate}, Chunk: {self.audio_chunk_size}")

        except Exception as e:
            # PyAudio başlatma veya akış açma sırasında beklenmedik bir istisna oluşursa.
            # (örn: PortAudio hatası, cihaz bulunamadı, geçersiz parametreler vb.)
            logger.error(f"AudioSensor başlatılırken hata oluştu: {e}", exc_info=True)
            self.is_audio_available = False # Hata durumunda ses aktif değil.
            # Hata durumunda açılmış olabilecek kaynakları (stream, pyaudio objesi) temizlemeyi dene.
            if self.stream:
                 try:
                      self.stream.stop_stream()
                      self.stream.close()
                      logger.debug("AudioSensor: Başlatma hatası sonrası stream temizlendi.")
                 except Exception:
                      pass # Temizleme hatasını yoksay
            if self.p:
                 try:
                      self.p.terminate()
                      logger.debug("AudioSensor: Başlatma hatası sonrası PyAudio sonlandırıldı.")
                 except Exception:
                      pass # Terminate hatasını yoksay
            self.stream = None # Objeleri None yap.
            self.p = None


        # Başlatma işleminin sonucunu logla.
        logger.info(f"AudioSensor başlatıldı. Ses aktif: {self.is_audio_available}")


    def capture_chunk(self):
        """
        Ses akışından bir chunk (parça) ses verisi yakalar.

        Eğer gerçek ses akışı aktifse, PyAudio stream objesinden bir chunk okur.
        Eğer gerçek ses akışı aktif değilse (başlatılamadı veya hata oluştu),
        yapılandırmada belirtilen boyutta simüle edilmiş (sessiz) ses verisi döndürür.
        Chunk yakalama sırasında hata oluşursa (gerçek akıştan), hatayı loglar
        ve None döndürerek main loop'un çökmesini engeller.

        Returns:
            numpy.ndarray or None: Başarıyla yakalanan (gerçek veya simüle) ses chunk'ı
                                    (dtype int16) veya
                                    gerçek akıştan yakalama sırasında hata oluştuysa None.
        """
        # Eğer gerçek ses akışı aktifse ve stream objesi mevcutsa
        if self.is_audio_available and self.stream is not None:
            try:
                # Akıştan belirtilen chunk boyutu kadar veri oku.
                # read() metodu blocking'tir (veri gelene kadar bekler).
                # Eğer timeout olursa IOError yükseltir.
                # Hata yönetimi bu IOError'ı yakalamalıdır.
                data = self.stream.read(self.audio_chunk_size)

                # Okunan byte verisini numpy array'e dönüştür.
                # np.frombuffer, byte'ları belirtilen dtype'a çevirir.
                audio_chunk = np.frombuffer(data, dtype=np.int16)

                # logger.debug(f"AudioSensor: Gerçek chunk yakalandı. Shape: {audio_chunk.shape}, Dtype: {audio_chunk.dtype}")
                # Başarılı durumda yakalanan gerçek ses chunk'ını döndür.
                return audio_chunk

            except IOError as e:
                 # PyAudio okuma sırasında oluşabilecek spesifik hatalar (örn: timeout, akış kesilmesi).
                 logger.error(f"AudioSensor: Ses chunk okuma hatası (IOError): {e}", exc_info=True)
                 # Bu tür bir hata akışın durduğu anlamına gelebilir.
                 self.is_audio_available = False # Geçici olarak akışı aktif değil yap.
                 # Hata durumunda None döndürerek main loop'un bu chunk'ı atlamasını sağla.
                 return None
            except Exception as e:
                # Okuma sırasında beklenmedik bir istisna oluşursa.
                logger.error(f"AudioSensor: Ses chunk yakalama sırasında beklenmedik hata: {e}", exc_info=True)
                self.is_audio_available = False # Hata durumunda akışı aktif değil yap.
                # Hata durumunda None döndürerek main loop'un çökmesini engelle.
                return None
        else:
            # Eğer gerçek ses akışı aktif değilse (başlatılamadı veya hata oluştu)
            # Yapılandırmada belirtilen chunk boyutunda simüle edilmiş (sessiz) bir ses verisi döndür.
            # logger.debug("AudioSensor: Simüle edilmiş chunk üretiliyor.")
            # Simüle edilmiş sessiz (sıfır değerleri) int16 formatında numpy array oluştur.
            dummy_chunk = np.zeros(self.audio_chunk_size, dtype=np.int16)
            # logger.debug(f"AudioSensor: Simüle edilmiş chunk üretildi. Shape: {dummy_chunk.shape}, Dtype: {dummy_chunk.dtype}")
            # Simüle edilmiş chunk'ı döndür.
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
                 # Akışı durdur ve kapat.
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
                 # PyAudio instance'ını sonlandır.
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
        Nesne temizlendiğinde çağrılır (Gelecekte __del__ metodunda veya explicit olarak kullanılabilir).
        Şimdilik stop_stream metodunu çağırmak için placeholder olarak eklendi,
        ancak kaynak temizliği genellikle stop_stream'de yapılır.
        module_loader.py bu metodu çağırır (varsa).
        """
        logger.info("AudioSensor objesi temizleniyor...")
        # Kaynak temizliği stop_stream metodunda yapıldığı için burada ekstra bir şey yapmaya gerek yok,
        # ancak stop_stream metodunun çağrıldığından emin olmalıyız (module_loader.py bunu yapıyor).
        # self.stop_stream() # cleanup içinde stop_stream çağırmak çift çağrıya neden olabilir.
        pass # Genellikle sadece loglamak yeterli.