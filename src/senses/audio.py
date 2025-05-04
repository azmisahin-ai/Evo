# src/senses/audio.py

import logging
import pyaudio
import numpy as np
import time # Ses akışı başlatılırken bekleme için

class AudioSensor:
    """
    Evo'nun işitsel duyu organını temsil eder.
    Mikrofondan ses akışını yakalamaktan sorumludur.
    """
    def __init__(self, config=None):
        logging.info("AudioSensor başlatılıyor...")
        self.config = config # Konfigürasyon ayarları burada kullanılabilir

        # Varsayılan ses ayarları
        self.format = pyaudio.paInt16 # 16-bit integer formatı
        self.channels = config.get('audio_channels', 1) # Tek kanal (mono)
        self.rate = config.get('audio_rate', 44100) # Örnekleme oranı (Hz)
        self.chunk_size = config.get('audio_chunk_size', 1024) # Okunacak frame sayısı (tampon boyutu)

        self.audio = None
        self.stream = None

        try:
            self.audio = pyaudio.PyAudio()

            # Varsayılan input cihazını bul veya config'den al
            input_device_index = config.get('audio_input_device_index')
            if input_device_index is None:
                 # Varsayılan input cihazını bulmaya çalış
                 try:
                     default_input_device_info = self.audio.get_default_input_device_info()
                     input_device_index = default_input_device_info.get('index')
                     logging.info(f"Varsayılan ses input cihazı bulundu: {default_input_device_info.get('name')} (Index: {input_device_index})")
                 except Exception as e:
                      logging.warning(f"Varsayılan ses input cihazı bulunamadı: {e}. İlk cihaz deneniyor.")
                      input_device_index = 0 # İlk cihazı dene

            # Ses akışını başlatma
            self.stream = self.audio.open(format=self.format,
                                          channels=self.channels,
                                          rate=self.rate,
                                          input=True, # Input akışı
                                          frames_per_buffer=self.chunk_size,
                                          input_device_index=input_device_index)

            logging.info(f"AudioSensor başarıyla başlatıldı. Cihaz: {input_device_index}, Rate: {self.rate}, Chunk: {self.chunk_size}")

        except Exception as e:
            logging.error(f"AudioSensor başlatılırken hata oluştu: {e}")
            # Hata durumunda kaynakları temizle
            if self.stream: self.stream.stop_stream(); self.stream.close()
            if self.audio: self.audio.terminate()
            self.audio = None
            self.stream = None
            # Hata yönetimi: Uygulama burada durdurulabilir veya ses almadan devam edilebilir.
            # Şimdilik hata loglayıp devam edeceğiz, capture_chunk None döndürecek.

    def capture_chunk(self):
        """
        Mikrofondan sesin küçük bir bölümünü (chunk) yakalar.
        Non-blocking okuma yapılır. Veri yoksa None döner.
        Gelecekte sürekli akış mantığı buraya eklenecek.
        """
        if self.stream is None or not self.stream.is_active():
            # logging.warning("Ses akışı hazır değil veya aktif değil. Chunk yakalanamadı.")
            return None # Akış hazır değilse veya durmuşsa None döndür

        try:
            # Non-blocking read: Timeout = 0
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            # PyAudio verisi bytes cinsindendir. NumPy array'e dönüştürelim.
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            # logging.debug(f"Ses chunk başarıyla yakalandı. Boyut: {len(audio_chunk)}")
            return audio_chunk # NumPy array döndürür

        except Exception as e:
            logging.error(f"Ses chunk yakalanamadı: {e}")
            return None # Hata durumunda None döndür


    def start_stream(self):
        """
        İşitsel akışı başlatır (Gerçek implementasyon bekleniyor).
        PyAudio stream zaten __init__ içinde başlatılıyor.
        Bu metot gelecekte stream'i thread içinde yönetmek için kullanılabilir.
        """
        logging.info("İşitsel akış başlatılıyor (Gerçek implementasyon bekleniyor)...")
        # if self.stream and not self.stream.is_active():
        #     self.stream.start_stream()
        pass

    def stop_stream(self):
        """
        İşitsel akışı durdurur ve PyAudio kaynağını sonlandırır.
        """
        logging.info("İşitsel akış durduruluyor...")
        if hasattr(self, 'stream') and self.stream is not None:
            if self.stream.is_active():
                 self.stream.stop_stream()
                 logging.info("Ses akışı durduruldu.")
            self.stream.close()
            logging.info("Ses akışı kapatıldı.")
        if hasattr(self, 'audio') and self.audio is not None:
             self.audio.terminate()
             logging.info("PyAudio sonlandırıldı.")
        elif hasattr(self, 'audio') and self.audio is None:
             logging.info("AudioSensor henüz tam başlatılmamıştı veya hata almıştı.")


    def __del__(self):
        """
        Nesne silindiğinde kaynakları serbest bırakır.
        """
        self.stop_stream()
        logging.info("AudioSensor objesi silindi.")

# Modülü bağımsız test etmek için örnek kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print("AudioSensor test ediliyor...")

    # Basit bir config objesi
    test_config = {
        'audio_rate': 44100,
        'audio_chunk_size': 1024,
        # 'audio_input_device_index': 1 # Belirli bir cihazı denemek için uncomment edin
    }

    sensor = None
    try:
        sensor = AudioSensor(test_config)

        if sensor.stream and sensor.stream.is_active():
            print("Mikrofon dinleniyor... Ses yakalamak için birkaç saniye bekleyin.")
            time.sleep(2) # Akışın oturması için biraz bekle

            chunk = sensor.capture_chunk()

            if chunk is not None:
                print(f"Yakalanan Ses Chunk Verisi (NumPy Array Shape): {chunk.shape}, Data Type: {chunk.dtype}")
                # Ses verisi üzerinde temel analizler yapılabilir (örn. enerji seviyesi)
                # energy = np.sum(chunk**2)
                # print(f"Yakalanan Ses Chunk Enerjisi: {energy}")
            else:
                 print("Ses chunk yakalama testi BAŞARISIZ oldu (Chunk alınamadı).")

        else:
            print("AudioSensor başlatma testi BAŞARISIZ oldu (Ses akışı açılamadı).")


    except Exception as e:
        logging.exception("AudioSensor test sırasında hata oluştu:")

    finally:
        if sensor:
            sensor.stop_stream() # Kaynakları temizle
        print("AudioSensor testi bitti.")