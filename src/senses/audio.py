# src/senses/audio.py

import logging
import pyaudio
import numpy as np
import time

class AudioSensor:
    """
    Evo'nun işitsel duyu organını temsil eder.
    Mikrofondan ses akışını yakalamaktan sorumludur.
    Mikrofon bulunamazsa veya başlatılamazsa simüle edilmiş (dummy) ses chunkları döndürebilir.
    """
    def __init__(self, config=None):
        logging.info("AudioSensor başlatılıyor...")
        self.config = config if config is not None else {}

        # Ses ayarları
        self.format = pyaudio.paInt16 # 16-bit integer formatı
        self.channels = self.config.get('audio_channels', 1) # Tek kanal (mono)
        self.rate = self.config.get('audio_rate', 44100) # Örnekleme oranı (Hz)
        self.chunk_size = self.config.get('audio_chunk_size', 1024) # Okunacak frame sayısı (tampon boyutu)

        self.audio = None
        self.stream = None
        self.is_audio_available = False # Ses akışının başarılı başlatılıp başlatılmadığı

        try:
            self.audio = pyaudio.PyAudio()

            # Varsayılan input cihazını bul veya config'den al
            input_device_index = self.config.get('audio_input_device_index')
            if input_device_index is None:
                 # Varsayılan input cihazını bulmaya çalış
                 try:
                     default_input_device_info = self.audio.get_default_input_device_info()
                     input_device_index = default_input_device_info.get('index')
                     logging.info(f"Varsayılan ses input cihazı bulundu: {default_input_device_info.get('name')} (Index: {input_device_index})")
                 except Exception as e:
                      logging.warning(f"Varsayılan ses input cihazı bulunamadı: {e}. İlk cihaz deneniyor veya simüle edilmiş girdi kullanılacak.")
                      # input_device_index 0 denenirse ve hata verirse de yakalanacak
                      input_device_index = 0 # İlk cihazı varsayılan olarak dene


            # Ses akışını başlatma
            # Eğer input_device_index hala None ise veya geçersizse open hata verecektir, bunu yakalayacağız.
            self.stream = self.audio.open(format=self.format,
                                          channels=self.channels,
                                          rate=self.rate,
                                          input=True, # Input akışı
                                          frames_per_buffer=self.chunk_size,
                                          input_device_index=input_device_index) # Belirlenen veya varsayılan index

            # Akış başarılı başladıysa
            self.is_audio_available = True
            logging.info(f"AudioSensor başarıyla başlatıldı. Cihaz: {input_device_index}, Rate: {self.rate}, Chunk: {self.chunk_size}")

        except Exception as e:
            logging.error(f"AudioSensor başlatılırken hata oluştu: {e}. Simüle edilmiş işitsel girdi kullanılacak.")
            self.is_audio_available = False
            # Hata durumunda kaynakları temizle (open metodu hata verdiğinde bunlar oluşmamış olabilir ama yine de kontrol edelim)
            if hasattr(self, 'stream') and self.stream:
                 if self.stream.is_active(): self.stream.stop_stream()
                 self.stream.close()
            if hasattr(self, 'audio') and self.audio:
                 self.audio.terminate()
            self.audio = None
            self.stream = None


        # Ses kullanılamıyorsa sahte veri boyutunu belirle (chunk_size ve channels)
        # Bunlar zaten init başında ayarlandı, sadece is_audio_available False ise kullanılacak.
        logging.info(f"AudioSensor başlatıldı. Ses aktif: {self.is_audio_available}")


    def capture_chunk(self):
        """
        Mikrofondan sesin küçük bir bölümünü (chunk) yakalar veya ses akışı aktif değilse sahte bir chunk döndürür.
        Blocking okuma (varsayılan timeout) veya non-blocking (timeout=0). Platform uyumluluğu için timeout kaldırıldı.
        Şu an blocking olmayan okuma için `stream.read(..., timeout=0)` kullanıldı. Eğer bu hata verirse,
        timeout=0 kaldırılıp blocking hale getirilebilir veya farklı bir non-blocking yöntem denenir.
        NOT: Önceki hata PyAudio versiyonundan veya platformdan kaynaklı timeout argümanı uyumsuzluğu olabilir.
        Bu düzeltmede timeout argümanı kaldırıldı. Sadece exception_on_overflow=False kaldı.
        """
        if self.is_audio_available and self.stream.is_active():
            try:
                # timeout=0 argümanı kaldırıldı
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)

                # PyAudio verisi bytes cinsindendir. NumPy array'e dönüştürelim.
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                # logging.debug(f"Gerçek ses chunk başarıyla yakalandı. Boyut: {len(audio_chunk)}")
                return audio_chunk # NumPy array döndürür

            except IOError as e:
                 # Ses tamponu henüz dolu değilse veya okuma hatası olursa
                 # logging.debug(f"Ses chunk okunamadı (IOError: {e}). Simüle edilmiş chunk döndürülüyor.")
                 # IOError durumunda (veri yok gibi) None döndürmek veya sahte dönmek tartışılır.
                 # Şimdilik simüle dönelim ki downstream processing patlamasın.
                 return self._generate_dummy_chunk() # Hata durumunda sahte chunk döndür

            except Exception as e:
                logging.error(f"Ses chunk yakalanamadı (Genel Hata: {e}). Simüle edilmiş chunk döndürülüyor.")
                return self._generate_dummy_chunk() # Diğer hatalarda da sahte chunk döndür

        else:
            # logging.debug("Ses akışı mevcut değil. Simüle edilmiş chunk döndürülüyor.")
            return self._generate_dummy_chunk() # Ses akışı mevcut değilse sahte chunk döndür

    def _generate_dummy_chunk(self):
        """Belirlenen boyut ve kanalda sessiz (sıfırlarla dolu) bir sahte ses chunk'ı oluşturur."""
        dummy_chunk = np.zeros(self.chunk_size * self.channels, dtype=np.int16)
        return dummy_chunk # NumPy array döndürür


    def start_stream(self):
        """
        İşitsel akışı başlatır (Gerçek implementasyon bekleniyor).
        PyAudio stream zaten __init__ içinde başlatılıyor.
        Bu metot gelecekte stream'i thread içinde yönetmek için kullanılabilir.
        """
        logging.info("İşitsel akış başlatılıyor (Gerçek implementasyon bekleniyor)...")
        # if self.is_audio_available and self.stream and not self.stream.is_active():
        #     self.stream.start_stream()
        pass

    def stop_stream(self):
        """
        İşitsel akışı durdurur ve PyAudio kaynağını sonlandırır.
        """
        logging.info("İşitsel akış durduruluyor...")
        if self.is_audio_available and hasattr(self, 'stream') and self.stream is not None:
            if self.stream.is_active():
                 self.stream.stop_stream()
                 logging.info("Ses akışı durduruldu.")
            self.stream.close()
            logging.info("Ses akışı kapatıldı.")
        if hasattr(self, 'audio') and self.audio is not None:
             self.audio.terminate()
             logging.info("PyAudio sonlandırıldı.")
        elif hasattr(self, 'audio') and self.audio is None:
             logging.info("AudioSensor tam başlatılamamıştı veya hata almıştı, serbest bırakılacak kaynak yok.")
        self.is_audio_available = False # Durduruldu olarak işaretle


    def __del__(self):
        """
        Nesne silindiğinde kaynakları serbest bırakır.
        """
        self.stop_stream()
        # logging.info("AudioSensor objesi silindi.") # __del__ içinde loglama bazen sorunlu olabilir


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

        # Ses akışının oturması için biraz bekle (gerçek mikrofon deneniyorsa)
        # if sensor.is_audio_available:
        #     print("Mikrofon dinleniyor... Ses yakalamak için birkaç saniye bekleyin.")
        #     time.sleep(2) # Akışın oturması için biraz bekle


        print("Ses chunk yakalama denemesi (gerçek veya simüle)...")
        # Döngüde birden fazla chunk yakalamak test etmek için daha iyi olabilir
        for _ in range(5): # 5 chunk dene
            chunk = sensor.capture_chunk()

            if chunk is not None:
                print(f"Yakalanan Ses Chunk Verisi (NumPy Array Shape): {chunk.shape}, Data Type: {chunk.dtype}")
                # Sahte olup olmadığını kontrol etmek için basit bir test (tüm değerler 0 mı)
                if not sensor.is_audio_available and np.all(chunk == 0):
                     print("  (Simüle edilmiş chunk)")
                elif sensor.is_audio_available:
                     # Gerçek veriyse, enerjiye göre basit bir ses aktivitesi kontrolü
                     energy = np.sum(chunk**2) / len(chunk) # Ortalama enerji
                     print(f"  (Gerçek chunk, Enerji: {energy:.2f})")
                     if energy < 100: # Çok düşük enerji = sessizlik olabilir (eşik değeri ayarlanmalı)
                          print("  (Muhtemelen sessizlik)")


            else:
                # Bu kısma normalde düşmemeli çünkü hata durumunda sahte chunk dönüyor.
                print("Hata: capture_chunk None döndürdü (beklenmeyen durum).")
            
            time.sleep(test_config.get('cognitive_loop_interval', 0.1)) # Döngü aralığı gibi bekle

    except Exception as e:
        logging.exception("AudioSensor test sırasında hata oluştu:")

    finally:
        if sensor:
            sensor.stop_stream() # Kaynakları temizle
        print("AudioSensor testi bitti.")