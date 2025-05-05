# src/interaction/api.py

import logging
import numpy as np
import threading # API sunucusunu ayrı thread'de çalıştırmak için
import time # Basit gecikmeler veya zamanlama için

# Flask kütüphanesini import et (requirements.txt'e eklenmeli: Flask)
try:
    from flask import Flask, jsonify, request
    # Flask logging'ini kontrol altına alalım
    logging.getLogger('flask').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    FLASK_AVAILABLE = True
except ImportError:
    logging.warning("Flask kütüphanesi bulunamadı. Web API kanalı kullanılamayacak. Yüklemek için: pip install Flask")
    FLASK_AVAILABLE = False


# Farklı çıktı kanalları için temel sınıf
class OutputChannel:
    """Tüm çıktı kanallarının miras alması gereken temel sınıf."""
    def __init__(self, name, config=None):
        self.name = name
        self.config = config if config is not None else {}
        logging.info(f"OutputChannel '{self.name}' başlatılıyor.")
        self.is_initialized = False # Başlatma durumu
        try:
            self._initialize() # Alt sınıfların başlatma metodu
            self.is_initialized = True
        except Exception as e:
            logging.critical(f"OutputChannel '{self.name}' başlatılırken hata: {e}", exc_info=True)
            self.is_initialized = False


    def _initialize(self):
         """Alt sınıfların kendi başlatma mantığını buraya koyması beklenir."""
         pass # Varsayılan olarak bir şey yapmaz

    def send(self, output_data):
        """Çıktı verisini bu kanala gönderir."""
        if not self.is_initialized:
             logging.warning(f"OutputChannel '{self.name}' başlatılmamış, çıktı gönderilemiyor.")
             return
        try:
            self._send_data(output_data) # Alt sınıfların gönderme metodu
        except Exception as e:
            logging.error(f"OutputChannel '{self.name}' gönderim sırasında hata: {e}", exc_info=True)


    def _send_data(self, output_data):
        """Alt sınıfların çıktı gönderme mantığını implement etmesi beklenir."""
        raise NotImplementedError(f"OutputChannel '{self.name}' _send_data metodunu implement etmelidir.")

    def cleanup(self):
        """Kanal kapatılırken kaynakları temizler."""
        if not self.is_initialized:
             # logging.debug(f"OutputChannel '{self.name}' tam başlatılmamış, temizlenecek kaynak yok.")
             return # Başlatılmamışsa temizlik gerekmez
        logging.info(f"OutputChannel '{self.name}' temizleniyor.")
        try:
            self._cleanup() # Alt sınıfların temizleme metodu
        except Exception as e:
            logging.error(f"OutputChannel '{self.name}' temizlenirken hata: {e}", exc_info=True)


    def _cleanup(self):
        """Alt sınıfların kendi temizleme mantığını buraya koyması beklenir."""
        pass # Varsayılan olarak bir şey yapmaz


# Konsola çıktı yazan kanal
class ConsoleOutputChannel(OutputChannel):
    """Çıktıyı doğrudan konsola (logging) yazan kanal."""
    def _initialize(self):
        super()._initialize()
        # Özel konsol ayarları burada kullanılabilir
        logging.info(f"ConsoleOutputChannel başlatıldı.")


    def _send_data(self, output_data):
        """Çıktı verisini INFO seviyesinde konsola yazar."""
        if output_data is not None:
            # logging.info(f"Evo Çıktısı [Console]: {output_data}")
            # Ana run_evo döngüsünde InteractionAPI'den loglandığı için burada tekrar INFO seviyesinde loglamaya gerek yok.
            # Debug seviyesinde ham çıktıyı loglayabiliriz veya hiç loglamayabiliriz.
             logging.debug(f"OutputChannel '{self.name}': Ham çıktı alindi, konsola yaziliyor.")
        # else: logging.debug(f"OutputChannel '{self.name}': Gönderilecek çıktı verisi None.")


# Çıktıyı bir Web API endpoint'i aracılığıyla dışarı sunan kanal
class WebAPIOutputChannel(OutputChannel):
    """Çıktıyı bir Web API endpoint'i aracılığıyla dışarı sunan kanal."""
    def _initialize(self):
        super()._initialize()
        if not FLASK_AVAILABLE:
             raise ImportError("Flask kütüphanesi bulunamadı. Web API kanalı başlatılamaz.")

        self.api_port = self.config.get('port', 5000)
        self.latest_output = None # Son çıktıyı saklamak için

        # Flask uygulamasını ayrı bir thread içinde başlat
        self.app = Flask(__name__)

        # API endpoint'ini tanımla
        @self.app.route('/evo_output', methods=['GET'])
        def get_evo_output():
            # Son çıktıyı JSON formatında döndür
            return jsonify({'output': self.latest_output, 'timestamp': time.time()})

        # API sunucusunu çalıştıracak thread'i oluştur
        # debug=False ve use_reloader=False thread içinde çalıştırmak için önemlidir
        self.api_thread = threading.Thread(
            target=self.app.run,
            kwargs={'port': self.api_port, 'debug': False, 'use_reloader': False}
        )
        self.api_thread.daemon = True # Ana program sonlandığında thread'in de sonlanmasını sağlar

        # Thread'i başlat
        self.api_thread.start()
        logging.info(f"WebAPIOutputChannel başlatıldı. API sunucusu Port {self.api_port} üzerinde çalışıyor.")


    def _send_data(self, output_data):
        """Çıktı verisini dahili olarak saklar."""
        if output_data is not None:
             # output_data string veya başka serialize edilebilir bir formatta olmalı
             self.latest_output = output_data # Son çıktıyı güncelle
             logging.debug(f"OutputChannel '{self.name}': Ham çıktı alindi, dahili olarak saklandi.")
        # else: logging.debug(f"OutputChannel '{self.name}': Gönderilecek çıktı verisi None.")


    def _cleanup(self):
        """Web API sunucu thread'ini durdurur."""
        logging.info(f"OutputChannel '{self.name}' Web API sunucusu durduruluyor...")
        # Flask thread'ini durdurmanın standart bir yolu yok, ancak daemon thread olduğu için
        # ana program sonlandığında o da sonlanacaktır. Daha kontrollü bir durdurma için
        # Flask uygulamasına bir shutdown endpoint'i eklenebilir. Şimdilik daemon yeterli.
        # API sunucusu durdurulurken biraz zaman alabilir.
        # Bir shutdown endpoint örneği:
        # func = request.environ.get('werkzeug.server.shutdown')
        # if func: func()
        # else: logging.warning("Could not shut down werkzeug server. May take time to exit.")
        logging.info(f"OutputChannel '{self.name}' Web API sunucusu temizlendi.")


# Diğer potansiyel kanallar:
# class AudioOutputChannel(OutputChannel): ... (TTS entegrasyonu)
# class FileOutputChannel(OutputChannel): ... (Çıktıyı dosyaya yazma)
# class RoboticsOutputChannel(OutputChannel): ... (Fiziksel komut gönderme)


class InteractionAPI:
    """
    Evo'nın dış dünya ile iletişim kurduğu arayüzü temsil eder.
    MotorControl modülünden gelen tepkileri farklı OutputChannel'lara iletir.
    """
    def __init__(self, config=None):
        logging.info("InteractionAPI modülü başlatılıyor...")
        self.config = config if config is not None else {}

        # Aktif çıktı kanallarını depolamak için sözlük
        self.output_channels = {}

        # --- Çıktı Kanallarını Başlatma ---
        enabled_channels = self.config.get('enabled_channels', ['console']) # Config'den aktif kanalları al, varsayılan console
        logging.info(f"InteractionAPI: Aktif kanallar: {enabled_channels}")

        # Mevcut Output Kanalları Mappingi
        available_channels_map = {
            'console': ConsoleOutputChannel,
            'web_api': WebAPIOutputChannel,
            # Diğer kanallar geldikçe buraya eklenecek
            # 'audio': AudioOutputChannel,
            # 'file': FileOutputChannel,
        }

        for channel_name in enabled_channels:
            if channel_name in available_channels_map:
                try:
                    # Kanalı config'den ilgili ayarları ile başlat
                    channel_config = self.config.get('channel_configs', {}).get(channel_name, {})
                    channel_instance = available_channels_map[channel_name](channel_config)

                    if channel_instance.is_initialized: # Sadece başarılı başlatılan kanalları ekle
                        self.output_channels[channel_name] = channel_instance
                        logging.info(f"InteractionAPI: OutputChannel '{channel_name}' başarıyla başlatıldı.")
                    else:
                        logging.error(f"InteractionAPI: OutputChannel '{channel_name}' başlatılamadı (is_initialized False). Atlanıyor.")


                except Exception as e: # OutputChannel init içindeki hata yakalanmalı ama burası da ek koruma.
                    logging.critical(f"InteractionAPI: OutputChannel '{channel_name}' başlatılırken beklenmeyen hata: {e}", exc_info=True)
                    # Hata veren kanalı başlatma, devam et
            else:
                logging.warning(f"InteractionAPI: Bilinmeyen OutputChannel adı '{channel_name}'. Atlanıyor.")


        if not self.output_channels:
             logging.warning("InteractionAPI: Hiçbir OutputChannel başarıyla başlatılamadı. Evo çıktı veremeyecek.")

        logging.info(f"InteractionAPI modülü başlatıldı. Aktif Output Kanalları: {list(self.output_channels.keys())}")


    def send_output(self, output_data):
        """
        MotorControl modülünden gelen çıktıyı alır ve tüm aktif kanallara gönderir.
        output_data: MotorControl modülünden gelen tepki (string veya başka format)
        """
        # logging.debug(f"InteractionAPI: send_output çağrıldı. Output data var mi: {output_data is not None}. Aktif kanal sayisi: {len(self.output_channels)}") # Çok sık loglanabilir.

        if output_data is None:
            # logging.debug("InteractionAPI: Gönderilecek çıktı verisi yok.")
            return # None çıktı gelirse gönderme

        if not self.output_channels:
            # logging.warning("InteractionAPI: Hiçbir OutputChannel aktif değil, çıktı gönderilemedi.")
            return # Aktif kanal yoksa gönderme

        # logging.debug(f"InteractionAPI: Çikti alindi. İçerik ilk 50 karakter: '{str(output_data)[:50]}', Gönderilecek kanal sayısı: {len(self.output_channels)}")

        # Tüm aktif kanallara çıktıyı gönder
        for channel_name, channel_obj in self.output_channels.items():
            # Kanalın başlatılmış olduğundan emin ol (zaten init'te kontrol ettik ama ek koruma)
            if channel_obj.is_initialized:
                 try:
                     # logging.debug(f"InteractionAPI: Çikti OutputChannel '{channel_name}' kanalına gönderiliyor.")
                     channel_obj.send(output_data)
                 except Exception as e:
                     logging.error(f"InteractionAPI: OutputChannel '{channel_name}' gönderim sırasında hata oluştu: {e}", exc_info=True)
            # else: logging.warning(f"InteractionAPI: OutputChannel '{channel_name}' başlatılmamış, gönderim atlandı.")


    def stop(self):
        """
        InteractionAPI modülünü ve tüm aktif çıktı kanallarını durdurur/temizler.
        """
        logging.info("InteractionAPI durduruluyor...")
        # Tüm aktif kanalların cleanup metodunu çağır
        for channel_name, channel_obj in self.output_channels.items():
            # Kanal objesinin var olduğundan emin ol
            if channel_obj:
                try:
                    # logging.info(f"InteractionAPI: OutputChannel '{channel_name}' temizleniyor...")
                    channel_obj.cleanup()
                    # logging.info(f"InteractionAPI: OutputChannel '{channel_name}' temizlendi.")
                except Exception as e:
                     logging.error(f"InteractionAPI: OutputChannel '{channel_name}' temizlenirken hata oluştu: {e}", exc_info=True)

        logging.info("InteractionAPI objesi temizlendi.")

    def __del__(self):
        """
        Nesne silindiğinde kaynakları temizler. stop() metodu çağrılmalıdır.
        """
        # logging.info("InteractionAPI objesi siliniyor...")
        # __del__ içinde stop çağırmak thread safety sorunlarına neden olabilir.
        # Main loop'un finally bloğunda stop() çağrıldığından emin olunmalıdır.
        pass


# Modülü bağımsız test etmek için örnek kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print("InteractionAPI modülü test ediliyor (Çikti Kanallari ile)...")

    # Konsol ve Placeholder Web API kanalını aktif et
    test_config = {
        'enabled_channels': ['console', 'web_api'],
        'channel_configs': {
            'web_api': {'port': 8080} # Web API için örnek ayar
        }
    }
    api = None
    try:
        api = InteractionAPI(test_config)

        if api.output_channels: # Başlatılan kanal varsa test et
             # Sahte çıktıları kanallara gönder
             dummy_output_1 = "Merhaba dünya!"
             dummy_output_2 = "Bu bir test mesajı."

             print("\nÇikti 'Merhaba dünya!' kanallara gönderiliyor:")
             api.send_output(dummy_output_1)

             print("\nÇikti 'Bu bir test mesajı.' kanallara gönderiliyor:")
             api.send_output(dummy_output_2)

             # None çıktı test et
             print("\nNone çikti kanallara gönderiliyor:")
             api.send_output(None) # None ise send metoduna hiç girmez

             # Boş string çıktı test et
             print("\nBoş string çikti kanallara gönderiliyor:")
             api.send_output("") # Boş string gönderilebilir

        else:
             print("\nHiçbir OutputChannel başlatılamadığı için send testleri atlandı.")


    except Exception as e:
        logging.exception("InteractionAPI test sırasında hata oluştu:")

    finally:
        if api:
            api.stop() # Kaynakları temizle

    print("\nInteractionAPI modülü testi bitti.")