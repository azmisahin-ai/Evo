# src/interaction/api.py

import logging
# Gelecekte ağ iletişimi veya cihaz kontrolü için kütüphaneler buraya eklenecek
# import flask # Örnek: Web API için (eğer burada başlatılacaksa)
# import robotics_interface_lib # Örnek: Robot kontrolü için

# Farklı çıktı kanalları için temel sınıf
class OutputChannel:
    """Tüm çıktı kanallarının miras alması gereken temel sınıf."""
    def __init__(self, name, config=None):
        self.name = name
        self.config = config if config is not None else {}
        logging.info(f"OutputChannel '{self.name}' başlatılıyor.")

    def send(self, output_data):
        """Çıktı verisini bu kanala gönderir."""
        raise NotImplementedError("Her OutputChannel sınıfı send metodunu implement etmelidir.")

    def cleanup(self):
        """Kanal kapatılırken kaynakları temizler."""
        logging.info(f"OutputChannel '{self.name}' temizleniyor.")
        pass # Varsayılan temizlik gerektirmez

# Konsola çıktı yazan kanal
class ConsoleOutputChannel(OutputChannel):
    """Çıktıyı doğrudan konsola (logging) yazan kanal."""
    def __init__(self, config=None):
        super().__init__("console", config)
        # Özel konsol ayarları burada kullanılabilir

    def send(self, output_data):
        """Çıktı verisini INFO seviyesinde konsola yazar."""
        if output_data is not None:
            # logging.info(f"Evo Çıktısı [Console]: {output_data}")
            # Ana run_evo döngüsünde loglandığı için burada tekrar INFO seviyesinde loglamaya gerek yok.
            # Debug seviyesinde ham çıktıyı loglayabiliriz.
             logging.debug(f"OutputChannel '{self.name}': Ham çıktı alindi, konsola yaziliyor.")
        # else: logging.debug(f"OutputChannel '{self.name}': Gönderilecek çıktı verisi None.")


# Gelecekteki kanallar için placeholderlar
class WebAPIOutputChannel(OutputChannel):
    """Çıktıyı bir Web API endpoint'i aracılığıyla dışarı sunan kanal (Placeholder)."""
    def __init__(self, config=None):
        super().__init__("web_api", config)
        # Web API başlatma mantığı (örn: thread içinde Flask sunucusu) buraya gelecek.
        # self.api_server = Flask(__name__)
        # self.api_thread = threading.Thread(target=self._run_server)
        # self.latest_output = None # Son çıktıyı saklamak için
        # self.api_port = self.config.get('port', 5000)

    def send(self, output_data):
        """Çıktı verisini dahili olarak saklar ve API üzerinden sunar (Placeholder)."""
        logging.debug(f"OutputChannel '{self.name}': Ham çıktı alindi, dahili olarak saklaniyor (Placeholder).")
        # if output_data is not None:
        #      self.latest_output = output_data # Son çıktıyı güncelle
        pass # Şimdilik bir şey yapmıyor

    # def _run_server(self):
    #     # Flask sunucusunu başlatma kodu
    #     # self.api_server.run(port=self.api_port)
    #     pass
    # def cleanup(self):
    #     logging.info(f"OutputChannel '{self.name}' sunucusu durduruluyor (Placeholder)...")
    #     # Sunucuyu durdurma kodu


# Diğer potansiyel kanallar:
# class AudioOutputChannel(OutputChannel): ... (TTS entegrasyonu)
# class FileOutputChannel(OutputChannel): ... (Çıktıyı dosyaya yazma)
# class RoboticsOutputChannel(OutputChannel): ... (Fiziksel komut gönderme)


class InteractionAPI:
    """
    Evo'nun dış dünya ile iletişim kurduğu arayüzü temsil eder.
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

        available_channels = {
            'console': ConsoleOutputChannel,
            'web_api': WebAPIOutputChannel, # Placeholder kanal
            # Diğer kanallar geldikçe buraya eklenecek
            # 'audio': AudioOutputChannel,
            # 'file': FileOutputChannel,
        }

        for channel_name in enabled_channels:
            if channel_name in available_channels:
                try:
                    # Kanalı config'den ilgili ayarları ile başlat
                    channel_config = self.config.get('channel_configs', {}).get(channel_name, {})
                    self.output_channels[channel_name] = available_channels[channel_name](channel_config)
                    logging.info(f"InteractionAPI: OutputChannel '{channel_name}' başarıyla başlatıldı.")
                except Exception as e:
                    logging.critical(f"InteractionAPI: OutputChannel '{channel_name}' başlatılırken kritik hata oluştu: {e}", exc_info=True)
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
        if output_data is None:
            # logging.debug("InteractionAPI: Gönderilecek çıktı verisi yok.")
            return # None çıktı gelirse gönderme

        if not self.output_channels:
            # logging.warning("InteractionAPI: Hiçbir OutputChannel aktif değil, çıktı gönderilemedi.")
            return # Aktif kanal yoksa gönderme

        # logging.debug(f"InteractionAPI: Çikti alindi. İçerik ilk 50 karakter: '{str(output_data)[:50]}', Gönderilecek kanal sayısı: {len(self.output_channels)}")

        # Tüm aktif kanallara çıktıyı gönder
        for channel_name, channel_obj in self.output_channels.items():
            try:
                # logging.debug(f"InteractionAPI: Çikti OutputChannel '{channel_name}' kanalına gönderiliyor.")
                channel_obj.send(output_data)
            except Exception as e:
                logging.error(f"InteractionAPI: OutputChannel '{channel_name}' kanalına çıktı gönderilirken hata oluştu: {e}", exc_info=True)


    def stop(self):
        """
        InteractionAPI modülünü ve tüm aktif çıktı kanallarını durdurur/temizler.
        """
        logging.info("InteractionAPI durduruluyor...")
        # Tüm aktif kanalların cleanup metodunu çağır
        for channel_name, channel_obj in self.output_channels.items():
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
    print("InteractionAPI modülü test ediliyor...")

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

    except Exception as e:
        logging.exception("InteractionAPI test sırasında hata oluştu:")

    finally:
        if api:
            api.stop() # Kaynakları temizle

    print("\nInteractionAPI modülü testi bitti.")