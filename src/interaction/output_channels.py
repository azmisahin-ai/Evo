# src/interaction/output_channels.py
import logging
# import time # Gerekirse
# import requests # WebAPI için gerekirse

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

# --- OutputChannel Base Class ---
class OutputChannel:
    """
    Farklı çıktı kanalları için temel sınıf.
    Diğer kanallar bu sınıftan miras almalıdır.
    """
    def __init__(self, name, config):
        self.name = name
        self.config = config
        # Her channel kendi logger'ını oluşturur
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.logger.info(f"OutputChannel '{self.name}' başlatılıyor.")

    def send(self, output_data):
        """
        İşlenmiş çıktıyı ilgili kanala gönderir.
        Alt sınıflar bu metodu implement etmelidir.

        Args:
            output_data (any): Gönderilecek çıktı verisi (format kanala bağlı).
        """
        raise NotImplementedError("Alt sınıflar 'send' metodunu implement etmelidir.")

    def cleanup(self):
        """
        Kanal kaynaklarını temizler. Gerekirse alt sınıflar override eder.
        """
        self.logger.info(f"OutputChannel '{self.name}' temizleniyor.")
        pass


# --- Console Output Channel ---
class ConsoleOutputChannel(OutputChannel):
    """
    Çıktıyı konsola yazdırır.
    """
    def __init__(self, config):
        super().__init__("console", config)
        self.logger.info("ConsoleOutputChannel başlatıldı.")

    def send(self, output_data):
        """
        Çıktıyı konsola yazdırır.

        Args:
            output_data (str): Konsola yazdırılacak metin.
        """
        # Temel hata yönetimi: Girdi tipi kontrolü
        if not isinstance(output_data, str):
             self.logger.warning(f"OutputChannel '{self.name}': Beklenmeyen çıktı tipi: {type(output_data)}. String bekleniyordu. Çıktı string'e çevriliyor.")
             try:
                 output_data = str(output_data)
             except Exception as e:
                 self.logger.error(f"OutputChannel '{self.name}': Çıktı string'e çevrilemedi: {e}", exc_info=True)
                 return # String'e çevrilemezse gönderme


        self.logger.debug(f"OutputChannel '{self.name}': Ham çıktı alindi, isleniyor/hazirlaniyor (Konsol).")

        try:
            # Konsola yazdırma işlemi
            print(f"Evo Çıktısı: {output_data}") # main loop'taki print yerine burası kullanılacak
            self.logger.debug(f"OutputChannel '{self.name}': Çıktı konsola yazdırıldı.")

        except Exception as e:
             # Konsola yazdırma sırasında beklenmedik hata (nadiren olur)
             self.logger.error(f"OutputChannel '{self.name}': Konsola yazdırma hatasi: {e}", exc_info=True)
             # Hata durumunda yapacak çok bir şey yok, loglamak yeterli.


    def cleanup(self):
        """Konsol kanalının temizliği (gerek yok)."""
        self.logger.info(f"OutputChannel '{self.name}' temizleniyor.")
        super().cleanup()


# --- Web API Output Channel (Placeholder) ---
class WebAPIOutputChannel(OutputChannel):
    """
    Çıktıyı bir Web API endpoint'ine gönderir (Placeholder).
    """
    def __init__(self, config):
        super().__init__("web_api", config)
        self.port = config.get('port', 5000)
        self.logger.info(f"WebAPIOutputChannel başlatıldı. Port: {self.port}")
        # API sunucusunu başlatma mantığı buraya gelebilir (ayrı bir thread/process?)

    def send(self, output_data):
        """
        Çıktıyı Web API'ye gönderir (Placeholder).

        Args:
            output_data (any): Gönderilecek veri (örn: dict, json string).
        """
        # Temel hata yönetimi ve placeholder mantığı
        self.logger.debug(f"OutputChannel '{self.name}': Ham çıktı alindi, isleniyor/hazirlaniyor (Web API).")

        try:
            # Web API'ye gönderme mantığı buraya gelecek (örn: requests.post)
            # Şimdilik sadece loglayalım
            self.logger.info(f"WebAPIOutputChannel: API endpoint'e çıktı gönderme simüle edildi: {output_data}")
            # Gelecekte:
            # api_url = f"http://localhost:{self.port}/output" # Örnek endpoint
            # try:
            #     response = requests.post(api_url, json={'data': output_data})
            #     response.raise_for_status() # HTTP hatalarını yakala
            #     self.logger.debug(f"WebAPIOutputChannel: Çıktı API'ye başarıyla gönderildi. Yanıt: {response.status_code}")
            # except requests.exceptions.RequestException as e:
            #     self.logger.error(f"WebAPIOutputChannel: API'ye gönderme hatasi: {e}", exc_info=True)
            # except Exception as e:
            #      self.logger.error(f"WebAPIOutputChannel: API gönderme sırasında beklenmedik hata: {e}", exc_info=True)


        except Exception as e:
             # Genel beklenmedik hata
             self.logger.error(f"OutputChannel '{self.name}': Gönderme sırasında beklenmedik hata: {e}", exc_info=True)
             # Hata durumunda yapacak çok bir şey yok, loglamak yeterli.


    def cleanup(self):
        """API sunucusunu kapatma vb."""
        self.logger.info(f"WebAPIOutputChannel '{self.name}' temizleniyor.")
        # API sunucusunu kapatma mantığı buraya gelebilir (eğer burada başlatıldıysa)
        super().cleanup()