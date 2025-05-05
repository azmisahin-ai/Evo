# src/interaction/output_channels.py
#
# Evo'nın dış dünyaya yönelik çıktı kanallarını tanımlar.
# Farklı türdeki çıktılar (metin, ses, görsel) belirli kanallara yönlendirilir.

import logging # Loglama için.
# import time # Gerekirse zamanlama için.
# import requests # WebAPI'ye istek göndermek için gerekebilir (Gelecek).
# import json # JSON formatı için gerekebilir (Gelecek).
# import flask # WebAPI sunucusu için gerekebilir (Gelecek).


# Bu modül için bir logger oluştur
# 'src.interaction.output_channels' adında bir logger döndürür.
logger = logging.getLogger(__name__)

# --- OutputChannel Base Class ---
class OutputChannel:
    """
    Farklı çıktı kanalları için temel (base) sınıf.

    Tüm özel çıktı kanalı sınıfları (ConsoleOutputChannel, WebAPIOutputChannel vb.)
    bu sınıftan miras almalıdır. Temel başlatma (__init__), çıktı gönderme (send)
    ve kaynak temizleme (cleanup) metotlarını tanımlar.
    """
    def __init__(self, name, config):
        """
        OutputChannel'ın temelini başlatır.

        Her kanalın bir adı ve yapılandırması olur. Kendi logger'ını oluşturur.

        Args:
            name (str): Kanalın adı (örn: 'console', 'web_api').
            config (dict): Bu kanala özel yapılandırma ayarları.
        """
        self.name = name
        self.config = config
        # Her channel kendi adlandırılmış logger'ını oluşturur.
        # Bu logger'ın adı 'src.interaction.output_channels.KanalAdı' şeklinde olur.
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.logger.info(f"OutputChannel '{self.name}' başlatılıyor.")
        # Base sınıf başlatma tamamlandı.

    def send(self, output_data):
        """
        İşlenmiş çıktıyı ilgili kanala gönderme metodu.

        Bu metot temel sınıfta implement edilmemiştir ve alt sınıflar tarafından
        override (geçersiz kılınarak kendi mantıklarıyla doldurulmalı) edilmelidir.
        output_data'nın formatı kanaldan kanala değişir.

        Args:
            output_data (any): Motor Control'den gelen gönderilecek çıktı verisi.
                               Formatı kanala bağlıdır (string, sayı, dict, numpy array vb.).
        """
        raise NotImplementedError("Alt sınıflar 'send' metodunu implement etmelidir.")

    def cleanup(self):
        """
        Kanal tarafından kullanılan kaynakları temizler.

        Bu metot temel sınıfta placeholder olarak tanımlanmıştır. Özel kaynak (dosya,
        ağ bağlantısı, thread vb.) kullanan alt sınıflar bu metodu override ederek
        kendi temizleme mantıklarını implement etmelidir.
        module_loader.py ve InteractionAPI.stop() bu metodu program sonlanırken çağırır (varsa).
        """
        self.logger.info(f"OutputChannel '{self.name}' temizleniyor.")
        pass


# --- Console Output Channel ---
class ConsoleOutputChannel(OutputChannel):
    """
    Çıktıyı sistem konsoluna (terminale) yazdıran çıktı kanalı.

    Genellikle metin tabanlı çıktılar için kullanılır.
    """
    def __init__(self, config):
        """
        ConsoleOutputChannel'ı başlatır.

        Args:
            config (dict): Kanal yapılandırma ayarları (bu kanal için özel ayar beklenmez şimdilik).
        """
        # Temel sınıfın __init__ metodunu çağır. Kanal adını "console" olarak belirler.
        super().__init__("console", config)
        self.logger.info("ConsoleOutputChannel başlatıldı.")

    def send(self, output_data):
        """
        Çıktıyı konsola yazdırır.

        Gelen output_data'yı string'e çevirmeyi dener ve konsola standart bir formatla yazdırır.
        string'e çevirme veya yazdırma sırasında hata oluşursa yakalar ve loglar.

        Args:
            output_data (any): Konsola yazdırılacak veri. Genellikle bir string beklenir.
                               String değilse str() ile çevrilmeye çalışılır.
        """
        # Hata yönetimi: Gelen verinin string olup olmadığını kontrol et.
        # String değilse string'e çevirmeyi dene.
        if not isinstance(output_data, str):
             # String değilse uyarı logu ver ve string'e çevirmeyi dene.
             self.logger.warning(f"OutputChannel '{self.name}': Beklenmeyen çıktı tipi: {type(output_data)}. String bekleniyordu. Çıktı string'e çevriliyor.")
             try:
                 # str() fonksiyonu çoğu Python objesini string'e çevirebilir.
                 output_to_print = str(output_data)
             except Exception as e:
                 # string'e çevirme sırasında hata oluşursa hata logu ver ve gönderme işlemini durdur.
                 self.logger.error(f"OutputChannel '{self.name}': Çıktı string'e çevrilemedi: {e}", exc_info=True)
                 return # String'e çevrilemezse gönderme işlemini atla.
        else:
             # Gelen veri zaten string ise doğrudan kullan.
             output_to_print = output_data


        # DEBUG logu: Ham çıktının alındığı ve işleneceği bilgisi.
        self.logger.debug(f"OutputChannel '{self.name}': Ham çıktı alindi, isleniyor/hazirlaniyor (Konsol).")

        try:
            # İşlenmiş çıktıyı (string) konsola yazdır.
            # Python'daki print() fonksiyonu StreamHandler ile loglamadan farklı olarak
            # doğrudan sys.stdout'a yazar. run_evo.py'deki main loop'ta log.info kullanılıyordu,
            # şimdi kanal kullandığımız için print kullanmak daha uygun olabilir.
            # Ancak loglama sistemini kullanarak INFO seviyesinde yazdırmak da bir seçenektir.
            # Şimdilik print() kullanalım.
            print(f"Evo Çıktısı: {output_to_print}") # Konsol çıktısını özelleştir.
            # Loglama ile de yazdırılabilir: self.logger.info(f"Evo Çıktısı: {output_to_print}")


            # DEBUG logu: Çıktının başarıyla konsola yazdırıldığı bilgisi.
            self.logger.debug(f"OutputChannel '{self.name}': Çıktı konsola yazdırıldı.")

        except Exception as e:
             # Konsola yazdırma sırasında beklenmedik bir hata oluşursa (nadiren olur).
             self.logger.error(f"OutputChannel '{self.name}': Konsola yazdırma hatasi: {e}", exc_info=True)
             # Hata durumında yapacak çok bir şey yok, hatayı loglamak yeterlidir.


    def cleanup(self):
        """
        ConsoleOutputChannel kaynaklarını temizler.

        Konsol çıktısı özel bir kaynak gerektirmediği için temizleme adımı içermez,
        sadece temel sınıfın temizleme metodunu çağırır.
        """
        # Bilgilendirme logu.
        self.logger.info(f"ConsoleOutputChannel '{self.name}' temizleniyor.")
        # Temel sınıfın cleanup metodunu çağır (sadece loglama yapar).
        super().cleanup()


# --- Web API Output Channel (Placeholder) ---
class WebAPIOutputChannel(OutputChannel):
    """
    Çıktıyı bir Web API endpoint'ine gönderen çıktı kanalı (Placeholder).

    Bu kanal, InteractionAPI'nin dış dünya ile HTTP veya benzeri protokoller
    üzerinden iletişim kurmasını sağlar.
    """
    def __init__(self, config):
        """
        WebAPIOutputChannel'ı başlatır.

        Args:
            config (dict): Kanal yapılandırma ayarları.
                           'port': API'nin çalıştığı port (int, varsayılan 5000).
                           'host': API'nin çalıştığı host (str, varsayılan '127.0.0.1').
        """
        # Temel sınıfın __init__ metodunu çağır. Kanal adını "web_api" olarak belirler.
        super().__init__("web_api", config)
        # Yapılandırmadan ayarları al, yoksa varsayılanları kullan.
        self.port = config.get('port', 5000)
        self.host = config.get('host', '127.0.0.1') # Gelecekte kullanılabilir

        self.logger.info(f"WebAPIOutputChannel başlatıldı. Port: {self.port}")
        # API sunucusunu başlatma mantığı buraya gelebilir (ayrı bir thread/process?) (Gelecek TODO).
        # self._start_api_server() # Gelecek TODO


    def send(self, output_data):
        """
        Çıktıyı Web API endpoint'ine gönderir (Placeholder implementasyon).

        Gelen output_data'yı (örn: dict, string) alır ve belirtilen bir
        API endpoint'ine HTTP POST isteği gibi gönderir. Şimdilik bu işlem
        sadece loglanır.
        Gönderme sırasında hata oluşursa yakalar ve loglar.

        Args:
            output_data (any): Gönderilecek çıktı verisi. Genellikle JSON'a çevrilebilecek
                               bir dict veya string beklenir.
        """
        # Hata yönetimi: Gelen verinin geçerliliğini kontrol et (isteğe bağlı).
        # None veya boş dict/string gibi durumlar burada yönetilebilir veya gönderilebilir.

        self.logger.debug(f"OutputChannel '{self.name}': Ham çıktı alindi, isleniyor/hazirlaniyor (Web API).")

        try:
            # Web API'ye gönderme mantığı buraya gelecek (Gelecek TODO).
            # Örneğin, requests kütüphanesini kullanarak bir POST isteği gönderme.
            # api_url = f"http://{self.host}:{self.port}/output" # Örnek endpoint URL'si
            # headers = {'Content-Type': 'application/json'}
            # try:
            #     # Çıktı verisini JSON formatına çevir (gerekirse)
            #     # json_data = json.dumps(output_data) # Eğer output_data dict/list ise
            #     # Veya doğrudan gönder: data=output_data

            #     # POST isteğini gönder
            #     # response = requests.post(api_url, headers=headers, data=json_data, timeout=5) # timeout eklemek iyi pratik
            #     # response.raise_for_status() # HTTP hatalarını (4xx, 5xx) bir Exception olarak yükseltir

            #     # Başarılı gönderme logu
            #     # self.logger.debug(f"WebAPIOutputChannel: Çıktı API'ye başarıyla gönderildi. Durum Kodu: {response.status_code}")

            # except requests.exceptions.RequestException as e:
            #     # requests kütüphanesinden kaynaklanan spesifik hatalar (bağlantı hatası, timeout, HTTP hata kodu vb.)
            #     self.logger.error(f"WebAPIOutputChannel: API'ye gönderme hatasi: {e}", exc_info=True)
            # except Exception as e:
            #      # JSON çevirme veya başka beklenmedik hatalar
            #      self.logger.error(f"WebAPIOutputChannel: API gönderme sırasında beklenmedik hata: {e}", exc_info=True)

            # Şimdilik sadece loglayalım ve gönderme işlemini simüle edelim.
            self.logger.info(f"WebAPIOutputChannel: API endpoint'e çıktı gönderme simüle edildi. Gönderilen veri: {output_data}")


        except Exception as e:
             # send metodu içindeki ana try bloğunu yakalayan genel hata yakalama.
             # Bu, içteki try-except blokları tarafından yakalanmayan bir hata ise devreye girer.
             self.logger.error(f"OutputChannel '{self.name}': Gönderme sırasında beklenmedik hata: {e}", exc_info=True)
             # Hata durumunda yapacak çok bir şey yok, loglamak yeterlidir.


    def cleanup(self):
        """
        WebAPIOutputChannel kaynaklarını temizler.

        API sunucusunu durdurma (eğer burada başlatıldıysa) veya açık bağlantıları kapatma
        mantığı buraya gelebilir.
        module_loader.py ve InteractionAPI.stop() bu metodu program sonlanırken çağırır (varsa).
        """
        # Bilgilendirme logu.
        self.logger.info(f"WebAPIOutputChannel '{self.name}' temizleniyor.")
        # API sunucusunu kapatma mantığı buraya gelebilir (eğer burada başlatıldıysa ve bir metodu varsa).
        # if hasattr(self, 'api_server') and self.api_server:
        #      self.logger.info(f"WebAPIOutputChannel: API sunucusu kapatılıyor (Port: {self.port})...")
        #      self.api_server.shutdown() # Flask development server veya başka bir server objesinin metodu


        # Temel sınıfın cleanup metodunu çağır (sadece loglama yapar).
        super().cleanup()

# TODO: Gelecekte eklenecek diğer çıktı kanalı sınıfları (örn: FileOutputChannel, RoboticArmChannel) buraya tanımlanacak.
# class FileOutputChannel(OutputChannel): ...
# class RoboticArmChannel(OutputChannel): ...