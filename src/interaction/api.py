# src/interaction/api.py
#
# Evo'nın dış dünya ile iletişim arayüzünü temsil eder.
# Motor Control'den gelen tepkileri alır ve aktif çıktı kanallarına gönderir.
# Farklı çıktı kanallarını yönetir.
# Kanal başlatma, gönderme ve temizleme sırasında oluşabilecek hataları yönetir.
# Gelecekte dış dünyadan girdi de alacak (Input kanalları).

import logging # Loglama için.
# import threading # Web API'si bir thread olarak çalışacaksa gerekebilir (Gelecek).
# import requests # API endpoint'lerine çıktı göndermek için gerekebilir (Gelecek).

# Yardımcı fonksiyonları import et
from src.core.config_utils import get_config_value
from src.core.utils import check_input_not_none, check_input_type # <<< check_input_not_none, check_input_type import edildi

# Output kanallarını import et
from .output_channels import ConsoleOutputChannel, WebAPIOutputChannel, OutputChannel # Base sınıfı da referans için import edildi.

# Bu modül için bir logger oluştur
# 'src.interaction.api' adında bir logger döndürür.
logger = logging.getLogger(__name__)


class InteractionAPI:
    """
    Evo'nın dış dünya ile iletişim arayüzü sınıfı.

    MotorControl modülünden gelen tepkileri (output_data) alır
    ve yapılandırmada belirtilen tüm aktif çıktı kanallarına gönderir.
    Farklı çıktı kanallarını (konsol, Web API vb.) yönetir.
    Kanal başlatma, gönderme ve temizleme sırasında oluşabilecek hataları yönetir.
    Gelecekte dış dünyadan girdi alımı için arayüzler de buraya eklenecek.
    """
    def __init__(self, config):
        """
        InteractionAPI modülünü başlatır.

        Yapılandırmadan aktif çıktı kanallarını okur ve bu kanalları başlatır.
        Kanal başlatma sırasında oluşabilecek hataları yönetir.

        Args:
            config (dict): Interaction modülü yapılandırma ayarları.
                           'enabled_channels': Aktif edilecek çıktı kanallarının adlarını içeren liste (örn: ['console', 'web_api']).
                                               Beklenen tip: liste.
                           'channel_configs': Kanal adlarına göre özel yapılandırma ayarları içeren sözlük (örn: {'web_api': {'port': 5000}}).
                                            Beklenen tip: sözlük.
        """
        self.config = config
        logger.info("InteractionAPI modülü başlatılıyor...")

        # Yapılandırmadan aktif kanalların listesini alırken get_config_value kullan.
        # enabled_channels için tip kontrolü (liste) yapalım. Varsayılan ['console'].
        self.enabled_channels = get_config_value(config, 'enabled_channels', ['console'], expected_type=list, logger_instance=logger)

        # Yapılandırmadan kanal bazlı özel ayarları alırken get_config_value kullan.
        # channel_configs için tip kontrolü (sözlük) yapalım. Varsayılan {}.
        self.channel_configs = get_config_value(config, 'channel_configs', {}, expected_type=dict, logger_instance=logger)


        self.output_channels = {} # Başlatılan aktif çıktı kanalı objelerini tutacak sözlük.

        logger.info(f"InteractionAPI: Konfigurasyondan aktif kanallar: {self.enabled_channels}")

        # Desteklenen çıktı kanalı sınıflarının eşleştirmesi.
        # Yeni kanal türleri eklendikçe bu sözlük güncellenmeli.
        channel_classes = {
            'console': ConsoleOutputChannel,
            'web_api': WebAPIOutputChannel, # Placeholder sınıfı
            # Gelecekte diğer kanallar buraya eklenecek (örn: 'file': FileOutputChannel, 'robot': RobotOutputChannel)
        }

        # Yapılandırmada belirtilen her aktif kanalı başlatmayı dene.
        # enabled_channels listesinin tipi zaten get_config_value ile kontrol edildi.
        # Şimdi listedeki her öğenin string olup olmadığını kontrol et.
        for channel_name in self.enabled_channels:
            if not isinstance(channel_name, str):
                 logger.warning(f"InteractionAPI: 'enabled_channels' listesinde beklenmeyen öğe tipi: {type(channel_name)}. String bekleniyordu. Bu öğe atlandı.")
                 continue # String değilse bu öğeyi atla.

            # İlgili kanal sınıfı tanımlanmış mı kontrol et.
            channel_class = channel_classes.get(channel_name)
            if channel_class:
                # Kanal sınıfı bulundu, şimdi başlatmayı dene.
                # Konfigurasyondan o kanala ait özel ayarları al, yoksa boş sözlük kullan.
                # self.channel_configs dict olduğundan emin olundu (get_config_value ile). get() güvenli.
                channel_config = self.channel_configs.get(channel_name, {})
                try:
                    # Kanal objesini oluştur ve başlat.
                    # Kanal init metotlarının hata durumunda None döndürmesi veya exception atması beklenir.
                    channel_instance = channel_class(channel_config)
                    # Başarıyla başlatıldıysa aktif kanallar sözlüğüne ekle.
                    if channel_instance is not None:
                         self.output_channels[channel_name] = channel_instance
                         logger.info(f"InteractionAPI: OutputChannel '{channel_name}' başarıyla başlatıldı.")
                    else:
                         # Kanal init None döndürdüyse (kendi içinde hata yönettiyse)
                         logger.error(f"InteractionAPI: OutputChannel '{channel_name}' başlatma sırasında None döndürdü.")

                except Exception as e:
                    # Kanal başlatılırken beklenmedik bir istisna oluştuysa logla.
                    # Bu tür bir hata başlatma sırasında kritik kabul edilmiyor policy gereği,
                    # sadece o kanal kullanılamaz hale gelir.
                    logger.error(f"InteractionAPI: OutputChannel '{channel_name}' başlatılırken hata oluştu: {e}", exc_info=True)
                    # Hata veren kanalı aktif kanallar sözlüğüne eklememek önemlidir.


            else:
                # Config'te adı geçen ama channel_classes sözlüğünde karşılığı olmayan kanal adları için uyarı.
                logger.warning(f"InteractionAPI: Konfigurasyonda bilinmeyen OutputChannel adı: '{channel_name}'. Bu kanal atlandı.")

        # Başlatılan InteractionAPI modülünün genel durumunu logla.
        # Aktif çıktı kanallarının listesini göster.
        logger.info(f"InteractionAPI modülü başlatıldı. Aktif Output Kanalları: {list(self.output_channels.keys())}")

        # Eğer Web API kanalı aktifse, API sunucusunu başlatma mantığı buraya gelebilir (ayrı bir thread/process?).
        # Bu, InteractionAPI.start() metoduna taşınmıştır.
        # if 'web_api' in self.output_channels and hasattr(self.output_channels['web_api'], 'start_server'):
        #      logger.info("InteractionAPI: Web API sunucusu başlatılıyor...")
        #      self.output_channels['web_api'].start_server() # Web API başlatma metodu


        # TODO: Girdi kanallarını başlatma mantığı buraya gelecek (Gelecekte TODO).
        # self._initialize_input_channels() # Gelecekte TODO


    def send_output(self, output_data):
        """
        MotorControl'den gelen çıktıyı (tepki) tüm aktif çıktı kanallarına gönderir.

        Her aktif OutputChannel objesinin `send` metodunu çağırır.
        Bir kanala gönderme sırasında hata oluşsa bile diğer kanallara gönderme devam eder.
        Gönderilecek veri None ise işlem yapmaz.

        Args:
            output_data (any): Motor Control modülünden gelen gönderilecek çıktı verisi (tepki).
                               Formatı kanaldan kanala değişebilir (str, dict, numpy array vb.).
                               None olabilir, bu durumda gönderme atlanır.
        """
        # Hata yönetimi: Gönderilecek veri None ise işlem yapma. check_input_not_none kullan.
        if not check_input_not_none(output_data, input_name="output_data", logger_instance=logger):
            # Çıktı verisi None ise, gönderme işlemi atlanır. Bu bir hata değil.
            return

        # DEBUG logu: Çıktının hangi kanallara gönderileceği.
        logger.debug(f"InteractionAPI: Çikti {list(self.output_channels.keys())} kanallarına gönderiliyor.")

        # Her aktif kanala çıktıyı gönderme döngüsü.
        # Hata yönetimi: Bir kanala gönderme hatası diğerlerini etkilememeli.
        # output_channels sözlüğü üzerinde dönerken, send metodu içinde bu sözlükte değişiklik
        # yapılmadığı varsayılır. Eğer send metodu kanalı pasifize edip sözlükten silerse,
        # döngünün bir kopyası üzerinde dönmek (list(self.output_channels.items())) daha güvenli olabilir.
        for channel_name, channel_instance in list(self.output_channels.items()):
            # channel_instance'ın None olup olmadığını kontrol et (başlatma hatası nedeniyle None olabilir).
            if channel_instance:
                try:
                    # Kanalın send metodunu çağır.
                    # send metotlarının kendi içlerinde de hata yakalama olmalı.
                    # Burada yakaladığımız hata, send metodunun kendisinin çağrılması sırasında
                    # veya send metodunun içindeki *işlenmemiş* bir hatadan kaynaklanır.
                    channel_instance.send(output_data)
                    # DEBUG logu: Hangi kanala gönderim yapıldığı.
                    # logger.debug(f"InteractionAPI: Çikti OutputChannel '{channel_name}' kanalına gönderildi.")

                except Exception as e:
                    # Kanalın send metodunu çağırırken veya çalıştırırken beklenmedik bir hata oluşursa logla.
                    logger.error(f"InteractionAPI: OutputChannel '{channel_name}' send metodu çalıştırılırken beklenmedik hata: {e}", exc_info=True)
                    # Hata veren bu kanalın bir daha kullanılmaması için aktif listesinden çıkarılması düşünülebilir
                    # (Gelecekteki bir iyileştirme/policy). Şu an sadece loglayıp devam ediyoruz.
                    # del self.output_channels[channel_name] # Döngü sırasında dict'i değiştirmek sorun yaratabilir!
            # else:
                 # Eğer channel_instance None ise, bu zaten başlatma sırasında loglanmıştır.
                 # Burada tekrar loglamaya gerek yok.
                 # logger.debug(f"InteractionAPI: OutputChannel '{channel_name}' objesi None, gönderme atlandı.")


    def start(self):
        """
        Interaction arayüzlerini başlatır (örn: Web API sunucusu).

        Bu metot, run_evo.py tarafından program başlatıldığında çağrılır.
        Şimdilik placeholder. Eğer bir API servisi thread veya process olarak
        çalışacaksa burası kullanılacak.
        """
        logger.info("InteractionAPI başlatılıyor (Placeholder)...")
        # Eğer Web API kanalı aktifse ve sunucuyu başlatma yeteneği varsa, burası çağrılabilir.
        # if 'web_api' in self.output_channels and hasattr(self.output_channels['web_api'], 'start_server'):
        #      logger.info("InteractionAPI: Web API sunucusu başlatılıyor...")
        #      self.output_channels['web_api'].start_server() # Web API başlatma metodu


    def stop(self):
        """
        Interaction arayüzlerini durdurur ve tüm aktif çıktı kanallarının kaynaklarını temizler.

        Bu metot, run_evo.py tarafından program sonlanırken çağrılır.
        """
        logger.info("InteractionAPI durduruluyor...")
        # Tüm aktif çıktı kanallarının cleanup metotlarını çağır.
        # Sözlük üzerinde dönerken değişiklik yapmamak için list() kullanarak bir kopya üzerinde dönülür.
        for channel_name, channel_instance in list(self.output_channels.items()):
            # Eğer obje None değilse (başlatma hatası olmadıysa) ve cleanup metodu varsa
            if channel_instance and hasattr(channel_instance, 'cleanup'):
                logger.info(f"InteractionAPI: OutputChannel '{channel_name}' temizleniyor...")
                try:
                    channel_instance.cleanup()
                    logger.info(f"InteractionAPI: OutputChannel '{channel_name}' temizlendi.")
                except Exception as e:
                     # Temizleme sırasında beklenmedik bir hata oluşursa logla.
                     logger.error(f"InteractionAPI: OutputChannel '{channel_name}' temizlenirken hata oluştu: {e}", exc_info=True)
                # Hata veren veya temizlenen kanalı listeden çıkarmak (isteğe bağlı).
                # if channel_name in self.output_channels: # Zaten kopya üzerinde dönüyoruz, bu kontrol gerekli olmayabilir.
                #      del self.output_channels[channel_name]

        # Aktif kanallar sözlüğünü tamamen boşalt (temizlendiğini belirtmek için).
        self.output_channels = {} # Veya self.output_channels = None

        # Eğer API sunucusu bir thread/process olarak başlatıldıysa, durdurma mantığı buraya gelecek (Gelecek TODO).
        # if hasattr(self, 'api_thread') and self.api_thread and self.api_thread.is_alive():
        #      logger.info("InteractionAPI: API sunucusu durduruluyor...")
        #      self.api_thread.stop() # API thread'inin stop metodu olmalı
        #      self.api_thread.join() # Thread'in bitmesini bekle


        logger.info("InteractionAPI objesi silindi.")


    # Gelecekte:
    # def receive_input(self):
    #     """Dış dünyadan girdi alır (örn: API endpoint'inden gelen istek)."""
    #     pass # Implement edilecek input alma mekanizmaları