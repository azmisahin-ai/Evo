# src/interaction/api.py
import logging

# Output kanallarını import et
from .output_channels import ConsoleOutputChannel, WebAPIOutputChannel, OutputChannel # OutputChannel base sınıfı da gerekli olabilir

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)


class InteractionAPI:
    """
    Evo'nın dış dünya ile iletişim arayüzü.
    Motor Control'den gelen tepkileri alır ve aktif çıktı kanallarına gönderir.
    Gelecekte dış dünyadan girdi de alacak (Input kanalları).
    """
    def __init__(self, config):
        self.config = config
        self.enabled_channels = config.get('enabled_channels', ['console']) # Config'ten aktif kanalları al
        self.channel_configs = config.get('channel_configs', {}) # Kanal bazlı özel ayarları al

        self.output_channels = {} # Aktif çıktı kanalı objelerini tutacak sözlük

        logger.info("InteractionAPI modülü başlatılıyor...")
        logger.info(f"InteractionAPI: Konfigurasyondan aktif kanallar: {self.enabled_channels}")


        # Çıktı kanallarını başlat
        # Hata yönetimi: Kanal başlatma hatalarını yakala
        channel_classes = { # Desteklenen kanal sınıfları
            'console': ConsoleOutputChannel,
            'web_api': WebAPIOutputChannel,
            # Diğer kanallar buraya eklenecek
        }

        for channel_name in self.enabled_channels:
            channel_class = channel_classes.get(channel_name)
            if channel_class:
                channel_config = self.channel_configs.get(channel_name, {})
                try:
                    # Kanal objesini oluştur
                    channel_instance = channel_class(channel_config)
                    self.output_channels[channel_name] = channel_instance
                    logger.info(f"InteractionAPI: OutputChannel '{channel_name}' başarıyla başlatıldı.")
                except Exception as e:
                    # Kanal başlatılırken hata oluştu
                    logger.error(f"InteractionAPI: OutputChannel '{channel_name}' başlatılırken hata oluştu: {e}", exc_info=True)
                    # Hata veren kanalı aktif listeye ekleme
                    if channel_name in self.output_channels:
                         del self.output_channels[channel_name]
            else:
                # Config'te tanımlı ama kodda karşılığı olmayan kanal
                logger.warning(f"InteractionAPI: Bilinmeyen OutputChannel adı konfigurasyonda: '{channel_name}'. Bu kanal atlandı.")

        logger.info(f"InteractionAPI modülü başlatıldı. Aktif Output Kanalları: {list(self.output_channels.keys())}")


    def send_output(self, output_data):
        """
        Motor Control'den gelen çıktıyı tüm aktif çıktı kanallarına gönderir.

        Args:
            output_data (any): Gönderilecek çıktı verisi.
        """
        # Temel hata yönetimi: Gönderilecek veri None ise işlem yapma
        if output_data is None:
            logger.debug("InteractionAPI: Gönderilecek çıktı verisi None. Gönderme atlanıyor.")
            return

        logger.debug(f"InteractionAPI: Çikti {list(self.output_channels.keys())} kanallarına gönderiliyor.")

        # Her aktif kanala çıktıyı gönder
        # Hata yönetimi: Bir kanal hata verse bile diğerleri devam etmeli
        for channel_name, channel_instance in self.output_channels.items():
            try:
                # Kanalın send metodunu çağır
                # send metotlarının kendi içlerinde de hata yakalama olmalı.
                # Burada yakaladığımız hata, send metodunun kendisinin çağrılması sırasında
                # veya send metodunun içindeki *işlenmemiş* bir hatadan kaynaklanır.
                channel_instance.send(output_data)
                # logger.debug(f"InteractionAPI: Çikti OutputChannel '{channel_name}' kanalına gönderildi.")
            except Exception as e:
                # Kanalın send metodunu çağırırken veya çalıştırırken hata
                logger.error(f"InteractionAPI: OutputChannel '{channel_name}' send metodu çalıştırılırken beklenmedik hata: {e}", exc_info=True)
                # Bu kanalın bir daha kullanılmaması için listeden çıkarılabilir (gelecekteki bir iyileştirme)
                # del self.output_channels[channel_name] # Dikkat: Döngü sırasında dict'i değiştirmek sorun yaratabilir


    def start(self):
        """API servisini veya diğer arayüzleri başlatır (Gelecek)."""
        # Eğer bir API servisi thread veya process olarak çalışacaksa burası kullanılacak
        logger.info("InteractionAPI başlatılıyor (Placeholder)...")
        # Örn: self.api_thread = threading.Thread(target=self._run_api_server); self.api_thread.start()

    def stop(self):
        """API servisini veya arayüzleri durdurur ve kaynakları temizler."""
        logger.info("InteractionAPI durduruluyor...")
        # Alt kanalların cleanup metodunu çağır
        for channel_name, channel_instance in list(self.output_channels.items()): # list() kopyası üzerinde dönmek güvenlidir
            try:
                channel_instance.cleanup()
                logger.info(f"InteractionAPI: OutputChannel '{channel_name}' temizlendi.")
            except Exception as e:
                 logger.error(f"InteractionAPI: OutputChannel '{channel_name}' temizlenirken hata oluştu: {e}", exc_info=True)
                 # Hata veren kanalı temizlenmiş gibi kabul edip listeden çıkarabiliriz.
                 if channel_name in self.output_channels:
                      del self.output_channels[channel_name]

        self.output_channels = {} # Listeyi tamamen boşalt

        # API servisi thread/process durdurma mantığı buraya gelecek (eğer start metodu kullanıldıysa)
        # if self.api_thread and self.api_thread.is_alive(): self.api_thread.stop(); self.api_thread.join()


        logger.info("InteractionAPI objesi temizlendi.")


    # Gelecekte:
    # def receive_input(self):
    #     """Dış dünyadan girdi alır (örn: API endpoint'inden gelen istek)."""
    #     pass