# src/core/logging_utils.py
import logging
import sys # Konsol çıktısı için StreamHandler'a gerekli olabilir
import logging.config # Dictionary tabanlı config için kullanabiliriz (şimdilik değil)

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

def setup_logging(config=None):
    """
    Configures the logging system for the Evo project based on provided config.

    Sets up console logging with a specific format.
    Sets the overall logging level for the root logger based on config or defaults.
    Ensures 'src' package logger is set appropriately (inherits from root).

    Args:
        config (dict, optional): Yapılandırma ayarları sözlüğü (genellikle main_config.yaml'den).
                                 Logging ayarları 'logging' anahtarı altında beklenir.
                                 Varsayılan olarak None.
    """
    # Mevcut logger ayarlarını temizle (run_evo'nun yeniden başlatılmasında çift loglama olmaması için)
    # Bu yöntem basicConfig kullanıldığında veya handler'lar manuel eklendiğinde işe yarar.
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    # handler'lar root logger'dan kaldırıldığında, formatter'ları da kaldırılabilir,
    # ama bu genellikle handler objesine bağlıdır. Yeni formatter tanımlamak daha güvenli.

    # --- Logging Level Ayarları ---
    default_level = logging.INFO # Varsayılan seviye INFO
    configured_level = default_level # Konfigurasyondan okunacak seviye

    if config and 'logging' in config and 'level' in config['logging']:
        level_name = config['logging']['level'].upper() # Config'teki string'i büyük harfe çevir
        try:
            # String log seviyesini (örn: "DEBUG") logging modülünün int karşılığına çevir
            configured_level = logging.getLevelName(level_name)
            # getLevelName string döndürürse (geçersiz isim), default kullanılır
            if isinstance(configured_level, str):
                 logger.warning(f"Logging: Konfigurasyonda geçersiz log seviyesi adı: '{level_name}'. Varsayılan seviye ({logging.getLevelName(default_level)}) kullanılıyor.")
                 configured_level = default_level # Geçersiz isimse varsayılanı kullan

        except Exception as e:
            logger.warning(f"Logging: Konfigurasyondaki log seviyesi okunurken hata: {e}. Varsayılan seviye ({logging.getLevelName(default_level)}) kullanılıyor.", exc_info=True)
            configured_level = default_level


    # Root logger seviyesini ayarla
    # Bu, log mesajlarının hangi minimum seviyeden itibaren root logger'a ulaşacağını belirler.
    # src modüllerinin DEBUG loglarını görmek için root logger'ın seviyesi en az DEBUG olmalı.
    # configure_level config'ten DEBUG gelirse, root DEBUG olur. INFO gelirse, root INFO olur.
    root_logger.setLevel(configured_level)
    logger.info(f"Logging: Root logger seviyesi ayarlandı: {logging.getLevelName(root_logger.level)}")


    # --- Handler (Çıktı Hedefi) Ayarları ---
    # Şimdilik sadece konsol çıktısı handler'ı ekleyeceğiz.
    # Gelecekte config'teki 'handlers' listesini okuyarak farklı handler'lar (dosya, vb.) eklenebilir.

    # Konsol handler'ı oluştur
    # StreamHandler default olarak sys.stderr'i kullanır, sys.stdout kullanmak tercih edilebilir.
    console_handler = logging.StreamHandler(sys.stdout)

    # Formatter oluştur (Modül adını içerecek şekilde)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Handler'a formatter'ı set et
    console_handler.setFormatter(formatter)

    # Handler seviyesini ayarla (İsteğe bağlı, handler kendi seviyesinin altındaki logları işlemez)
    # Genellikle handler seviyesi, root logger seviyesinden daha yüksek veya eşittir.
    # Konsol handler'ının her şeyi göstermesi için seviyesini root logger seviyesiyle aynı yapalım.
    console_handler.setLevel(configured_level) # Handler seviyesi root seviyesiyle aynı

    # Root logger'a handler'ı ekle
    root_logger.addHandler(console_handler)

    # --- Specifc Logger Levels ---
    # 'src' paketi logger seviyesini ayarla (isteğe bağlı ama netlik için iyi)
    # Eğer root_logger seviyesi DEBUG ise, src otomatik olarak DEBUG'i miras alır.
    # Ancak config'te genel seviye INFO iken src DEBUG loglarını görmek istersek,
    # burada src logger seviyesini DEBUG'e çekmeli ve handler seviyesini DEBUG'e indirmeliyiz.
    # Mevcut yapılandırmada root seviyesi config'e göre ayarlandığı için
    # src logger seviyesini manuel ayarlamaya gerek yok, root'tan miras alacaktır.
    # Yorum satırı olarak bırakalım veya kaldıralım:
    # logging.getLogger('src').setLevel(logging.DEBUG) # Genellikle gerekli değil eğer root DEBUG ise

    # setup_logging fonksiyonunun kendi logları (bu fonksiyondan gelen) artık görünür olacaktır.
    #logger.info("Logging system configured successfully based on config.")
    #logger.debug("This is a debug message from logging_utils (should only appear if level is DEBUG).")


# Örnek: Log seviyesi ismini int seviyeye çevirme yardımcı (logging.getLevelName zaten bunu yapıyor)
# def _get_logging_level(level_name):
#     """String log seviyesi adını logging modülünün seviyesine çevirir."""
#     level_map = {
#         'DEBUG': logging.DEBUG,
#         'INFO': logging.INFO,
#         'WARNING': logging.WARNING,
#         'ERROR': logging.ERROR,
#         'CRITICAL': logging.CRITICAL,
#     }
#     return level_map.get(level_name.upper(), logging.INFO) # Varsayılan INFO