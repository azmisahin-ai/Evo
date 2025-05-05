# src/core/logging_utils.py
#
# Evo projesi için merkezi loglama yardımcı fonksiyonlarını içerir.
# Loglama sistemini yapılandırma dosyasına göre ayarlar.

import logging
import sys # Konsol çıktısı için StreamHandler'a gerekli olabilir
# import logging.config # Dictionary tabanlı config için kullanılabilir (şimdilik değil)

# Bu modül için bir logger oluştur
# logging.getLogger(__name__) çağrısı, modül hiyerarşisine uygun olarak
# 'src.core.logging_utils' adında bir logger döndürür.
logger = logging.getLogger(__name__)

class AnsiColorFormatter(logging.Formatter):
    COLOR_MAP = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[1;31m' # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        return super().format(record)

def setup_logging(config=None):
    """
    Evo projesinin loglama sistemini sağlanan yapılandırmaya göre ayarlar.

    Mevcut handler'ları temizler, root logger seviyesini yapılandırmaya göre ayarlar,
    bir konsol handler'ı ekler ve log formatını belirler.
    Config'te 'logging' anahtarı altında 'level' belirtilerek log seviyesi ayarlanabilir.
    Örn: config = {'logging': {'level': 'DEBUG'}}

    Args:
        config (dict, optional): Yapılandırma ayarları sözlüğü (genellikle main_config.yaml'den).
                                 Loglama ayarları 'logging' anahtarı altında beklenir.
                                 'level' anahtarı loglama seviyesini (string olarak: DEBUG, INFO, vb.) belirtir.
                                 Varsayılan olarak None. Config yoksa veya geçersizse
                                 varsayılan INFO seviyesi kullanılır.
    """
    # Mevcut logger ayarlarını temizle
    # Bu, run_evo'nun yeniden başlatılmasında veya farklı config'lerle test yaparken
    # aynı handler'ların tekrar tekrar eklenmesini önler.
    root_logger = logging.getLogger()
    # Root logger'a eklenmiş handler'ları kontrol et
    if root_logger.handlers:
        # Var olan tüm handler'ları kaldır
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
        # Root logger'ın seviyesini de sıfırlayabiliriz, böylece config'e göre yeniden ayarlanır.
        # root_logger.setLevel(logging.NOTSET) # Veya logging.DEBUG gibi bir başlangıç seviyesi


    # --- Logging Level Ayarları ---
    default_level = logging.INFO # Konfigurasyon sağlanmazsa kullanılacak varsayılan seviye
    configured_level = default_level # Başlangıçta varsayılan seviyeyi ata

    # Konfigurasyonda loglama seviyesi belirtilmiş mi kontrol et
    if config and 'logging' in config and 'level' in config['logging']:
        # Config'teki seviye değerini al, string yap ve büyük harfe çevir
        level_name = str(config['logging']['level']).upper()
        try:
            # String log seviyesini (örn: "DEBUG", "INFO") logging modülünün int karşılığına çevir.
            # logging.getLevelName() geçerli isimler için int seviye (10, 20, ...), geçersiz isimler için string seviye adını döndürür.
            level_value = logging.getLevelName(level_name)
            if isinstance(level_value, int):
                 # Eğer dönen değer bir int ise, geçerli bir seviye bulundu demektir.
                 configured_level = level_value
            else:
                 # Eğer dönen değer int değilse (yani string seviye adı ise), isim geçersiz demektir.
                 logger.warning(f"Logging: Konfigurasyonda geçersiz log seviyesi adı: '{level_name}'. Varsayılan seviye ({logging.getLevelName(default_level)}) kullanılıyor.")
                 configured_level = default_level # Geçersiz isimse varsayılanı kullan

        except Exception as e:
            # getLevelName veya config okuma sırasında beklenmedik bir hata olursa
            logger.warning(f"Logging: Konfigurasyondaki log seviyesi okunurken hata: {e}. Varsayılan seviye ({logging.getLevelName(default_level)}) kullanılıyor.", exc_info=True)
            configured_level = default_level # Hata olursa varsayılanı kullan


    # Root logger seviyesini ayarla
    # Bu seviye, alt loggerlardan yayılan (propagate eden) logların root'a ulaştığında
    # hangi minimum seviyeden itibaren root handler'ları tarafından işleneceğini belirler.
    # Örneğin, root seviyesi INFO iken, alt loggerdan gelen DEBUG logu root'a ulaşır ama handler'a geçemez.
    # Bu nedenle, gösterilmek istenen en düşük seviye (config'teki configured_level) root logger seviyesi olarak ayarlanır.
    root_logger.setLevel(configured_level)
    logger.info(f"Logging: Root logger seviyesi ayarlandı: {logging.getLevelName(root_logger.level)}")


    # --- Handler (Çıktı Hedefi) Ayarları ---
    # Şimdilik sadece konsol çıktısı handler'ı ekleyeceğiz.
    # Gelecekte config'teki 'handlers' listesini okuyarak farklı handler'lar (dosya, vb.) eklenebilir.
    # Config formatı için Python'ın logging.config.dictConfig dokümantasyonuna bakılabilir.

    # Konsol handler'ı oluştur
    # StreamHandler default olarak sys.stderr'i kullanır. sys.stdout genellikle konsol çıktısı için tercih edilir.
    # Özellikle konsol çıktılarını dosyalara yönlendirmek veya yakalamak istendiğinde sys.stdout kullanılır.
    console_handler = logging.StreamHandler(sys.stdout)

    # Formatter oluştur (Zaman damgası, logger adı, seviye ve mesajı içerecek şekilde)
    # %(asctime)s: Loglama zamanı (örneğin, 2023-10-27 10:30:00,123)
    # %(name)s: Loglayan logger'ın adı (örneğin, 'src.run_evo', 'src.senses.vision', 'src.core.logging_utils')
    # %(levelname)s: Loglama seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    # %(message)s: Log mesajının içeriği
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = AnsiColorFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)-40s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler'a formatter'ı set et
    console_handler.setFormatter(formatter)

    # Handler seviyesini ayarla
    # Handler seviyesi, o handler'ın hangi minimum seviyedeki logları işleyeceğini belirler.
    # Genellikle handler seviyesi, root logger seviyesiyle aynı veya daha düşüktür (daha fazla logu işlemesi için).
    # Eğer handler seviyesi root'tan yüksekse, root'tan gelen düşük seviyeli logları filtreler.
    # Eğer handler seviyesi root'tan düşükse, root tarafından zaten filtrelenmiş logları tekrar filtreler (etkisizdir).
    # Basitlik ve tutarlılık için, handler seviyesini root logger seviyesiyle aynı yapalım.
    console_handler.setLevel(configured_level) # Handler seviyesi root seviyesiyle aynı ayarlandı

    # Root logger'a handler'ı ekle
    # Root logger'a eklenen handler'lar, tüm alt loggerlardan yayılan (propagate eden) logları işler.
    root_logger.addHandler(console_handler)

    # --- Özel Logger Seviyeleri (İsteğe Bağlı / Gelecek TODO) ---
    # Belirli logger'ların (örn: 'src.senses') seviyelerini genel root seviyesinden farklı yapmak istersek,
    # logging.getLogger('logger.adı').setLevel(logging.SEVIYE) şeklinde buraya ekleyebiliriz.
    # Config'teki 'loggers' bölümü üzerinden yönetilmesi daha esnek olacaktır.
    # Örneğin:
    # if config and 'logging' in config and 'loggers' in config['logging']:
    #     for logger_name, logger_config in config['logging']['loggers'].items():
    #         if 'level' in logger_config:
    #             try:
    #                 alt_level = logging.getLevelName(str(logger_config['level']).upper())
    #                 if isinstance(alt_level, int):
    #                     logging.getLogger(logger_name).setLevel(alt_level)
    #                     logger.debug(f"Logging: Alt logger '{logger_name}' seviyesi ayarlandı: {logging.getLevelName(alt_level)}")
    #                 else:
    #                      logger.warning(f"Logging: Konfigurasyonda geçersiz alt logger seviyesi adı: '{logger_config['level']}' for logger '{logger_name}'.")
    #             except Exception as e:
    #                  logger.warning(f"Logging: Alt logger '{logger_name}' seviyesi okunurken hata: {e}", exc_info=True)


    # Loglama sisteminin başarıyla yapılandırıldığına dair bilgi logu (bu fonksiyonun kendi logger'ını kullanır).
    # Bu log, setup tamamlandıktan sonra çıkar ve yapılandırılmış seviyeye göre gösterilir.
    # logger.info("Loglama sistemi başarıyla yapılandırıldı.")
    # DEBUG seviyesinde ekstra bir test logu (sadece seviye DEBUG veya altıysa görünür).
    # logger.debug("Bu, logging_utils'tan gelen bir DEBUG mesajıdır.")


# Örnek: Dictionary tabanlı loglama yapılandırması için placeholder (Gelecek TODO)
# Daha karmaşık handler, formatter ve filter senaryoları için kullanılır.
# def setup_logging_from_dict(config_dict):
#     """
#     Dictionary tabanlı loglama yapılandırmasını uygular.
#     config_dict, Python logging.config.dictConfig'in beklediği formatta olmalıdır.
#     """
#     # Mevcut logger'ları temizleme adımı burada da düşünülmeli (dictConfig temizlemez).
#     # try:
#     #     logging.config.dictConfig(config_dict)
#     #     logger = logging.getLogger(__name__) # Kendi logger'ını tekrar al
#     #     logger.info("Loglama sistemi dictionary config ile yapılandırıldı.")
#     # except Exception as e:
#     #     # Hata durumunda en temel konsol loglamayı kurmayı düşünebiliriz.
#     #     # Şimdilik sadece hata logla.
#     #     logging.critical(f"Hata: Loglama sistemi dictionary config ile yapılandırılamadı: {e}", exc_info=True)
#     #     # Burada temel loglama setup'ı (console handler vb.) yeniden kurulabilir.
#     pass