# src/core/logging_utils.py
#
# Evo projesi için merkezi loglama yardımcı fonksiyonlarını içerir.
# Loglama sistemini yapılandırma dosyasına göre ayarlar.

import logging
import sys # Konsol çıktısı için StreamHandler'a gerekli olabilir
import logging.config # Dictionary tabanlı config için kullanılabilir (şimdilik değil)

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

def setup_logging(config=None):
    """
    Evo projesinin loglama sistemini sağlanan yapılandırmaya göre ayarlar.

    Mevcut handler'ları temizler, root logger seviyesini yapılandırmaya göre ayarlar,
    bir konsol handler'ı ekler ve log formatını belirler.

    Args:
        config (dict, optional): Yapılandırma ayarları sözlüğü (genellikle main_config.yaml'den).
                                 Loglama ayarları 'logging' anahtarı altında beklenir.
                                 'level' anahtarı loglama seviyesini (string olarak) belirtir.
                                 Varsayılan olarak None. Config yoksa veya geçersizse
                                 varsayılan INFO seviyesi kullanılır.
    """
    # Mevcut logger ayarlarını temizle
    # Bu, run_evo'nun yeniden başlatılmasında veya farklı config'lerle test yaparken
    # aynı handler'ların tekrar tekrar eklenmesini önler.
    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Var olan handler'ları kaldır
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
        # Root logger seviyesini de varsayılana sıfırlayabiliriz (isteğe bağlı)
        # root_logger.setLevel(logging.NOTSET)


    # --- Logging Level Ayarları ---
    default_level = logging.INFO # Varsayılan seviye INFO
    configured_level = default_level # Konfigurasyondan okunacak seviye

    # Konfigurasyonda loglama seviyesi belirtilmiş mi kontrol et
    if config and 'logging' in config and 'level' in config['logging']:
        level_name = str(config['logging']['level']).upper() # Config'teki değeri string yap ve büyük harfe çevir
        try:
            # String log seviyesini (örn: "DEBUG", "INFO") logging modülünün int karşılığına çevir.
            # logging.getLevelName() geçerli isimler için int, geçersizler için string döndürür.
            level_value = logging.getLevelName(level_name)
            if isinstance(level_value, int):
                 configured_level = level_value # Geçerli seviye bulundu, kullan
            else:
                 # logging.getLevelName string döndürdüyse isim geçersiz demektir.
                 logger.warning(f"Logging: Konfigurasyonda geçersiz log seviyesi adı: '{level_name}'. Varsayılan seviye ({logging.getLevelName(default_level)}) kullanılıyor.")
                 configured_level = default_level # Geçersiz isimse varsayılanı kullan

        except Exception as e:
            # getLevelName veya config okuma sırasında hata
            logger.warning(f"Logging: Konfigurasyondaki log seviyesi okunurken hata: {e}. Varsayılan seviye ({logging.getLevelName(default_level)}) kullanılıyor.", exc_info=True)
            configured_level = default_level


    # Root logger seviyesini ayarla
    # Bu, herhangi bir logger'dan yayılan (propagate eden) log mesajlarının
    # root logger'a ulaştığında hangi minimum seviyeden itibaren işleneceğini belirler.
    # Eğer alt logger seviyesi DEBUG ise ve root logger seviyesi INFO ise, DEBUG logları root'a ulaşır ama filtrelemeden geçemez.
    # Bu yüzden genellikle root logger seviyesi, gösterilmek istenen en düşük seviyeye ayarlanır.
    root_logger.setLevel(configured_level)
    logger.info(f"Logging: Root logger seviyesi ayarlandı: {logging.getLevelName(root_logger.level)}")


    # --- Handler (Çıktı Hedefi) Ayarları ---
    # Şimdilik sadece konsol çıktısı handler'ı ekleyeceğiz.
    # Gelecekte config'teki 'handlers' listesini okuyarak farklı handler'lar (dosya, vb.) eklenebilir.

    # Konsol handler'ı oluştur
    # StreamHandler default olarak sys.stderr'i kullanır, sys.stdout kullanmak tercih edilebilir
    # özellikle konsol çıktılarını dosyalara yönlendirmek veya yakalamak istendiğinde.
    console_handler = logging.StreamHandler(sys.stdout)

    # Formatter oluştur (Zaman, logger adı, seviye ve mesajı içerecek şekilde)
    # %(name)s: Loglayan logger'ın adı (örn: 'src.run_evo', 'src.senses.vision')
    # %(asctime)s: Loglama zamanı
    # %(levelname)s: Loglama seviyesi (DEBUG, INFO vb.)
    # %(message)s: Log mesajının içeriği
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Handler'a formatter'ı set et
    console_handler.setFormatter(formatter)

    # Handler seviyesini ayarla
    # Handler seviyesi, o handler'ın hangi minimum seviyedeki logları işleyeceğini belirler.
    # Genellikle handler seviyesi, root logger seviyesiyle aynı veya daha düşüktür (daha fazla logu işlemesi için).
    # Eğer handler seviyesi root'tan yüksekse, root'tan gelen düşük seviyeli logları filtreler.
    # Eğer handler seviyesi root'tan düşükse, root tarafından zaten filtrelenmiş logları tekrar filtreler (etkisizdir).
    # Basitlik için, handler seviyesini root logger seviyesiyle aynı yapalım.
    console_handler.setLevel(configured_level) # Handler seviyesi root seviyesiyle aynı

    # Root logger'a handler'ı ekle
    # Root logger'a eklenen handler'lar, tüm alt logger'lardan yayılan (propagate eden) logları işler.
    root_logger.addHandler(console_handler)

    # --- Özel Logger Seviyeleri (İsteğe Bağlı) ---
    # Belirli logger'ların seviyelerini genel root seviyesinden farklı yapmak istersek buraya ekleyebiliriz.
    # Örneğin, 'src.senses' logger'ından sadece WARNING ve üstünü görmek istersek:
    # logging.getLogger('src.senses').setLevel(logging.WARNING)
    # Ancak mevcut yapıda, tüm src modüllerinin DEBUG loglarını görmek istediğimiz için
    # root logger'ı DEBUG'e ayarladık ve bu alt loggerlar root'tan miras alacaktır.
    # Bu yüzden bu kısım şimdilik gerekli değil.

    # Loglama sisteminin başarıyla yapılandırıldığına dair bilgi logu (bu fonksiyonun kendi logger'ını kullanır)
    # Bu log, setup tamamlandıktan sonra çıkar ve yapılandırılmış seviyeye göre gösterilir.
    # logger.info("Loglama sistemi başarıyla yapılandırıldı.")
    # DEBUG seviyesinde ekstra bir test logu
    # logger.debug("Bu, logging_utils'tan gelen bir DEBUG mesajıdır (sadece log seviyesi DEBUG veya altıysa görünür).")


# Örnek: Dictionary tabanlı loglama yapılandırması için placeholder (Gelecek TODO)
# def setup_logging_from_dict(config_dict):
#     """
#     Dictionary tabanlı loglama yapılandırmasını uygular.
#     config_dict, Python logging.config.dictConfig'in beklediği formatta olmalıdır.
#     """
#     # Mevcut logger'ları temizleme adımı burada da düşünülmeli.
#     # logging.config.dictConfig(config_dict)
#     pass