# src/core/config_utils.py
#
# Evo projesi için merkezi yapılandırma (config) yükleme yardımcı fonksiyonlarını içerir (Konsolide Edilmiş).
# YAML dosyasından config yükler ve iç içe geçmiş değerlere güvenli erişim sağlar.

import yaml
import os
import logging
import numpy as np # get_config_value'da np.number kontrolü için kullanılıyor

# Bu modül için bir logger oluştur. Loglama seviyesi ve handler'lar dışarıdan (setup_logging) ayarlanacak.
logger = logging.getLogger(__name__)

def load_config_from_yaml(filepath="config/main_config.yaml"):
    """
    Belirtilen YAML dosyasından yapılandırmayı yükler.

    Args:
        filepath (str): Yüklemek için YAML dosyasının yolu.

    Returns:
        dict: Yüklenen yapılandırma sözlüğü veya hata durumunda boş sözlük.
    """
    if not isinstance(filepath, str) or not filepath:
        logger.error("Yapılandırma dosyası yüklenirken geçersiz dosya yolu belirtildi.")
        return {}

    if not os.path.exists(filepath):
        logger.error(f"Yapılandırma dosyası bulunamadı: {filepath}. Lütfen yolun doğru olduğundan emin olun.")
        return {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logger.info(f"Yapılandırma başarıyla yüklendi: {filepath}")
            # yaml.safe_load boş dosya için None dönebilir
            return config if config is not None else {}
    except yaml.YAMLError as e:
        logger.error(f"Yapılandırma dosyası okunurken YAML hatası oluştu: {filepath}", exc_info=True)
        return {}
    except Exception as e:
        logger.error(f"Yapılandırma dosyası yüklenirken beklenmedik bir hata oluştu: {filepath}", exc_info=True)
        return {}


# --- get_config_value fonksiyonu (nested anahtar yolu destekler) ---
def get_config_value(config: dict, *keys, default=None, expected_type=None, logger_instance=None):
    """
    İç içe geçmiş bir sözlükten anahtar zinciri (*keys) ile değer alır.
    Anahtar bulunamazsa veya yol geçersizse None veya belirtilen varsayılan değeri döndürür.
    Opsiyonel olarak beklenen tipi kontrol eder ve loglama için logger instance alabilir.

    Args:
        config (dict): Bakılacak yapılandırma sözlüğü.
        *keys (str): İç içe geçmiş anahtar adımları.
        default (any, optional): Anahtar bulunamazsa döndürülecek varsayılan değer.
                                 Varsayılanı None'dır.
        expected_type (type or tuple of types, optional): Beklenen değer tipi veya tipleri tuple'ı. None ise tip kontrolü yapılmaz.
                                                       np.number gibi özel tipler numpy'den alınmalıdır.
        logger_instance (logging.Logger, optional): Loglama için kullanılacak logger instance'ı. None ise modül logger'ı kullanılır.

    Returns:
        any: Bulunan değer veya varsayılan değer.
    """
    log = logger_instance if logger_instance is not None else logger
    # Config objesi büyük olabilir, sadece tipini loglayalım
    #log.debug(f"get_config_value: Başladı. config tip: {type(config)}, keys: {keys}, default: {default}, expected_type: {expected_type}") # Gürültülü olabilir

    current_value = config
    # Başlangıç config'in sözlük olduğundan emin ol (None veya başka tip gelirse hata vermemeli)
    if not isinstance(current_value, dict):
        log.debug(f"get_config_value: Başlangıç config geçerli bir sözlük değil (tip: {type(current_value)}). Varsayılan ({default}) dönülüyor.")
        return default

    # Anahtar yolu boyunca ilerle
    try:
        for i, key in enumerate(keys):
            #log.debug(f"get_config_value: Adım {i+1}/{len(keys)}. Mevcut tip: {type(current_value)}, Anahtar: '{key}'") # Gürültülü olabilir

            # Eğer mevcut değer dict değilse, yol takip edilemez.
            if not isinstance(current_value, dict):
                 log.debug(f"get_config_value: Adım {i+1}/{len(keys)}. Mevcut değer sözlük değil (tip: {type(current_value)}). Path takip edilemiyor. Varsayılan ({default}) dönülüyor.")
                 # Path burada kesildi, hata fırlatmadan varsayılan dönmeliyiz.
                 return default

            # Anahtara erişmeyi dene. Anahtar yoksa KeyError fırlayacak.
            # .get() kullanmak yerine KeyError'i yakalamak, anahtarın gerçekten olmadığını
            # veya değerin None olup olmadığını ayırt etmemize yardımcı olur.
            try:
                 current_value = current_value[key]
                 #log.debug(f"get_config_value: Adım {i+1}/{len(keys)}. Başarılı. Yeni tip: {type(current_value)}") # Gürültülü olabilir
            except KeyError:
                 log.debug(f"get_config_value: Anahtar '{key}' bulunamadı. Yol: {' -> '.join(keys[:i+1])}. Varsayılan ({default}) dönüyor.")
                 return default


    except (TypeError, AttributeError):
        # Path boyunca mevcut değer dict değil veya None'dı, veya geçersiz erişim denendi.
        log.debug(f"get_config_value: Path takip edilirken TypeError/AttributeError yakalandı. Yol: {' -> '.join(keys)}. Varsayılan değer ({default}) dönüyor.")
        return default
    except Exception as e:
        # Diğer tüm beklenmedik hatalar.
        log.error(f"get_config_value: Beklenmedik hata: {e}", exc_info=True)
        log.debug(f"get_config_value: Beklenmedik hata sonrası varsayılan değer ({default}) dönüyor.")
        return default

    # Döngü tamamlandı, değer bulundu. Tip kontrolü yapalım.
    # None değeri için tip kontrolü yapma (None her zaman None'dır)
    if current_value is not None and expected_type is not None:
        # numpy number gibi özel tipler için isinstance yerine np.issubdtype kullan
        if expected_type == np.number:
             # Eğer değer numpy array değilse veya sayısal dtype değilse tip uyuşmazlığı.
             # Not: get_config_value genelde sayılar, stringler vb. alır, numpy array'in kendisini config'e koymak nadirdir.
             # Bu np.number kontrolü config'teki sayısal değerler içindir (int, float).
             # isinstance kontrolü burada np.number için doğru değil. Değerin int/float olup olmadığına bakalım.
             if not isinstance(current_value, (int, float)):
                  log.warning(f"get_config_value: Config yolu {' -> '.join(keys)} için bulunan değer beklenen sayısal tipte değil (tip: {type(current_value)}). Beklenen tip: np.number (int veya float). Varsayılan değer ({default}) dönülüyor.")
                  return default # Tip uyuşmazsa varsayılan dön

        elif expected_type == np.ndarray:
             # Eğer değer numpy array değilse tip uyuşmazlığı.
             if not isinstance(current_value, np.ndarray):
                  log.warning(f"get_config_value: Config yolu {' -> '.join(keys)} için bulunan değer beklenen numpy array tipi değil (tip: {type(current_value)}). Beklenen tip: np.ndarray. Varsayılan değer ({default}) dönülüyor.")
                  return default # Tip uyuşmazsa varsayılan dön

        # Normal tipler için isinstance kontrolü. Tuple of types da desteklenir.
        # expected_type np.number veya np.ndarray değilse buraya gelir.
        elif not isinstance(current_value, expected_type):
             log.warning(f"get_config_value: Config yolu {' -> '.join(keys)} için bulunan değer beklenen tipte değil (beklenen: {expected_type}, geldi: {type(current_value)}). Varsayılan değer ({default}) dönülüyor.")
             return default # Tip uyuşmazsa varsayılan dön

    # Değer bulundu ve tip kontrolünden geçti (veya tip kontrolü istenmedi). Return the value.
    # Log the final successful retrieval (handle potentially large values)
    # value_repr = str(current_value) # Gürültülü olabilir
    # if len(value_repr) > 200:
    #      value_repr = value_repr[:100] + "..." + value_repr[-100:]
    # log.debug(f"get_config_value: Başarılı tamamlandı. Dönüş değeri tipi: {type(current_value)}, Değer: {value_repr}") # Gürültülü olabilir

    return current_value


# Gelecekte diğer config ile ilgili yardımcı fonksiyonlar eklenebilir.