# src/core/config_utils.py

import yaml
import os
import logging
import numpy as np # get_config_value'da np.number kontrolü için eklendi

# Bu modül için bir logger oluştur (varsayılan loglama ayarları kullanılıyor)
# setup_logging çağrıldığında bu logger güncellenecektir.
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
        logger.error(f"Yapılandırma dosyası bulunamadı: {filepath}")
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
        logger.error(f"Yapılandırma dosyası yüklenirken beklenmeyen bir hata oluştu: {filepath}", exc_info=True)
        return {}


def get_config_value(config: dict, *keys, default=None, expected_type=None, logger_instance=None):
    """
    İç içe geçmiş bir sözlükten anahtar zinciri ile değer alır.
    Anahtar bulunamazsa veya yol geçersizse None veya belirtilen varsayılan değeri döndürür.
    Opsiyonel olarak beklenen tipi kontrol eder ve loglama için logger instance alabilir.

    Örnek: get_config_value(config, 'vision', 'input_width', default=640, expected_type=int, logger_instance=my_logger)

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
    # Fonksiyonun başladığını ve aldığı argümanları logla
    # Config objesi büyük olabilir, sadece tipini loglayalım
    log = logger_instance if logger_instance is not None else logger
    log.debug(f"get_config_value: Başladı. config tip: {type(config)}, keys: {keys}, default: {default}, expected_type: {expected_type}")


    current_value = config
    try:
        # Config'in dict olduğundan emin ol (veya None değil)
        if not isinstance(current_value, dict):
            log.debug(f"get_config_value: Başlangıç config geçerli bir sözlük değil (tip: {type(current_value)}). Varsayılan dönülüyor.")
            return default

        for i, key in enumerate(keys):
            # Her anahtar erişimi öncesinde mevcut değeri ve erişilmeye çalışılan anahtarı logla
            log.debug(f"get_config_value: Adım {i+1}/{len(keys)}. Mevcut tip: {type(current_value)}, Anahtar: '{key}'")
            # Mevcut değerin dict olduğundan ve anahtarın var olduğundan emin olalım
            if not isinstance(current_value, dict) or key not in current_value:
                 log.debug(f"get_config_value: Adım {i+1}/{len(keys)}. '{key}' anahtarı bulunamadı veya mevcut değer sözlük değil (tip: {type(current_value)}). Path tamamlaniyor.")
                 # Path burada kesildi, varsayılan değeri döndürmeliyiz.
                 # Ancak, döngü bitmeden varsayılan dönmek yerine, döngüden sonra kontrol edelim.
                 # Şimdilik mevcut exception handling yeterli olmalı.
                 current_value = current_value[key] # Eğer dict değilse veya key yoksa burada exception fırlayacak

            # Başarılı erişim sonrası sonucu logla
            log.debug(f"get_config_value: Adım {i+1}/{len(keys)}. Başarılı. Yeni tip: {type(current_value)}")


        # Döngü tamamlandı, değer bulundu. Tip kontrolü yapalım.
        # None değeri için tip kontrolü yapma (None her zaman None'dır)
        if current_value is not None and expected_type is not None:
             # numpy number gibi özel tipler için isinstance yerine np.issubdtype kullan
             if expected_type == np.number:
                  if not isinstance(current_value, np.ndarray) or not np.issubdtype(current_value.dtype, np.number):
                       log.warning(f"get_config_value: Config yolu {' -> '.join(keys)} için bulunan değer beklenen sayısal numpy array tipi değil (tip: {type(current_value)}, dtype: {getattr(current_value, 'dtype', 'N/A')}). Varsayılan değer dönülüyor.")
                       return default # Tip uyuşmazsa varsayılan dön

             elif expected_type == np.ndarray:
                  if not isinstance(current_value, np.ndarray):
                       log.warning(f"get_config_value: Config yolu {' -> '.join(keys)} için bulunan değer beklenen numpy array tipi değil (tip: {type(current_value)}). Varsayılan değer dönülüyor.")
                       return default # Tip uyuşmazsa varsayılan dön

             elif not isinstance(current_value, expected_type):
                  log.warning(f"get_config_value: Config yolu {' -> '.join(keys)} için bulunan değer beklenen tipte değil (beklenen: {expected_type}, geldi: {type(current_value)}). Varsayılan değer dönülüyor.")
                  return default # Tip uyuşmazsa varsayılan dön


        # Değer bulundu ve tip kontrolünden geçti (veya tip kontrolü istenmedi).
        # Dönmeden önce değeri logla. Çok büyük değerleri kısaltalım.
        value_repr = str(current_value)
        if len(value_repr) > 200:
             value_repr = value_repr[:100] + "..." + value_repr[-100:]

        log.debug(f"get_config_value: Başarılı tamamlandı. Dönüş değeri tipi: {type(current_value)}, Değer: {value_repr}")
        return current_value


    except (KeyError, TypeError, AttributeError) as e:
        # Path boyunca bir anahtar bulunamadı, mevcut değer dict değil veya None'dı.
        log.debug(f"get_config_value: Path takip edilirken hata yakalandı ({type(e).__name__}). Yol: {' -> '.join(keys)}. Varsayılan değer dönüyor: {default}")
        return default

    except Exception as e:
        # Diğer tüm beklenmedik hatalar.
        log.error(f"get_config_value: Beklenmedik hata: {e}", exc_info=True)
        log.debug(f"get_config_value: Beklenmedik hata sonrası varsayılan değer dönüyor: {default}")
        return default