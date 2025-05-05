# src/core/utils.py
#
# Evo projesinde farklı modüllerde kullanılabilecek genel yardımcı fonksiyonlar ve sınıflar.

import logging
import numpy as np # Numpy array kontrolleri için

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

def get_config_value(config, key, default, expected_type=None, logger_instance=None):
    """
    Yapılandırma sözlüğünden belirtilen anahtara ait değeri güvenli bir şekilde alır.

    Anahtar bulunamazsa varsayılan değeri döner.
    Beklenen tip belirtilmişse tip kontrolü yapar ve uymuyorsa uyarı loglar.

    Args:
        config (dict): Yapılandırma sözlüğü.
        key (str): Alınacak ayarın anahtarı.
        default (any): Anahtar bulunamazsa veya tip uymazsa dönecek varsayılan değer.
        expected_type (type or tuple, optional): Beklenen veri tipi (örn: int, float, str, (int, float)). None ise tip kontrolü yapılmaz.
        logger_instance (logging.Logger, optional): Loglama için kullanılacak logger objesi. Yoksa bu modülün logger'ı kullanılır.

    Returns:
        any: Konfigürasyondan alınan değer veya varsayılan değer.
    """
    # Kullanılacak logger objesini belirle
    log = logger_instance if logger_instance is not None else logger

    value = config.get(key, default)

    # Beklenen tip kontrolü
    if expected_type is not None:
        # Eğer değer varsayılan değer *değilse* ve tipi uymuyorsa uyarı logla.
        # Varsayılan değerin tipini kontrol etmek her zaman anlamlı olmayabilir.
        if value is not default and not isinstance(value, expected_type):
            log.warning(f"ConfigUtils: Yapılandırma anahtarı '{key}' için beklenmeyen tip: {type(value)}. {expected_type} bekleniyordu. Varsayılan değer ({default}) kullanılıyor.")
            return default # Tip uymuyorsa varsayılanı döndür

    # Eğer değer None ise ve varsayılan None değilse, belki uyarı loglamak iyi olabilir?
    # Veya bu durum load_config_from_yaml'de mi yönetilmeli?
    # Şimdilik sadece tip kontrolü yapalım.

    return value

def check_input_not_none(input_data, input_name="input", logger_instance=None):
    """
    Bir girdinin None olup olmadığını kontrol eder ve None ise DEBUG loglar.

    Args:
        input_data (any): Kontrol edilecek girdi verisi.
        input_name (str, optional): Log mesajında kullanılacak girdi adı. Varsayılan 'input'.
        logger_instance (logging.Logger, optional): Loglama için kullanılacak logger objesi. Yoksa bu modülün logger'ı kullanılır.

    Returns:
        bool: Girdi None ise False, değilse True.
    """
    log = logger_instance if logger_instance is not None else logger
    if input_data is None:
        log.debug(f"Input '{input_name}' None. İşlem atlanıyor veya None döndürülüyor.")
        return False
    return True

def check_input_type(input_data, expected_type, input_name="input", logger_instance=None):
    """
    Bir girdinin beklenen tipte olup olmadığını kontrol eder ve uymuyorsa ERROR loglar.

    Args:
        input_data (any): Kontrol edilecek girdi verisi.
        expected_type (type or tuple): Beklenen veri tipi (örn: int, float, str, (int, float)).
        input_name (str, optional): Log mesajında kullanılacak girdi adı. Varsayılan 'input'.
        logger_instance (logging.Logger, optional): Loglama için kullanılacak logger objesi. Yoksa bu modülün logger'ı kullanılır.

    Returns:
        bool: Girdi beklenen tipte ise True, değilse False.
    """
    log = logger_instance if logger_instance is not None else logger
    if not isinstance(input_data, expected_type):
        log.error(f"Beklenmeyen '{input_name}' tipi: {type(input_data)}. {expected_type} bekleniyordu.")
        return False
    return True

def check_numpy_input(input_data, expected_dtype=None, expected_ndim=None, input_name="input", logger_instance=None):
    """
    Bir girdinin numpy array olup olmadığını ve isteğe bağlı dtype/ndim kontrolünü yapar.

    Args:
        input_data (any): Kontrol edilecek girdi verisi.
        expected_dtype (numpy.dtype or type, optional): Beklenen dtype (örn: np.float32, np.uint8, np.number). None ise dtype kontrolü yapılmaz.
        expected_ndim (int, optional): Beklenen boyut sayısı (ndim). None ise boyut kontrolü yapılmaz.
        input_name (str, optional): Log mesajında kullanılacak girdi adı. Varsayılan 'input'.
        logger_instance (logging.Logger, optional): Loglama için kullanılacak logger objesi. Yoksa bu modülün logger'ı kullanılır.

    Returns:
        bool: Girdi numpy array ise ve belirtilen kriterlere uyuyorsa True, değilse False.
    """
    log = logger_instance if logger_instance is not None else logger

    # Önce genel numpy array kontrolü
    if not isinstance(input_data, np.ndarray):
        log.error(f"Beklenmeyen '{input_name}' tipi: {type(input_data)}. numpy.ndarray bekleniyordu.")
        return False

    # Dtype kontrolü (belirtilmişse)
    if expected_dtype is not None:
         # np.issubdtype daha esnek, alt tipleri de yakalar (örn: np.issubdtype(np.int16, np.number) -> True)
         # isinstance sadece tam tipi kontrol eder.
         if not np.issubdtype(input_data.dtype, expected_dtype):
              log.error(f"Beklenmeyen '{input_name}' dtype: {input_data.dtype}. {expected_dtype} (veya alt tipi) bekleniyordu.")
              return False

    # Boyut (ndim) kontrolü (belirtilmişse)
    if expected_ndim is not None:
        if input_data.ndim != expected_ndim:
            log.error(f"Beklenmeyen '{input_name}' boyut sayısı (ndim): {input_data.ndim}. {expected_ndim} bekleniyordu.")
            return False

    return True # Tüm kontroller başarılı

# Gelecekte diğer yardımcı fonksiyonlar eklenebilir:
# - Veri formatlama/dönüştürme (örn: normalizasyon, one-hot encoding)
# - Ortak matematiksel işlemler
# - Belirli desenleri kontrol etme
# - Threading/Process yardımcıları