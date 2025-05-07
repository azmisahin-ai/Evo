# src/core/utils.py
#
# Evo projesinde farklı modüllerde kullanılabilecek genel yardımcı fonksiyonlar ve sınıflar (Konsolide Edilmiş).

import logging
import numpy as np # Numpy array kontrolleri için

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

# NOT: get_config_value fonksiyonu bu dosyadan kaldırıldı ve src/core/config_utils.py dosyasına taşındı.

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
    # Eski utils'daki DEBUG loglama seviyesini koruyoruz.
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
    # Eski utils'daki ERROR loglama seviyesini koruyoruz.
    if not isinstance(input_data, expected_type):
        log.error(f"Beklenmeyen '{input_name}' tipi: {type(input_data)}. {expected_type} bekleniyordu.")
        return False
    return True

def check_numpy_input(input_data, expected_dtype=None, expected_ndim=None, input_name="input", logger_instance=None):
    """
    Bir girdinin numpy array olup olmadığını ve isteğe bağlı dtype/ndim kontrolünü yapar.
    Hata durumunda loglar (ERROR seviyesinde) ve False döner, exception fırlatmaz.

    Args:
        input_data (any): Kontrol edilecek girdi verisi.
        expected_dtype (numpy.dtype or type or tuple, optional): Beklenen dtype (örn: np.float32, np.uint8, np.number, (np.float32, np.float64)). None ise dtype kontrolü yapılmaz. Tuple verilirse tiplerden biri beklenir.
        expected_ndim (int or tuple, optional): Beklenen boyut sayısı (ndim). None ise boyut kontrolü yapılmaz. Tuple verilirse boyut sayılarından biri beklenir (örn: (2, 3)).
        input_name (str, optional): Log mesajında kullanılacak girdi adı. Varsayılan 'input'.
        logger_instance (logging.Logger, optional): Loglama için kullanılacak logger objesi. Yoksa bu modülün logger'ı kullanılır.

    Returns:
        bool: Girdi numpy array ise ve belirtilen kriterlere uyuyorsa True, değilse False.
    """
    log = logger_instance if logger_instance is not None else logger

    # 1. Genel numpy array kontrolü
    if not isinstance(input_data, np.ndarray):
        log.error(f"Beklenmeyen '{input_name}' tipi: {type(input_data)}. numpy.ndarray bekleniyordu.")
        return False

    # 2. Dtype kontrolü (belirtilmişse)
    if expected_dtype is not None:
         # expected_dtype bir tuple ise içindeki tiplerden birine mi sahip?
         # değilse tek bir expected_dtype ile issubdtype kontrolü yap.
         dtype_match = False
         # np.issubdtype ikinci argüman olarak tuple alabilir (NumPy 1.7+), bunu kullanalım
         try:
            if np.issubdtype(input_data.dtype, expected_dtype):
                 dtype_match = True
         except TypeError:
             # Eğer expected_dtype bir tuple değilse veya eski numpy versiyonuysa, tek tek kontrol et
             if isinstance(expected_dtype, tuple):
                  for d in expected_dtype:
                       if np.issubdtype(input_data.dtype, d):
                            dtype_match = True
                            break
             else: # expected_dtype tek bir değer ama issubdtype hata verdi? (nadiren olabilir)
                 log.error(f"'{input_name}' için dtype kontrolü sırasında beklenmedik hata. Beklenen dtype: {expected_dtype}", exc_info=True)
                 return False # Hata durumunda False dön

         if not dtype_match:
              # Log mesajını daha bilgilendirici yapalım
              expected_dtype_str = str(expected_dtype)
              if isinstance(expected_dtype, tuple):
                   expected_dtype_str = " veya ".join([str(d) for d in expected_dtype])
              log.error(f"Beklenmeyen '{input_name}' dtype: {input_data.dtype}. Beklenen: {expected_dtype_str}.")
              return False


    # 3. Boyut (ndim) kontrolü (belirtilmişse)
    if expected_ndim is not None:
        # expected_ndim bir tuple ise, girdinin ndim'i tuple içindeki değerlerden birine mi eşit?
        if isinstance(expected_ndim, tuple):
            if input_data.ndim not in expected_ndim:
                 log.error(f"Beklenmeyen '{input_name}' boyut sayısı (ndim): {input_data.ndim}. Beklenen: {expected_ndim}.")
                 return False
        else: # expected_ndim tek bir int değer
            if input_data.ndim != expected_ndim:
                log.error(f"Beklenmeyen '{input_name}' boyut sayısı (ndim): {input_data.ndim}. {expected_ndim} bekleniyordu.")
                return False

    return True # Tüm kontroller başarılı

def run_safely(func, *args, logger_instance, error_message="İşlem sırasında beklenmedik hata", error_level=logging.ERROR, **kwargs):
    """
    Bir fonksiyonu try-except bloğu içinde güvenli bir şekilde çalıştırır.

    Hata oluşursa yakalar, loglar ve None döndürür.

    Args:
        func (callable): Çalıştırılacak fonksiyon.
        *args: Fonksiyona gönderilecek pozisyonel argümanlar.
        logger_instance (logging.Logger): Loglama için kullanılacak logger objesi (Zorunlu).
        error_message (str, optional): Hata oluştuğunda loglanacak mesajın başlangıcı.
        error_level (int, optional): Hata logunun seviyesi (örn: logging.ERROR, logging.CRITICAL).
        **kwargs: Fonksiyona gönderilecek anahtar kelime argümanları.

    Returns:
        any: Fonksiyonun başarıyla döndürdüğü değer veya hata durumunda None.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # Hata oluştuğunda belirtilen logger objesi ile logla
        logger_instance.log(error_level, f"{error_message}: {e}", exc_info=True)
        return None # Hata durumunda None döndür


def cleanup_safely(cleanup_func, logger_instance, error_message="Temizleme sırasında hata", error_level=logging.ERROR):
    """
    Bir temizleme fonksiyonunu try-except bloğu içinde güvenli bir şekilde çalıştırır.

    Hata oluşursa yakalar ve loglar. Temizleme fonksiyonları genellikle bir değer döndürmez.

    Args:
        cleanup_func (callable): Çalıştırılacak temizleme fonksiyonu.
        logger_instance (logging.Logger): Loglama için kullanılacak logger objesi (Zorunlu).
        error_message (str, optional): Hata oluştuğunda loglanacak mesajın başlangıcı.
        error_level (int, optional): Hata logunun seviyesi.
    """
    try:
        cleanup_func()
    except Exception as e:
        # Hata oluştuğunda belirtilen logger objesi ile logla
        logger_instance.log(error_level, f"{error_message}: {e}", exc_info=True)

# Gelecekte diğer yardımcı fonksiyonlar eklenebilir:
# - Veri formatlama/dönüştürme (örn: normalizasyon, one-hot encoding)
# - Ortak matematiksel işlemler
# - Belirli desenleri kontrol etme
# - Threading/Process yardımcıları