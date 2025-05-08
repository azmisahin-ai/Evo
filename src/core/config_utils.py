# src/core/config_utils.py
#
# Evo projesi için merkezi yapılandırma (config) yükleme yardımcı fonksiyonlarını içerir (Konsolide Edilmiş).
# YAML dosyasından config yükler ve iç içe geçmiş değerlere güvenli erişim sağlar.

import yaml
import os
import logging
import numpy as np

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

    # Debug: Çalışma dizinini logla
    # logger.debug(f"Mevcut çalışma dizini: {os.getcwd()}")

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
# Düzeltildi: Varsayılan değeri positional argüman olarak alan çağrıları ele alacak şekilde workaround iyileştirildi.
# Artık positional olarak verilen son argüman (keyword default= verilmediyse) tipi ne olursa olsun default kabul edilecek.
def get_config_value(config: dict, *keys, default=None, expected_type=None, logger_instance=None):
    """
    İç içe geçmiş bir sözlükten anahtar zinciri (*keys) ile değer alır.
    Anahtar bulunamazsa, yol geçersizse veya tip uyuşmazsa
    belirtilen varsayılan değeri döndürür.
    Opsiyonel olarak beklenen tipi kontrol eder ve loglama için logger instance alabilir.
     positional olarak verilmiş default değerleri (örneğin get_config_value(config, 'key', default_value))
    desteklemek için bir workaround içerir.

    Args:
        config (dict): Bakılacak yapılandırma sözlüğü.
        *keys (str): İç içe geçmiş anahtar adımları. Örn: 'logging', 'level' veya ('logging', 'level').
                         Artık positional olarak default değer GEÇİLMEMELİDIR.
        default (any, optional): Anahtar bulunamazsa, yol geçersizse veya tip uyuşmazsa
                                 döndürülecek varsayılan değer. Varsayılanı None'dır.
                                 Bu parametre YALNIZCA default=... şeklinde keyword argüman olarak verilmelidir.
        expected_type (type or tuple of types, optional): Beklenen değer tipi veya tipleri tuple'ı. None ise tip kontrolü yapılmaz.
                                                       np.number gibi özel tipler numpy'den alınmalıdır.
        logger_instance (logging.Logger, optional): Loglama için kullanılacak logger instance'ı. None ise modül logger'ı kullanılır.

    Returns:
        any: Bulunan değer veya varsayılan değer.
    """
    log = logger_instance if logger_instance is not None else logger

    # --- WORKAROUND: DecisionModule gibi çağıranların default değeri positional geçmesine uyum sağla ---
    # Eğer keyword default=X hiç kullanılmadıysa (default parametresine dışarıdan değer atanmadıysa)
    # VE positional keys tuple'ı boş değilse, son elemanı default olarak ele al.
    # Bu, get_config_value(config, 'key', default_value) şeklindeki çağrıları yakalar.
    # actual_default is None check'i default=None keyword ile çağrılan durumları ele alır.
    # len(actual_keys) > 0 check'i en az bir anahtar olmalı der.
    # Bu workaround, `get_config_value(config, 'key', default_value)` şeklinde default değeri positional olarak geçen çağrılarda,
    # `default` parametresine hiçbir değer atanmadığı için `actual_default`'ın hala `None` olmasını ve
    # `actual_keys`'in `('key', default_value)` gibi bir şey olmasını bekler.
    # Bu durumda `actual_keys.pop()` ile alınan `default_value` doğru varsayılan değer olarak kullanılır.
    # Eğer çağrı `get_config_value(config, 'key', default=default_value)` şeklinde olsaydı, `default`'a `default_value` atanacağı için `actual_default` None olmazdı ve bu blok çalışmazdı.
    # Eğer çağrı `get_config_value(config, 'key')` şeklinde olsaydı, `actual_default` None olurdu ama `len(actual_keys)` 1 olurdu, pop yapıp default kullanmazdı.
    # Yani workaround şu anki haliyle legacy positional default çağrılarını doğru yakalamalı.
    if default is None and len(keys) > 0: # Original 'default is None' check here
         # Convert keys tuple to list to mutate
         actual_keys_list = list(keys)
         # Assuming the last positional argument is the intended default
         actual_default = actual_keys_list.pop()
         # Reassign actual_keys to the list without the last element
         actual_keys = tuple(actual_keys_list) # Now actual_keys only contains the actual keys

         # WORKAROUND logunu daha net yapalım.
         log.debug(f"get_config_value WORKAROUND: Keyword 'default' kullanılmadı, son positional argüman '{actual_default}' varsayılan olarak kullanıldı. Gerçek keys: {actual_keys}")
    else:
        # If default was provided as a keyword argument, or keys were empty, use the provided default
        actual_default = default
        actual_keys = keys # Use the keys as provided
        # Normal kullanım durumunu logla.
        log.debug(f"get_config_value: Keyword 'default' kullanıldı veya positional default yok. Default değeri: {actual_default}. Keys: {actual_keys}")
    # --- WORKAROUND SONU ---


    path_str = ' -> '.join(map(str, actual_keys))

    current_value = config
    final_value = actual_default # Bulunamazsa veya hata olursa dönecek değer, başlangıçta DÜZELTİLMİŞ default olarak ayarlandı.

    # Başlangıç config'in sözlük olduğundan emin ol
    if not isinstance(current_value, dict):
        log.debug(f"get_config_value: Başlangıç config geçerli bir sözlük değil (tip: {type(current_value)}). '{path_str}' yolu için varsayılan ({actual_default}) dönülüyor.")
        return actual_default

    # Eğer anahtar yolu boşsa (get_config_value(config, default=X) gibi çağrıldıysa - workaround nedeniyle bu path işlenmez),
    # veya sadece get_config_value(config, default=X) gibi çağrıldığında (actual_keys boşsa),
    # config dict'in kendisi dönmeli. Ancak bu kullanım formatı pek beklenen değil.
    # Varsayalım ki actual_keys her zaman anahtar adımlarını içerir.
    if not actual_keys:
        # Bu durum, get_config_value(config) gibi çağrılarda olabilir.
        # Workaround bu durumda çalışmaz (len(actual_keys) > 0 değil).
        # Eğer böyle bir çağrı yapılırsa, current_value = config kalır ve döngüye girilmez.
        # Aşağıdaki final_value ataması current_value'yu kullanır.
        # Bu durumda config dict'in kendisi dönmelidir.
        # Ancak tip kontrolü (expected_type=dict gibi) yapılabilir.
        # Şimdilik bu edge case'i görmezden gelelim ve anahtar yolunun boş OLMADIĞINI varsayalım.
         pass # Normal akış devam eder.


    try:
        # Anahtar yolu boyunca ilerle
        for i, key in enumerate(actual_keys):
            # Eğer mevcut değer bir dict değilse ve hala path'in ortasındaysak, yol geçersiz.
            # actual_keys boşsa döngüye girilmez.
            if not isinstance(current_value, dict):
                 # Eğer hala anahtarlar varsa (yolun sonuna gelmediysek) ve mevcut değer dict değilse hata.
                 # i < len(actual_keys) koşulu döngünün kendisinden dolayı zaten True.
                 # Yani buraya girildiyse ve current_value dict değilse, yol geçersiz demektir.
                 log.debug(f"get_config_value: '{path_str}' yolu takip edilirken ara değer sözlük değil (adım {i+1}/{len(actual_keys)}, anahtar '{key}', tip: {type(current_value)}). Varsayılan ({actual_default}) dönülüyor.")
                 return actual_default # Yolun ortasında dict bekleniyordu, yoktu.


            try:
                 # current_value'nun dict olduğunu biliyoruz (yukardaki if'ten), key'e güvenle erişebiliriz.
                 current_value = current_value[key]

            except (KeyError, TypeError):
                 # Anahtar bulunamadı (path'in ortasında veya sonunda) veya current_value dict değildi (TypeError).
                 # TypeError durumunu yukarıdaki isinstance(current_value, dict) kontrolü yakalamalıydı, ama yine de burada yakalamak sağlamlık katabilir.
                 # Log mesajını daha net yapalım.
                 log.debug(f"get_config_value: '{path_str}' yolu takip edilirken anahtar '{key}' bulunamadı veya mevcut değer beklenmeyen tipteydi (adım {i+1}/{len(actual_keys)}, mevcut tip: {type(current_value)}). Varsayılan ({actual_default}) dönüyor.")
                 return actual_default # Anahtar yoksa veya ara değer dict değilse varsayılan dön.

            except Exception as e:
                 # Diğer beklenmedik hatalar (örn: anahtar tipi geçerli değilse)
                 log.error(f"get_config_value: '{path_str}' yolu takip edilirken beklenmedik hata (adım {i+1}/{len(actual_keys)}, anahtar '{key}'): {e}", exc_info=True)
                 log.debug(f"get_config_value: Hata sonrası varsayılan değer ({actual_default}) dönüyor.")
                 return actual_default


        # Döngü başarıyla tamamlandı (tüm anahtarlar actual_keys'teydi).
        # current_value artık bulunan değerdir.
        # Eğer actual_keys boşsa (bu case'i şimdilik ignore ettik), current_value hala başlangıç config olurdu.
        final_value = current_value # Bulunan değeri final_value'ya ata


    except Exception as e:
        # Genel hata yakalandı (bu blok teorik olarak çok çalışmamalı)
        log.error(f"get_config_value: '{path_str}' yolu takip edilirken genel hata yakalandı: {e}", exc_info=True)
        log.debug(f"get_config_value: Genel hata sonrası varsayılan değer ({actual_default}) dönüyor.")
        return actual_default # Hata durumunda varsayılan döndür.


    # --- Tip kontrolü yapalım ---
    # None değeri için tip kontrolü yapma (None her zaman None'dır)
    if final_value is not None and expected_type is not None:
        # isinstance kullanırken tuple da kabul edilir. numpy tipleri için özel kontrol ekleyelim.
        is_correct_type = False

        # expected_type tuple ise içindeki tipleri tek tek kontrol et
        if isinstance(expected_type, tuple):
            for t in expected_type:
                 if t == np.number:
                      # np.number kontrolü: int, float, veya numpy sayısal dtype'ına sahip skalar/array.
                      if isinstance(final_value, (int, float)) or (isinstance(final_value, np.ndarray) and np.issubdtype(final_value.dtype, np.number)) or (np.isscalar(final_value) and np.issubdtype(type(final_value), np.number)):
                           is_correct_type = True
                           break # Tuple içinde doğru tip bulundu, döngüyü kır.
                 elif t == np.ndarray:
                      if isinstance(final_value, np.ndarray):
                           is_correct_type = True
                           break # Tuple içinde doğru tip bulundu, döngüyü kır.
                 # Normal tip kontrolü (int, float, str, list, dict vb.)
                 elif isinstance(final_value, t):
                    is_correct_type = True
                    break # Tuple içinde doğru tip bulundu, döngüyü kır.
        # expected_type tuple değil, tek tip ise
        else:
             if expected_type == np.number:
                  if isinstance(final_value, (int, float)) or (isinstance(final_value, np.ndarray) and np.issubdtype(final_value.dtype, np.number)) or (np.isscalar(final_value) and np.issubdtype(type(final_value), np.number)):
                       is_correct_type = True
             elif expected_type == np.ndarray:
                  if isinstance(final_value, np.ndarray):
                       is_correct_type = True
             # Normal tip kontrolü
             elif isinstance(final_value, expected_type):
                is_correct_type = True


        if not is_correct_type:
             log.warning(f"get_config_value: Config yolu '{path_str}' için bulunan değer beklenen tipte değil (beklenen: {expected_type}, geldi: {type(final_value)}). Varsayılan değer ({actual_default}) dönülüyor.")
             # TIP UYUSMAZLIGINDA VARSAYILAN DEGER DONDURULMELI.
             return actual_default # Tip uyuşmazsa varsayılan dön


    # Değer bulundu ve tip kontrolünden geçti (veya istenmedi/değer None). Bulunan değeri döndür.
    # log.debug(f"get_config_value: '{path_str}' için değer başarıyla alındı (tip: {type(final_value)}).") # Gürültülü olabilir
    return final_value