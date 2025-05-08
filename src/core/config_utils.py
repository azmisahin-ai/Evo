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
        *keys (str): İç içe geçmiş anahtar adımları.
        default (any, optional): Anahtar bulunamazsa, yol geçersizse veya tip uyuşmazsa
                                 döndürülecek varsayılan değer. Varsayılanı None'dır.
                                 Bu varsayılan değer, fonksiyona default=... olarak GİRİLEN değerdir.
        expected_type (type or tuple of types, optional): Beklenen değer tipi veya tipleri tuple'ı. None ise tip kontrolü yapılmaz.
                                                       np.number gibi özel tipler numpy'den alınmalıdır.
        logger_instance (logging.Logger, optional): Loglama için kullanılacak logger instance'ı. None ise modül logger'ı kullanılır.

    Returns:
        any: Bulunan değer veya varsayılan değer.
    """
    log = logger_instance if logger_instance is not None else logger

    # --- WORKAROUND: DecisionModule gibi çağıranların default değeri positional geçmesine uyum sağla ---
    # Eğer keys tuple'ı birden fazla elemanlıysa ve keyword default=X verilmediyse (default None kaldıysa),
    # son elemanı default değer olarak ele alalım ve keys listesinden çıkaralım.
    # Bu çirkin bir çözümdür ve gelecekte çağıranlar düzeltildiğinde kaldırılmalıdır.
    actual_default = default # Çağrıda default=X şeklinde keyword ile gelen değer
    actual_keys = list(keys) # Positional olarak gelen keys tuple'ını listeye çevir

    # Eğer keyword default=X verilmediyse (actual_default hala None ise) VE positional keys'in son elemanı default değer gibi görünüyorsa...
    # default olarak geçirilmesi amaçlanan değerin tipine bakmaksızın son positional argümanı default kabul et.
    # Argüman sayısına bakarak default değeri positional olarak geçildi mi anlamaya çalışıyoruz.
    # Eğer sadece 1 positional argüman varsa ve o config dict'i değilse bu durum test edilebilir.
    # Ama burada iç içe geçmiş anahtar yolu (*keys) alıyoruz, yani en az bir anahtar beklenir.
    # Eğer keyword default=X kullanılmadıysa (default hala None ise) VE *keys tuple'ı boş değilse...
    # Asıl amaç get_config_value(config, 'anahtar', default_deger, expected_type, ...) gibi kullanımları yakalamak.
    # Bu durumda *keys tuple'ı ('anahtar', default_deger) şeklinde olur.
    # Yani len(actual_keys) > 0 olmalı ve actual_default None kalmış olmalı.
    # Bu koşul gerçekleşirse, actual_keys'in son elemanını default kabul et.
    if actual_default is None and len(actual_keys) > 0:
         # Son elemanın default değer olması en olası senaryo, bunu default olarak ele alalım.
         actual_default = actual_keys.pop() # Son elemanı keys'ten çıkar ve default olarak kullan.
         log.debug(f"get_config_value WORKAROUND: Positional argüman '{actual_default}' varsayılan değer olarak kullanıldı. Gerçek keys: {tuple(actual_keys)}")
    # Else durumda, keys normal anahtarları içeriyor olmalı, actual_default ise çağrıda belirtilen default=X değeridir (veya None).
    else:
        log.debug(f"get_config_value: keys tuple'ı normal görünüyor: {tuple(actual_keys)}. Default değeri: {actual_default}")
    # --- WORKAROUND SONU ---


    path_str = ' -> '.join(map(str, actual_keys))

    current_value = config
    final_value = actual_default # Bulunamazsa veya hata olursa dönecek değer, başlangıçta DÜZELTİLMİŞ default olarak ayarlandı.

    # Başlangıç config'in sözlük olduğundan emin ol
    if not isinstance(current_value, dict):
        log.debug(f"get_config_value: Başlangıç config geçerli bir sözlük değil (tip: {type(current_value)}). '{path_str}' yolu için varsayılan ({actual_default}) dönülüyor.")
        return actual_default

    try:
        # Anahtar yolu boyunca ilerle
        # Düzeltilmiş keys listesini kullan!
        for i, key in enumerate(actual_keys):
            # Eğer mevcut değer bir dict değilse ve hala path'in ortasındaysak, yol geçersiz.
            # actual_keys boşsa döngüye girilmez.
            if not isinstance(current_value, dict):
                 # Eğer hala anahtarlar varsa (yolun sonuna gelmediysek) ve mevcut değer dict değilse hata.
                 if i < len(actual_keys):
                      log.debug(f"get_config_value: '{path_str}' yolu takip edilirken ara değer sözlük değil (adım {i+1}/{len(actual_keys)}, anahtar '{key}', tip: {type(current_value)}). Varsayılan ({actual_default}) dönülüyor.")
                      return actual_default # Yolun ortasında dict bekleniyordu, yoktu.
                 # Eğer anahtar kalmadıysa (yolun sonuna geldik), current_value zaten bulunan değerdir.
                 # Bu case aşağıdaki döngü sonrası final_value atamasında ele alınacak.


            try:
                 # current_value'nun dict olduğunu biliyoruz, key'e güvenle erişebiliriz.
                 current_value = current_value[key]

            except (KeyError, TypeError):
                 # Anahtar bulunamadı (path'in ortasında veya sonunda) veya current_value dict değildi (TypeError).
                 log.debug(f"get_config_value: '{path_str}' yolu takip edilirken anahtar '{key}' bulunamadı veya mevcut değer dict değil (adım {i+1}/{len(actual_keys)}, mevcut tip: {type(current_value)}). Varsayılan ({actual_default}) dönüyor.")
                 return actual_default # Anahtar yoksa veya ara değer dict değilse varsayılan dön.

            except Exception as e:
                 # Diğer beklenmedik hatalar (örn: anahtar tipi geçerli değilse)
                 log.error(f"get_config_value: '{path_str}' yolu takip edilirken beklenmedik hata (adım {i+1}/{len(actual_keys)}, anahtar '{key}'): {e}", exc_info=True)
                 log.debug(f"get_config_value: Hata sonrası varsayılan değer ({actual_default}) dönüyor.")
                 return actual_default


        # Döngü başarıyla tamamlandı (tüm anahtarlar actual_keys'teydi).
        # current_value artık bulunan değerdir.
        # Eğer actual_keys boşsa, döngü hiç çalışmaz ve current_value hala başlangıç config'dir.
        # Bu durumda config'in kendisi dönülmeli.
        # Ancak actual_keys'in boş olması durumu (get_config_value(config)) bu workaround ile desteklenmiyor,
        # çünkü tek positional argüman config olduğunda o default sanılabilir.
        # Workaround, *keys'te en az bir eleman (ilk anahtar veya default) olduğunu varsayar.
        # Eğer get_config_value(config) şeklinde çağrı desteklenecekse, bu workaround'un dışında ele alınmalıdır.
        # Şu anki kullanımda hep en az bir anahtar bekliyoruz (örn: get_config_value(config, 'level')).
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