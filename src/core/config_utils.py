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
# İMZA DEĞİŞTİRİLDİ: Sadece config ve keys positional, default ve expected_type keyword olmalı.
# Workaround kaldırıldı.
def get_config_value(config: dict, *keys: str, default=None, expected_type=None, logger_instance=None):
    """
    İç içe geçmiş bir sözlükten anahtar zinciri (*keys) ile değer alır.
    Anahtar bulunamazsa, yol geçersizse veya tip uyuşmazsa
    belirtilen varsayılan değeri döndürür.
    Opsiyonel olarak beklenen tipi kontrol eder ve loglama için logger instance alabilir.

    Args:
        config (dict): Bakılacak yapılandırma sözlüğü.
        *keys (str): İç içe geçmiş anahtar adımları. Örn: 'logging', 'level' veya ('logging', 'level').
                     Bu parametre YALNIZCA string veya tuple olarak GEÇİLMELİDİR.
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

    # Workaround tamamen kaldırıldı. Çağrı formatı artık daha katı.
    # Eğer *keys içinde string olmayan bir şey varsa (eski positional default gibi),
    # bu aşağıdaki for döngüsünde TypeError veya başka bir hata fırlatır ve yakalanır.
    # Bu, yanlış çağrı formatlarının derleme zamanı yerine çalışma zamanında hata vermesine neden olur,
    # ama en azından get_config_value'nun iç mantığı temiz kalır.

    path_str = ' -> '.join(map(str, keys)) # keys artık hep anahtarlar olmalı

    current_value = config
    final_value = default # Varsayılan değer default= ile gelen değerdir.

    # Başlangıç config'in sözlük olduğundan emin ol
    if not isinstance(current_value, dict):
        log.debug(f"get_config_value: Başlangıç config geçerli bir sözlük değil (tip: {type(current_value)}). '{path_str}' yolu için varsayılan ({default}) dönülüyor.")
        return default

    # Eğer anahtar yolu boşsa, config dict'in kendisi dönmeli (ve tip kontrolü yapılmalı).
    if not keys:
         log.debug("get_config_value: Anahtar belirtilmedi. Config dict'in kendisi döndürülüyor.")
         final_value = config # Bulunan değer config dict'in kendisi

         # Anahtar yolu boşken de tip kontrolü yapılabilir (örn: expected_type=dict)
         if final_value is not None and expected_type is not None:
             # isinstance kullanırken tuple da kabul edilir. numpy tipleri için özel kontrol ekleyelim.
             is_correct_type = False
             # ... (tip kontrol mantığı aynı) ...
             # Bu blok aşağıdaki genel tip kontrolüyle aynı, kodu tekrarlamamak için atlayabiliriz.
             pass # Genel tip kontrolü aşağıda yapılacak


    try:
        # Anahtar yolu boyunca ilerle
        # keys tuple'ını kullan (listeye çevirmeye gerek yok artık)
        for i, key in enumerate(keys):
            # Eğer mevcut değer bir dict değilse ve hala path'in ortasındaysak, yol geçersiz.
            # keys boş değilse döngüye girilir.
            if not isinstance(current_value, dict):
                 # Eğer hala anahtarlar varsa (yolun sonuna gelmediysek) ve mevcut değer dict değilse hata.
                 # i < len(keys) koşulu döngünün kendisinden dolayı zaten True.
                 # Yani buraya girildiyse ve current_value dict değilse, yol geçersiz demektir.
                 log.debug(f"get_config_value: '{path_str}' yolu takip edilirken ara değer sözlük değil (adım {i+1}/{len(keys)}, anahtar '{key}', tip: {type(current_value)}). Varsayılan ({default}) dönülüyor.")
                 return default # Yolun ortasında dict bekleniyordu, yoktu.


            try:
                 # current_value'nun dict olduğunu biliyoruz (yukardaki if'ten), key'e güvenle erişebiliriz.
                 current_value = current_value[key]

            except (KeyError, TypeError):
                 # Anahtar bulunamadı (path'in ortasında veya sonunda) veya current_value dict değildi (TypeError).
                 # TypeError durumunu yukarıdaki isinstance(current_value, dict) kontrolü yakalamalıydı, ama yine de burada yakalamak sağlamlık katabilir.
                 # Log mesajını daha net yapalım.
                 log.debug(f"get_config_value: '{path_str}' yolu takip edilirken anahtar '{key}' bulunamadı veya mevcut değer beklenmeyen tipteydi (adım {i+1}/{len(keys)}, mevcut tip: {type(current_value)}). Varsayılan ({default}) dönüyor.")
                 return default # Anahtar yoksa veya ara değer dict değilse varsayılan dön.

            except Exception as e:
                 # Diğer beklenmedik hatalar (örn: anahtar tipi geçerli değilse)
                 log.error(f"get_config_value: '{path_str}' yolu takip edilirken beklenmedik hata (adım {i+1}/{len(keys)}, anahtar '{key}'): {e}", exc_info=True)
                 log.debug(f"get_config_value: Hata sonrası varsayılan değer ({default}) dönüyor.")
                 return default


        # Döngü başarıyla tamamlandı (tüm anahtarlar keys'teydi).
        # current_value artık bulunan değerdir.
        # Eğer keys boşsa (bu durum yukarıda ele alındı), current_value hala başlangıç config olurdu.
        final_value = current_value # Bulunan değeri final_value'ya ata


    except Exception as e:
        # Genel hata yakalandı (bu blok teorik olarak çok çalışmamalı)
        log.error(f"get_config_value: '{path_str}' yolu takip edilirken genel hata yakalandı: {e}", exc_info=True)
        log.debug(f"get_config_value: Genel hata sonrası varsayılan değer ({default}) dönüyor.")
        return default # Hata durumunda varsayılan döndür.


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
                 elif isinstance(final_value, expected_type): # Fixed: Use expected_type directly for normal types
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
             log.warning(f"get_config_value: Config yolu '{path_str}' için bulunan değer beklenen tipte değil (beklenen: {expected_type}, geldi: {type(final_value)}). Varsayılan değer ({default}) dönülüyor.")
             # TIP UYUSMAZLIGINDA VARSAYILAN DEGER DONDURULMELI.
             return default # Tip uyuşmazsa varsayılan dön


    # Değer bulundu ve tip kontrolünden geçti (veya istenmedi/değer None). Bulunan değeri döndür.
    # log.debug(f"get_config_value: '{path_str}' için değer başarıyla alındı (tip: {type(final_value)}).") # Gürültülü olabilir
    return final_value
