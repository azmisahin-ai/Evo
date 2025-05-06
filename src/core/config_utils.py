# src/core/config_utils.py

import yaml
import os
import logging

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
    # Proje kök dizinini bul (config/main_config.yaml yolu göreceliyse gerekli)
    # Bu fonksiyon genellikle projenin ana giriş noktasından çağrılır
    # O yüzden buradaki yolun zaten proje kökünden doğru olduğunu varsayabiliriz
    # veya scriptin nerede çalıştırıldığına bağlı olarak yolu ayarlayabiliriz.
    # Şimdilik ana akışın proje kökünden çalıştığını varsayalım.

    if not os.path.exists(filepath):
        logger.error(f"Yapılandırma dosyası bulunamadı: {filepath}")
        return {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logger.info(f"Yapılandırma başarıyla yüklendi: {filepath}")
            return config if config is not None else {} # yaml.safe_load boş dosya için None dönebilir
    except yaml.YAMLError as e:
        logger.error(f"Yapılandırma dosyası okunurken YAML hatası oluştu: {filepath} - {e}")
        return {}
    except Exception as e:
        logger.error(f"Yapılandırma dosyası yüklenirken beklenmeyen bir hata oluştu: {filepath} - {e}", exc_info=True)
        return {}

def get_config_value(config: dict, *keys, default=None):
    """
    İç içe geçmiş bir sözlükten anahtar zinciri ile değer alır.
    Anahtar bulunamazsa veya yol geçersizse None veya belirtilen varsayılan değeri döndürür.

    Örnek: get_config_value(config, 'vision', 'input_width', default=640)

    Args:
        config (dict): Bakılacak yapılandırma sözlüğü.
        *keys (str): İç içe geçmiş anahtar adımları.
        default (any, optional): Anahtar bulunamazsa döndürülecek varsayılan değer.
                                 Varsayılanı None'dır.

    Returns:
        any: Bulunan değer veya varsayılan değer.
    """
    current_value = config
    try:
        for key in keys:
            # current_value dict değilse veya key yoksa KeyError/AttributeError oluşur
            current_value = current_value[key]
        return current_value
    except (KeyError, TypeError, AttributeError):
        # Anahtar yolu boyunca bir hata oluştu
        # logger.debug(f"Config yolunda değer bulunamadı: {' -> '.join(keys)}. Varsayılan değer kullanılıyor.") # Çok sık loglama yapabilir, debug seviyesinde kalsın veya yorum satırı yapılsın
        return default
    except Exception as e:
        # Bu çok nadir olmalı, genellikle yukarıdaki except'lere takılır
        logger.error(f"get_config_value sırasında beklenmeyen bir hata oluştu: {e}", exc_info=True)
        return default