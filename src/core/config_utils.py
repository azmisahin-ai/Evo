# src/core/config_utils.py
#
# Yapılandırma dosyalarını (YAML formatında) yüklemek için yardımcı fonksiyonları içerir.

import yaml # YAML dosyalarını okumak için
import logging # Loglama için
import os # Dosya varlığını kontrol etmek için

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

def load_config_from_yaml(config_path="config/main_config.yaml"):
    """
    Belirtilen YAML yapılandırma dosyasından ayarları yükler.

    Dosya bulunamazsa veya YAML formatında hata olursa, durumu loglar
    ve boş bir sözlük (`{}`) döndürür.

    Args:
        config_path (str): Yüklenecek YAML yapılandırma dosyasının yolu.
                           Varsayılan değer 'config/main_config.yaml'.

    Returns:
        dict: Yüklenen yapılandırma ayarları bir Python sözlüğü olarak.
              Hata durumunda boş sözlük `{}` döner.
    """
    # Yapılandırma dosyasının varlığını kontrol et
    if not os.path.exists(config_path):
        # Dosya yoksa kritik hata logla ve boş sözlük döndür.
        # Bu hata run_evo.py tarafından yakalanıp program sonlandırılır.
        logger.critical(f"Hata: Yapılandırma dosyası bulunamadı: {config_path}")
        return {}

    # Dosyayı okumayı ve YAML formatını işlemeyi dene
    try:
        with open(config_path, 'r', encoding='utf-8') as f: # utf-8 encoding kullanmak iyi pratik
            config = yaml.safe_load(f) # Güvenli yükleme yöntemi

        # Yüklenen config'in None veya boş olup olmadığını kontrol et
        # Boş YAML dosyası None olarak yüklenebilir.
        if config is None:
            logger.warning(f"Yapılandırma dosyası boş veya geçersiz içerik barındırıyor: {config_path}. Boş sözlük olarak yorumlanıyor.")
            config = {} # None ise boş sözlüğe çevir

        # Başarılı yükleme logu
        logger.info(f"Yapılandırma dosyası başarıyla yüklendi: {config_path}")
        # DEBUG seviyesinde tüm config içeriğini loglamak hata ayıklama için faydalı olabilir,
        # ancak hassas bilgiler içerebileceği durumlara dikkat edilmeli.
        # logger.debug(f"Yüklenen Konfigürasyon Detayları: {config}")

        return config

    except yaml.YAMLError as e:
        # YAML formatında bir hata oluşursa
        # Kritik hata logla ve boş sözlük döndür.
        logger.critical(f"Hata: Yapılandırma dosyası okunurken YAML format hatası oluştu: {config_path}\nHata: {e}", exc_info=True)
        return {}
    except Exception as e:
        # Dosya okuma veya başka beklenmedik bir hata oluşursa
        # Kritik hata logla ve boş sözlük döndür.
        logger.critical(f"Hata: Yapılandırma dosyası yüklenirken beklenmedik bir hata oluştu: {config_path}\nHata: {e}", exc_info=True)
        return {}

# # İsteğe bağlı olarak, yüklenen config'i tek bir yerde tutmak için bir sınıf veya singleton eklenebilir
# # class ConfigManager: ... (Mevcut load_config_from_yaml yeterli ve daha basit)