# src/core/config_utils.py
#
# Yapılandırma dosyalarını (YAML formatında) yüklemek için yardımcı fonksiyonları içerir.
# Uygulama ayarlarını harici bir dosyadan okumayı sağlar.

import yaml # YAML dosyalarını okumak için PyYAML kütüphanesi gereklidir. requirements.txt'e eklenmeli.
import logging # Loglama için
import os # Dosya varlığını kontrol etmek için

# Bu modül için bir logger oluştur
# 'src.core.config_utils' adında bir logger döndürür.
logger = logging.getLogger(__name__)

def load_config_from_yaml(config_path="config/main_config.yaml"):
    """
    Belirtilen YAML yapılandırma dosyasından ayarları yükler.

    Dosya bulunamazsa veya YAML formatında hata olursa, durumu kritik olarak loglar
    ve boş bir sözlük (`{}`) döndürür. Bu, run_evo.py'nin hatayı tespit edip
    başlatma sürecini durdurmasını sağlar.

    Args:
        config_path (str): Yüklenecek YAML yapılandırma dosyasının yolu.
                           Varsayılan değer 'config/main_config.yaml'.

    Returns:
        dict: Yüklenen yapılandırma ayarları bir Python sözlüğü olarak.
              Hata durumunda boş sözlük `{}` döner.
              Yüklenen config objesi None ise de boş sözlüğe çevrilir.
    """
    # Yapılandırma dosyasının belirtilen yolda varlığını kontrol et
    if not os.path.exists(config_path):
        # Dosya yoksa kritik hata logla ve boş sözlük döndür.
        # Bu, run_evo.py'deki hata kontrolü tarafından yakalanır.
        logger.critical(f"Hata: Yapılandırma dosyası bulunamadı: {config_path}")
        return {}

    # Dosyayı okumayı ve YAML formatını işlemeyi dene
    try:
        # Dosyayı okuma modu ('r') ve metin formatında ('utf-8' encoding) aç.
        with open(config_path, 'r', encoding='utf-8') as f:
            # yaml.safe_load() kullanarak YAML verisini güvenli bir şekilde Python objesine yükle.
            # safe_load, rastgele Python objeleri oluşturmayı engelleyerek güvenlik açığı riskini azaltır.
            config = yaml.safe_load(f)

        # Yüklenen config'in None veya boş olup olmadığını kontrol et
        # Boş YAML dosyası (veya sadece yorum içeren) yaml.safe_load tarafından None olarak yüklenebilir.
        # Uygulamada config'in bir sözlük olması beklendiği için None'ı boş sözlüğe çeviririz.
        if config is None:
            logger.warning(f"Yapılandırma dosyası boş veya geçersiz içerik barındırıyor: {config_path}. Boş sözlük olarak yorumlanıyor.")
            config = {} # None ise boş sözlüğe çevir

        # Başarılı yükleme logu
        # logger.info(f"Yapılandırma dosyası başarıyla yüklendi: {config_path}")
        # INFO seviyesinde loglamak çok sık olabilir, sadece DEBUG seviyesinde detayları loglamak daha iyi.
        # DEBUG seviyesinde tüm config içeriğini loglamak, hata ayıklama için faydalı olabilir,
        # ancak hassas bilgiler (API keyleri gibi) içerebileceği durumlarda dikkatli olunmalı veya loglama filtrelenmeli.
        # logger.debug(f"Yüklenen Konfigürasyon Detayları: {config}")

        return config # Başarılı durumda yüklenen config sözlüğünü döndür

    except yaml.YAMLError as e:
        # YAML formatında bir hata oluşursa (örn: syntax hatası)
        # Kritik hata logla ve boş sözlük döndür. Bu hata run_evo.py'de yakalanır.
        logger.critical(f"Hata: Yapılandırma dosyası okunurken YAML format hatası oluştu: {config_path}\nHata: {e}", exc_info=True)
        return {}
    except Exception as e:
        # Dosya okuma (izin hatası vb.) veya başka beklenmedik bir hata oluşursa
        # Kritik hata logla ve boş sözlük döndür. Bu hata run_evo.py'de yakalanır.
        logger.critical(f"Hata: Yapılandırma dosyası yüklenirken beklenmedik bir hata oluştu: {config_path}\nHata: {e}", exc_info=True)
        return {}

# # İsteğe bağlı olarak, yüklenen config'i uygulamanın her yerinden erişilebilen tek bir yerde (singleton) tutmak için bir sınıf eklenebilir
# # class ConfigManager: ...
# # Mevcut load_config_from_yaml fonksiyonunu run_evo.py'nin başında bir kere çağırmak şimdilik yeterli ve daha basit.