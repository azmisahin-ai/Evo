# src/core/config_utils.py
import yaml
import logging
import os # Dosya varlığını kontrol etmek için

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

def load_config_from_yaml(config_path="config/main_config.yaml"):
    """
    Belirtilen YAML dosyasından yapılandırma ayarlarını yükler.

    Args:
        config_path (str): Yüklenecek YAML yapılandırma dosyasının yolu.

    Returns:
        dict: Yüklenen yapılandırma ayarları bir sözlük olarak.
              Dosya bulunamazsa veya format hatası olursa boş sözlük döner.
    """
    if not os.path.exists(config_path):
        logger.critical(f"Hata: Yapılandırma dosyası bulunamadı: {config_path}")
        return {} # Dosya yoksa boş sözlük döndür

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Yapılandırma dosyası başarıyla yüklendi: {config_path}")
        # logger.debug(f"Yüklenen Konfigürasyon: {config}") # DEBUG seviyesinde tüm config'i logla (çok detaylı olabilir)
        return config
    except yaml.YAMLError as e:
        logger.critical(f"Hata: Yapılandırma dosyası okunurken YAML format hatası oluştu: {config_path}\nHata: {e}", exc_info=True)
        return {} # Format hatası varsa boş sözlük döndür
    except Exception as e:
        logger.critical(f"Hata: Yapılandırma dosyası yüklenirken beklenmedik bir hata oluştu: {config_path}\nHata: {e}", exc_info=True)
        return {} # Diğer hatalarda boş sözlük döndür

# # İsteğe bağlı olarak, tek bir instance olarak config'i tutmak için bir sınıf veya singleton eklenebilir
# class ConfigManager:
#     _instance = None
#     _config = None
#     _config_path = "config/main_config.yaml"

#     def __new__(cls, config_path=None):
#         if cls._instance is None:
#             cls._instance = super(ConfigManager, cls).__new__(cls)
#             # Yükleme sadece ilk instance oluşturulduğunda yapılır
#             cls._config_path = config_path if config_path else cls._config_path
#             cls._config = load_config_from_yaml(cls._config_path)
#         return cls._instance

#     def get_config(self):
#         return self._config

#     # Örnek kullanım:
#     # config_manager = ConfigManager()
#     # app_config = config_manager.get_config()
#     # Veya doğrudan fonksiyonu kullanmak daha basit olabilir:
#     # app_config = load_config_from_yaml()