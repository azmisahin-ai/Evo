# tests/conftest.py

import pytest
import os
import sys
import logging

# Projenin kök dizinini sys.path'e ekleme fixture'ı
# conftest.py'nin 'tests' dizininin kökünde olduğunu varsayar.
@pytest.fixture(scope='session', autouse=True)
def add_project_root_to_sys_path():
    """
    Projenin kök dizinini sys.path'e ekleyerek src ve diğer üst seviye
    modüllerin testler tarafından import edilebilmesini sağlar.
    Bu fixture test oturumu başladığında otomatik olarak çalışır.
    """
    # conftest.py'nin bulunduğu dizin (tests/)
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    # Projenin kök dizini (tests/'ın bir üst dizini)
    project_root = os.path.dirname(tests_dir)

    # Eğer kök dizin zaten sys.path'te değilse ekle (genellikle ilk sıraya)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        # print(f"\nAdded {project_root} to sys.path") # DEBUG amaçlı, pytest çıktısında görünür


# Test oturumu için loglama ayarı fixture'ı
@pytest.fixture(scope='session', autouse=True)
def setup_test_logging():
    """
    Pytest test oturumu için loglama seviyesini DEBUG olarak ayarlar
    ve konsola çıktı veren bir handler ekler.
    Bu fixture test oturumu başladığında otomatik olarak çalışır.
    """
    # Mevcut handler'ları kaldırarak çift loglama çıktısını önle
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    for filter in logging.root.filters[:]:
         logging.root.removeFilter(filter) # Filtreleri de kaldır

    # logging.basicConfig, handler yoksa temel bir handler kurar.
    # Formatı DEBUG seviyesindeki logları gösterecek şekilde ayarla.
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s')

    # src ve tests loggers'ının seviyelerini DEBUG olarak ayarla
    logging.getLogger('src').setLevel(logging.DEBUG)
    logging.getLogger('tests').setLevel(logging.DEBUG) # Test logları için
    # Daha az detay görmek istediğiniz diğer modüller için seviyeyi artırabilirsiniz:
    # logging.getLogger('some.noisy.module').setLevel(logging.INFO)

    #print("\nTest logging configured to DEBUG level.") # DEBUG amaçlı, pytest çıktısında görünür

    # yield ifadesi testlerin çalışmasına izin verir.
    yield

    # Teardown: Test oturumu bittikten sonra handler'ları temizlemek isteğe bağlıdır.
    # Genellikle process kapanacağı için gerekli değildir.
    # for handler in logging.root.handlers[:]:
    #     logging.root.removeHandler(handler)


# İsteğe bağlı olarak buraya genel test fixture'ları eklenebilir.
# Örneğin, tüm testlerin kullanabileceği bir config objesi fixture'ı.
# Şu an için her test kendi dummy config'ini oluştursa da,
# merkezi bir config fixture'ı düşünülürse buraya gelebilir.
# Ancak unit testlerde dummy config daha izole test sağlar.

# @pytest.fixture(scope="session")
# def main_app_config():
#     """load_config_from_yaml ile gerçek config dosyasını yükler."""
#     from src.core.config_utils import load_config_from_yaml
#     config = load_config_from_yaml("config/main_config.yaml")
#     # Config yüklenemezse testleri atla veya hata ver
#     if not config:
#          pytest.skip("Ana yapılandırma dosyası yüklenemedi.")
#          # Veya pytest.fail("Ana yapılandırma dosyası yüklenemedi.")
#     return config