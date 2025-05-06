# tests/conftest.py

import pytest
import os
import sys

# Pytest testleri başlamadan önce çalışacak hook
# Bu hook, projenin kök dizinini Python'ın sys.path'ine ekler.
# Bu sayede test dosyaları 'src' paketini ve diğer kök dizin altındaki
# modülleri kolayca içe aktarabilir.
# 'conftest.py' dosyasının 'tests' dizininin kökünde olduğunu varsayar.

@pytest.fixture(scope='session', autouse=True)
def add_project_root_to_sys_path():
    """
    Projenin kök dizinini sys.path'e ekleyerek src ve diğer üst seviye
    modüllerin testler tarafından import edilebilmesini sağlar.
    """
    # conftest.py'nin bulunduğu dizin (tests/)
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    # Projenin kök dizini (tests/'ın bir üst dizini)
    project_root = os.path.dirname(tests_dir)

    # Eğer kök dizin zaten sys.path'te değilse ekle
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"\nAdded {project_root} to sys.path") # Pytest çıktısında görünmesi için

    # Testlerin çalışmasına izin ver
    yield

    # İsteğe bağlı: Testler bittikten sonra sys.path'ten kaldırılabilir
    # Ancak genellikle bu gerekli değildir ve bırakılabilir.
    # if project_root in sys.path:
    #     sys.path.remove(project_root)

# İsteğe bağlı olarak buraya genel test fixture'ları eklenebilir.
# Örneğin, tüm testlerin kullanabileceği bir config objesi fixture'ı.
# Şu an için her test kendi dummy config'ini oluştursa da,
# merkezi bir config fixture'ı düşünülürse buraya gelebilir.

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