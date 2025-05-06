# tests/processing/test_vision.py

import pytest
import numpy as np
import os
import sys

# Test edilecek modülü içe aktar
try:
    from src.processing.vision import VisionProcessor
    # src.core.config_utils'u da içe aktaralım, config oluşturmak için gerekebilir
    from src.core.config_utils import load_config_from_yaml
    # setup_logging'i de içe aktaralım, testler çalışırken log görmek faydalı olabilir
    from src.core.logging_utils import setup_logging
    import logging
    # Testler için loglamayı yapılandır (INFO seviyesinde)
    setup_logging(config=None)
    test_logger = logging.getLogger(__name__)
    test_logger.info("src modülleri ve loglama başarıyla içe aktarıldı.")

except ImportError as e:
    pytest.fail(f"src modülleri içe aktarılamadı. Proje kök dizininden çalıştığınızdan emin olun veya PYTHONPATH'i ayarlayın. Hata: {e}")

# VisionProcessor testi için temel bir yapılandırma objesi (pytest fixture daha iyi olabilir gelecekte)
# Bu, VisionProcessor'ın init metodunun veya process metodunun ihtiyaç duyabileceği config değerlerini içermelidir.
# scripts/test_module.py'deki create_dummy_method_inputs fonksiyonundan ipuçları alarak oluşturuldu.
@pytest.fixture(scope="module")
def dummy_vision_config():
    """VisionProcessor testi için sahte yapılandırma sözlüğü sağlar."""
    # VisionProcessor'ın init veya process sırasında ihtiyaç duyabileceği config değerleri
    # Gerçek config dosyasını yüklemek yerine manuel olarak oluşturuyoruz.
    config = {
        'vision': {
            'input_width': 640,
            'input_height': 480,
            # ... VisionProcessor'ın kullandığı diğer vision configleri ...
        },
        'processors': {
             'vision': {
                 'output_width': 64, # Bu değer process çıktısının şeklini etkileyebilir
                 'output_height': 64, # Bu değer process çıktısının şeklini etkileyebilir
                 'grayscale_weight': 0.7, # Process logic'te kullanılabilir
                 'edge_threshold': 100, # Process logic'te kullanılabilir
                 # ... VisionProcessor'ın kullandığı diğer processor.vision configleri ...
             }
        },
        # ... diğer genel configler ...
    }
    test_logger.debug("Sahte vision config fixture oluşturuldu.")
    return config


@pytest.fixture(scope="module")
def vision_processor_instance(dummy_vision_config):
    """Sahte yapılandırma ile VisionProcessor örneği sağlar."""
    try:
        # VisionProcessor'ın __init__ metodunun config aldığını varsayarak başlatıyoruz
        processor = VisionProcessor(dummy_vision_config)
        test_logger.debug("VisionProcessor instance oluşturuldu.")
        yield processor # Test fonksiyonuna instance'ı ver
        # Testler bittikten sonra cleanup (varsa) çağrılabilir
        if hasattr(processor, 'cleanup'):
             processor.cleanup()
             test_logger.debug("VisionProcessor cleanup çağrıldı.")
    except Exception as e:
        pytest.fail(f"VisionProcessor başlatılırken hata oluştu: {e}", exc_info=True)


def test_vision_processor_basic_processing(vision_processor_instance):
    """
    VisionProcessor'ın process metodunun sahte bir görüntü ile doğru çıktıyı ürettiğini test eder.
    """
    test_logger.info("test_vision_processor_basic_processing testi başlatıldı.")

    # VisionProcessor.process metodunun beklediği sahte girdi verisi (BGR numpy array)
    # Boyutlar config ile uyumlu olmalı veya process metodunun esnek olması gerekir.
    # scripts/test_module.py'deki create_dummy_method_inputs'a benzer bir input oluşturalım.
    dummy_height = 480
    dummy_width = 640
    dummy_frame = np.random.randint(0, 256, size=(dummy_height, dummy_width, 3), dtype=np.uint8)
    test_logger.debug(f"Sahte girdi görüntüsü oluşturuldu: {dummy_frame.shape}, {dummy_frame.dtype}")


    # Process metodunu çağır
    try:
        processed_output = vision_processor_instance.process(dummy_frame)
        test_logger.debug(f"VisionProcessor.process çağrıldı. Çıktı tipi: {type(processed_output)}")

    except Exception as e:
        pytest.fail(f"VisionProcessor.process çalıştırılırken hata oluştu: {e}", exc_info=True)


    # --- Çıktıyı Kontrol Et (Assert) ---
    # scripts/test_module.py'deki create_dummy_method_inputs RepresentationLearner için
    # VisionProcessor çıktısının {'grayscale': array, 'edges': array} dictionary'si olduğunu varsayıyor.
    # Testimiz de bu varsayımı kontrol etsin.

    # 1. Çıktı bir sözlük mü?
    assert isinstance(processed_output, dict), f"Process çıktısı dict olmalı, alınan tip: {type(processed_output)}"
    test_logger.debug("Assert geçti: Çıktı tipi dict.")

    # 2. Beklenen anahtarları içeriyor mu?
    expected_keys = ['grayscale', 'edges']
    for key in expected_keys:
        assert key in processed_output, f"Çıktı sözlüğü '{key}' anahtarını içermeli."
        assert isinstance(processed_output[key], np.ndarray), f"'{key}' değeri numpy array olmalı."
        # İsteğe bağlı: Çıktı array'lerinin şeklini veya dtype'ını kontrol et
        # Bu, VisionProcessor'ın implementasyonuna bağlıdır.
        # Örneğin, 64x64 gri tonlama ve kenar haritaları bekleniyorsa:
        expected_output_shape = (
             vision_processor_instance.config['processors']['vision']['output_height'],
             vision_processor_instance.config['processors']['vision']['output_width']
        )
        assert processed_output[key].shape == expected_output_shape, f"'{key}' çıktısı beklenen şekle sahip olmalı. Beklenen: {expected_output_shape}, Alınan: {processed_output[key].shape}"
        assert processed_output[key].dtype == np.uint8, f"'{key}' çıktısı beklenen dtype'a sahip olmalı. Beklenen: np.uint8, Alınan: {processed_output[key].dtype}"
        test_logger.debug(f"Assert geçti: Çıktı sözlüğü '{key}' anahtarını içeriyor, numpy array ve beklenen şekil/dtype'a sahip.")


    # 3. Çıktı None değil mi? (Zaten isinstance kontrolü ile örtülü ama netleştirebilir)
    assert processed_output is not None, "Process çıktısı None olmamalı."
    test_logger.debug("Assert geçti: Çıktı None değil.")


    test_logger.info("test_vision_processor_basic_processing testi başarıyla tamamlandı.")


# TODO: VisionProcessor için daha fazla test senaryosu eklenebilir:
# - Farklı girdi boyutları (eğer destekleniyorsa)
# - Boş veya geçersiz girdi
# - Config değerlerinin (örn. grayscale_weight, edge_threshold) çıktıyı etkilediği durumlar
# - Hata işleme senaryoları