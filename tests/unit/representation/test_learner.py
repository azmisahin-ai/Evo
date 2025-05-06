# tests/unit/representation/test_learner.py

import pytest
import numpy as np
import os # BASE_DIR kalktı ama os hala gerekebilir (kullanılmıyorsa kaldırılabilir)
import sys # BASE_DIR kalktı ama sys hala gerekebilir (kullanılmıyorsa kaldırılabilir)
import logging

# Test edilecek modülü içe aktar
try:
    # Artık BASE_DIR hesaplamasına gerek yok, src doğrudan import edilebilir
    from src.representation.models import RepresentationLearner
    # get_config_value artık config_utils'da olduğu için doğru yerden import edilecek
    from src.core.config_utils import get_config_value, load_config_from_yaml # load_config_from_yaml testte kullanılmasa da dursun
    from src.core.logging_utils import setup_logging
    # Testler için loglamayı yapılandır
    setup_logging(config=None)
    test_logger = logging.getLogger(__name__)
    test_logger.info("src.representation.models ve loglama başarıyla içe aktarıldı.")

except ImportError as e:
    # sys.path düzeltildiğinden bu hata oluşmamalı, oluşursa ciddi bir sorun var demektir.
    pytest.fail(f"src modülleri içe aktarılamadı. conftest.py'nin doğru çalıştığından emin olun. Hata: {e}")

@pytest.fixture(scope="module")
def dummy_learner_config():
    """RepresentationLearner testi için sahte yapılandırma sözlüğü sağlar."""
    # RepresentationLearner'ın init veya learn metodunda ihtiyaç duyabileceği config değerleri
    config = {
        'processors': { # Processors'ın çıktı boyutlarına ihtiyaç duyar (sahte)
             'vision': {
                 'output_width': 64,
                 'output_height': 64,
             },
             'audio': {
                 'output_dim': 2, # AudioProcessor'ın gerçek implementasyonuna göre 2
             }
        },
        'representation': {
             'representation_dim': 128, # Öğrencinin çıktı vektörü boyutu (bu testte beklenen)
             # ... RepresentationLearner'ın kullandığı diğer representation configleri ...
        },
        # ... diğer genel configler ...
    }
    test_logger.debug("Sahte learner config fixture oluşturuldu.")
    return config


@pytest.fixture(scope="module")
def representation_learner_instance(dummy_learner_config):
    """Sahte yapılandırma ile RepresentationLearner örneği sağlar."""
    try:
        # RepresentationLearner'ın __init__ metodunun config aldığını varsayıyoruz
        learner = RepresentationLearner(dummy_learner_config)
        test_logger.debug("RepresentationLearner instance oluşturuldu.")
        yield learner # Test fonksiyonuna instance'ı ver
        # Testler bittikten sonra cleanup (varsa) çağrılabilir
        if hasattr(learner, 'cleanup'):
             learner.cleanup()
             test_logger.debug("RepresentationLearner cleanup çağrıldı.")
    except Exception as e:
        pytest.fail(f"RepresentationLearner başlatılırken hata oluştu: {e}", exc_info=True)


def test_representation_learner_basic_learn(representation_learner_instance, dummy_learner_config):
    """
    RepresentationLearner'ın learn metodunun sahte girdilerle doğru çıktıyı ürettiğini test eder.
    """
    test_logger.info("test_representation_learner_basic_learn testi başlatıldı.")

    # RepresentationLearner.learn metodunun beklediği sahte girdi verisi
    # Bu girdi {'visual': dict, 'audio': np.ndarray} formatında olmalı.
    # Bu dictionary, VisionProcessor ve AudioProcessor'ın process çıktılarının birleşimidir.

    # Sahte VisionProcessor çıktısı dictionary'si
    vis_out_w = get_config_value(dummy_learner_config, 'processors', {}).get('vision', {}).get('output_width', 64)
    vis_out_h = get_config_value(dummy_learner_config, 'processors', {}).get('vision', {}).get('output_height', 64)
    # Sahte grayscale ve edges arrayleri (VisionProcessor çıktısının formatı)
    dummy_processed_visual_gray = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
    dummy_processed_visual_edges = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
    dummy_visual_input = {'grayscale': dummy_processed_visual_gray, 'edges': dummy_processed_visual_edges}
    test_logger.debug(f"Sahte visual input oluşturuldu: {list(dummy_visual_input.keys())}")


    # Sahte AudioProcessor çıktısı array'i
    audio_out_dim = get_config_value(dummy_learner_config, 'processors', {}).get('audio', {}).get('output_dim', 2)
    dummy_audio_input = np.random.rand(audio_out_dim).astype(np.float32) # Float32 veya float64 olabilir? RepLearner implementasyonuna bağlı
    test_logger.debug(f"Sahte audio input oluşturuldu: {dummy_audio_input.shape}, {dummy_audio_input.dtype}")


    # RepresentationLearner.learn metoduna verilecek birleşik girdi
    dummy_processed_inputs = {
        'visual': dummy_visual_input,
        'audio': dummy_audio_input
    }
    test_logger.debug(f"Sahte learn metod girdisi oluşturuldu: {list(dummy_processed_inputs.keys())}")


    # Learn metodunu çağır
    try:
        learned_representation = representation_learner_instance.learn(dummy_processed_inputs)
        test_logger.debug(f"RepresentationLearner.learn çağrıldı. Çıktı tipi: {type(learned_representation)}")

    except Exception as e:
        pytest.fail(f"RepresentationLearner.learn çalıştırılırken hata oluştu: {e}", exc_info=True)


    # --- Çıktıyı Kontrol Et (Assert) ---
    # RepresentationLearner'ın Representation vektörü (numpy array) döndürmesi beklenir.
    # Boyutu config'deki representation_dim ile uyumlu olmalıdır.
    expected_representation_dim = get_config_value(dummy_learner_config, 'representation', {}).get('representation_dim', 128)


    # 1. Çıktı bir numpy array mi?
    assert isinstance(learned_representation, np.ndarray), f"Learn çıktısı numpy array olmalı, alınan tip: {type(learned_representation)}"
    test_logger.debug("Assert geçti: Çıktı tipi numpy array.")

    # 2. Beklenen şekle sahip mi? (Tek boyutlu representation vektörü bekleniyor)
    expected_representation_shape = (expected_representation_dim,)
    assert learned_representation.shape == expected_representation_shape, f"Representation vektörü beklenen şekle sahip olmalı. Beklenen: {expected_representation_shape}, Alınan: {learned_representation.shape}"
    test_logger.debug("Assert geçti: Çıktı beklenen şekle sahip.")

    # 3. Beklenen dtype'a sahip mi? (Genellikle float beklenir)
    assert np.issubdtype(learned_representation.dtype, np.floating), f"Representation vektörü float tipi olmalı, alınan dtype: {learned_representation.dtype}"
    test_logger.debug("Assert geçti: Çıktı float dtype'a sahip.")

    # 4. Çıktı None değil mi?
    assert learned_representation is not None, "Representation vektörü None olmamalı."
    test_logger.debug("Assert geçti: Çıktı None değil.")

    # 5. Değerler makul mü? (Örneğin, NaN veya inf içermemeli)
    # Bu, modelin ağırlıkları başlatıldığında bazen olabilir.
    assert not np.isnan(learned_representation).any(), "Representation vektörü NaN değerler içeriyor."
    assert not np.isinf(learned_representation).any(), "Representation vektörü Inf değerler içeriyor."
    test_logger.debug("Assert geçti: Çıktı NaN veya Inf içermiyor.")


    test_logger.info("test_representation_learner_basic_learn testi başarıyla tamamlandı.")


# TODO: RepresentationLearner için daha fazla test senaryosu eklenebilir:
# - Boş veya geçersiz girdi formatları
# - Farklı config değerleri (örn. representation_dim)
# - Öğrenme adımlarının (eğer learn metodu tek adım yapıyorsa) doğru çalıştığını doğrulama (daha ileri testler)
# - Hata işleme senaryoları