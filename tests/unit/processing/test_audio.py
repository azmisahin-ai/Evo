# tests/processing/test_audio.py

import pytest
import numpy as np
import os
import sys
import logging

# src dizinini Python yoluna ekle (eğer testler kök dizinden çalışmıyorsa gerekli olabilir)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Test edilecek modülü içe aktar
try:
    from src.processing.audio import AudioProcessor
    from src.core.config_utils import load_config_from_yaml # Fixture için gerekmeyebilir ama dursun
    from src.core.logging_utils import setup_logging
    # Testler için loglamayı yapılandır (Sadece config ile, level argümanı yok)
    setup_logging(config=None)
    test_logger = logging.getLogger(__name__)
    test_logger.info("src.processing.audio ve loglama başarıyla içe aktarıldı.")

except ImportError as e:
    pytest.fail(f"src modülleri içe aktarılamadı. Proje kök dizininden çalıştığınızdan emin olun veya PYTHONPATH'i ayarlayın. Hata: {e}")


@pytest.fixture(scope="module")
def dummy_audio_config():
    """AudioProcessor testi için sahte yapılandırma sözlüğü sağlar."""
    # AudioProcessor'ın init veya process sırasında ihtiyaç duyabileceği config değerleri
    # scripts/test_module.py'deki create_dummy_method_inputs'dan ipuçları alındı.
    config = {
        'audio': {
            'audio_rate': 44100,        # Ses örnekleme hızı
            'audio_chunk_size': 1024,   # İşlenecek ses bloğu boyutu
            # ... AudioProcessor'ın kullandığı diğer audio configleri ...
        },
         'processors': {
             'audio': {
                 'output_dim': 2, # Process çıktısının boyutu (örneğin, özellik vektörü boyutu)
                 'n_mfcc': 13,     # MFCC hesaplaması için kullanılan config (varsa)
                 # ... AudioProcessor'ın kullandığı diğer processor.audio configleri ...
             }
         },
        # ... diğer genel configler ...
    }
    test_logger.debug("Sahte audio config fixture oluşturuldu.")
    return config


@pytest.fixture(scope="module")
def audio_processor_instance(dummy_audio_config):
    """Sahte yapılandırma ile AudioProcessor örneği sağlar."""
    try:
        # AudioProcessor'ın __init__ metodunun config aldığını varsayarak başlatıyoruz
        processor = AudioProcessor(dummy_audio_config)
        test_logger.debug("AudioProcessor instance oluşturuldu.")
        yield processor # Test fonksiyonuna instance'ı ver
        # Testler bittikten sonra cleanup (varsa) çağrılabilir
        if hasattr(processor, 'cleanup'):
             processor.cleanup()
             test_logger.debug("AudioProcessor cleanup çağrıldı.")
    except Exception as e:
        pytest.fail(f"AudioProcessor başlatılırken hata oluştu: {e}", exc_info=True)


def test_audio_processor_basic_processing(audio_processor_instance, dummy_audio_config):
    """
    AudioProcessor'ın process metodunun sahte bir ses bloğu ile doğru çıktıyı ürettiğini test eder.
    """
    test_logger.info("test_audio_processor_basic_processing testi başlatıldı.")

    # AudioProcessor.process metodunun beklediği sahte girdi verisi (int16 numpy array)
    # Boyutlar config ile uyumlu olmalı.
    chunk_size = dummy_audio_config['audio']['audio_chunk_size']
    # Sahte int16 ses verisi (örneğin, rastgele gürültü veya basit bir ton)
    # np.iinfo(np.int16).max * 0.1 genlikte sinüs dalgası daha gerçekçi olabilir
    frequency = 440 # A4 nota
    amplitude = np.iinfo(np.int16).max * 0.1
    t = np.linspace(0., chunk_size / dummy_audio_config['audio']['audio_rate'], chunk_size)
    dummy_chunk = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)

    test_logger.debug(f"Sahte girdi ses bloğu oluşturuldu: {dummy_chunk.shape}, {dummy_chunk.dtype}")

    # Process metodunu çağır
    try:
        processed_output = audio_processor_instance.process(dummy_chunk)
        test_logger.debug(f"AudioProcessor.process çağrıldı. Çıktı tipi: {type(processed_output)}")

    except Exception as e:
        pytest.fail(f"AudioProcessor.process çalıştırılırken hata oluştu: {e}", exc_info=True)


    # --- Çıktıyı Kontrol Et (Assert) ---
    # scripts/test_module.py'deki create_dummy_method_inputs RepresentationLearner için
    # AudioProcessor çıktısının np.random.rand(audio_out_dim).astype(np.float32) formatında olduğunu varsayıyor.
    # Testimiz de bu varsayımı kontrol etsin.

    # Beklenen çıktı boyutu config'den
    expected_output_dim = dummy_audio_config['processors']['audio']['output_dim']

    # 1. Çıktı bir numpy array mi?
    assert isinstance(processed_output, np.ndarray), f"Process çıktısı numpy array olmalı, alınan tip: {type(processed_output)}"
    test_logger.debug("Assert geçti: Çıktı tipi numpy array.")

    # 2. Beklenen şekle sahip mi? (Tek boyutlu özellik vektörü bekleniyor)
    expected_output_shape = (expected_output_dim,)
    assert processed_output.shape == expected_output_shape, f"Çıktı beklenen şekle sahip olmalı. Beklenen: {expected_output_shape}, Alınan: {processed_output.shape}"
    test_logger.debug("Assert geçti: Çıktı beklenen şekle sahip.")

    # 3. Beklenen dtype'a sahip mi? (Genellikle float beklenir)
    # RepresentationLearner float32 beklediği için float32 veya float64 kontrol edebiliriz.
    # process çıktısının float tipinde olduğunu kontrol edelim.
    assert np.issubdtype(processed_output.dtype, np.floating), f"Çıktı float tipi olmalı, alınan dtype: {processed_output.dtype}"
    test_logger.debug("Assert geçti: Çıktı float dtype'a sahip.")

    # 4. Çıktı None değil mi?
    assert processed_output is not None, "Process çıktısı None olmamalı."
    test_logger.debug("Assert geçti: Çıktı None değil.")


    test_logger.info("test_audio_processor_basic_processing testi başarıyla tamamlandı.")


# TODO: AudioProcessor için daha fazla test senaryosu eklenebilir:
# - Farklı girdi boyutları (eğer process esnekse)
# - Boş veya geçersiz girdi (örn: 0 boyutta array)
# - Config değerlerinin (örn. n_mfcc) çıktıyı etkilediği durumlar
# - Hata işleme senaryoları (örn: yanlış dtype girdi)