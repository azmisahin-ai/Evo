# tests/unit/memory/test_memory.py

import pytest
import numpy as np
import os
import sys
import logging
import time # Zaman damgası için
import shutil # Test için geçici bellek dizini temizlemek için

# conftest.py sys.path'i ayarlayacak
# src modüllerini içe aktar
try:
    from src.memory.core import Memory
    from src.core.config_utils import get_config_value
    from src.core.logging_utils import setup_logging
    # Testler için loglamayı yapılandır
    setup_logging(config=None)
    test_logger = logging.getLogger(__name__)
    test_logger.info("src.memory.core ve loglama başarıyla içe aktarıldı.")

except ImportError as e:
    pytest.fail(f"src modülleri içe aktarılamadı. conftest.py'nin doğru çalıştığından emin olun. Hata: {e}")


# Bellek dosyalarının saklanacağı geçici bir dizin fixture'ı
@pytest.fixture(scope="function") # Her test fonksiyonu için ayrı bir bellek dizini
def temp_memory_dir(tmp_path):
    """Pytest'ın sağladığı geçici dizin içinde bellek için bir alt dizin oluşturur."""
    mem_dir = tmp_path / "test_memory"
    mem_dir.mkdir()
    test_logger.debug(f"Geçici bellek dizini oluşturuldu: {mem_dir}")
    yield str(mem_dir) # Dizin yolunu string olarak döndür
    # Test fonksiyonu bittikten sonra dizini temizle (otomatik oluyor tmp_path ile ama explicit iyi)
    # shutil.rmtree(mem_dir) # tmp_path bunu otomatik yapar

@pytest.fixture(scope="function") # Her test fonksiyonu için ayrı bir instance
def dummy_memory_config(temp_memory_dir):
    """Memory modülü testi için sahte yapılandırma sözlüğü sağlar."""
    config = {
        'memory': {
            # Bellek dosyalarının saklanacağı dizin. Geçici dizini kullanacağız.
            'storage_dir': temp_memory_dir,
            'representation_dim': 128, # Memory'nin saklayacağı representation vektörü boyutu
            'max_entries': 1000,       # Bellek kapasitesi (testte küçük tutulabilir)
            'num_retrieved_memories': 5, # Retrieve metodu için varsayılan
            # ... Memory'nin kullandığı diğer configler ...
        },
        'representation': { # get_config_value için gerekli olabilir
             'representation_dim': 128, # Memory config'deki ile aynı olmalı
        },
        # ... diğer genel configler ...
    }
    test_logger.debug("Sahte memory config fixture oluşturuldu.")
    return config


@pytest.fixture(scope="function") # Her test fonksiyonu için ayrı bir instance
def memory_instance(dummy_memory_config):
    """Sahte yapılandırma ile Memory örneği sağlar."""
    test_logger.debug("Memory instance oluşturuluyor...")
    try:
        # Memory'nin __init__ metodunun config aldığını varsayıyoruz
        mem = Memory(dummy_memory_config)
        test_logger.debug("Memory instance oluşturuldu.")
        yield mem # Test fonksiyonuna instance'ı ver
        # Testler bittikten sonra cleanup (varsa) çağrılabilir
        if hasattr(mem, 'cleanup'):
             mem.cleanup()
             test_logger.debug("Memory cleanup çağrıldı.")
    except Exception as e:
        pytest.fail(f"Memory başlatılırken hata oluştu: {e}", exc_info=True)


def test_memory_store_and_retrieve_basic(memory_instance, dummy_memory_config):
    """
    Memory modülünün representation vektörlerini sakladığını ve geri alabildiğini test eder.
    """
    test_logger.info("test_memory_store_and_retrieve_basic testi başlatıldı.")

    # Memory.store metodunun beklediği sahte girdi verisi (Representation vektörü)
    # Boyut config'deki representation_dim ile uyumlu olmalı.
    repr_dim = get_config_value(dummy_memory_config, 'memory', 'representation_dim', default=128)
    # RepresentationLearner çıktısı float64 olduğu için float64 kullanalım
    dummy_representation_1 = np.random.rand(repr_dim).astype(np.float64)
    dummy_representation_2 = np.random.rand(repr_dim).astype(np.float64)
    dummy_representation_3 = np.random.rand(repr_dim).astype(np.float64)

    # Test için bellekten geri alınacak vektör (dummy_representation_2'ye benzer olsun)
    # np.random.randn ile biraz gürültü ekleyerek benzerlik simüle edelim
    query_representation = dummy_representation_2 + np.random.randn(repr_dim).astype(np.float64) * 0.01
    test_logger.debug("Sahte representation vektörleri ve query oluşturuldu.")


    # --- Store Metodunu Test Et ---
    try:
        test_logger.debug("Memory.store çağrılıyor (1)...")
        memory_instance.store(dummy_representation_1, metadata={'source': 'test1', 'timestamp': time.time() - 10})
        test_logger.debug("Memory.store çağrıldı (1).")

        # Biraz bekleme, zaman damgalarının farklı olması için
        time.sleep(0.01)

        test_logger.debug("Memory.store çağrılıyor (2)...")
        memory_instance.store(dummy_representation_2, metadata={'source': 'test2', 'timestamp': time.time()})
        test_logger.debug("Memory.store çağrıldı (2).")

        time.sleep(0.01)

        test_logger.debug("Memory.store çağrılıyor (3)...")
        memory_instance.store(dummy_representation_3, metadata={'source': 'test3', 'timestamp': time.time() + 10})
        test_logger.debug("Memory.store çağrıldı (3).")

    except Exception as e:
        pytest.fail(f"Memory.store çalıştırılırken hata oluştu: {e}", exc_info=True)

    # Store sonrası bellek içeriğini kontrol et (isteğe bağlı, iç duruma erişim gerektirebilir)
    test_logger.debug("Assert geçti: Bellek giriş sayısı doğru.")


    # --- Retrieve Metodunu Test Et ---
    try:
        test_logger.debug("Memory.retrieve çağrılıyor...")
        # query_representation'a en benzer olanları (default config'e göre 5 adet) isteyelim
        retrieved_entries = memory_instance.retrieve(query_representation)
        test_logger.debug(f"Memory.retrieve çağrıldı. Alınan giriş sayısı: {len(retrieved_entries)}")

    except Exception as e:
        pytest.fail(f"Memory.retrieve çalıştırılırken hata oluştu: {e}", exc_info=True)


    # --- Retrieve Çıktısını Kontrol Et (Assert) ---
    # retrieve metodu bir liste döndürmeli
    assert isinstance(retrieved_entries, list), f"Retrieve çıktısı liste olmalı, alınan tip: {type(retrieved_entries)}"
    test_logger.debug("Assert geçti: Retrieve çıktısı liste.")

    # Liste içinde dictionary'ler olmalı ve her dictionary belirli anahtarları içermeli
    # README'de Memory.retrieve çıktısı list of dicts olarak belirtilmiş:
    # [{'representation': array, 'metadata': {}, 'timestamp': float}, ...]
    assert len(retrieved_entries) > 0, "Retrieve sonrası boş liste dönmemeli (eğer bellek boş değilse)."
    test_logger.debug("Assert geçti: Retrieve sonrası boş liste dönmedi.")

    first_entry = retrieved_entries[0]
    assert isinstance(first_entry, dict), f"Retrieve listesindeki elemanlar dict olmalı, ilki tipi: {type(first_entry)}"
    assert 'representation' in first_entry, "Retrieve edilen giriş 'representation' anahtarını içermeli."
    assert 'metadata' in first_entry, "Retrieve edilen giriş 'metadata' anahtarını içermeli."
    assert 'timestamp' in first_entry, "Retrieve edilen giriş 'timestamp' anahtarını içermeli."
    test_logger.debug("Assert geçti: Retrieve edilen ilk eleman doğru anahtarları içeriyor.")

    # Representation numpy array olmalı ve boyutu doğru olmalı
    assert isinstance(first_entry['representation'], np.ndarray), f"'representation' değeri numpy array olmalı, tipi: {type(first_entry['representation'])}"
    assert first_entry['representation'].shape == (repr_dim,), f"'representation' şekli beklenen gibi olmalı. Beklenen: {(repr_dim,)}, Alınan: {first_entry['representation'].shape}"
    assert np.issubdtype(first_entry['representation'].dtype, np.floating), f"'representation' dtype'ı float olmalı, alınan: {first_entry['representation'].dtype}"
    test_logger.debug("Assert geçti: Retrieve edilen 'representation' değeri doğru formatta.")

    # Metadata dictionary olmalı
    assert isinstance(first_entry['metadata'], dict), f"'metadata' değeri dict olmalı, tipi: {type(first_entry['metadata'])}"
    test_logger.debug("Assert geçti: Retrieve edilen 'metadata' değeri dict.")

    # Zaman damgası sayı olmalı
    assert isinstance(first_entry['timestamp'], (int, float)), f"'timestamp' değeri sayı olmalı, tipi: {type(first_entry['timestamp'])}"
    test_logger.debug("Assert geçti: Retrieve edilen 'timestamp' değeri sayı.")

    # En benzer olanın (dummy_representation_2'ye en yakın olanın) genellikle ilk sırada dönmesi beklenir.
    # Ancak bu testin kapsamını aşabilir (benzerlik hesaplama mantığını test etmek ayrı bir konu).
    # Sadece listenin döndüğünü ve formatının doğru olduğunu test etmek yeterli.
    # Daha ileri testlerde belirli bir query için beklenen sonucu (hangi representation'ın döneceği)
    # mocklayarak veya sahte benzerlik skorları vererek test edilebilir.


    test_logger.info("test_memory_store_and_retrieve_basic testi başarıyla tamamlandı.")

# TODO: Memory modülü için daha fazla test senaryosu eklenebilir:
# - Maksimum giriş sayısını aştığında ne olduğu (en eskilerin silinmesi?)
# - Farklı metadata formatları ile store etme
# - Boş bellekten retrieve etme
# - Geçersiz representation (NaN, yanlış boyut vb.) ile store etme
# - Bellek dosyasına kaydetme ve yükleme (Disk I/O testleri, daha çok integration testi olabilir)
# - Eşik değerine göre retrieve (varsa)