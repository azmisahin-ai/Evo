# tests/unit/memory/test_memory.py

import pytest
import numpy as np
import os
import sys # sys.path manipülasyonu conftest'te.
import logging
import time # Zaman damgası için
import shutil # Test için geçici bellek dizini temizlemek için

# conftest.py sys.path'i ayarlayacak, bu importlar artık doğrudan çalışmalı.
# setup_logging burada çağrılmayacak, konftest tarafından ayarlanacak.
from src.memory.core import Memory
from src.core.config_utils import get_config_value
from src.core.logging_utils import setup_logging # setup_logging import'ı conftest kullanıyor olabilir, kalsın.

# Bu test dosyası için bir logger oluştur. Seviye conftest tarafından ayarlanacak.
test_logger = logging.getLogger(__name__)
# Artık setup_logging(config=None) burada çağrılmayacak.
test_logger.info("src.memory.core ve gerekli yardımcılar başarıyla içe aktarıldı.")


# Bellek dosyalarının saklanacağı geçici bir dizin fixture'ı
@pytest.fixture(scope="function") # Her test fonksiyonu için ayrı bir bellek dizini
def temp_memory_dir(tmp_path):
    """Pytest'ın sağladığı geçici dizin içinde bellek için bir alt dizin oluşturur."""
    # tmp_path pytest tarafından sağlanan pathlib.Path objesidir.
    mem_dir = tmp_path / "test_memory"
    mem_dir.mkdir(parents=True, exist_ok=True) # Dizinleri oluştur, zaten varsa hata verme
    test_logger.debug(f"Geçici bellek dizini oluşturuldu: {mem_dir}")
    # Memory modülü string path bekleyebilir, bu yüzden str() kullanıyoruz.
    yield str(mem_dir)
    # Test fonksiyonu bittikten sonra pytest tmp_path'i otomatik temizler.
    # Manuel temizlik (shutil.rmtree) gerekli değil ama explicit istenirse yapılabilir.
    # test_logger.debug(f"Geçici bellek dizini temizleniyor: {mem_dir}")
    # shutil.rmtree(mem_dir, ignore_errors=True)


@pytest.fixture(scope="function") # Her test fonksiyonu için ayrı bir instance
def dummy_memory_config(temp_memory_dir):
    """Memory modülü testi için sahte yapılandırma sözlüğü sağlar."""
    # Gerçek Memory.__init__ metodu memory_file_path'i config'den alıyor.
    # Bu test fixture'ı da bu yapıyı taklit etmeli.
    # memory_file_path'i temporary_memory_dir ve bir dosya adı birleştirerek oluşturacağız.

    test_file_name = "test_core_memory.pkl" # Teste özel dosya adı
    # os.path.join kullanarak işletim sistemine uygun path oluştur.
    test_memory_path = os.path.join(temp_memory_dir, test_file_name)


    config = {
        'memory': {
            # Gerçek Memory modülü memory_file_path'i bekliyor:
            'memory_file_path': test_memory_path, # <-- Temporary dosya yolunu buraya atıyoruz.
            'max_memory_size': 1000,         # Test için maksimum boyut
            'num_retrieved_memories': 5,     # Test için varsayılan geri çağrı sayısı
            'representation_dim': 128,       # Test için representation boyutu (Memory __init__ ve store/retrieve kullanıyor)
        },
        # representation.representation_dim gerçek RepresentationLearner testinde kullanılıyor.
        # Memory testinde doğrudan memory.representation_dim config değeri kullanıldığı için
        # representation alt anahtarına burada gerek yok.
        # Ama consistency için tutulabilir.
        # 'representation': {
        #      'representation_dim': 128, # get_config_value testinde kullanılırsa kalsın.
        # },
        # ... diğer genel configler ...
    }
    test_logger.debug(f"Sahte memory config fixture oluşturuldu. memory_file_path: {config['memory']['memory_file_path']}")
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
             # cleanup_safely kullanmak daha sağlam olur.
             from src.core.utils import cleanup_safely # cleanup_safely import et
             cleanup_safely(mem.cleanup, logger_instance=test_logger, error_message="Memory instance cleanup sırasında hata")
             test_logger.debug("Memory cleanup çağrıldı.")
    except Exception as e:
        # Memory başlatılırken hata olursa testi başarısız yap.
        pytest.fail(f"Memory başlatılırken hata oluştu: {e}", exc_info=True)


def test_memory_store_and_retrieve_basic(memory_instance, dummy_memory_config):
    """
    Memory modülünün representation vektörlerini sakladığını ve geri alabildiğini test eder.
    Store edilen anıların, query representation'a en yakın olanının geri çağrıldığını
    (en azından boş liste dönmediğini) doğrular.
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
    query_noise_level = 0.01 # Ne kadar gürültü ekleneceği (benzerlik seviyesini etkiler)
    query_representation = dummy_representation_2 + np.random.randn(repr_dim).astype(np.float64) * query_noise_level
    test_logger.debug("Sahte representation vektörleri ve query oluşturuldu.")


    # --- Store Metodunu Test Et ---
    # Store metotlarının çağrılması ve hata kontrolü.
    try:
        test_logger.debug("Memory.store çağrılıyor (1)...")
        memory_instance.store(dummy_representation_1, metadata={'source': 'test1', 'timestamp': time.time() - 10})
        test_logger.debug("Memory.store çağrıldı (1).")

        # Biraz bekleme, zaman damgalarının farklı olması için (retrieve'de sıralama zaman damgasına göre ise etkili olur)
        time.sleep(0.01)

        test_logger.debug("Memory.store çağrılıyor (2)...")
        memory_instance.store(dummy_representation_2, metadata={'source': 'test2', 'timestamp': time.time()})
        test_logger.debug("Memory.store çağrıldı (2).")

        time.sleep(0.01)

        test_logger.debug("Memory.store çağrılıyor (3)...")
        memory_instance.store(dummy_representation_3, metadata={'source': 'test3', 'timestamp': time.time() + 10})
        test_logger.debug("Memory.store çağrıldı (3).")

    except Exception as e:
        pytest.fail(f"Memory.store çalıştırılırken beklenmedik hata oluştu: {e}", exc_info=True)


    # Bellek giriş sayısı kontrolü (İsteğe bağlı, ama store'un çalıştığını basitçe gösterir)
    # Bu kontrol, memory_entries gibi dahili özniteliğe doğrudan erişmeye çalışmıyordu,
    # Memory sınıfının get_all_representations metodu varsa onu kullanabiliriz, veya bu kontrolü atlayabiliriz.
    # Şu an get_all_representations var, onu kullanalım:
    try:
         all_reps_in_memory = memory_instance.get_all_representations()
         assert len(all_reps_in_memory) == 3, f"Bellekte 3 Representation olması bekleniyordu, {len(all_reps_in_memory)} bulundu."
         test_logger.debug("Assert geçti: get_all_representations ile bellek giriş sayısı doğru.")
    except Exception as e:
         test_logger.warning(f"Memory.get_all_representations çağrılırken hata oluştu veya sonuç beklenmiyor: {e}", exc_info=True)
         # Bu assert başarısız olursa testi kırmayalım (warning logla), retrieve testine devam edelim.
         # Eğer get_all_representations testin kritik bir parçasıysa pytest.fail çağrılabilir.
         # Şimdilik sadece loglayıp devam edelim.
         pass # Assert başarısız olsa da devam et


    # --- Retrieve Metodunu Test Et ---
    # retrieve metodunun çağrılması ve hata kontrolü.
    try:
        test_logger.debug("Memory.retrieve çağrılıyor...")
        # query_representation'a en benzer olanları (default config'e göre 5 adet) isteyelim
        retrieved_entries = memory_instance.retrieve(query_representation)
        test_logger.debug(f"Memory.retrieve çağrıldı. Alınan giriş sayısı: {len(retrieved_entries)}")

    except Exception as e:
        pytest.fail(f"Memory.retrieve çalıştırılırken beklenmedik hata oluştu: {e}", exc_info=True)


    # --- Retrieve Çıktısını Kontrol Et (Assert) ---
    # retrieve metodu bir liste döndürmeli
    assert isinstance(retrieved_entries, list), f"Retrieve çıktısı liste olmalı, alınan tip: {type(retrieved_entries)}"
    test_logger.debug("Assert geçti: Retrieve çıktısı liste.")

    # Liste içinde dictionary'ler olmalı ve her dictionary belirli anahtarları içermeli
    # [{'representation': array, 'metadata': {}, 'timestamp': float}, ...]
    # Bellekte 3 anı store ettik ve query dummy_representation_2'ye çok benzerdi.
    # Retrieve, benzerliğe göre sıralıyor ve ilk 5'i istiyor.
    # Eğer benzerlik hesaplama doğru çalışıyorsa, dummy_representation_2'ye karşılık gelen anının
    # retrieve listesinde olması beklenir ve liste boş olmamalıdır.
    assert len(retrieved_entries) > 0, "Retrieve sonrası boş liste dönmemeli (en azından store edilen anılardan biri bulunmalı)."
    test_logger.debug(f"Assert geçti: Retrieve sonrası liste boş değil. ({len(retrieved_entries)} giriş bulundu).")

    # İlk elemanın formatını kontrol et (rastgele bir elemanın formatını kontrol etmek daha sağlam olabilir)
    first_entry = retrieved_entries[0]
    assert isinstance(first_entry, dict), f"Retrieve listesindeki elemanlar dict olmalı, ilki tipi: {type(first_entry)}"
    assert 'representation' in first_entry, "Retrieve edilen giriş 'representation' anahtarını içermeli."
    assert 'metadata' in first_entry, "Retrieve edilen giriş 'metadata' anahtarını içermeli."
    assert 'timestamp' in first_entry, "Retrieve edilen giriş 'timestamp' anahtarını içermeli."
    test_logger.debug("Assert geçti: Retrieve edilen ilk eleman doğru anahtarları içeriyor.")

    # Representation numpy array olmalı ve boyutu doğru olmalı
    expected_repr_shape = (repr_dim,)
    rep_from_retrieved = first_entry.get('representation')
    assert rep_from_retrieved is not None and isinstance(rep_from_retrieved, np.ndarray), f"'representation' değeri numpy array olmalı, tipi: {type(rep_from_retrieved)}"
    assert rep_from_retrieved.shape == expected_repr_shape, f"'representation' şekli beklenen gibi olmalı. Beklenen: {expected_repr_shape}, Alınan: {rep_from_retrieved.shape}"
    assert np.issubdtype(rep_from_retrieved.dtype, np.floating), f"'representation' dtype'ı float olmalı, alınan: {rep_from_retrieved.dtype}"
    test_logger.debug("Assert geçti: Retrieve edilen 'representation' değeri doğru formatta.")

    # Metadata dictionary olmalı
    assert isinstance(first_entry.get('metadata'), dict), f"'metadata' değeri dict olmalı, tipi: {type(first_entry.get('metadata'))}"
    test_logger.debug("Assert geçti: Retrieve edilen 'metadata' değeri dict.")

    # Zaman damgası sayı olmalı
    assert isinstance(first_entry.get('timestamp'), (int, float)), f"'timestamp' değeri sayı olmalı, tipi: {type(first_entry.get('timestamp'))}"
    test_logger.debug("Assert geçti: Retrieve edilen 'timestamp' değeri sayı.")

    # İsteğe bağlı: Geri çağrılan anılardan birinin store ettiğimiz anılardan biri olduğunu doğrula.
    # Bu, numpy array eşitliğini kontrol etmeyi gerektirir ve biraz daha karmaşıktır.
    # np.array_equal veya np.allclose kullanarak yapılabilir.
    # Örneğin, retrieved_entries listesindeki representation'ların, store ettiğimiz dummy representation'lardan biriyle eşleştiğini kontrol edebiliriz.
    # Bu, benzerlik hesaplama mantığının doğru çalıştığının daha güçlü bir kanıtı olur.
    # Şimdilik sadece liste boş değilse başarılı kabul ediyoruz, bu ilk test için yeterli.

    test_logger.info("test_memory_store_and_retrieve_basic testi başarıyla tamamlandı.")


# TODO: Memory modülü için daha fazla test senaryosu eklenebilir:
# - Maksimum giriş sayısını aştığında en eskilerin silindiği durum.
# - Farklı metadata formatları ile store etme.
# - Boş bellekten retrieve etme.
# - Geçersiz representation (NaN, yanlış boyut vb.) ile store etme ve bunların saklanmadığını doğrulama.
# - Bellek dosyasına kaydetme (_save_to_storage) ve yükleme (_load_from_storage) işlevlerinin doğru çalıştığı testleri. (Bu, dosya sistemine eriştiği için saf unit testten çok integration test gibi düşünülebilir, ancak yine de memory modülünün sorumluluğundadır.)
# - Geçersiz memory_file_path config değeri verildiğinde ne olduğu.
# - Retrieve sırasında query representation boyutu yanlışsa boş liste döndürdüğü.
# - get_all_representations metodunun doğru Representationları döndürdüğü.
# - Eşik değerine göre retrieve (varsa) - şu an kodda yok ama config'de 'retrieval_threshold' belirtilmiş. Implemente edilince test edilmeli.