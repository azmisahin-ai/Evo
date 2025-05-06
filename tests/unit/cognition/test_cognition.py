# tests/unit/cognition/test_cognition.py

import time
import pytest
import numpy as np
import os
import sys
import logging
from unittest.mock import MagicMock # Mock objeleri için

# conftest.py sys.path'i ayarlayacak
# src modüllerini import et
from src.cognition.understanding import UnderstandingModule # UnderstandingModule import edildi
from src.core.config_utils import get_config_value
# setup_logging conftest'te, burada gerek yok.
# from src.core.logging_utils import setup_logging
from src.core.utils import cleanup_safely # Fixture cleanup için


# Bu test dosyası için bir logger oluştur. Seviye conftest tarafından ayarlanacak.
test_logger = logging.getLogger(__name__)
test_logger.info("src.cognition.understanding modülü ve gerekli yardımcılar başarıyla içe aktarıldı.")


# UnderstandingModule Testleri
@pytest.fixture(scope="function")
def dummy_understanding_config():
    """UnderstandingModule testi için sahte yapılandırma sözlüğü sağlar."""
    # UnderstandingModule'un ihtiyaç duyabileceği config değerleri
    config = {
        'cognition': {
            # Anlama (Understanding) ayarları (örn: eşik değerleri)
            # scripts/test_module.py'deki dummy input oluşturma kısmından alınabilir.
            'familiarity_threshold': 0.8,
            'audio_energy_threshold': 1000.0,
            'visual_edges_threshold': 50.0,
            'brightness_threshold_high': 200.0,
            'brightness_threshold_low': 50.0,
            # max_concept_similarity ve most_similar_concept_id için belki eşikler olabilir.
            # 'concept_recognition_threshold': 0.85, # Concept recognition DecisionModule'da mı?
        },
        # Processor ve Representation çıktı boyutları UnderstandingModule içinde kullanılabilir
        # (örn. parlaklık hesaplarken görsel çıktı boyutu veya ses enerjisi hesaplarken ses çıktı boyutu).
        'processors': {
             'vision': {
                 'output_width': 64,
                 'output_height': 64,
             },
             'audio': {
                 'output_dim': 2, # AudioProcessor çıktısı boyutu
             }
        },
         'representation': {
             'representation_dim': 128, # Representation boyutu
        },
        # ... diğer genel configler ...
    }
    test_logger.debug("Sahte understanding config fixture oluşturuldu.")
    return config


@pytest.fixture(scope="function")
def understanding_module_instance(dummy_understanding_config):
    """Sahte yapılandırma ile UnderstandingModule örneği sağlar."""
    test_logger.debug("UnderstandingModule instance oluşturuluyor...")
    try:
        # UnderstandingModule'un __init__ metodunun config aldığını varsayıyoruz
        module = UnderstandingModule(dummy_understanding_config)

        test_logger.debug("UnderstandingModule instance oluşturuldu.")
        yield module # Test fonksiyonuna instance'ı ver

        # Testler bittikten sonra cleanup (varsa) çağrılabilir.
        if hasattr(module, 'cleanup'):
             cleanup_safely(module.cleanup, logger_instance=test_logger, error_message="UnderstandingModule instance cleanup sırasında hata (teardown)")
             test_logger.debug("UnderstandingModule cleanup çağrıldı.")

    except Exception as e:
        # pytest.fail artık exc_info argümanı almayabilir.
        test_logger.error(f"UnderstandingModule fixture başlatılırken veya cleanup sırasında hata: {e}", exc_info=True)
        pytest.fail(f"UnderstandingModule fixture hatası: {e}")


def test_understanding_module_process_basic(understanding_module_instance, dummy_understanding_config, mocker):
    """
    UnderstandingModule'un process metodunun sahte girdilerle
    beklenen formatta anlama sinyalleri (dictionary) ürettiğini test eder.
    """
    test_logger.info("test_understanding_module_process_basic testi başlatıldı.")

    # UnderstandingModule.process metodunun beklediği sahte girdi verisi.
    # Bu girdi, Processors ve RepresentationLearner çıktıları formatında olmalı.
    # Processors çıktısı: {'visual': dict, 'audio': np.ndarray}
    # RepresentationLearner çıktısı (Representation vektörü): np.ndarray (shape (D,))
    # UnderstandingModule.process(self, processed_inputs, learned_representation, relevant_memory_entries) bekliyor olabilir
    # README'ye göre UnderstandingModule'ın tek başına bir process metodu var ve sadece processed_inputs alıyor gibi görünüyor?
    # "CognitionCore: UnderstandingModule, DecisionModule, LearningModule başlatılır..., decide metodu sahte girdilerle çağrılır, anlama sinyalleri ve karar loglanır."
    # Bu, anlama sinyallerinin CognitionCore içinde veya UnderstandingModule'ın ayrı bir metodunda üretildiğini düşündürüyor.
    # STRUCTURE.md'ye bakalım... "Understanding Module: Gelen işlenmiş duyusal veriyi yorumlar ve temel anlama sinyalleri (parlaklık, hareket, ses seviyesi, tanıdık eşleşme vb.) üretir."
    # INTERACTION_GUIDE.md'de "UnderstandingModule çıktısı: {'similarity_score': float, 'high_audio_energy': bool, ...}" formatı belirtilmiş.

    # Varsayım: UnderstandingModule'ın ana metodu `process` veya `generate_signals` gibi bir şey olabilir
    # ve Processors + Representation + Memory çıktılarını girdi olarak alıyordur.
    # scripts/test_module.py'deki dummy input oluşturma mantığına bakalım...
    # scripts/test_module.py create_dummy_method_inputs CognitionCore için dummy processed_inputs, dummy_representation, dummy_memory_entries oluşturuyor.
    # Bu girdiler UnderstandingModule tarafından kullanılıyor olmalı.
    # UnderstandingModule'ın process metodunu bulalım src/cognition/understanding.py içinde...
    # src/cognition/understanding.py inceleniyor... Sınıf adı UnderstandingModule. Metot adı `process`.
    # process(self, processed_inputs, learned_representation, relevant_memory_entries): <-- Evet, bu girdileri alıyor.

    # Şimdi bu metoda uygun sahte girdileri oluşturalım.
    # Processors çıktısı (visual dict, audio np array)
    vis_out_w = get_config_value(dummy_understanding_config, 'processors', 'vision', 'output_width', default=64, expected_type=int)
    vis_out_h = get_config_value(dummy_understanding_config, 'processors', 'vision', 'output_height', default=64, expected_type=int)
    dummy_processed_visual = {'grayscale': np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8),
                              'edges': np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)}
    audio_out_dim = get_config_value(dummy_understanding_config, 'processors', 'audio', 'output_dim', default=2, expected_type=int)
    dummy_processed_audio = np.random.rand(audio_out_dim).astype(np.float32)
    dummy_processed_inputs = {'visual': dummy_processed_visual, 'audio': dummy_processed_audio}
    test_logger.debug("Sahte processed_inputs oluşturuldu.")

    # RepresentationLearner çıktısı (Representation vektörü)
    repr_dim = get_config_value(dummy_understanding_config, 'representation', 'representation_dim', default=128, expected_type=int)
    dummy_representation = np.random.rand(repr_dim).astype(np.float64)
    test_logger.debug("Sahte representation oluşturuldu.")

    # Memory retrieve çıktısı (list of dicts)
    # Mock Memory objesinden retrieve çıktısı alıyormuş gibi simüle edelim.
    # UnderstandingModule'ın retrieve metodunu çağırmadığını, dışarıdan aldığını varsayıyoruz.
    dummy_memory_entries = [] # Boş liste veya birkaç sahte giriş
    # Birkaç sahte bellek girişi ekleyelim.
    num_dummy_memories = 3
    for i in range(num_dummy_memories):
         dummy_mem_rep = np.random.rand(repr_dim).astype(np.float64)
         # Bir tanesi query representation'a çok yakın olsun (familiarity testi için)
         if i == 0: dummy_mem_rep = dummy_representation.copy() + np.random.randn(repr_dim) * 0.001
         dummy_memory_entries.append({
             'representation': dummy_mem_rep,
             'metadata': {'source': f'test_mem_{i}'},
             'timestamp': time.time() - (num_dummy_memories - i) * 10 # Zaman damgaları farklı olsun
         })
    test_logger.debug(f"Sahte {len(dummy_memory_entries)} memory entries oluşturuldu.")


    # UnderstandingModule.process metodunun beklediği dördüncü argüman: current_concepts listesi
    # Bu, LearningModule.get_concepts() çıktısıdır.
    # Test için boş bir liste veya birkaç sahte kavram Representation'ı kullanabiliriz.
    # Boş liste başlangıç için yeterlidir.
    dummy_current_concepts = []
    test_logger.debug(f"Sahte current_concepts listesi oluşturuldu ({len(dummy_current_concepts)} kavram).")


    # process metodunu çağır
    try:
        # process(self, processed_inputs, learned_representation, relevant_memory_entries)
        understanding_signals = understanding_module_instance.process(
            dummy_processed_inputs,
            dummy_representation,
            dummy_memory_entries,
             dummy_current_concepts # <-- Bu satırı ekle
        )
        test_logger.debug(f"UnderstandingModule.process çağrıldı. Çıktı tipi: {type(understanding_signals)}")

    except Exception as e:
        # pytest.fail artık exc_info argümanı almayabilir.
        test_logger.error(f"UnderstandingModule.process çalıştırılırken hata: {e}", exc_info=True)
        pytest.fail(f"UnderstandingModule.process çalıştırılırken beklenmedik hata: {e}")


    # --- Çıktıyı Kontrol Et (Assert) ---
    # UnderstandingModule.process çıktısı beklenen dictionary formatında olmalı.
    # INTERACTION_GUIDE.md'deki format:
    # {'similarity_score': float, 'high_audio_energy': bool, 'high_visual_edges': bool,
    #  'is_bright': bool, 'is_dark': bool,
    #  'max_concept_similarity': float, 'most_similar_concept_id': int or None,
    #  'mood_signal': str # Veya başka içsel durum sinyalleri
    # }

    # 1. Çıktı bir sözlük mü?
    assert isinstance(understanding_signals, dict), f"process çıktısı dict olmalı, alınan tip: {type(understanding_signals)}"
    test_logger.debug("Assert geçti: Çıktı tipi dict.")

    # 2. Beklenen anahtarları içeriyor mu ve tipleri doğru mu?
    expected_keys_and_types = {
        'similarity_score': float, # Bellek benzerliği skoru
        'high_audio_energy': bool, # Yüksek ses enerjisi algılandı mı?
        'high_visual_edges': bool, # Yüksek görsel kenarlar (karmaşıklık) algılandı mı?
        'is_bright': bool,         # Ortam parlak mı?
        'is_dark': bool,           # Ortam karanlık mı?
        'max_concept_similarity': float, # En benzer kavrama olan benzerlik skoru
        'most_similar_concept_id': (int, type(None)), # En benzer kavramın ID'si veya None
        # Eklenen diğer sinyaller (örn. mood_signal, novelty_score vb.)
        # 'novelty_score': float, # Örneğin, yeni girdilere dayalı yenilik skoru
        # 'sensory_conflict': bool, # Örneğin, görsel ve işitsel girdiler tutarsız mı?
        # ... UnderstandingModule'ın ürettiği diğer sinyaller ...
    }
    # Gerçek UnderstandingModule implementasyonunda üretilen sinyallere göre bu dictionary güncellenmeli.
    # Şu anki implementasyonda hangi sinyaller üretiliyor bakmak lazım src/cognition/understanding.py'ye.
    # process metoduna bakınca üretilen sinyaller şunlar:
    # similarity_score (float), high_audio_energy (bool), high_visual_edges (bool), is_bright (bool), is_dark (bool),
    # max_concept_similarity (float), most_similar_concept_id (int veya None), novelty_score (float), sensory_conflict (bool).
    # Yukarıdaki expected_keys_and_types dictionary'si güncel implementasyon ile uyumlu.

    for key, expected_type in expected_keys_and_types.items():
        assert key in understanding_signals, f"Çıktı sözlüğü '{key}' anahtarını içermeli."
        # Değerin None olup olmadığını ve tipini kontrol et
        value = understanding_signals[key]
        if value is not None:
             # expected_type tuple ise (örn: (int, type(None))), isinstance doğru çalışır.
             assert isinstance(value, expected_type), f"'{key}' değeri beklenen tipte olmalı. Beklenen: {expected_type}, Alınan: {type(value)}."
        else:
             # Eğer değer None ise, expected_type içinde type(None) olmalı.
             assert expected_type == type(None) or (isinstance(expected_type, tuple) and type(None) in expected_type), \
                 f"'{key}' değeri None, ancak beklenen tip {expected_type} içinde None yok."

        test_logger.debug(f"Assert geçti: Çıktı sözlüğü '{key}' anahtarını içeriyor ve tipi doğru.")


    # 3. Çıktı None değil mi?
    assert understanding_signals is not None, "process çıktısı None olmamalı."
    test_logger.debug("Assert geçti: Çıktı None değil.")

    # 4. Mantıksal tutarlılık assertleri (isteğe bağlı ama unit testin gücünü artırır)
    # Örneğin, hem parlak hem karanlık olamaz.
    assert not (understanding_signals.get('is_bright', False) and understanding_signals.get('is_dark', False)), \
        "Çıktı aynı anda hem 'is_bright' hem 'is_dark' True olamaz."
    test_logger.debug("Assert geçti: Parlaklık/Karanlık tutarlılığı.")

    # Örneğin, similarity_score ve max_concept_similarity 0.0 ile 1.0 arasında olmalı (veya np.nan olabilir hata durumunda).
    # Eğer np.nan dönmesi hata değilse bu assertler kaldırılabilir veya nan kontrolü yapılabilir.
    # Varsayım: Skarlar 0-1 arası (veya NaN değil).
    sim_score = understanding_signals.get('similarity_score')
    if sim_score is not None and not np.isnan(sim_score):
         assert 0.0 <= sim_score <= 1.0, f"'similarity_score' 0.0 ile 1.0 arasında olmalı. Alınan: {sim_score}"
    concept_sim_score = understanding_signals.get('max_concept_similarity')
    if concept_sim_score is not None and not np.isnan(concept_sim_score):
         assert 0.0 <= concept_sim_score <= 1.0, f"'max_concept_similarity' 0.0 ile 1.0 arasında olmalı. Alınan: {concept_sim_score}"
    test_logger.debug("Assert geçti: Skorlar 0-1 aralığında (veya NaN değil).")


    test_logger.info("test_understanding_module_process_basic testi başarıyla tamamlandı.")


# TODO: UnderstandingModule için daha fazla test senaryosu eklenebilir:
# - Farklı girdi kombinasyonları ve eşik değerleri için beklenen sinyalleri test etme.
#   (örn. yüksek ses enerjisi ve kenar girdileri => high_audio_energy=True, high_visual_edges=True)
#   (örn. parlak ortam ve düşük eşik => is_bright=True)
#   (örn. query representation bellektekine çok benzer => similarity_score yüksek)
# - Boş veya geçersiz girdi formatları verildiğinde ne olduğu (hata loglayıp varsayılan sinyaller mi dönüyor?).
# - Config değerleri (eşikler) değiştirildiğinde çıktının nasıl değiştiğini test etme.
# - Bağımlı modüllere (Processors, Representation) ne tür çağrılar yaptığını test etme (eğer process içinde çağrılıyorsa, şu an dışarıdan girdi alıyor).
# DecisionModule, LearningModule testleri de bu dosyaya eklenebilir.