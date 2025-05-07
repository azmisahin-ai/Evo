import unittest
import sys
import os
import numpy as np
import logging

# Mock kullanabilmek için (bazı gelişmiş senaryolar veya bağımlılıkları izole etmek için gerekebilir,
# ancak mevcut testler için check_* fonksiyonlarını mock'lamıyoruz)
# from unittest.mock import patch, MagicMock

# Proje kök dizinini sys.path'e ekleyin. Test dosyasının tests/unit/cognition içinde olduğunu varsayar.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

# Şimdi modülleri import edebiliriz.
# check_* fonksiyonları src.core.utils'tan gelir.
# get_config_value src.core.config_utils'tan gelir.
try:
    from src.cognition.understanding import UnderstandingModule
    from src.core.utils import check_input_not_none, check_input_type, check_numpy_input # check_* buradan
    from src.core.config_utils import get_config_value # get_config_value buradan
except ImportError as e:
     raise ImportError(f"Temel modüller import edilemedi. PYTHONPATH doğru ayarlanmış mı? Hata: {e}")


# Testler sırasında logger çıktılarını görmek isterseniz bu satırları etkinleştirebilirsiniz.
# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger('src.cognition.understanding').setLevel(logging.DEBUG)
# logging.getLogger('src.core.utils').setLevel(logging.DEBUG) # Utils logger'ını da görmek için
# logging.getLogger('src.core.config_utils').setLevel(logging.DEBUG) # Config Utils logger'ını da görmek için


# Varsayılan çıktı sözlüğü, testlerde tekrar kullanmak için
DEFAULT_UNDERSTANDING_SIGNALS = {
    'similarity_score': 0.0,
    'high_audio_energy': False,
    'high_visual_edges': False,
    'is_bright': False,
    'is_dark': False,
    'max_concept_similarity': 0.0,
    'most_similar_concept_id': None,
}


class TestUnderstanding(unittest.TestCase):

    def setUp(self):
        """Her test metodundan önce çalışır, varsayılan bir konfigürasyon ve UnderstandingModule örneği oluşturur."""
        # Varsayılan, geçerli konfigürasyon
        self.default_config = {
            'audio_energy_threshold': 1000.0,
            'visual_edges_threshold': 50.0,
            'brightness_threshold_high': 200.0,
            'brightness_threshold_low': 50.0,
        }
        # UnderstandingModule __init__ içinde float çevirme yaptığı için buraya gerek yok.
        self.module = UnderstandingModule(self.default_config)

    def tearDown(self):
        """Her test metodundan sonra çalışır."""
        pass

    # --- __init__ Testleri (Bu testler pytest çıktısında geçiyordu) ---
    # ConfigUtils'taki get_config_value artık int'i float'a otomatik çevirmez,
    # ancak UnderstandingModule.__init__ içinde manuel çeviri eklendiği için bu testler hala geçmeli.

    def test_init_with_valid_config(self):
        """Geçerli bir konfigürasyon ile başlatmayı test eder."""
        self.assertEqual(self.module.config, self.default_config)
        # ConfigUtils'taki get_config_value int döndürebilir ama __init__ içinde float'a çevriliyor.
        self.assertEqual(self.module.audio_energy_threshold, 1000.0)
        self.assertIsInstance(self.module.audio_energy_threshold, float)
        self.assertEqual(self.module.visual_edges_threshold, 50.0)
        self.assertIsInstance(self.module.visual_edges_threshold, float)
        self.assertEqual(self.module.brightness_threshold_high, 200.0)
        self.assertIsInstance(self.module.brightness_threshold_high, float)
        self.assertEqual(self.module.brightness_threshold_low, 50.0)
        self.assertIsInstance(self.module.brightness_threshold_low, float)

    def test_init_with_missing_config_values(self):
        """Bazı konfigürasyon değerleri eksikken başlatmayı test eder (varsayılanlar kullanılmalı)."""
        incomplete_config = {
            'audio_energy_threshold': 500, # int olarak verelim
            'brightness_threshold_high': 150.0,
            # visual_edges_threshold ve brightness_threshold_low eksik
        }
        module = UnderstandingModule(incomplete_config)
        self.assertEqual(module.audio_energy_threshold, 500.0) # int verilse de float'a çevrilmiş olmalı
        self.assertIsInstance(module.audio_energy_threshold, float)
        self.assertEqual(module.visual_edges_threshold, 50.0) # Varsayılan
        self.assertIsInstance(module.visual_edges_threshold, float)
        self.assertEqual(module.brightness_threshold_high, 150.0)
        self.assertIsInstance(module.brightness_threshold_high, float)
        self.assertEqual(module.brightness_threshold_low, 50.0) # Varsayılan
        self.assertIsInstance(module.brightness_threshold_low, float)


    def test_init_with_invalid_config_types(self):
        """Geçersiz tipte konfigürasyon değerleri ile başlatmayı test eder (varsayılanlar kullanılmalı)."""
        invalid_type_config = {
            'audio_energy_threshold': "not a float", # Geçersiz tip -> Varsayılan (1000.0 float)
            'visual_edges_threshold': 60, # Geçerli int -> Float'a çevrilmeli (60.0)
            'brightness_threshold_high': [250], # Geçersiz tip -> Varsayılan (200.0 float)
            'brightness_threshold_low': 30.0, # Geçerli float -> Float kalmalı (30.0)
        }
        # ConfigUtils'taki get_config_value tip hatasında varsayılan döner ve WARN loglar.
        # __init__ içinde float çevirme bunu etkilemez (varsayılan da float olduğu için).
        module = UnderstandingModule(invalid_type_config)
        self.assertEqual(module.audio_energy_threshold, 1000.0) # Varsayılan
        self.assertIsInstance(module.audio_energy_threshold, float)
        self.assertEqual(module.visual_edges_threshold, 60.0) # Doğru parse edildi ve çevrildi
        self.assertIsInstance(module.visual_edges_threshold, float)
        self.assertEqual(module.brightness_threshold_high, 200.0) # Varsayılan
        self.assertIsInstance(module.brightness_threshold_high, float)
        self.assertEqual(module.brightness_threshold_low, 30.0) # Doğru parse edildi
        self.assertIsInstance(module.brightness_threshold_low, float)

    # --- process Testleri ---

    def test_process_all_valid_inputs_above_thresholds(self):
        """Tüm girdilerin geçerli ve Process çıktılarının eşiklerin üzerinde olduğu durumu test eder."""
        learned_representation = np.array([1.0, 2.0], dtype=np.float32)
        relevant_memory_entries = [{'representation': np.array([1.1, 2.2], dtype=np.float32)}]
        processed_inputs = {
            'audio': np.array([self.module.audio_energy_threshold + 1.0, 0.5], dtype=np.float32),
            'visual': {
                'edges': np.array([[self.module.visual_edges_threshold + 1.0]], dtype=np.float32),
                'grayscale': np.array([[self.module.brightness_threshold_high + 1.0]], dtype=np.float32)
            }
        }
        current_concepts = [np.array([1.2, 2.4], dtype=np.float32), np.array([-1.0, 3.0], dtype=np.float32)] # İlk kavram daha benzer

        # Beklenen hesaplamalar
        # Bellek benzerliği
        norm_repr = np.linalg.norm(learned_representation)
        norm_mem = np.linalg.norm(relevant_memory_entries[0]['representation'])
        expected_memory_similarity = np.dot(learned_representation, relevant_memory_entries[0]['representation']) / (norm_repr * norm_mem)

        # Kavram benzerlikleri
        norm_conc0 = np.linalg.norm(current_concepts[0])
        norm_conc1 = np.linalg.norm(current_concepts[1])
        sim0 = np.dot(learned_representation, current_concepts[0]) / (norm_repr * norm_conc0)
        sim1 = np.dot(learned_representation, current_concepts[1]) / (norm_repr * norm_conc1)
        expected_max_concept_similarity = max(sim0, sim1)
        expected_most_similar_concept_id = 0 if sim0 >= sim1 else 1 # sim0 ~1.0, sim1 ~0.707, so 0

        expected_output = {
            'similarity_score': expected_memory_similarity,
            'high_audio_energy': True,
            'high_visual_edges': True,
            'is_bright': True,
            'is_dark': False,
            'max_concept_similarity': expected_max_concept_similarity,
            'most_similar_concept_id': expected_most_similar_concept_id,
        }

        result = self.module.process(processed_inputs, learned_representation, relevant_memory_entries, current_concepts)

        self.assertIsInstance(result, dict)
        self.assertIn('similarity_score', result)
        self.assertIn('high_audio_energy', result)
        self.assertIn('high_visual_edges', result)
        self.assertIn('is_bright', result)
        self.assertIn('is_dark', result)
        self.assertIn('max_concept_similarity', result)
        self.assertIn('most_similar_concept_id', result)

        self.assertAlmostEqual(result['similarity_score'], expected_output['similarity_score'], places=6)
        self.assertEqual(result['high_audio_energy'], expected_output['high_audio_energy'])
        self.assertEqual(result['high_visual_edges'], expected_output['high_visual_edges'])
        self.assertEqual(result['is_bright'], expected_output['is_bright'])
        self.assertEqual(result['is_dark'], expected_output['is_dark'])
        self.assertAlmostEqual(result['max_concept_similarity'], expected_output['max_concept_similarity'], places=6)
        self.assertEqual(result['most_similar_concept_id'], expected_output['most_similar_concept_id'])


    def test_process_all_valid_inputs_below_thresholds(self):
        """Tüm girdilerin geçerli ve Process çıktılarının eşiklerin altında olduğu durumu test eder."""
        learned_representation = np.array([1.0, 2.0], dtype=np.float32)
        relevant_memory_entries = [{'representation': np.array([1.1, 2.2], dtype=np.float32)}] # Benzerlik > 0
        processed_inputs = {
            'audio': np.array([self.module.audio_energy_threshold - 1.0, 0.1], dtype=np.float32), # Düşük
            'visual': {
                'edges': np.array([[self.module.visual_edges_threshold - 1.0]], dtype=np.float32), # Düşük
                'grayscale': np.array([[self.module.brightness_threshold_low - 1.0]], dtype=np.float32) # Karanlık
            }
        }
        current_concepts = [np.array([1.2, 2.4], dtype=np.float32), np.array([-1.0, 3.0], dtype=np.float32)] # Kavram benzerliği > 0

        # Beklenen hesaplamalar (Yukarıdaki testle aynı benzerlikler)
        norm_repr = np.linalg.norm(learned_representation)
        norm_mem = np.linalg.norm(relevant_memory_entries[0]['representation'])
        expected_memory_similarity = np.dot(learned_representation, relevant_memory_entries[0]['representation']) / (norm_repr * norm_mem)

        norm_conc0 = np.linalg.norm(current_concepts[0])
        norm_conc1 = np.linalg.norm(current_concepts[1])
        sim0 = np.dot(learned_representation, current_concepts[0]) / (norm_repr * norm_conc0)
        sim1 = np.dot(learned_representation, current_concepts[1]) / (norm_repr * norm_conc1)
        expected_max_concept_similarity = max(sim0, sim1)
        expected_most_similar_concept_id = 0 if sim0 >= sim1 else 1 # sim0 ~1.0, sim1 ~0.707, so 0


        expected_output = {
            'similarity_score': expected_memory_similarity,
            'high_audio_energy': False, # Eşik altı
            'high_visual_edges': False, # Eşik altı
            'is_bright': False,
            'is_dark': True, # Eşik altı (karanlık)
            'max_concept_similarity': expected_max_concept_similarity,
            'most_similar_concept_id': expected_most_similar_concept_id,
        }

        result = self.module.process(processed_inputs, learned_representation, relevant_memory_entries, current_concepts)

        self.assertAlmostEqual(result['similarity_score'], expected_output['similarity_score'], places=6)
        self.assertEqual(result['high_audio_energy'], expected_output['high_audio_energy'])
        self.assertEqual(result['high_visual_edges'], expected_output['high_visual_edges'])
        self.assertEqual(result['is_bright'], expected_output['is_bright'])
        self.assertEqual(result['is_dark'], expected_output['is_dark'])
        self.assertAlmostEqual(result['max_concept_similarity'], expected_output['max_concept_similarity'], places=6)
        self.assertEqual(result['most_similar_concept_id'], expected_output['most_similar_concept_id'])


    def test_process_all_inputs_none(self):
        """Tüm girdilerin None olduğu durumu test eder."""
        # check_input_not_none False döner. is_valid_* bayrakları False olur.
        # Hiçbir if bloğuna girilmez. Varsayılan sözlük döndürülür.
        result = self.module.process(None, None, None, None)
        self.assertEqual(result, DEFAULT_UNDERSTANDING_SIGNALS)

    def test_process_empty_lists_or_dicts(self):
        """Girdilerin boş liste veya boş dictionary olduğu durumu test eder."""
        learned_representation = np.array([1.0, 2.0], dtype=np.float32) # Representation geçerli
        # Process inputs boş dict ({}) - is_valid_processed_inputs True olur ama içindeki .get() None döner veya boş array check'leri fail olur.
        # Memory boş liste ([]) - is_valid_memory_list True olur ama döngüye girilmez.
        # Concepts boş liste ([]) - is_valid_concepts_list True olur ama döngüye girilmez.
        result = self.module.process({}, learned_representation, [], [])

        # Benzerlikler hesaplanmaz (listeler boş). Process bayrakları hesaplanmaz (processed_inputs içi boş/geçersiz).
        # Varsayılan sözlük döndürülür.
        self.assertEqual(result, DEFAULT_UNDERSTANDING_SIGNALS)

    # DÜZELTİLEN TEST: Invalid 'learned_representation' verildiğinde, Process inputlar işlenecek.
    # check_numpy_input exception fırlatmadığı için process devam eder.
    def test_process_invalid_representation(self):
        """Geçersiz 'learned_representation' tipi veya şekli verildiğinde test eder."""
        invalid_representations = [
            "not a numpy array", # check_numpy_input yakalar
            np.array([1, 2, 3]), # int dtype (check_numpy_input yakalar)
            np.array([[1.0, 2.0]], dtype=np.float32), # 2D shape (check_numpy_input yakalar)
        ]
        # Diğer girdiler geçerli OLSUN ki Process bayrakları hesaplanabilsin ve beklenti doğru kurulsun.
        valid_processed_inputs = {
            'audio': np.array([self.module.audio_energy_threshold + 1.0, 0.5], dtype=np.float32), # Yüksek ses enerjisi -> True
            'visual': {
                'edges': np.array([[self.module.visual_edges_threshold + 1.0]], dtype=np.float32), # Yüksek görsel kenar -> True
                'grayscale': np.array([[self.module.brightness_threshold_high + 1.0]], dtype=np.float32) # Parlak ortam -> True
            }
        }
        # Bellek ve kavramlar boş olsun, çünkü representation geçersizse bunlar hesaplanmayacak.
        empty_memory = []
        empty_concepts = []

        # Beklenen çıktı: Representation geçersiz olduğu için benzerlikler default (0.0/None).
        # Process input geçerli olduğu ve içindeki değerler eşiklerin üzerinde olduğu için Process bayrakları True.
        expected_output = {
            'similarity_score': 0.0,
            'high_audio_energy': True,
            'high_visual_edges': True,
            'is_bright': True,
            'is_dark': False, # Parlak olduğu için is_dark False
            'max_concept_similarity': 0.0,
            'most_similar_concept_id': None,
        }

        for invalid_repr in invalid_representations:
            with self.subTest(msg=f"Testing with invalid representation: {invalid_repr}"):
                # is_valid_representation False olacak (check_numpy_input False dönecek).
                # Benzerlik hesaplama if'leri atlanacak (is_valid_representation False).
                # Process input geçerli olduğu için process bayrakları hesaplanacak.
                result = self.module.process(valid_processed_inputs, invalid_repr, empty_memory, empty_concepts)
                # Sonuç, Process bayrakları hesaplanmış ve benzerlikler default kalmış olmalı.
                self.assertEqual(result, expected_output)


    # DÜZELTİLEN TEST: Bellek listesindeki *bazı* girdiler geçersiz olsa da, geçerli olanlar üzerinden
    # hesaplama yapılıp beklentinin karşılanması test edilir. check_numpy_input exception fırlatmaz.
    def test_process_memory_with_invalid_entries(self):
        """Bellek listesi içinde geçersiz girdiler olduğunda test eder."""
        learned_representation = np.array([1.0, 2.0], dtype=np.float32) # Geçerli Representation
        relevant_memory_entries = [
            {'representation': np.array([1.1, 2.2], dtype=np.float32)}, # Geçerli (ID 0)
            {'representation': "not a numpy array"}, # Geçersiz tip, check_numpy_input yakalar, atlar
            {'representation': np.array([[3.0, 4.0]], dtype=np.float32)}, # Geçersiz şekil, check_numpy_input yakalar, atlar
            {'other_key': 'no representation'}, # 'representation' yok, atlar
            {'representation': np.array([0.0, 0.0], dtype=np.float32)}, # Sıfır norm, atlar
            {'representation': np.array([5.0, 6.0], dtype=np.float32)}, # Geçerli (ID 5)
        ]
        # Beklenen en yüksek benzerlik, sadece geçerli girdilerden (ID 0 ve ID 5) hesaplanmalı
        norm_repr = np.linalg.norm(learned_representation)
        norm_mem0 = np.linalg.norm(relevant_memory_entries[0]['representation'])
        norm_mem5 = np.linalg.norm(relevant_memory_entries[5]['representation'])
        sim0 = np.dot(learned_representation, relevant_memory_entries[0]['representation']) / (norm_repr * norm_mem0)
        sim5 = np.dot(learned_representation, relevant_memory_entries[5]['representation']) / (norm_repr * norm_mem5)
        expected_max_similarity = max(sim0, sim5) # sim0 ~1.0, sim5 ~0.92, so max is sim0

        # Diğer girdiler boş/None olsun
        empty_processed_inputs = {}
        empty_concepts = []

        result = self.module.process(empty_processed_inputs, learned_representation, relevant_memory_entries, empty_concepts)

        # Bellek benzerliği geçerli girdiden hesaplanmalı ve atanmalı
        self.assertAlmostEqual(result['similarity_score'], expected_max_similarity, places=6)
        # Diğer alanlar varsayılan olmalı çünkü diğer girdiler boş/None
        self.assertEqual(result['high_audio_energy'], False)
        self.assertEqual(result['high_visual_edges'], False)
        self.assertEqual(result['is_bright'], False)
        self.assertEqual(result['is_dark'], False)
        self.assertEqual(result['max_concept_similarity'], 0.0) # Kavram listesi boştu
        self.assertIsNone(result['most_similar_concept_id']) # Kavram listesi boştu


    # DÜZELTİLEN TEST: Kavram listesindeki *bazı* girdiler geçersiz olsa da, geçerli olanlar üzerinden
    # hesaplama yapılıp beklentinin karşılanması test edilir. check_numpy_input exception fırlatmaz.
    def test_process_concepts_with_invalid_entries(self):
        """Kavram listesi içinde geçersiz girdiler olduğığında test eder."""
        learned_representation = np.array([1.0, 2.0], dtype=np.float32) # Geçerli Representation
        current_concepts = [
            np.array([1.2, 2.4], dtype=np.float32), # Geçerli (ID 0)
            "not a numpy array", # Geçersiz tip, check_numpy_input yakalar, atlar
            np.array([[3.0, 4.0]], dtype=np.float32), # Geçersiz şekil, check_numpy_input yakalar, atlar
            np.array([0.0, 0.0], dtype=np.float32), # Sıfır norm, atlar
            np.array([5.0, 6.0], dtype=np.float32), # Geçerli (ID 4)
        ]
         # Beklenen en yüksek benzerlik, sadece geçerli girdilerden (ID 0 ve ID 4) hesaplanmalı
        norm_repr = np.linalg.norm(learned_representation)
        norm_conc0 = np.linalg.norm(current_concepts[0])
        norm_conc4 = np.linalg.norm(current_concepts[4])
        sim0 = np.dot(learned_representation, current_concepts[0]) / (norm_repr * norm_conc0)
        sim4 = np.dot(learned_representation, current_concepts[4]) / (norm_repr * norm_conc4)
        expected_max_similarity = max(sim0, sim4) # sim0 ~1.0, sim4 ~0.92, so max is sim0

        # En benzer olan ID 0'daki kavram olmalı
        if sim0 >= sim4:
             expected_concept_id = 0
        else:
             expected_concept_id = 4 # Bu örnekte 0 olacak ama sim'e göre karar veriyoruz.

        # Diğer girdiler boş/None olsun
        empty_processed_inputs = {}
        empty_memory = []

        result = self.module.process(empty_processed_inputs, learned_representation, empty_memory, current_concepts)

        # Kavram benzerliği geçerli girdiden hesaplanmalı ve atanmalı
        self.assertAlmostEqual(result['max_concept_similarity'], expected_max_similarity, places=6)
        self.assertEqual(result['most_similar_concept_id'], expected_concept_id)
        # Diğer alanlar varsayılan olmalı çünkü diğer girdiler boş/None
        self.assertEqual(result['similarity_score'], 0.0) # Bellek listesi boştu
        self.assertEqual(result['high_audio_energy'], False)
        self.assertEqual(result['high_visual_edges'], False)
        self.assertEqual(result['is_bright'], False)
        self.assertEqual(result['is_dark'], False)

    # DÜZELTİLEN TEST: Representation normu sıfıra yakınsa, Process inputlar işlenecek.
    # is_valid_representation False olur.
    def test_process_representation_with_zero_norm(self):
        """Learned representation'ın normu sıfıra yakınsa test eder."""
        # Learned representation sıfıra yakınsa, process metodundaki kontrol onu is_valid_representation = False yapacak.
        learned_representation_zero_norm = np.array([1e-10, -1e-10], dtype=np.float32)
        # Diğer girdiler geçerli OLSUN ki Process bayrakları hesaplanabilsin ve beklenti doğru kurulsun.
        # Bellek ve kavramlar boş olabilir, çünkü Representation geçersiz olduğu için onlar da hesaplanmayacak.
        empty_memory = []
        empty_concepts = []
        valid_processed_inputs = {
            'audio': np.array([self.module.audio_energy_threshold + 1.0, 0.5], dtype=np.float32), # Yüksek ses enerjisi -> True
            'visual': {
                'edges': np.array([[self.module.visual_edges_threshold + 1.0]], dtype=np.float32), # Yüksek görsel kenar -> True
                'grayscale': np.array([[self.module.brightness_threshold_high + 1.0]], dtype=np.float32) # Parlak ortam -> True
            }
        }

        # Beklenen çıktı: Representation geçersiz olduğu için benzerlikler default (0.0/None).
        # Process input geçerli olduğu ve içindeki değerler eşiklerin üzerinde olduğu için Process bayrakları True.
        expected_output = {
            'similarity_score': 0.0,
            'high_audio_energy': True,
            'high_visual_edges': True,
            'is_bright': True,
            'is_dark': False, # Parlak olduğu için is_dark False
            'max_concept_similarity': 0.0,
            'most_similar_concept_id': None,
        }


        result = self.module.process(valid_processed_inputs, learned_representation_zero_norm, empty_memory, empty_concepts)

        # Representation normu sıfıra yakın olduğu için is_valid_representation False olur.
        # Benzerlik hesaplama if'leri atlanır.
        # Process input geçerli olduğu için process bayrakları hesaplanacak.
        # Sonuç, Process bayrakları hesaplanmış ve benzerlikler default kalmış olmalı.
        self.assertEqual(result, expected_output)


    # DÜZELTİLEN TEST: Process inputs geçersiz olduğunda, ilgili check_* fonksiyonları False döner
    # veya Process içindeki np.mean/indexleme exception fırlatırsa try...except çalışır.
    # Her durumda varsayılan sözlük beklenmelidir.
    def test_process_process_inputs_missing_keys_or_invalid_data(self):
        """Processed inputs dictionary'sinde eksik anahtarlar veya geçersiz veri olduğunda test eder."""
        learned_representation = np.array([1.0, 2.0], dtype=np.float32) # Representation geçerli olsun ki sadece Process input test edilsin.

        # Test senaryoları: Farklı geçersiz/eksik processed_inputs durumları.
        # Bunlardan herhangi biri Process input işleme bloğunda bir False koşula veya exception'a yol açmalı.
        test_cases = [
            {}, # Boş dict - isinstance(visual_features, dict) geçebilir, ama get('audio') None, get('visual') None. check_numpy_input False döner. Process bayrakları False kalır.
            {'audio': "not an array"}, # Ses geçersiz tip - check_numpy_input False döner.
            {'visual': {'edges': "not an array"}}, # Kenar geçersiz tip - check_numpy_input False döner.
            {'visual': {'grayscale': "not an array"}}, # Parlaklık geçersiz tip - check_numpy_input False döner.
            {'visual': {'edges': np.array([[]], dtype=np.float32)}}, # Kenar boş array - check_numpy_input True, size>0 False.
            {'visual': {'grayscale': np.array([[]], dtype=np.float32)}}, # Parlaklık boş array - check_numpy_input True, size>0 False.
            {'visual': 'not a dict'}, # Visual girdisi dict değil - isinstance(visual_features, dict) False döner.
            # Diğer Process bayraklarını False yapacak, ama exception fırlatmayacak senaryolar da eklenebilir
            # {'audio': np.array([self.module.audio_energy_threshold - 1.0], dtype=np.float32)}, # Sadece audio eşik altı
            # {'visual': {'edges': np.array([[self.module.visual_edges_threshold - 1.0]], dtype=np.float32)}}, # Sadece edges eşik altı
        ]

        # Bu testin beklentisi, Process input düzgün değilse (boş, eksik veya hatalı veri içeriyorsa)
        # Process input işleme bloğunun beklenen bayrakları ayarlayamaması ve sonuç olarak
        # Process bayraklarının default (False) kalmasıdır.
        # Eğer bu sırada bir exception olursa, except bloğu çalışır ve default sözlük döner.
        # Yani her durumda default sözlük beklenebilir.
        expected_output = DEFAULT_UNDERSTANDING_SIGNALS

        # Benzerlik hesaplamalarını etkilememesi için bellek ve kavramlar boş olsun
        empty_memory = []
        empty_concepts = []

        for inputs in test_cases:
            with self.subTest(msg=f"Testing with processed_inputs: {inputs}"):
                 result = self.module.process(inputs, learned_representation, empty_memory, empty_concepts)
                 # Process input geçerli olmadığında veya içindeki numpy işlemleri hata verdiğinde (veya sadece boş olduğunda),
                 # ya try...except çalışacak ya da ilgili if koşulları False dönecek.
                 # Her durumda Process bayrakları default (False) kalacak. Benzerlikler de default (0.0) kalacak.
                 self.assertEqual(result, expected_output)

    # DÜZELTİLEN TEST: Ses enerjisi eşikte olduğunda ilgili bayrak True olmalı.
    def test_process_audio_energy_at_threshold(self):
        """Ses enerjisinin tam eşikte olduğu durumu test eder."""
        learned_representation = np.array([1.0, 2.0], dtype=np.float32) # Repr geçerli olsun
        processed_inputs = {'audio': np.array([self.module.audio_energy_threshold, 0.5], dtype=np.float32)}
        # Diğer girdiler boş
        empty_memory = []
        empty_concepts = []
        result = self.module.process(processed_inputs, learned_representation, empty_memory, empty_concepts)
        # Benzerlikler 0.0/None dönmeli (bellek/kavram boş). Audio enerjisi eşitte olduğu için True dönmeli. Diğer process bayrakları False.
        self.assertEqual(result['high_audio_energy'], True)
        self.assertEqual(result['similarity_score'], 0.0)
        self.assertEqual(result['max_concept_similarity'], 0.0)
        self.assertEqual(result['high_visual_edges'], False)
        self.assertEqual(result['is_bright'], False)
        self.assertEqual(result['is_dark'], False)


    # DÜZELTİLEN TEST: Görsel kenar eşikte olduğunda ilgili bayrak True olmalı.
    def test_process_visual_edges_at_threshold(self):
        """Görsel kenar yoğunluğunun tam eşikte olduğu durumu test eder."""
        learned_representation = np.array([1.0, 2.0], dtype=np.float32) # Repr geçerli olsun
        processed_inputs = {'visual': {'edges': np.array([[self.module.visual_edges_threshold]], dtype=np.float32)}}
         # Diğer girdiler boş
        empty_memory = []
        empty_concepts = []
        result = self.module.process(processed_inputs, learned_representation, empty_memory, empty_concepts)
        # Benzerlikler 0.0/None dönmeli (bellek/kavram boş). Görsel kenar eşikte olduğu için True dönmeli. Diğer process bayrakları False.
        self.assertEqual(result['high_visual_edges'], True)
        self.assertEqual(result['similarity_score'], 0.0)
        self.assertEqual(result['max_concept_similarity'], 0.0)
        self.assertEqual(result['high_audio_energy'], False)
        self.assertEqual(result['is_bright'], False)
        self.assertEqual(result['is_dark'], False)


    # DÜZELTİLEN TEST: Parlaklık/Karanlık eşiklerinde doğru bayrakların True/False dönmesi gerekir.
    def test_process_brightness_at_thresholds(self):
        """Parlaklık/Karanlık eşiklerinin tam üzerinde/altında olduğu durumları test eder."""
        learned_representation = np.array([1.0, 2.0], dtype=np.float32) # Repr geçerli olsun
        empty_memory = []
        empty_concepts = []

        # Tam high eşiğinde
        processed_inputs_high = {'visual': {'grayscale': np.array([[self.module.brightness_threshold_high]], dtype=np.float32)}}
        result_high = self.module.process(processed_inputs_high, learned_representation, empty_memory, empty_concepts)
        self.assertEqual(result_high['is_bright'], True)
        self.assertEqual(result_high['is_dark'], False)
        self.assertEqual(result_high['similarity_score'], 0.0)
        self.assertEqual(result_high['max_concept_similarity'], 0.0)
        self.assertEqual(result_high['high_audio_energy'], False)
        self.assertEqual(result_high['high_visual_edges'], False)


        # Tam low eşiğinde
        processed_inputs_low = {'visual': {'grayscale': np.array([[self.module.brightness_threshold_low]], dtype=np.float32)}}
        result_low = self.module.process(processed_inputs_low, learned_representation, empty_memory, empty_concepts)
        self.assertEqual(result_low['is_bright'], False)
        self.assertEqual(result_low['is_dark'], True)
        self.assertEqual(result_low['similarity_score'], 0.0)
        self.assertEqual(result_low['max_concept_similarity'], 0.0)
        self.assertEqual(result_low['high_audio_energy'], False)
        self.assertEqual(result_low['high_visual_edges'], False)


        # Eşikler arasında
        processed_inputs_between = {'visual': {'grayscale': np.array([[(self.module.brightness_threshold_low + self.module.brightness_threshold_high) / 2]], dtype=np.float32)}}
        result_between = self.module.process(processed_inputs_between, learned_representation, empty_memory, empty_concepts)
        self.assertEqual(result_between['is_bright'], False)
        self.assertEqual(result_between['is_dark'], False)
        self.assertEqual(result_between['similarity_score'], 0.0)
        self.assertEqual(result_between['max_concept_similarity'], 0.0)
        self.assertEqual(result_between['high_audio_energy'], False)
        self.assertEqual(result_between['high_visual_edges'], False)


    # Bu test pytest çıktısında geçiyordu.
    def test_cleanup(self):
        """cleanup metodunun sorunsuz çalıştığını test eder (şimdilik içi boş)."""
        try:
            self.module.cleanup()
        except Exception as e:
            self.fail(f"cleanup metodu bir hata fırlattı: {e}")
        # Başka bir assert'e gerek yok, hata fırlatmaması başarıdır.


# Testleri çalıştırmak için boilerplate kod
if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]], exit=False)