# tests/unit/cognition/test_understanding.py
import unittest
import sys
import os
import numpy as np
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

try:
    from src.cognition.understanding import UnderstandingModule
    from src.core.utils import check_input_not_none, check_input_type, check_numpy_input
    from src.core.config_utils import get_config_value
except ImportError as e:
     raise ImportError(f"Temel modüller import edilemedi. PYTHONPATH doğru ayarlanmış mı? Hata: {e}")

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
        # Config yapısı main_config.yaml'ı yansıtmalı
        self.default_config_dict = { # __init__ tüm config'i alır
            'cognition': { # get_config_value 'cognition' altından okur
                'audio_energy_threshold': 1000.0,
                'visual_edges_threshold': 50.0,
                'brightness_threshold_high': 200.0,
                'brightness_threshold_low': 50.0,
            }
        }
        self.module = UnderstandingModule(self.default_config_dict)
        # Testler için varsayılan eşik değerlerini de saklayabiliriz
        self.audio_thresh = self.default_config_dict['cognition']['audio_energy_threshold']
        self.edge_thresh = self.default_config_dict['cognition']['visual_edges_threshold']
        self.bright_high_thresh = self.default_config_dict['cognition']['brightness_threshold_high']
        self.bright_low_thresh = self.default_config_dict['cognition']['brightness_threshold_low']


    def tearDown(self):
        self.module.cleanup() # cleanup'ı çağırmak iyi bir pratik

    # --- __init__ Testleri ---
    def test_init_with_valid_config(self):
        self.assertEqual(self.module.config, self.default_config_dict)
        self.assertEqual(self.module.audio_energy_threshold, self.audio_thresh)
        self.assertIsInstance(self.module.audio_energy_threshold, float)
        # ... diğer eşikler için benzer assert'ler ...

    def test_init_with_missing_config_values(self):
        incomplete_config_dict = {
            'cognition': { # 'cognition' anahtarı olmalı
                'audio_energy_threshold': 500, # int olarak verelim
                # visual_edges_threshold eksik, brightness_threshold_low eksik
                'brightness_threshold_high': 150.0,
            }
        }
        module = UnderstandingModule(incomplete_config_dict)
        self.assertEqual(module.audio_energy_threshold, 500.0)
        self.assertIsInstance(module.audio_energy_threshold, float)
        self.assertEqual(module.visual_edges_threshold, 50.0) # Varsayılan
        self.assertIsInstance(module.visual_edges_threshold, float)
        self.assertEqual(module.brightness_threshold_high, 150.0)
        self.assertIsInstance(module.brightness_threshold_high, float)
        self.assertEqual(module.brightness_threshold_low, 50.0) # Varsayılan
        self.assertIsInstance(module.brightness_threshold_low, float)

    def test_init_with_invalid_config_types(self):
        invalid_type_config_dict = {
            'cognition': {
                'audio_energy_threshold': "not a float",
                'visual_edges_threshold': 60,
                'brightness_threshold_high': [250],
                'brightness_threshold_low': 30.0,
            }
        }
        module = UnderstandingModule(invalid_type_config_dict)
        self.assertEqual(module.audio_energy_threshold, 1000.0)
        self.assertIsInstance(module.audio_energy_threshold, float)
        self.assertEqual(module.visual_edges_threshold, 60.0)
        self.assertIsInstance(module.visual_edges_threshold, float)
        self.assertEqual(module.brightness_threshold_high, 200.0)
        self.assertIsInstance(module.brightness_threshold_high, float)
        self.assertEqual(module.brightness_threshold_low, 30.0)
        self.assertIsInstance(module.brightness_threshold_low, float)

    # --- process Testleri ---

    def _calculate_expected_similarities(self, learned_representation, relevant_memory_entries, current_concepts):
        """Testler için beklenen benzerlik skorlarını hesaplayan yardımcı fonksiyon."""
        expected_mem_sim = 0.0
        expected_con_sim = 0.0
        expected_con_id = None
        norm_repr = np.linalg.norm(learned_representation)

        if norm_repr > 1e-8:
            if relevant_memory_entries:
                valid_mem_sims = []
                for entry in relevant_memory_entries:
                    if isinstance(entry, dict) and 'representation' in entry:
                        mem_rep = entry['representation']
                        if isinstance(mem_rep, np.ndarray) and mem_rep.ndim == 1 and np.issubdtype(mem_rep.dtype, np.number) and mem_rep.shape == learned_representation.shape:
                            norm_mem_rep = np.linalg.norm(mem_rep)
                            if norm_mem_rep > 1e-8:
                                sim = np.dot(learned_representation, mem_rep) / (norm_repr * norm_mem_rep)
                                if not np.isnan(sim):
                                    valid_mem_sims.append(sim)
                if valid_mem_sims:
                    expected_mem_sim = max(valid_mem_sims) if valid_mem_sims else 0.0


            if current_concepts:
                max_s = -1.0 # Kosinüs benzerliği -1 ile 1 arasında
                best_id = None
                for i, con_rep in enumerate(current_concepts):
                    if isinstance(con_rep, np.ndarray) and con_rep.ndim == 1 and np.issubdtype(con_rep.dtype, np.number) and con_rep.shape == learned_representation.shape:
                        norm_con_rep = np.linalg.norm(con_rep)
                        if norm_con_rep > 1e-8:
                            sim = np.dot(learned_representation, con_rep) / (norm_repr * norm_con_rep)
                            if not np.isnan(sim) and sim > max_s:
                                max_s = sim
                                best_id = i
                if best_id is not None:
                    expected_con_sim = max_s
                    expected_con_id = best_id
                else: # Eğer tüm kavramlar geçersizse veya benzerlikler NaN ise
                    expected_con_sim = 0.0


        return expected_mem_sim, expected_con_sim, expected_con_id

    def test_process_all_valid_inputs_above_thresholds(self):
        learned_representation = np.array([1.0, 2.0], dtype=np.float32)
        relevant_memory_entries = [{'representation': np.array([1.1, 2.2], dtype=np.float32)}]
        processed_inputs = {
            'audio': np.array([self.audio_thresh + 1.0, 0.5], dtype=np.float32),
            'visual': {
                'edges': np.array([[self.edge_thresh + 1.0]], dtype=np.float32),
                'grayscale': np.array([[self.bright_high_thresh + 1.0]], dtype=np.float32)
            }
        }
        current_concepts = [np.array([1.2, 2.4], dtype=np.float32), np.array([-1.0, 3.0], dtype=np.float32)]

        exp_mem_s, exp_con_s, exp_con_id = self._calculate_expected_similarities(learned_representation, relevant_memory_entries, current_concepts)

        expected_output = {
            'similarity_score': exp_mem_s,
            'high_audio_energy': True,
            'high_visual_edges': True,
            'is_bright': True,
            'is_dark': False,
            'max_concept_similarity': exp_con_s,
            'most_similar_concept_id': exp_con_id,
        }
        result = self.module.process(processed_inputs, learned_representation, relevant_memory_entries, current_concepts)
        for key in expected_output:
            if isinstance(expected_output[key], float):
                self.assertAlmostEqual(result[key], expected_output[key], places=6, msg=f"Key: {key}")
            else:
                self.assertEqual(result[key], expected_output[key], msg=f"Key: {key}")

    def test_process_all_valid_inputs_below_thresholds(self):
        learned_representation = np.array([1.0, 2.0], dtype=np.float32)
        relevant_memory_entries = [{'representation': np.array([1.1, 2.2], dtype=np.float32)}]
        processed_inputs = {
            'audio': np.array([self.audio_thresh - 1.0, 0.1], dtype=np.float32),
            'visual': {
                'edges': np.array([[self.edge_thresh - 1.0]], dtype=np.float32),
                'grayscale': np.array([[self.bright_low_thresh - 1.0]], dtype=np.float32)
            }
        }
        current_concepts = [np.array([1.2, 2.4], dtype=np.float32), np.array([-1.0, 3.0], dtype=np.float32)]
        exp_mem_s, exp_con_s, exp_con_id = self._calculate_expected_similarities(learned_representation, relevant_memory_entries, current_concepts)

        expected_output = {
            'similarity_score': exp_mem_s,
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': True,
            'max_concept_similarity': exp_con_s,
            'most_similar_concept_id': exp_con_id,
        }
        result = self.module.process(processed_inputs, learned_representation, relevant_memory_entries, current_concepts)
        for key in expected_output:
            if isinstance(expected_output[key], float):
                self.assertAlmostEqual(result[key], expected_output[key], places=6, msg=f"Key: {key}")
            else:
                self.assertEqual(result[key], expected_output[key], msg=f"Key: {key}")

    def test_process_all_inputs_none(self):
        result = self.module.process(None, None, None, None)
        self.assertEqual(result, DEFAULT_UNDERSTANDING_SIGNALS)

    def test_process_empty_lists_or_dicts_but_valid_repr(self):
        """Representation geçerli, diğerleri boş/None."""
        learned_representation = np.array([1.0, 2.0], dtype=np.float32)
        # Process inputs boş, memory boş, concepts boş.
        # Benzerlikler 0.0/None kalmalı. Process bayrakları False kalmalı.
        result = self.module.process({}, learned_representation, [], [])
        self.assertEqual(result, DEFAULT_UNDERSTANDING_SIGNALS)


    def test_process_invalid_representation_dtype_or_shape(self):
        """Geçersiz 'learned_representation' tipi veya şekli (dtype int veya shape 2D)."""
        # np.number int'i kapsar, bu yüzden int dtype burada `is_valid_representation`'ı False yapmaz.
        # Sadece shape 2D olanı test edelim.
        invalid_representations = [
             np.array([[1.0, 2.0]], dtype=np.float32), # 2D shape
        ]
        valid_processed_inputs = {
            'audio': np.array([self.audio_thresh + 1.0], dtype=np.float32),
            'visual': {
                'edges': np.array([[self.edge_thresh + 1.0]], dtype=np.float32),
                'grayscale': np.array([[self.bright_high_thresh + 1.0]], dtype=np.float32)
            }
        }
        empty_memory = []
        empty_concepts = []
        expected_output_for_invalid_repr = {
            'similarity_score': 0.0, # Benzerlikler hesaplanmaz
            'high_audio_energy': True, # Process bayrakları hesaplanır
            'high_visual_edges': True,
            'is_bright': True,
            'is_dark': False,
            'max_concept_similarity': 0.0, # Benzerlikler hesaplanmaz
            'most_similar_concept_id': None,
        }
        for invalid_repr in invalid_representations:
            with self.subTest(msg=f"Testing with invalid representation shape: {invalid_repr.shape}"):
                result = self.module.process(valid_processed_inputs, invalid_repr, empty_memory, empty_concepts)
                self.assertEqual(result, expected_output_for_invalid_repr)

    def test_process_representation_non_numpy(self):
        """Learned representation numpy array değilse."""
        invalid_repr_non_numpy = "not a numpy array"
        valid_processed_inputs = {
            'audio': np.array([self.audio_thresh + 1.0], dtype=np.float32),
        } # Diğerleri testin odak noktası değil
        empty_memory = []
        empty_concepts = []
        expected_output_for_invalid_repr = {**DEFAULT_UNDERSTANDING_SIGNALS, 'high_audio_energy': True}

        result = self.module.process(valid_processed_inputs, invalid_repr_non_numpy, empty_memory, empty_concepts)
        self.assertEqual(result, expected_output_for_invalid_repr)


    def test_process_memory_with_invalid_entries(self):
        learned_representation = np.array([1.0, 2.0], dtype=np.float32)
        # Sadece ID 0 ve ID 4'teki girdiler geçerli ve benzerlik hesaplamasına dahil olacak
        relevant_memory_entries = [
            {'representation': np.array([1.1, 2.2], dtype=np.float32)}, # Geçerli
            {'representation': "not a numpy array"},
            {'representation': np.array([[3.0, 4.0]], dtype=np.float32)},
            {'other_key': 'no representation'},
            {'representation': np.array([0.0, 0.0], dtype=np.float32)}, # Sıfır norm
            {'representation': np.array([5.0, 6.0], dtype=np.float32)}, # Geçerli
            {'representation': np.array([1.0, 2.0, 3.0], dtype=np.float32)}, # Yanlış boyut
        ]
        exp_mem_s, _, _ = self._calculate_expected_similarities(learned_representation, relevant_memory_entries, [])

        expected_output = {**DEFAULT_UNDERSTANDING_SIGNALS, 'similarity_score': exp_mem_s}
        result = self.module.process({}, learned_representation, relevant_memory_entries, [])
        self.assertAlmostEqual(result['similarity_score'], exp_mem_s, places=6)
        for k,v in expected_output.items():
            if k != 'similarity_score':
                self.assertEqual(result[k], v, msg=f"Key: {k}")


    def test_process_concepts_with_invalid_entries(self):
        learned_representation = np.array([1.0, 2.0], dtype=np.float32)
        # Sadece ID 0 ve ID 4'teki kavramlar geçerli
        current_concepts = [
            np.array([1.2, 2.4], dtype=np.float32), # Geçerli
            "not a numpy array",
            np.array([[3.0, 4.0]], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32), # Sıfır norm
            np.array([5.0, 6.0], dtype=np.float32), # Geçerli
            np.array([1.0, 2.0, 3.0], dtype=np.float32), # Yanlış boyut
        ]
        _, exp_con_s, exp_con_id = self._calculate_expected_similarities(learned_representation, [], current_concepts)

        expected_output = {
            **DEFAULT_UNDERSTANDING_SIGNALS,
            'max_concept_similarity': exp_con_s,
            'most_similar_concept_id': exp_con_id
        }
        result = self.module.process({}, learned_representation, [], current_concepts)
        self.assertAlmostEqual(result['max_concept_similarity'], exp_con_s, places=6)
        self.assertEqual(result['most_similar_concept_id'], exp_con_id)
        for k,v in expected_output.items():
            if k not in ['max_concept_similarity', 'most_similar_concept_id']:
                self.assertEqual(result[k], v, msg=f"Key: {k}")


    def test_process_representation_with_zero_norm(self):
        learned_representation_zero_norm = np.array([1e-10, -1e-10], dtype=np.float32)
        valid_processed_inputs = {
            'audio': np.array([self.audio_thresh + 1.0], dtype=np.float32),
        }
        expected_output = {**DEFAULT_UNDERSTANDING_SIGNALS, 'high_audio_energy': True}
        result = self.module.process(valid_processed_inputs, learned_representation_zero_norm, [], [])
        self.assertEqual(result, expected_output)


    def test_process_process_inputs_missing_keys_or_invalid_data(self):
        learned_representation = np.array([1.0, 2.0], dtype=np.float32)
        # Process input'un çeşitli geçersiz durumları
        test_cases_processed_inputs = [
            {}, # Boş dict
            {'audio': "not an array"},
            {'visual': {'edges': "not an array"}},
            {'visual': {'grayscale': "not an array"}},
            {'visual': {'edges': np.array([[]], dtype=np.float32)}}, # Boş array
            {'visual': {'grayscale': np.array([[]], dtype=np.float32)}}, # Boş array
            {'visual': 'not a dict'},
        ]
        # Bu durumlarda Process bayrakları False kalmalı.
        # Eğer learned_representation ve bellek/kavramlar geçerliyse, benzerlikler hesaplanmalı.
        # Şimdilik, bellek ve kavramları da boş tutarak DEFAULT_UNDERSTANDING_SIGNALS'ı bekleyelim.
        for inputs in test_cases_processed_inputs:
            with self.subTest(msg=f"Testing with processed_inputs: {inputs}"):
                 result = self.module.process(inputs, learned_representation, [], [])
                 self.assertEqual(result, DEFAULT_UNDERSTANDING_SIGNALS)

    def test_process_audio_energy_at_threshold(self):
        learned_representation = np.array([1.0, 2.0], dtype=np.float32)
        processed_inputs = {'audio': np.array([self.audio_thresh], dtype=np.float32)} # Tek elemanlı, eşikte
        expected_output = {**DEFAULT_UNDERSTANDING_SIGNALS, 'high_audio_energy': True}
        result = self.module.process(processed_inputs, learned_representation, [], [])
        self.assertEqual(result, expected_output)


    def test_process_visual_edges_at_threshold(self):
        learned_representation = np.array([1.0, 2.0], dtype=np.float32)
        processed_inputs = {'visual': {'edges': np.array([[self.edge_thresh]], dtype=np.float32)}}
        expected_output = {**DEFAULT_UNDERSTANDING_SIGNALS, 'high_visual_edges': True}
        result = self.module.process(processed_inputs, learned_representation, [], [])
        self.assertEqual(result, expected_output)


    def test_process_brightness_at_thresholds(self):
        learned_representation = np.array([1.0, 2.0], dtype=np.float32)

        # Tam high eşiğinde
        processed_inputs_high = {'visual': {'grayscale': np.array([[self.bright_high_thresh]], dtype=np.float32)}}
        expected_high = {**DEFAULT_UNDERSTANDING_SIGNALS, 'is_bright': True}
        result_high = self.module.process(processed_inputs_high, learned_representation, [], [])
        self.assertEqual(result_high, expected_high)

        # Tam low eşiğinde
        processed_inputs_low = {'visual': {'grayscale': np.array([[self.bright_low_thresh]], dtype=np.float32)}}
        expected_low = {**DEFAULT_UNDERSTANDING_SIGNALS, 'is_dark': True}
        result_low = self.module.process(processed_inputs_low, learned_representation, [], [])
        self.assertEqual(result_low, expected_low)

        # Eşikler arasında (is_bright: False, is_dark: False olmalı, yani default)
        processed_inputs_between = {'visual': {'grayscale': np.array([[(self.bright_low_thresh + self.bright_high_thresh) / 2]], dtype=np.float32)}}
        result_between = self.module.process(processed_inputs_between, learned_representation, [], [])
        self.assertEqual(result_between, DEFAULT_UNDERSTANDING_SIGNALS)


    def test_cleanup(self):
        try:
            self.module.cleanup()
        except Exception as e:
            self.fail(f"cleanup metodu bir hata fırlattı: {e}")

if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]], exit=False)