# tests/unit/cognition/test_decision.py
import unittest
import sys
import os
import numpy as np
import logging
import random

from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

try:
    from src.cognition.decision import DecisionModule
    # check_* ve get_config_value import'ları artık try block'una dahil değil
    # Çünkü production kodunda da bu modüllerin import edilmesi beklenir.
    # Eğer import hatası hala olursa, test ortamı kurulumu sorunludur.
    from src.core.utils import check_input_not_none, check_input_type # Sadece import edildiğini varsayalım, mocklamayalım
    from src.core.config_utils import get_config_value # Sadece import edildiğini varsayalım, mocklamayalım

except ImportError as e:
     print(f"Temel modüller import edilemedi. PYTHONPATH doğru ayarlanmış mı? Hata: {e}")
     raise e


# Testler sırasında logger çıktılarını görmek isterseniz bu satırları etkinleştirebilirsiniz.
# import logging
# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger('src.cognition.decision').setLevel(logging.DEBUG)
# logging.getLogger('src.core.utils').setLevel(logging.DEBUG)
# logging.getLogger('src.core.config_utils').setLevel(logging.DEBUG)


class TestDecisionModule(unittest.TestCase):

    def setUp(self):
        """Her test metodundan önce çalışır, varsayılan bir konfigürasyon ve DecisionModule örneği oluşturur."""
        self.default_config = {
            'familiarity_threshold': 0.8,
            'audio_energy_threshold': 1000.0,
            'visual_edges_threshold': 50.0,
            'brightness_threshold_high': 200.0,
            'brightness_threshold_low': 50.0,
            'concept_recognition_threshold': 0.85,
            'curiosity_threshold': 5.0,
            'curiosity_increment_new': 1.0,
            'curiosity_decrement_familiar': 0.5,
            'curiosity_decay': 0.1,
        }
        # DecisionModule init çağrısı config_utils'taki workaround'a uygun kalıyor.
        self.module = DecisionModule(self.default_config)

        # Her test başında merak seviyesini varsayılan (0.0) olarak ayarlayalım (setUp'ta zaten yapılıyor ama emin olalım)
        self.module.curiosity_level = 0.0


    def tearDown(self):
        """Her test metodundan sonra çalışır."""
        self.module.cleanup()


    # --- __init__ Testleri (Eşik kontrol hatalarını test etmeli) ---

    def test_init_with_valid_config(self):
        """Geçerli bir konfigürasyon ile başlatmayı test eder."""
        self.assertEqual(self.module.config, self.default_config)
        self.assertEqual(self.module.familiarity_threshold, 0.8)
        self.assertEqual(self.module.audio_energy_threshold, 1000.0)
        self.assertEqual(self.module.visual_edges_threshold, 50.0)
        self.assertEqual(self.module.brightness_threshold_high, 200.0)
        self.assertEqual(self.module.brightness_threshold_low, 50.0)
        self.assertEqual(self.module.concept_recognition_threshold, 0.85)
        self.assertEqual(self.module.curiosity_threshold, 5.0)
        self.assertEqual(self.module.curiosity_increment_new, 1.0)
        self.assertEqual(self.module.curiosity_decrement_familiar, 0.5)
        self.assertEqual(self.module.curiosity_decay, 0.1)
        self.assertEqual(self.module.curiosity_level, 0.0)

    def test_init_with_missing_config_values(self):
        """Bazı konfigürasyon değerleri eksikken başlatmayı test eder (varsayılanlar kullanılmalı)."""
        # get_config_value artık None dönmemeli (workaround sayesinde), varsayılanlar doğru dönmeli.
        incomplete_config = {
            'familiarity_threshold': 0.9,
            # Diğerleri eksik, get_config_value default değerlerini döndürmeli
        }
        # init metodu config_utils'tan değerleri alacak ve aralık kontrolü yapacak.
        # Bu durumda eksik değerler get_config_value'dan varsayılan dönecek ve aralık kontrolünden geçecek.
        module = DecisionModule(incomplete_config)
        self.assertEqual(module.familiarity_threshold, 0.9)
        self.assertEqual(module.audio_energy_threshold, 1000.0) # Varsayılan
        self.assertEqual(module.visual_edges_threshold, 50.0) # Varsayılan
        self.assertEqual(module.brightness_threshold_high, 200.0) # Varsayılan
        self.assertEqual(module.brightness_threshold_low, 50.0) # Varsayılan
        self.assertEqual(module.concept_recognition_threshold, 0.85) # Varsayılan
        self.assertEqual(module.curiosity_threshold, 5.0) # Varsayılan
        self.assertEqual(module.curiosity_increment_new, 1.0) # Varsayılan
        self.assertEqual(module.curiosity_decrement_familiar, 0.5) # Varsayılan
        self.assertEqual(module.curiosity_decay, 0.1) # Varsayılan


    def test_init_with_invalid_config_types(self):
        """Geçersiz tipte konfigürasyon değerleri ile başlatmayı test eder (varsayılanlar kullanılmalı)."""
        # config_utils'taki get_config_value'nun expected_type kontrolü ve default döndürmesi test ediliyor.
        # Bu durumda DecisionModule'ün init'i get_config_value'dan varsayılanları alacak ve aralık kontrolünden geçecek.
        invalid_type_config = {
            'familiarity_threshold': "0.5", # Invalid type -> get_config_value default 0.8 döndürmeli
            'audio_energy_threshold': [1000], # Invalid type -> get_config_value default 1000.0 döndürmeli
            'curiosity_threshold': "5.0" # Invalid type -> get_config_value default 5.0 döndürmeli
        }
        module = DecisionModule(invalid_type_config)
        self.assertEqual(module.familiarity_threshold, 0.8) # Default used by get_config_value
        self.assertEqual(module.audio_energy_threshold, 1000.0) # Default used by get_config_value
        self.assertEqual(module.curiosity_threshold, 5.0) # Default used by get_config_value
        # Diğer varsayılanları da kontrol edilebilir.


    def test_init_thresholds_out_of_range(self):
        """Bazı eşiklerin geçerli aralık dışında ayarlanması durumunu test eder."""
        # Bu test DecisionModule'ün kendi içindeki aralık kontrol mantığını test eder.
        out_of_range_config = {
            'familiarity_threshold': 1.1, # Should be reset to 0.8 by DecisionModule's own check
            'concept_recognition_threshold': -0.1, # Should be reset to 0.85 by DecisionModule's own check
            'audio_energy_threshold': -10.0, # Should be reset to 1000.0 by DecisionModule's own check
            'brightness_threshold_low': 100.0, # lower > higher -> should be reset to 50.0 (low)
            'brightness_threshold_high': 80.0, # lower > higher -> should be reset to 200.0 (high)
            'curiosity_increment_new': -5.0 # Should be reset to 1.0 by DecisionModule's own check
        }
        module = DecisionModule(out_of_range_config)
        # DecisionModule'ün kendi aralık kontrolü default değerleri zorla atamalı.
        self.assertEqual(module.familiarity_threshold, 0.8)
        self.assertEqual(module.concept_recognition_threshold, 0.85)
        self.assertEqual(module.audio_energy_threshold, 1000.0)
        self.assertEqual(module.brightness_threshold_low, 50.0) # Reset edildi
        self.assertEqual(module.brightness_threshold_high, 200.0) # Reset edildi
        self.assertEqual(module.curiosity_increment_new, 1.0)
        # Merak eşiği ve decay kontrolü de eklenebilir.


    # --- decide Input Validation Tests ---

    def test_decide_input_none(self):
        """understanding_signals None ise decide metodunu test eder."""
        initial_curiosity = self.module.curiosity_level # 0.0
        result = self.module.decide(None, [])
        self.assertIsNone(result)
        # Input None ise merak seviyesi güncellenmemeli (finally bloğu decision is not None koşulu sayesinde)
        self.assertEqual(self.module.curiosity_level, initial_curiosity)


    def test_decide_input_not_dict(self):
        """understanding_signals dictionary değilse decide metodunu test eder."""
        initial_curiosity = self.module.curiosity_level # 0.0
        result = self.module.decide("not a dict", [])
        self.assertIsNone(result)
        # Input dict değilse merak seviyesi güncellenmemeli
        self.assertEqual(self.module.curiosity_level, initial_curiosity)


    def test_decide_input_empty_dict(self):
        """understanding_signals boş dictionary ise decide metodunu test eder (varsayılan değerler kullanılmalı)."""
        # Boş dict'teki tüm flag'ler False, skorlar 0.0 olur. Hiçbir öncelikli eşik aşılmaz.
        # Temel durum 'new' olur. Nihai karar "new_input_detected" veya "new_input_detected_fallback" olmalı.
        # DecisionModule artık "new_input_detected" döndürüyor None olursa.
        # Merak seviyesi: Başlangıç 0.0 -> new_input_detected kararı ile +increment -> decay ile -decay
        initial_curiosity = 0.0
        self.module.curiosity_level = initial_curiosity # Emin olalım
        # Karar "new_input_detected" olacak.
        expected_curiosity = initial_curiosity + self.module.curiosity_increment_new - self.module.curiosity_decay
        expected_curiosity = max(0.0, expected_curiosity)

        result = self.module.decide({}, []) # Boş dict gönder
        # Fallback karar "new_input_detected" olmalı
        self.assertEqual(result, "new_input_detected") # Fallback artık new_input_detected olarak güncellendi
        # Merak seviyesi güncellenmiş olmalı (inc + decay)
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    # --- decide Karar Önceliği Testleri ---
    # Merak seviyelerini test başında 0.0'dan başlatacağız ve expected curiosity hesaplarken
    # karara göre inc/dec ve decay'i ekleyeceğiz.

    @patch('random.choice', return_value='explore_randomly') # random.choice'u mocklayalım
    def test_decide_priority_curiosity_threshold(self, mock_random_choice):
        """Merak eşiği aşıldığında decide metodunu test eder."""
        initial_curiosity = self.module.curiosity_threshold + 1.0 # Eşiğin üstüne çıkar
        self.module.curiosity_level = initial_curiosity

        # Diğer sinyallerin hiçbiri yüksek öncelikli bir kararı tetiklememeli
        signals = {
            'similarity_score': 0.1, # Bellek tanıdıklık eşiği altında
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.1, # Kavram tanıma eşiği altında
            'most_similar_concept_id': None,
        }

        # Merak seviyesi: Başlangıç > eşik -> Karar "explore_randomly" (mocklandı) -> Bu karar merakı inc/dec etmez -> decay ile -decay
        expected_curiosity = initial_curiosity - self.module.curiosity_decay
        expected_curiosity = max(0.0, expected_curiosity) # Merak negatif olmamalı

        result = self.module.decide(signals, [])

        self.assertEqual(result, 'explore_randomly') # random.choice'un döndürdüğü değer olmalı
        mock_random_choice.assert_called_once_with(["explore_randomly", "make_noise"]) # random.choice doğru seçeneklerle çağrılmalı
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) # Merak sadece decay olmalı


    def test_decide_priority_sound_detected(self):
        """Yüksek ses enerjisi algılandığında decide metodunu test eder."""
        # Merak eşiği altında olsun ki merak kararı tetiklenmesin.
        initial_curiosity = self.module.curiosity_threshold - 1.0 # örn: 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': 0.1, # Bellek tanıdıklık eşiği altında
            'high_audio_energy': True, # Ses algılandı - yüksek öncelik
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.9, # Kavram tanıma eşiği üstünde (ama ses daha öncelikli)
            'most_similar_concept_id': 1,
        }

        # Merak seviyesi: Başlangıç < eşik -> Karar "sound_detected" -> Bu karar merakı inc/dec etmez -> decay ile -decay
        expected_curiosity = initial_curiosity - self.module.curiosity_decay # örn: 4.0 - 0.1 = 3.9
        expected_curiosity = max(0.0, expected_curiosity)

        result = self.module.decide(signals, [])
        self.assertEqual(result, "sound_detected")
        # Merak sadece decay olmalı.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_complex_visual_detected(self):
        """Yüksek görsel kenar yoğunluğu algılandığında decide metodunu test eder."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # örn: 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': 0.1, # Bellek tanıdıklık eşiği altında
            'high_audio_energy': False, # Ses yok
            'high_visual_edges': True, # Görsel kenar algılandı - öncelikli
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.9, # Kavram tanıma eşiği üstünde (ama görsel daha öncelikli)
            'most_similar_concept_id': 1,
        }

        # Merak seviyesi: Başlangıç < eşik -> Karar "complex_visual_detected" -> Bu karar merakı inc/dec etmez -> decay ile -decay
        expected_curiosity = initial_curiosity - self.module.curiosity_decay # örn: 4.0 - 0.1 = 3.9
        expected_curiosity = max(0.0, expected_curiosity)

        result = self.module.decide(signals, [])
        self.assertEqual(result, "complex_visual_detected")
        # Merak sadece decay olmalı.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_bright_light_detected(self):
        """Parlak ortam algılandığında decide metodunu test eder."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # örn: 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': 0.1, # Bellek tanıdıklık eşiği altında
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': True, # Parlak algılandı - öncelikli
            'is_dark': False,
            'max_concept_similarity': 0.9, # Kavram tanıma eşiği üstünde (ama parlaklık daha öncelikli)
            'most_similar_concept_id': 1,
        }

        # Merak seviyesi: Başlangıç < eşik -> Karar "bright_light_detected" -> Bu karar merakı inc/dec etmez -> decay ile -decay
        expected_curiosity = initial_curiosity - self.module.curiosity_decay # örn: 4.0 - 0.1 = 3.9
        expected_curiosity = max(0.0, expected_curiosity)

        result = self.module.decide(signals, [])
        self.assertEqual(result, "bright_light_detected")
        # Merak sadece decay olmalı.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_dark_environment_detected(self):
        """Karanlık ortam algılandığında decide metodunu test eder."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # örn: 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': 0.1, # Bellek tanıdıklık eşiği altında
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': True, # Karanlık algılandı - öncelikli
            'max_concept_similarity': 0.9, # Kavram tanıma eşiği üstünde (ama karanlık daha öncelikli)
            'most_similar_concept_id': 1,
        }

        # Merak seviyesi: Başlangıç < eşik -> Karar "dark_environment_detected" -> Bu karar merakı inc/dec etmez -> decay ile -decay
        expected_curiosity = initial_curiosity - self.module.curiosity_decay # örn: 4.0 - 0.1 = 3.9
        expected_curiosity = max(0.0, expected_curiosity)

        result = self.module.decide(signals, [])
        self.assertEqual(result, "dark_environment_detected")
        # Merak sadece decay olmalı.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_recognized_concept(self):
        """Kavram tanındığında decide metodunu test eder."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # örn: 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': 0.1, # Bellek tanıdıklık eşiği altında
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': self.module.concept_recognition_threshold + 0.01, # Kavram tanıma eşiği üstünde (örn: 0.86)
            'most_similar_concept_id': 42, # Kavram ID'si var - öncelikli (Process sinyallerinden sonra)
        }

        # Merak seviyesi: Başlangıç < eşik -> Karar "recognized_concept_42" -> Bu karar merakı decrement eder -> decay ile -decay
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay # örn: 4.0 - 0.5 - 0.1 = 3.4
        expected_curiosity = max(0.0, expected_curiosity) # Merak negatif olmasın

        result = self.module.decide(signals, [])
        self.assertEqual(result, "recognized_concept_42")
        # Merak azalmalı ve decay olmalı.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_recognized_concept_at_threshold(self):
        """Kavram tanıma benzerlik skoru eşiğe eşit olduğunda decide metodunu test eder."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # örn: 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': 0.1, # Bellek tanıdıklık eşiği altında
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': self.module.concept_recognition_threshold, # Eşiğe eşit (0.85)
            'most_similar_concept_id': 99, # Kavram ID'si var
        }

        # Eşitlik durumunda da kavram tanınmalı (>= kullandığımız için)
        # Merak seviyesi: Başlangıç < eşik -> Karar "recognized_concept_99" -> decrement eder -> decay ile -decay
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay # örn: 4.0 - 0.5 - 0.1 = 3.4
        expected_curiosity = max(0.0, expected_curiosity)

        result = self.module.decide(signals, [])
        self.assertEqual(result, "recognized_concept_99")
        # Merak azalmalı ve decay olmalı.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_recognized_concept_similarity_below_threshold(self):
        """Kavram tanıma benzerlik skoru eşik altında kaldığında decide metodunu test eder."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # örn: 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': 0.1, # Bellek tanıdıklık eşiği altında
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': self.module.concept_recognition_threshold - 0.01, # Eşik altında (örn: 0.84)
            'most_similar_concept_id': 123, # ID olsa da benzerlik düşük
        }

        # Kavram tanıma koşulu sağlanmadı. Karar, bir sonraki öncelikli koşula düşmeli (Bellek Tanıdıklığı veya Yeni).
        # similarity_score 0.1, familiarity_threshold 0.8 -> 0.1 < 0.8 -> Temel durum is_fundamentally_new.
        # Karar "new_input_detected" olmalı.
        # Merak seviyesi: Başlangıç < eşik -> Karar "new_input_detected" -> increment eder -> decay ile -decay
        expected_curiosity = initial_curiosity + self.module.curiosity_increment_new - self.module.curiosity_decay # örn: 4.0 + 1.0 - 0.1 = 4.9
        expected_curiosity = max(0.0, expected_curiosity)

        result = self.module.decide(signals, [])
        # Kavram tanıma tetiklenmedi, default 'new'e düştü
        self.assertEqual(result, "new_input_detected")
        # Merak artmalı ve decay olmalı.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_recognized_concept_id_none(self):
        """Kavram tanıma benzerlik skoru yüksek ama most_similar_concept_id None olduğunda test eder."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # örn: 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': 0.1, # Bellek tanıdıklık eşiği altında
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.9, # Eşiğin üstünde
            'most_similar_concept_id': None, # ID None -> Kavram tanıma koşulu sağlanmaz
        }

        # most_similar_concept_id None olduğu için kavram tanıma koşulu sağlanmadı.
        # Karar, bir sonraki öncelikli koşula düşmeli (Bellek Tanıdıklığı veya Yeni).
        # similarity_score 0.1, familiarity_threshold 0.8 -> 0.1 < 0.8 -> Temel durum is_fundamentally_new.
        # Karar "new_input_detected" olmalı.
        # Merak seviyesi: Başlangıç < eşik -> Karar "new_input_detected" -> increment eder -> decay ile -decay
        expected_curiosity = initial_curiosity + self.module.curiosity_increment_new - self.module.curiosity_decay # örn: 4.0 + 1.0 - 0.1 = 4.9
        expected_curiosity = max(0.0, expected_curiosity)

        result = self.module.decide(signals, [])
        # Kavram tanıma tetiklenmedi, default 'new'e düştü
        self.assertEqual(result, "new_input_detected")
        # Merak artmalı ve decay olmalı.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_familiar_input_detected(self):
        """Bellek benzerlik skoru eşiği aştığında decide metodunu test eder."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # örn: 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': self.module.familiarity_threshold + 0.01, # Tanıdıklık eşiği üstünde (örn: 0.81)
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.1, # Kavram tanıma eşiği altında
            'most_similar_concept_id': None, # ID yok
        }

        # Önceki tüm öncelikler False. similarity_score >= familiarity_threshold True -> "familiar_input_detected"
        # Merak seviyesi: Başlangıç < eşik -> Karar "familiar_input_detected" -> decrement eder -> decay ile -decay
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay # örn: 4.0 - 0.5 - 0.1 = 3.4
        expected_curiosity = max(0.0, expected_curiosity)

        result = self.module.decide(signals, [])
        self.assertEqual(result, "familiar_input_detected")
        # Merak azalmalı ve decay olmalı.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_familiar_input_at_threshold(self):
        """Bellek benzerlik skoru eşiğe eşit olduğunda decide metodunu test eder."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # örn: 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': self.module.familiarity_threshold, # Eşiğe eşit (0.8)
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.1, # Kavram tanıma eşiği altında
            'most_similar_concept_id': None, # ID yok
        }

        # Eşitlik durumunda da tanıdık kabul edilmeli (>= kullandığımız için)
        # Merak seviyesi: Başlangıç < eşik -> Karar "familiar_input_detected" -> decrement eder -> decay ile -decay
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay # örn: 4.0 - 0.5 - 0.1 = 3.4
        expected_curiosity = max(0.0, expected_curiosity)

        result = self.module.decide(signals, [])
        self.assertEqual(result, "familiar_input_detected")
        # Merak azalmalı ve decay olmalı.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_decide_priority_new_input_detected(self):
        """Hiçbir öncelikli veya tanıdık koşul sağlanmadığında decide metodunu test eder."""
        initial_curiosity = self.module.curiosity_threshold - 1.0 # örn: 4.0
        self.module.curiosity_level = initial_curiosity

        signals = {
            'similarity_score': self.module.familiarity_threshold - 0.01, # Tanıdıklık eşiği altında (örn: 0.79)
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.1, # Kavram tanıma eşiği altında
            'most_similar_concept_id': None, # ID yok
        }

        # Önceki tüm öncelikler False. similarity_score < familiarity_threshold True -> Temel durum 'new'.
        # Decision Module None döndürecek -> Fallback "new_input_detected"
        # Merak seviyesi: Başlangıç < eşik -> Karar "new_input_detected" -> increment eder -> decay ile -decay
        expected_curiosity = initial_curiosity + self.module.curiosity_increment_new - self.module.curiosity_decay # örn: 4.0 + 1.0 - 0.1 = 4.9
        expected_curiosity = max(0.0, expected_curiosity)

        result = self.module.decide(signals, [])
        # Hiçbir öncelikli durum algılanamadığı için varsayılan "new_input_detected" olmalı.
        self.assertEqual(result, "new_input_detected")
        # Merak artmalı ve decay olmalı.
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    # --- Merak Güncelleme Testleri ---
    # Bu testler artık yukarıdaki öncelik testleri içinde merak seviyesi kontrolü ile birleştirildi.
    # Ancak ayrı testler olarak da tutmak temizlik açısından iyi olabilir.
    # Mevcut test seti zaten merak güncellemesini öncelik testleri içinde kontrol ediyor.
    # Sadece ayrı testler olarak kalmasını istiyorsanız, duplicate testler gibi görünebilir.
    # Mevcut test setinde Merak güncellemesi testleri zaten var, sadece beklenen merak değerlerini düzeltelim.

    def test_curiosity_update_new_input(self):
        """'new_input_detected' kararı merakı artırmalı ve decay olmalı."""
        initial_curiosity = 1.0 # Başlangıç merak seviyesi
        self.module.curiosity_level = initial_curiosity

        signals = { # new_input_detected kararına yol açacak sinyaller (tüm öncelikler False, sim < threshold)
            'similarity_score': self.module.familiarity_threshold - 0.01, # 0.79
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }

        # Beklenen: initial + increment - decay
        expected_curiosity = initial_curiosity + self.module.curiosity_increment_new - self.module.curiosity_decay # 1.0 + 1.0 - 0.1 = 1.9
        expected_curiosity = max(0.0, expected_curiosity)

        result = self.module.decide(signals, [])
        self.assertEqual(result, "new_input_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_curiosity_update_familiar_input(self):
        """'familiar_input_detected' kararı merakı azaltmalı ve decay olmalı (negatif olmamalı)."""
        initial_curiosity = 1.0 # Başlangıç merak seviyesi
        self.module.curiosity_level = initial_curiosity

        signals = { # familiar_input_detected kararına yol açacak sinyaller (öncelikler False, sim >= threshold)
            'similarity_score': self.module.familiarity_threshold + 0.01, # 0.81
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }

        # Beklenen: initial - decrement - decay
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay # 1.0 - 0.5 - 0.1 = 0.4
        expected_curiosity = max(0.0, expected_curiosity) # Negatif olmamalı

        result = self.module.decide(signals, [])
        self.assertEqual(result, "familiar_input_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_curiosity_update_recognized_concept(self):
        """'recognized_concept_X' kararı merakı azaltmalı ve decay olmalı (negatif olmamalı)."""
        initial_curiosity = 1.0 # Başlangıç merak seviyesi
        self.module.curiosity_level = initial_curiosity

        signals = { # recognized_concept_X kararına yol açacak sinyaller (process false, sim < threshold, concept_sim >= threshold)
            'similarity_score': 0.1, # Bellek eşiği altında olsun ki familiar tetiklenmesin
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': self.module.concept_recognition_threshold + 0.01, 'most_similar_concept_id': 77, # 0.86
        }

        # Beklenen: initial - decrement - decay
        expected_curiosity = initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay # 1.0 - 0.5 - 0.1 = 0.4
        expected_curiosity = max(0.0, expected_curiosity) # Negatif olmamalı

        result = self.module.decide(signals, [])
        self.assertEqual(result, "recognized_concept_77")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6)


    def test_curiosity_update_other_decisions_only_decay(self):
        """Process tabanlı kararlar (ses, görsel vb.) merakı sadece decay etmeli."""
        initial_curiosity = 1.0 # Başlangıç merak seviyesi
        self.module.curiosity_level = initial_curiosity

        signals = { # sound_detected kararına yol açacak sinyaller (yüksek öncelikli olduğu için diğerleri önemsiz)
            'similarity_score': 0.1,
            'high_audio_energy': True, # Bu karar tetiklenir
            'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }

        # Beklenen: initial - decay (inc/dec yok)
        expected_curiosity = initial_curiosity - self.module.curiosity_decay # 1.0 - 0.1 = 0.9
        expected_curiosity = max(0.0, expected_curiosity) # Negatif olmamalı

        result = self.module.decide(signals, [])
        self.assertEqual(result, "sound_detected") # Karar doğru olmalı
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) # Sadece decay olmalı

        # Diğer process tabanlı kararlar için de aynı behavior beklenir.
        self.module.curiosity_level = initial_curiosity # Sıfırla
        signals_visual = { 'high_audio_energy': False, 'high_visual_edges': True, 'is_bright': False, 'is_dark': False, 'similarity_score': 0.1, 'max_concept_similarity': 0.1, 'most_similar_concept_id': None, }
        result_visual = self.module.decide(signals_visual, [])
        self.assertEqual(result_visual, "complex_visual_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) # Sadece decay olmalı (1.0 -> 0.9)

        self.module.curiosity_level = initial_curiosity # Sıfırla
        signals_bright = { 'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': True, 'is_dark': False, 'similarity_score': 0.1, 'max_concept_similarity': 0.1, 'most_similar_concept_id': None, }
        result_bright = self.module.decide(signals_bright, [])
        self.assertEqual(result_bright, "bright_light_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) # Sadece decay olmalı (1.0 -> 0.9)


        self.module.curiosity_level = initial_curiosity # Sıfırla
        signals_dark = { 'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': True, 'similarity_score': 0.1, 'max_concept_similarity': 0.1, 'most_similar_concept_id': None, }
        result_dark = self.module.decide(signals_dark, [])
        self.assertEqual(result_dark, "dark_environment_detected")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) # Sadece decay olmalı (1.0 -> 0.9)


    @patch('random.choice', return_value='explore_randomly') # random.choice'u mocklayalım
    def test_curiosity_update_explore_randomly_only_decay(self, mock_random_choice):
        """'explore_randomly' kararı merakı sadece decay etmeli."""
        initial_curiosity = self.module.curiosity_threshold + 1.0 # Merak eşiği üstünde (örn: 6.0)
        self.module.curiosity_level = initial_curiosity

        signals = { # Merak kararını tetikleyecek (diğerleri düşük öncelikli)
            'similarity_score': 0.1,
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }

        # Beklenen: initial - decay (inc/dec yok)
        expected_curiosity = initial_curiosity - self.module.curiosity_decay # örn: 6.0 - 0.1 = 5.9
        expected_curiosity = max(0.0, expected_curiosity)

        result = self.module.decide(signals, [])
        self.assertEqual(result, "explore_randomly")
        self.assertAlmostEqual(self.module.curiosity_level, expected_curiosity, places=6) # Sadece decay olmalı


    def test_curiosity_does_not_go_below_zero(self):
        """Merak seviyesi sıfırın altına düşmemeli."""
        initial_curiosity = 0.1 # Çok düşük bir başlangıç değeri
        self.module.curiosity_level = initial_curiosity

        signals = { # familiar_input_detected kararına yol açacak sinyaller (decrement tetikler)
            'similarity_score': self.module.familiarity_threshold + 0.01, # 0.81
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }

        # initial_curiosity (0.1) - decrement (0.5) - decay (0.1) = -0.5. Negatif olmamalı, 0.0 olmalı.
        expected_curiosity = max(0.0, initial_curiosity - self.module.curiosity_decrement_familiar - self.module.curiosity_decay)

        result = self.module.decide(signals, [])
        self.assertEqual(result, "familiar_input_detected")
        self.assertAlmostEqual(self.module.curiosity_level, 0.0, places=6) # Sıfıra eşit olmalı


    # --- Exception Handling Test ---

    # decide metodunun try bloğu içinde hata fırlatacak bir şey mocklayalım.
    # random.choice merak eşiği aşıldığında çağrılıyor. Onu mocklayıp hata fırlatalım.
    @patch('random.choice', side_effect=RuntimeError("Simüle Edilmiş Karar Hatası"))
    def test_decide_exception_handling_during_decision_logic(self, mock_random_choice):
        """Karar alma mantığı sırasında hata oluşursa None döndürmesini test eder."""
        initial_curiosity = self.module.curiosity_threshold + 1.0 # Merak eşiği üstüne çıkarak random.choice çağrılmasını sağla (örn: 6.0)
        self.module.curiosity_level = initial_curiosity

        signals = { # Merak kararını tetikleyecek sinyaller
            'similarity_score': 0.1,
            'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False,
            'max_concept_similarity': 0.1, 'most_similar_concept_id': None,
        }

        # Beklenti: Karar alma sırasında hata fırlatılacak (mock sayesinde).
        # except bloğu yakalayacak ve None döndürecek.
        # finally bloğu çalışacak.
        # finally içindeki if decision is not None: false olacak (decision = None kaldı).
        # decay çalışMAYACAK (çünkü decision None). Initial merak seviyesi aynı kalmalı.
        # Düzeltme: finally bloğu decision is not None kontrolü kaldırıldı. Merak seviyesi *her zaman* decay olmalı.
        # Yeni Merak Logic'ine göre: Hata durumunda decision=None olur. Finally çalışır. decision is not None false olur. Merak değişmez.
        # O zaman expected_curiosity initial_curiosity olmalı.
        # Tekrar kontrol: finally bloğunda decay decision is not None kontrolünün dışında mıydı?
        # HAYIR, decay de decision is not None kontrolünün içindeymiş.
        # Demek ki hata durumunda NE ARTIS NE AZALIS NE DE DECAY uygulanıyor.
        # initial_curiosity = 6.0
        # Expected curiosity should be 6.0

        initial_curiosity_for_test = self.module.curiosity_threshold + 1.0
        self.module.curiosity_level = initial_curiosity_for_test

        result = self.module.decide(signals, [])

        self.assertIsNone(result) # Hata durumunda None dönmeli
        mock_random_choice.assert_called_once() # Mocklanan fonksiyon çağrılmış olmalı

        # Merak seviyesi kontrolü: Hata durumunda merak seviyesi güncellenmemeli (artış/azalış veya decay).
        # DecisionModule finally bloğundaki if decision is not None kontrolü sayesinde.
        self.assertEqual(self.module.curiosity_level, initial_curiosity_for_test)


    # --- cleanup Testleri ---

    def test_cleanup(self):
        """cleanup metodunun kaynakları temizlediğini test eder (şimdilik sadece loglama)."""
        # cleanup şu an bir state değiştirmiyor, sadece logluyor.
        # Log çıktısını mocklayarak çağrıldığını doğrulayabiliriz.
        with patch('src.cognition.decision.logger.info') as mock_logger_info:
             self.module.cleanup()
             # Doğru log mesajı ve çağrı.
             mock_logger_info.assert_called_with("DecisionModule objesi temizleniyor.")
        # Curiosity seviyesi zaten setUp'ta ve teardown'dan sonra sıfırlanmış oluyor,
        # cleanup metodu da aslında bir state değiştirmiyor.


# Testleri çalıştırmak için boilerplate kod
if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]], exit=False)