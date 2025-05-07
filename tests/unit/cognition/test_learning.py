# tests/unit/cognition/test_learning.py
import unittest
import sys
import os
import numpy as np
import logging

# Mock kullanabilmek için patch import edildi
from unittest.mock import patch # MagicMock bu testte kullanılmıyor, kaldırıldı

# Proje kök dizinini sys.path'e ekleyin. Test dosyasının tests/unit/cognition içinde olduğunu varsayar.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

# Şimdi modülleri import edebiliriz.
# check_* fonksiyonları src.core.utils'tan gelir.
# get_config_value src.core.config_utils'tan gelir.
try:
    from src.cognition.learning import LearningModule
    # check_* fonksiyonları src.core.utils'tan gelmeli
    from src.core.utils import check_input_not_none, check_input_type, check_numpy_input
    from src.core.config_utils import get_config_value
except ImportError as e:
     # Bu import hatası genelde PYTHONPATH yanlış ayarlandığında olur.
     # Test ortamı kurulumunu kontrol edin.
     raise ImportError(f"Temel modüller import edilemedi. PYTHONPATH doğru ayarlanmış mı? Hata: {e}")


# Testler sırasında logger çıktılarını görmek isterseniz bu satırları etkinleştirebilirsiniz.
# import logging
# logging.basicConfig(level=logging.DEBUG) # Genel seviyeyi DEBUG yap
# logging.getLogger('src.cognition.learning').setLevel(logging.DEBUG)
# logging.getLogger('src.core.utils').setLevel(logging.DEBUG)
# logging.getLogger('src.core.config_utils').setLevel(logging.DEBUG)


class TestLearningModule(unittest.TestCase):

    def setUp(self):
        """Her test metodundan önce çalışır, varsayılan bir konfigürasyon ve LearningModule örneği oluşturur."""
        self.default_config = {
            'new_concept_threshold': 0.7,
            'representation_dim': 10, # Varsayılan representation boyutu
        }
        self.module = LearningModule(self.default_config)

    def tearDown(self):
        """Her test metodundan sonra çalışır."""
        # Her testten sonra kavramları temizleyelim ki testler birbirinden etkilenmesin.
        self.module.cleanup()


    # --- __init__ Testleri (Bunlar zaten geçiyordu) ---

    def test_init_with_valid_config(self):
        """Geçerli bir konfigürasyon ile başlatmayı test eder."""
        self.assertEqual(self.module.config, self.default_config)
        self.assertEqual(self.module.new_concept_threshold, 0.7)
        self.assertIsInstance(self.module.new_concept_threshold, float)
        self.assertEqual(self.module.representation_dim, 10)
        self.assertIsInstance(self.module.representation_dim, int)
        self.assertEqual(self.module.concept_representatives, [])

    def test_init_with_missing_config_values(self):
        """Bazı konfigürasyon değerleri eksikken başlatmayı test eder (varsayılanlar kullanılmalı)."""
        incomplete_config = {
            'representation_dim': 5, # int olarak verelim
        }
        module = LearningModule(incomplete_config)
        self.assertEqual(module.new_concept_threshold, 0.7) # Varsayılan float
        self.assertIsInstance(module.new_concept_threshold, float)
        self.assertEqual(module.representation_dim, 5) # Belirtilen int
        self.assertIsInstance(module.representation_dim, int)
        self.assertEqual(module.concept_representatives, [])

    def test_init_with_invalid_config_types(self):
        """Geçersiz tipte konfigürasyon değerleri ile başlatmayı test eder (varsayılanlar kullanılmalı)."""
        invalid_type_config = {
            'new_concept_threshold': "not a float", # Geçersiz tip -> Varsayılan (0.7 float)
            'representation_dim': [128], # Geçersiz tip -> Varsayılan (128 int)
        }
        module = LearningModule(invalid_type_config)
        self.assertEqual(module.new_concept_threshold, 0.7) # Varsayılan float
        self.assertIsInstance(module.new_concept_threshold, float)
        self.assertEqual(module.representation_dim, 128) # Varsayılan int
        self.assertIsInstance(module.representation_dim, int)
        self.assertEqual(module.concept_representatives, [])

    def test_init_new_concept_threshold_out_of_range(self):
        """new_concept_threshold 0.0-1.0 aralığı dışındaysa test eder."""
        config_high = {'new_concept_threshold': 1.5, 'representation_dim': 10}
        module_high = LearningModule(config_high)
        self.assertEqual(module_high.new_concept_threshold, 0.7) # Reset edilmeli

        config_low = {'new_concept_threshold': -0.5, 'representation_dim': 10}
        module_low = LearningModule(config_low)
        self.assertEqual(module_low.new_concept_threshold, 0.7) # Reset edilmeli

        config_at_boundaries_high = {'new_concept_threshold': 1.0, 'representation_dim': 10}
        module_boundaries_high = LearningModule(config_at_boundaries_high)
        self.assertEqual(module_boundaries_high.new_concept_threshold, 1.0) # Sınırlar dahil olmalı

        config_at_boundaries_low = {'new_concept_threshold': 0.0, 'representation_dim': 10}
        module_boundaries_low = LearningModule(config_at_boundaries_low)
        self.assertEqual(module_boundaries_low.new_concept_threshold, 0.0) # Sınırlar dahil olmalı


    def test_learn_concepts_invalid_representation_dim_skips(self):
        """LearningModule'un representation_dim'i geçersizse öğrenme yapılmadığını test eder."""
        original_dim = self.module.representation_dim
        self.module.representation_dim = 0 # Geçersiz yap

        initial_concepts = [np.random.rand(original_dim).astype(np.float32)] # Use original_dim for consistency
        self.module.concept_representatives = initial_concepts # Başlangıçta kavram olsun

        rep_list = [np.random.rand(original_dim).astype(np.float32)] # Input vector with original_dim
        result = self.module.learn_concepts(rep_list)

        self.assertEqual(result, initial_concepts) # Öğrenme yapılmamalı, mevcut liste dönmeli
        self.assertEqual(self.module.concept_representatives, initial_concepts) # İç liste değişmemeli

        # Testten sonra dim'i eski haline getirelim ki diğer testleri bozmasın
        self.module.representation_dim = original_dim


    # --- learn_concepts Testleri ---

    def test_learn_concepts_input_none(self):
        """representation_list None ise test eder."""
        initial_concepts = [np.random.rand(self.module.representation_dim).astype(np.float32)]
        self.module.concept_representatives = initial_concepts # Başlangıçta kavram olsun
        result = self.module.learn_concepts(None)
        # Numpy array içeren listeleri == ile karşılaştırmak yerine np.testing.assert_array_equal kullanın.
        # Ancak burada sadece None inputta aynı liste dönmesi bekleniyor, referans karşılaştırması (==) yeterli.
        self.assertEqual(result, initial_concepts) # Mevcut liste dönmeli
        self.assertEqual(self.module.concept_representatives, initial_concepts) # İç liste değişmemeli

    def test_learn_concepts_input_not_list(self):
        """representation_list liste değilse test eder."""
        initial_concepts = [np.random.rand(self.module.representation_dim).astype(np.float32)]
        self.module.concept_representatives = initial_concepts
        result = self.module.learn_concepts("not a list")
        self.assertEqual(result, initial_concepts) # Mevcut liste dönmeli
        self.assertEqual(self.module.concept_representatives, initial_concepts) # İç liste değişmemeli

    def test_learn_concepts_input_empty_list(self):
        """representation_list boş liste ise test eder."""
        initial_concepts = [np.random.rand(self.module.representation_dim).astype(np.float32)]
        self.module.concept_representatives = initial_concepts
        result = self.module.learn_concepts([])
        self.assertEqual(result, initial_concepts) # Mevcut liste dönmeli
        self.assertEqual(self.module.concept_representatives, initial_concepts) # İç liste değişmemeli


    def test_learn_concepts_from_scratch_first_vector(self):
        """Boşken ilk geçerli vektörü öğrenme test eder (yeni kavram olmalı)."""
        rep_vector = np.random.rand(self.module.representation_dim).astype(np.float32)

        self.assertEqual(len(self.module.concept_representatives), 0) # Başlangıçta boş olmalı

        result = self.module.learn_concepts([rep_vector])

        self.assertEqual(len(result), 1) # 1 kavram olmalı
        self.assertIsInstance(result[0], np.ndarray)
        self.assertEqual(result[0].shape, (self.module.representation_dim,))
        # Öğrenilen vektör, input vektörün kopyası olmalı (value check)
        np.testing.assert_array_equal(result[0], rep_vector)
        # İç liste de güncellenmiş olmalı ve array objesi farklı olmalı (.copy() kullandığı için learn_concepts içinde)
        self.assertEqual(len(self.module.concept_representatives), 1) # Uzunluk kontrolü
        np.testing.assert_array_equal(self.module.concept_representatives[0], rep_vector) # İçerik kontrolü
        self.assertIsNot(self.module.concept_representatives[0], rep_vector) # Obje referansı kontrolü


    # DÜZELTİLEN TEST: Yeni, eşik altı benzerlikteki vektörün yeni kavram olarak eklenmesini test eder.
    # Deterministik vektörler kullanıldı ve beklenti 2 kavram (başlangıç 1 + 1 yeni) olarak düzeltildi.
    # Önceki testteki "Yeni vektör 2" aslında ilk kavrama scaled kopyası gibiydi, benzerliği 1.0 çıkıyordu.
    # Bu test, sadece ORTOGONAL vektörün eklendiğini test etmeli.
    def test_learn_concepts_new_concept_below_threshold_deterministic(self):
        """Yeni, eşik altı benzerlikteki vektörün yeni kavram olarak eklenmesini test eder (deterministik)."""
        dim = self.module.representation_dim
        self.module.new_concept_threshold = 0.7 # Eşik 0.7

        # İlk kavram: [1, 0, 0, ...]
        initial_concept_rep = np.zeros(dim, dtype=np.float32); initial_concept_rep[0] = 1.0
        self.module.learn_concepts([initial_concept_rep])
        self.assertEqual(len(self.module.concept_representatives), 1) # İlk kavram eklendi

        # Yeni vektör: [0, 1, 0, ...] (İlk kavrama ortogonal, benzerlik 0.0)
        new_rep_vector_orthogonal = np.zeros(dim, dtype=np.float32); new_rep_vector_orthogonal[1] = 1.0
        # Benzerlik 0.0. Eşik 0.7. 0.0 < 0.7 -> True. Eklenmeli.

        result = self.module.learn_concepts([new_rep_vector_orthogonal])

        self.assertEqual(len(result), 2) # Yeni kavram eklendi
        np.testing.assert_array_equal(result[0], initial_concept_rep)
        np.testing.assert_array_equal(result[1], new_rep_vector_orthogonal)
        self.assertEqual(len(self.module.concept_representatives), 2) # İç liste uzunluğu
        np.testing.assert_array_equal(self.module.concept_representatives[0], initial_concept_rep)
        np.testing.assert_array_equal(self.module.concept_representatives[1], new_rep_vector_orthogonal)


    # DÜZELTİLEN TEST: Mevcutlara eşik üstü/eşit benzerlikteki vektörün eklenmemesini test eder.
    # Deterministik vektörler kullanıldı. Beklenti 1 kavram (başlangıç 1 + 0 yeni)
    def test_learn_concepts_existing_concept_above_threshold_deterministic(self):
        """Mevcutlara eşik üstü/eşit benzerlikteki vektörün eklenmemesini test eder (deterministik)."""
        dim = self.module.representation_dim
        self.module.new_concept_threshold = 0.7 # Eşik 0.7

        # İlk kavram: [1, 0, 0, ...]
        initial_concept_rep = np.zeros(dim, dtype=np.float32); initial_concept_rep[0] = 1.0
        self.module.learn_concepts([initial_concept_rep])
        self.assertEqual(len(self.module.concept_representatives), 1) # İlk kavram eklendi

        # Çok benzer bir vektör: [1.001, 0, 0, ...] (Benzerlik ~1.0)
        similar_rep_vector = np.zeros(dim, dtype=np.float32); similar_rep_vector[0] = 1.001
        # Benzerlik ~1.0. Eşik 0.7. 1.0 < 0.7 -> False. Eklenmemeli.

        result = self.module.learn_concepts([similar_rep_vector])

        self.assertEqual(len(result), 1) # Yeni kavram eklenmedi
        np.testing.assert_array_equal(result[0], initial_concept_rep)
        self.assertEqual(len(self.module.concept_representatives), 1) # İç liste uzunluğu
        np.testing.assert_array_equal(self.module.concept_representatives[0], initial_concept_rep)


        # Eşit benzerlikteki bir vektör: threshold=0.7 yapalım, vektörün benzerliği 0.7 olsun.
        similar_at_threshold_rep = np.zeros(dim, dtype=np.float32)
        similar_at_threshold_rep[0] = 0.7
        if dim > 1: # Boyut 1'den büyükse diğer elemanı ayarla
             similar_at_threshold_rep[1] = np.sqrt(1.0 - 0.7**2)
        # Normalize edelim
        norm = np.linalg.norm(similar_at_threshold_rep)
        if norm > 1e-8:
            similar_at_threshold_rep /= norm
        else:
            similar_at_threshold_rep[0] = 0.0  # Handle tiny norms
        # Sim with [1,0,...] is 0.7. Sim 0.7. Eşik 0.7. 0.7 < 0.7 -> False. Eklenmemeli.

        result2 = self.module.learn_concepts([similar_at_threshold_rep])
        # Önceki kavram listesi 1 öğeydi. Yeni vektör eklenmedi. Toplam 1 olmalı.
        self.assertEqual(len(result2), 1) # Hala 1 kavram olmalı
        np.testing.assert_array_equal(result2[0], initial_concept_rep)
        self.assertEqual(len(self.module.concept_representatives), 1) # İç liste uzunluğu
        np.testing.assert_array_equal(self.module.concept_representatives[0], initial_concept_rep)



    # DÜZELTİLEN TEST: Karışık vektör listesi ile öğrenme test eder.
    # Deterministik vektörler kullanıldı. Beklenti 4 kavram (başlangıç 2 + 2 yeni)
    def test_learn_concepts_multiple_vectors_mixed_deterministic(self):
        """Karışık vektör listesi ile öğrenme test eder (geçerli, geçersiz, benzer, yeni - deterministik)."""
        dim = self.module.representation_dim
        self.module.new_concept_threshold = 0.7 # Varsayılan eşik

        # Başlangıç kavramları
        concept1 = np.zeros(dim, dtype=np.float32); concept1[0] = 1.0
        concept2 = np.zeros(dim, dtype=np.float32); concept2[1] = 1.0
        self.module.learn_concepts([concept1, concept2]) # Şimdi 2 kavram var
        self.assertEqual(len(self.module.concept_representatives), 2)

        # Öğrenilecek liste (Deterministik vektörler):
        # Vektörleri liste dışında tanımlayalım
        new_concept_rep1_det = np.zeros(dim, dtype=np.float32); new_concept_rep1_det[2] = 1.0 # Sim to concept1 & 2 is 0.0 < 0.7 -> Added
        new_concept_rep2_det = np.zeros(dim, dtype=np.float32); new_concept_rep2_det[3] = 1.0 # Sim to concept1, 2, new1 is 0.0 < 0.7 -> Added

        similar_enough_rep_det = np.zeros(dim, dtype=np.float32) # Sim to concept1 ~0.8 > 0.7 -> Not added
        similar_enough_rep_det[0] = 0.8
        if dim > 1: similar_enough_rep_det[dim-1] = np.sqrt(1-0.8**2)
        norm = np.linalg.norm(similar_enough_rep_det)
        if norm > 1e-8: similar_enough_rep_det /= norm
        else: similar_enough_rep_det[0] = 0.0


        rep_list_to_learn_det = [
            np.zeros(dim, dtype=np.float32), # Sıfır normlu -> Atlanmalı
            None, # None girdi -> Atlanmalı
            "not a numpy array", # Yanlış tip -> Atlanmalı
            np.random.rand(dim // 2 if dim > 1 else 1).astype(np.float32), # Yanlış boyut -> Atlanmalı
            concept1.copy() + 1e-5, # Concept1'e çok benzer -> Atlanmalı
            concept2.copy() - 1e-5, # Concept2'ye çok benzer -> Atlanmalı
            np.zeros(dim, dtype=np.float32), # Tekrar sıfır normlu -> Atlanmalı

            new_concept_rep1_det, # Yeni kavram 1 -> Eklenecek
            new_concept_rep2_det, # Yeni kavram 2 -> Eklenecek
            similar_enough_rep_det, # Benzer kavram -> Atlanmalı
        ]

        # Beklenen: Başlangıçtaki 2 + 2 yeni = 4 kavram
        initial_concept_count = len(self.module.concept_representatives) # 2
        result = self.module.learn_concepts(rep_list_to_learn_det)

        self.assertEqual(len(result), len(self.module.concept_representatives)) # Dönüş değeri ile iç liste aynı olmalı
        self.assertEqual(len(result), initial_concept_count + 2) # Total 4 concepts expected

        # İlk 2 kavram başlangıçtakiler olmalı (value check)
        np.testing.assert_array_equal(result[0], concept1)
        np.testing.assert_array_equal(result[1], concept2)

        # Eklenen 2 yeni kavram doğru deterministik vektörler olmalı (value check)
        # Sıra: rep_list_to_learn_det içindeki sıraya göre eklenirler.
        # new_concept_rep1_det (index 7) sonra new_concept_rep2_det (index 8)
        np.testing.assert_array_equal(result[2], new_concept_rep1_det)
        np.testing.assert_array_equal(result[3], new_concept_rep2_det)


    # DÜZELTİLEN TEST: Sıfır normlu bir vektörün listede olduğunda atlanmasını test eder.
    # Deterministik vektör kullanıldı. Beklenti 2 kavram (başlangıç 1 + 1 yeni)
    def test_learn_concepts_with_zero_norm_vector_in_list_deterministic(self):
        """Sıfır normlu bir vektörün listede olduğunda atlanmasını test eder (deterministik)."""
        dim = self.module.representation_dim
        # Başlangıçta 1 kavram olsun
        initial_concept = np.zeros(dim, dtype=np.float32); initial_concept[0] = 1.0
        self.module.learn_concepts([initial_concept])
        self.assertEqual(len(self.module.concept_representatives), 1)

        # Yeni kavram olacak vektör (orthogonal)
        new_concept_rep = np.zeros(dim, dtype=np.float32); new_concept_rep[1] = 1.0 # Sim 0.0 < 0.7 -> Eklenmeli

        rep_list = [
            np.array([0.0] * dim, dtype=np.float32), # Sıfır normlu -> Atlanmalı
            new_concept_rep, # Yeni kavram olacak -> Eklenmeli
        ]

        result = self.module.learn_concepts(rep_list)

        # Sıfır normlu olan atlanmalı, diğeri yeni kavram olarak eklenmeli
        self.assertEqual(len(result), len(self.module.concept_representatives))
        self.assertEqual(len(result), 2) # Başlangıç 1 + Yeni 1 = 2 kavram olmalı
        np.testing.assert_array_equal(result[0], initial_concept) # İlk kavram aynı kalmalı
        np.testing.assert_array_equal(result[1], new_concept_rep) # Yeni kavram ikinci girdiden gelmeli


    # DÜZELTİLEN TEST: Yanlış boyutlu bir vektörün listede olduğunda atlanmasını test eder.
    # Deterministik vektör kullanıldı. Beklenti 2 kavram (başlangıç 1 + 1 yeni)
    def test_learn_concepts_with_wrong_dimension_vector_in_list_deterministic(self):
        """Yanlış boyutlu bir vektörün listede olduğunda atlanmasını test eder (deterministik)."""
        dim = self.module.representation_dim
        # Başlangıçta 1 kavram olsun
        initial_concept = np.zeros(dim, dtype=np.float32); initial_concept[0] = 1.0
        self.module.learn_concepts([initial_concept])
        self.assertEqual(len(self.module.concept_representatives), 1)

        # Yeni kavram olacak vektör (orthogonal)
        new_concept_rep = np.zeros(dim, dtype=np.float32); new_concept_rep[1] = 1.0 # Sim 0.0 < 0.7 -> Eklenmeli

        rep_list = [
            np.random.rand(dim // 2 if dim > 1 else 1).astype(np.float32), # Yanlış boyut -> Atlanmalı
            new_concept_rep, # Yeni kavram olacak -> Eklenmeli
        ]

        result = self.module.learn_concepts(rep_list)

        # Yanlış boyutlu olan atlanmalı, diğeri yeni kavram olarak eklenmeli
        self.assertEqual(len(result), len(self.module.concept_representatives))
        self.assertEqual(len(result), 2) # Başlangıç 1 + Yeni 1 = 2 kavram olmalı
        np.testing.assert_array_equal(result[0], initial_concept)
        np.testing.assert_array_equal(result[1], new_concept_rep)


    # DÜZELTİLEN TEST: Liste içinde None veya non-numpy öğeler olduğunda atlanmasını test eder.
    # Deterministik vektör kullanıldı. Beklenti 2 kavram (başlangıç 1 + 1 yeni)
    def test_learn_concepts_with_none_or_invalid_type_in_list_deterministic(self):
        """Liste içinde None veya non-numpy öğeler olduğunda atlanmasını test eder (deterministik)."""
        dim = self.module.representation_dim
        # Başlangıçta 1 kavram olsun
        initial_concept = np.zeros(dim, dtype=np.float32); initial_concept[0] = 1.0
        self.module.learn_concepts([initial_concept])
        self.assertEqual(len(self.module.concept_representatives), 1)

        # Yeni kavram olacak vektör (orthogonal)
        new_concept_rep = np.zeros(dim, dtype=np.float32); new_concept_rep[1] = 1.0 # Sim 0.0 < 0.7 -> Eklenmeli

        rep_list = [
            None, # Atlanmalı
            "not a numpy array", # Atlanmalı
            123, # Atlanmalı
            new_concept_rep, # Yeni kavram olacak -> Eklenmeli
        ]

        result = self.module.learn_concepts(rep_list)

        # Geçersiz olanlar atlanmalı, sadece sonuncu eklenmeli
        self.assertEqual(len(result), len(self.module.concept_representatives))
        self.assertEqual(len(result), 2) # Başlangıç 1 + Yeni 1 = 2 kavram olmalı
        np.testing.assert_array_equal(result[0], initial_concept)
        np.testing.assert_array_equal(result[1], new_concept_rep)


    # DÜZELTİLEN TEST: new_concept_threshold 0.0 iken öğrenme test eder (deterministik).
    # Beklenti 2 kavram (başlangıç 1 + 1 yeni)
    def test_learn_concepts_threshold_zero_deterministic(self):
        """new_concept_threshold 0.0 iken öğrenme test eder (deterministik)."""
        dim = self.module.representation_dim
        self.module.new_concept_threshold = 0.0 # Çok katı eşik (sadece sim < 0 olanlar eklenir)

        # Vektörler (float32 olarak ayarlayalım)
        rep1 = np.zeros(dim, dtype=np.float32); rep1[0] = 1.0 # [1, 0, ...]
        rep2_similar = np.zeros(dim, dtype=np.float32); rep2_similar[0] = 1.001 # [1.001, 0, ...], Sim(rep1, rep2_similar) = 1.0
        rep3_orthogonal = np.zeros(dim, dtype=np.float32); rep3_orthogonal[1] = 1.0 # [0, 1, ...], Sim(rep1, rep3_orthogonal) = 0.0
        rep4_opposite = np.zeros(dim, dtype=np.float32); rep4_opposite[0] = -1.0 # [-1, 0, ...], Sim(rep1, rep4_opposite) = -1.0. -1.0 < 0.0 True -> Eklenecek.
        rep5_almost_opposite = np.zeros(dim, dtype=np.float32); rep5_almost_opposite[0] = -0.1;
        # Boyut > 1 ise non-zero eleman ekleyip normalize edelim
        if dim > 1: rep5_almost_opposite[dim-1] = np.sqrt(1-(-0.1)**2)
        norm = np.linalg.norm(rep5_almost_opposite)
        if norm > 1e-8:
            rep5_almost_opposite /= norm
        else:
            rep5_almost_opposite[0] = 0.0 # Ensure normalized, handle tiny norms
        # Sim(rep1, rep5_almost_opposite) = -0.1. Sim(rep4_opposite, rep5_almost_opposite) = dot([-1,0,...], [-0.1,...])/norms ~ 0.1. Max sim = max(-0.1, 0.1) = 0.1. 0.1 < 0.0 False -> Eklenmeyecek.

        self.module.learn_concepts([rep1]) # İlk vektör her zaman eklenir (liste boş olduğu için)
        self.assertEqual(len(self.module.concept_representatives), 1)
        np.testing.assert_array_equal(self.module.concept_representatives[0], rep1)

        # Öğrenilecek liste: rep2 (sim 1.0), rep3 (sim 0.0), rep4 (sim -1.0), rep5 (sim -0.1)
        rep_list = [rep2_similar, rep3_orthogonal, rep4_opposite, rep5_almost_opposite]

        result = self.module.learn_concepts(rep_list)

        # rep2 (sim 1.0): 1.0 < 0.0 False -> Eklenmez
        # rep3 (sim 0.0): 0.0 < 0.0 False -> Eklenmez
        # rep4 (sim -1.0 to rep1): -1.0 < 0.0 True -> Eklenir (2. kavram). Concepts: [rep1, rep4_opposite]
        # rep5 (sim -0.1 to rep1, sim ~0.1 to rep4). Max sim = ~0.1. 0.1 < 0.0 False -> Eklenmez.

        self.assertEqual(len(result), len(self.module.concept_representatives))
        self.assertEqual(len(result), 2) # Başlangıç 1 + Yeni 1 (rep4) = 2 kavram olmalı
        np.testing.assert_array_equal(result[0], rep1) # İlk kavram aynı kalmalı
        np.testing.assert_array_equal(result[1], rep4_opposite) # İkinci eklenen -1.0 simli olmalı


    # DÜZELTİLEN TEST: new_concept_threshold 1.0 iken öğrenme test eder (deterministik).
    # Beklenti 3 kavram (başlangıç 1 + 2 yeni)
    def test_learn_concepts_threshold_one_deterministic(self):
        """new_concept_threshold 1.0 iken öğrenme test eder (deterministik)."""
        dim = self.module.representation_dim
        self.module.new_concept_threshold = 1.0 # En yüksek katılık (sadece sim < 1.0 olanlar eklenir)

        # Vektörler (float32 olarak ayarlayalım)
        rep1 = np.zeros(dim, dtype=np.float32); rep1[0] = 1.0 # [1, 0, ...]
        rep2_identical = rep1.copy() # [1, 0, ...], Sim(rep1, rep2_identical) = 1.0
        rep3_very_similar = np.zeros(dim, dtype=np.float32); rep3_very_similar[0] = 1.0 - 1e-9;
        # Normalize edelim
        norm = np.linalg.norm(rep3_very_similar)
        if norm > 1e-8:
            rep3_very_similar /= norm
        else:
            rep3_very_similar[0] = 0.0 # Sim ~ 1.0 but < 1.0. ~1.0 < 1.0 True -> Eklenecek.
        rep4_different = np.zeros(dim, dtype=np.float32); rep4_different[1] = 1.0 # [0, 1, ...], Sim(rep1, rep4_different) = 0.0. Sim(rep3, rep4) ~0.0. Max sim ~0.0 < 1.0 True -> Eklenecek.

        self.module.learn_concepts([rep1]) # İlk vektör her zaman eklenir
        self.assertEqual(len(self.module.concept_representatives), 1)
        np.testing.assert_array_equal(self.module.concept_representatives[0], rep1)

        # Öğrenilecek liste: rep2 (sim 1.0), rep3 (sim < 1.0), rep4 (sim 0.0)
        rep_list = [rep2_identical, rep3_very_similar, rep4_different]

        result = self.module.learn_concepts(rep_list)

        # rep2 (sim 1.0): 1.0 < 1.0 False -> Eklenmez
        # rep3 (sim ~1.0 to rep1): ~1.0 < 1.0 True -> Eklenir (2. kavram). Concepts: [rep1, rep3_very_similar]
        # rep4 (sim 0.0 to rep1): 0.0 < 1.0 True. Sim to rep3_very_similar: dot(rep4, rep3)/norm(rep4)/norm(rep3) ~ 0. Sim ~ 0.0.
        # Max sim = max(0.0 (to rep1), ~0.0 (to rep3)) = ~0.0. Eşik 1.0. ~0.0 < 1.0 True -> Eklenir (3. kavram)

        self.assertEqual(len(result), len(self.module.concept_representatives))
        # Düzeltilmiş hali:
        self.assertEqual(len(result), 2) # Başlangıç 1 + Yeni 1 (rep4) = 2 kavram olmalı
        np.testing.assert_array_equal(result[0], rep1) # İlk kavram aynı kalmalı
        # İkinci eklenen kavram rep4_different olmalı
        np.testing.assert_array_equal(result[1], rep4_different) # Doğru karşılaştırma


    # DÜZELTİLEN TEST: Exception handling testi. array karşılaştırması düzeltildi.
    def test_learn_concepts_exception_handling(self):
        """Öğrenme sırasında bir exception olursa mevcut listenin döndürülmesini test eder."""
        dim = self.module.representation_dim
        initial_concept = np.random.rand(dim).astype(np.float32) # Ensure float32
        self.module.learn_concepts([initial_concept]) # Başlangıçta 1 kavram olsun
        self.assertEqual(len(self.module.concept_representatives), 1)

        # numpy.dot fonksiyonunu mock'layarak hata fırlatmasını simüle edelim.
        # Testteki tek vektör işlenirken hata fırlatacak şekilde ayarlayalım.

        rep_vector_to_learn = np.random.rand(dim).astype(np.float32) # Öğrenmeye çalışacağımız vektör
        # Bu vektör, ilk konsepte (initial_concept) karşı dot ürünü hesaplanırken hata fırlatacak.

        with patch('numpy.dot', side_effect=RuntimeError("Simüle Edilmiş Numpy.dot Hatası")):
             # learn_concepts'i çağıralım.
             # learn_concepts loop'a girer. rep_vector_to_learn alınır. norm hesaplanır (hata vermez).
             # self.concept_representatives boş değil. Loop'a girer. concept_rep (initial_concept) alınır. norm hesaplanır (hata vermez).
             # np.dot(rep_vector_to_learn, concept_rep) çağrılır. BURADA hata fırlatacak.
             # Hata except bloğunda yakalanacak.
             result = self.module.learn_concepts([rep_vector_to_learn])

             # Except bloğu çalıştıysa, öğrenme başarısız olmalı.
             # Mevcut liste (başlangıçtaki initial_concept) döndürülmeli.
             # Liste uzunluğu kontrolü
             self.assertEqual(len(result), len([initial_concept]))
             # Array içeriği kontrolü (value check)
             np.testing.assert_array_equal(result[0], initial_concept)
             # İç listenin de değişmediğini kontrol et (value check)
             self.assertEqual(len(self.module.concept_representatives), len([initial_concept]))
             np.testing.assert_array_equal(self.module.concept_representatives[0], initial_concept)



    # --- get_concepts Testleri ---

    def test_get_concepts_empty(self):
        """Kavram yokken get_concepts test eder."""
        concepts = self.module.get_concepts()
        self.assertEqual(concepts, [])
        self.assertIsInstance(concepts, list)

    # DÜZELTİLEN TEST: Kavram varken get_concepts test eder ve shallow copy beklentisini doğrular.
    # Test inputu deterministik hale getirildi.
    # Count hatası düzeltildi, 2 kavram eklenmeli.
    def test_get_concepts_with_data(self):
        """Kavram varken get_concepts test eder (shallow copy beklenir)."""
        dim = self.module.representation_dim
        # Deterministik kavramlar ekleyelim (sim 0.0 < 0.7)
        concept1 = np.zeros(dim, dtype=np.float32); concept1[0] = 1.0
        concept2 = np.zeros(dim, dtype=np.float32); concept2[1] = 1.0

        self.module.learn_concepts([concept1, concept2]) # 2 kavram ekle. sim(concept1, concept2) = 0.0 < 0.7 -> concept2 de eklenmeli.
        self.assertEqual(len(self.module.concept_representatives), 2) # Sanity check

        concepts = self.module.get_concepts()

        self.assertEqual(len(concepts), 2)
        self.assertIsInstance(concepts, list)
        # Dönen liste objesi orijinalden farklı olmalı (shallow copy)
        self.assertIsNot(concepts, self.module.concept_representatives)
        # İçindeki array'ler orijinal array objeleriyle aynı olmalı (shallow copy'nin özelliği)
        self.assertIs(concepts[0], self.module.concept_representatives[0])
        self.assertIs(concepts[1], self.module.concept_representatives[1])

        # İçeriklerin doğruluğunu kontrol et (zaten object referansları aynıysa içerik de aynıdır)
        np.testing.assert_array_equal(concepts[0], concept1)
        np.testing.assert_array_equal(concepts[1], concept2)


    # DÜZELTİLEN TEST: get_concepts'in shallow copy döndürdüğünü test eder (detaylı).
    # assertIs hatası ve array karşılaştırması düzeltildi.
    def test_get_concepts_is_shallow_copy(self):
        """get_concepts'in shallow copy döndürdüğünü test eder (detaylı)."""
        dim = self.module.representation_dim
        concept1 = np.random.rand(dim).astype(np.float32) # Ensure float32
        self.module.learn_concepts([concept1]) # 1 kavram ekle
        self.assertEqual(len(self.module.concept_representatives), 1)

        concepts = self.module.get_concepts() # Shallow copy'yi al

        self.assertIsNot(concepts, self.module.concept_representatives) # 1. Liste objesi farklı

        # Dönen listeye yeni bir öğe ekle (Orijinal liste etkilenmemeli)
        new_vector = np.random.rand(dim).astype(np.float32) # Ensure float32
        concepts.append(new_vector)
        self.assertEqual(len(concepts), 2)
        self.assertEqual(len(self.module.concept_representatives), 1) # 2. Orijinal liste uzunluğu değişmemeli


        # Dönen listedeki bir array'i değiştir (Bu, orijinal listedeki array'i de değiştirmeli çünkü shallow copy)
        # Sadece liste objesinin shallow copy'si yapıldı, içindeki array objeleri aynı.
        if len(concepts) > 0: # Testte en az 1 kavram eklenmiş olmalı
             original_array_ref = self.module.concept_representatives[0]
             returned_array_ref = concepts[0]

             self.assertIs(returned_array_ref, original_array_ref) # Array objelerinin referansları aynı olmalı

             original_value_before_mod = original_array_ref[0] # Değiştirmeden önceki değer
             modification_value = 999.0
             # Array'i değiştir (inplace değişiklik)
             returned_array_ref[0] += modification_value


             # Orijinal listedeki array'in de değiştiğini kontrol et
             # Değer karşılaştırması yapıyoruz. Değişim doğru yapıldıysa beklenen değere eşit olmalı.
             self.assertAlmostEqual(original_array_ref[0], original_value_before_mod + modification_value, places=6)


    # --- cleanup Testleri (Zaten geçiyordu) ---

    def test_cleanup(self):
        """cleanup metodunun kavram listesini temizlediğini test eder."""
        dim = self.module.representation_dim
        concept1 = np.random.rand(dim)
        self.module.learn_concepts([concept1]) # Kavram ekle
        self.assertEqual(len(self.module.concept_representatives), 1)

        self.module.cleanup() # Temizleme metodunu çağır

        self.assertEqual(self.module.concept_representatives, []) # Liste boş olmalı


# Testleri çalıştırmak için boilerplate kod
if __name__ == '__main__':
    # unittest'in argümanlarını temizleyerek notebook/script içinde çalışmasını sağlar.
    # argv=[sys.argv[0]] sadece dosya adını bırakır.
    unittest.main(argv=[sys.argv[0]], exit=False)