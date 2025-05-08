# src/cognition/learning.py
#
# Evo'nın denetimsiz öğrenme (kavram keşfi) modülünü temsil eder.
# Bellekteki Representation vektörlerini analiz ederek temel desenleri/kümeleri (kavramları) keşfeder.
# Basit eşik tabanlı yeni kavram ekleme mantığı kullanır.

import logging
import numpy as np # For vector operations and similarity calculation

# Import utility functions
# check_* functions come from src/core/utils
# get_config_value comes from src/core/config_utils
try:
    from src.core.utils import check_input_not_none, check_input_type, check_numpy_input
    from src.core.config_utils import get_config_value
except ImportError as e:
    # Log critical error if fundamental utility modules cannot be imported (for development/debug)
    logging.critical(f"Fundamental utility modules could not be imported: {e}. Please ensure src/core/utils.py and src/core/config_utils.py exist and PYTHONPATH is configured correctly.")
    # Placeholder functions (only used in case of import error)
    def get_config_value(config, *keys, default=None, expected_type=None, logger_instance=None):
         return default
    def check_input_not_none(input_data, input_name="Input", logger_instance=None):
         return input_data is not None
    def check_input_type(input_data, expected_type, input_name="Input", logger_instance=None):
         return isinstance(input_data, expected_type)
    def check_numpy_input(input_data, expected_dtype=None, expected_ndim=None, input_name="Input", logger_instance=None):
         return isinstance(input_data, np.ndarray)


# Create a logger for this module
logger = logging.getLogger(__name__)

class LearningModule:
    """
    Evo's unsupervised learning (concept discovery) module class (Phase 4 implementation).
    ... (Docstring same) ...
    """
    def __init__(self, config):
        """
        Initializes the LearningModule.

        Args:
            config (dict): Learning module configuration settings.
                           'new_concept_threshold': The similarity threshold (float, default 0.7) below which
                                                    a Representation vector is considered a new concept
                                                    relative to existing concept representatives.
                           'representation_dim': The dimension of the Representation vectors to be processed (int).
                                                  This should match the output dimension of the RepresentationLearner.
                                                  It can be obtained from the main config's representation section.
        """
        self.config = config
        logger.info("LearningModule initializing (Phase 4)...")

        # Get threshold from config using get_config_value with keyword arguments.
        # Based on config, new_concept_threshold is under the 'cognition' key.
        self.new_concept_threshold = get_config_value(config, 'cognition', 'new_concept_threshold', default=0.7, expected_type=(float, int), logger_instance=logger)
        # Get the representation dimension from config. Obtain it from the representation.representation_dim key.
        self.representation_dim = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)

        # Cast threshold to float to ensure correct comparison arithmetic later.
        # While expected_type checks type, explicit cast ensures float.
        self.new_concept_threshold = float(self.new_concept_threshold)
        # representation_dim is expected to be an int, no casting needed after expected_type check.

        # Simple value check for the threshold (must be between 0.0 and 1.0)
        if not (0.0 <= self.new_concept_threshold <= 1.0):
             logger.warning(f"LearningModule: Config 'new_concept_threshold' out of expected range ({self.new_concept_threshold}). Expected between 0.0 and 1.0. Using default 0.7.")
             self.new_concept_threshold = 0.7

        # Check if the representation dimension is positive.
        if self.representation_dim <= 0:
             # This could be a critical error but we don't crash during initialization.
             logger.error(f"LearningModule: Invalid 'representation_dim' config value ({self.representation_dim}). Expected a positive value. Concept learning might not function correctly.")
             # Subsequent methods (learn_concepts) should handle this state.


        # List to store concept representatives (vectors).
        self.concept_representatives = []

        # TODO: Persistence for concepts (saving/loading to file) will be added in the future.

        logger.info(f"LearningModule initialized. New Concept Threshold: {self.new_concept_threshold}, Representation Dimension: {self.representation_dim}. Initial Concept Count: {len(self.concept_representatives)}")

    # ... (learn_concepts, get_concepts, cleanup methods - same as before) ...

    def learn_concepts(self, representation_list):
        """
        Verilen Representation vektör listesini kullanarak yeni kavramları keşfeder veya mevcutları günceller.
        ... (Docstring aynı) ...
        """
        # Girdi kontrolü. check_* fonksiyonları artık src.core.utils'tan geliyor.
        if not check_input_not_none(representation_list, input_name="representation_list for LearningModule", logger_instance=logger):
             logger.debug("LearningModule.learn_concepts: representation_list None. Kavram öğrenme atlandi.")
             return self.concept_representatives

        if not check_input_type(representation_list, list, input_name="representation_list for LearningModule", logger_instance=logger):
             logger.error(f"LearningModule.learn_concepts: representation_list liste değil ({type(representation_list)}). Kavram öğrenme atlandi.")
             return self.concept_representatives

        if not representation_list:
             logger.debug("LearningModule.learn_concepts: representation_list boş liste. Kavram öğrenme atlandi.")
             return self.concept_representatives

        # Representation boyutu geçerli değilse öğrenme yapma.
        if self.representation_dim <= 0:
             logger.error("LearningModule.learn_concepts: Representation boyutu geçersiz. Kavram öğrenme atlandi.")
             return self.concept_representatives


        logger.debug(f"LearningModule.learn_concepts: {len(representation_list)} representation vektörü öğrenme için alındı. Mevcut {len(self.concept_representatives)} kavram var.")

        try:
            # Gelen her Representation vektörünü işle.
            for rep_vector in representation_list:
                # Representation vektörünün geçerli (numpy array, 1D, sayısal, doğru boyut) olduğundan emin ol.
                if rep_vector is not None and check_numpy_input(rep_vector, expected_dtype=np.number, expected_ndim=1, input_name="item in representation_list", logger_instance=logger):
                    if rep_vector.shape[0] == self.representation_dim:
                        # Vektörün normalizasyonunu hesapla (benzerlik için).
                        rep_norm = np.linalg.norm(rep_vector)

                        # Eğer vektör sıfır değilse (an anlamlıysa)
                        if rep_norm > 1e-8:
                            # Bu vektörün mevcut kavram temsilcilerine olan en yüksek benzerliğini bul.
                            # max_similarity_to_concepts başlangıcını -1.0 yap (maksimum alırken negatif benzerlikleri de yakalamak için)
                            # max_similarity_to_concepts = -1.0 başlangıcı ve threshold 1.0/0.0 karşılaştırma mantığı doğru uygulandığından emin olun.
                            max_similarity_to_concepts = -1.0 # Kosinüs benzerliği -1.0 ile 1.0 arasındadır.

                            if self.concept_representatives: # Öğrenilmiş kavram varsa benzerlik hesapla.
                                for concept_rep in self.concept_representatives:
                                     if concept_rep is not None and check_numpy_input(concept_rep, expected_dtype=np.number, expected_ndim=1, input_name="existing concept representative", logger_instance=logger):
                                          concept_norm = np.linalg.norm(concept_rep)
                                          if concept_norm > 1e-8:
                                               # Kosinüs benzerliği hesapla.
                                               similarity = np.dot(rep_vector, concept_rep) / (rep_norm * concept_norm)
                                               # NaN veya sonsuzluk kontrolü
                                               if not np.isnan(similarity) and np.isfinite(similarity):
                                                    max_similarity_to_concepts = max(max_similarity_to_concepts, similarity)
                                          # else: logger.debug("LM.learn_concepts: Concept rep near zero norm, skipping similarity.")
                                     # else: logger.warning("LM.learn_concepts: Existing concept representative invalid, skipping similarity.")

                                # logger.debug(f"LM.learn_concepts: Vektör için en yüksek kavram benzerliği: {max_similarity_to_concepts:.4f}")

                            # Eğer en yüksek benzerlik eşiğin altındaysa, yeni kavram olarak ekle.
                            # Veya hiç kavram yoksa ilk vektörü ilk kavram yap.
                            should_add_concept = False
                            if not self.concept_representatives:
                                # İlk vektör her zaman eklenir (boş liste kontrolü)
                                should_add_concept = True
                            else:
                                # Karşılaştırma yap
                                # Eşik 1.0 olduğunda floating point toleransı kullan.
                                if abs(self.new_concept_threshold - 1.0) < 1e-9: # Eşik 1.0'a çok yakınsa
                                    # Çok yakınsa bile 1.0'dan KESİNLİKLE küçük kabul et (epsilon toleransı)
                                    should_add_concept = max_similarity_to_concepts < (1.0 - 1e-9)
                                elif abs(self.new_concept_threshold - 0.0) < 1e-9: # Eşik 0.0'a çok yakınsa
                                    # Kesinlikle negatifse ekle
                                     should_add_concept = max_similarity_to_concepts < 0.0
                                else:
                                     # Genel durum
                                     should_add_concept = max_similarity_to_concepts < self.new_concept_threshold

                            if should_add_concept:
                                # Yeni kavram temsilcisi olarak mevcut Representation vektörünü ekle (kopyası).
                                self.concept_representatives.append(rep_vector.copy()) # Numpy array'in kopyasını sakla.
                                logger.info(f"LearningModule.learn_concepts: Yeni kavram keşfedildi! Toplam kavram: {len(self.concept_representatives)}. En yüksek mevcut benzerlik: {max_similarity_to_concepts:.4f} < Eşik {self.new_concept_threshold:.4f}")
                            # else: logger.debug(f"LM.learn_concepts: Vektör yeterince tanıdık (benzerlik {max_similarity_to_concepts:.4f} >= eşik {self.new_concept_threshold:.4f}). Yeni kavram eklenmedi.")

                        # else: logger.debug("LM.learn_concepts: Representation vektörü sıfır norm, işlenmedi.")
                    # else: logger.warning(f"LM.learn_concepts: Representation vektörü yanlış boyut ({rep_vector.shape[0]} vs {self.representation_dim}), işlenmedi.")
                # else: logger.warning("LM.learn_concepts: Listedeki öğe geçerli Representation vektörü değil, işlenmedi.")

            logger.debug(f"LearningModule.learn_concepts: Kavram öğrenme tamamlandı. Güncel kavram sayısı: {len(self.concept_representatives)}")

        except Exception as e:
            # Öğrenme işlemi sırasında beklenmedik bir hata olursa logla.
            logger.error(f"LearningModule.learn_concepts: Kavram öğrenme sırasında beklenmedik hata: {e}", exc_info=True)
            # Hata durumunda mevcut kavram listesini döndür.

        return self.concept_representatives # Güncellenmiş kavram temsilcileri listesini döndür.


    def get_concepts(self):
        """
        Öğrenilmiş kavram temsilcileri listesini döndürür (shallow copy).

        Returns:
            list: Öğrenilmiş kavram temsilcileri numpy array listesi.
        """
        # Güvenlik için listenin shallow kopyasını döndürmek daha iyi olabilir.
        # Bu, listenin kendisi kopyalanır ancak içindeki array objeleri referans olarak kalır.
        return self.concept_representatives[:]


    def cleanup(self):
        """
        LearningModule kaynaklarını temizler.
        ... (Docstring aynı) ...
        """
        logger.info("LearningModule objesi siliniyor...")
        # TODO: Öğrenilen kavramları kaydetme mantığı buraya gelecek (gelecek TODO).

        # Öğrenilen kavramlar listesini temizle.
        self.concept_representatives = []
        logger.info("LearningModule: Öğrenilen kavramlar temizlendi.")
        pass

    # TODO: Kavramları kaydetme/yükleme metotları eklenecek (gelecek TODO).