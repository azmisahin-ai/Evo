# src/cognition/understanding.py
#
# Evo'nın anlama modülünü temsil eder.
# İşlenmiş temsilleri, bellek girdilerini, anlık duyu özelliklerini ve öğrenilmiş kavramları kullanarak dünyayı anlamaya çalışır.

import logging
import numpy as np # For similarity calculation, array operations and mean

# Import utility functions (input checks from src/core/utils, config from src/core/config_utils)
try:
    from src.core.utils import check_input_not_none, check_input_type, check_numpy_input
    from src.core.config_utils import get_config_value
except ImportError as e:
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

class UnderstandingModule:
    """
    Evo's understanding capability module class (Phase 3/4 implementation).

    Receives Representation from RepresentationLearner, relevant memory entries from Memory,
    instantaneous sensory features from Process (low-level features), and learned concept
    representatives from LearningModule. It uses these inputs to generate a primitive
    "understanding" output (currently a dictionary).
    Current implementation: Calculates the highest memory similarity score, boolean flags
    based on instantaneous sensory features, AND the highest similarity to learned concept
    representatives.
    More complex understanding algorithms will be implemented in the future.
    """
    def __init__(self, config):
        """
        Initializes the UnderstandingModule.

        Args:
            config (dict): Understanding module configuration settings.
                           'audio_energy_threshold': Threshold for detecting high audio energy (float, default 1000.0).
                           'visual_edges_threshold': Threshold for detecting high visual edge density (float, default 50.0).
                           'brightness_threshold_high': Threshold for detecting a bright environment (float, default 200.0).
                           'brightness_threshold_low': Threshold for detecting a dark environment (float, default 50.0).
                           Future settings like model paths, understanding strategies could go here.
        """
        self.config = config
        logger.info("UnderstandingModule initializing (Phase 3/4)...")

        # Get thresholds from config using get_config_value with keyword arguments.
        # Based on config, these settings are directly under the 'cognition' key.
        # CognitionCore init passes the whole config dict, so the path starts directly with the key name.
        # Corrected: Use default= keyword format.
        self.audio_energy_threshold = get_config_value(config, 'cognition', 'audio_energy_threshold', default=1000.0, expected_type=(float, int), logger_instance=logger)
        self.visual_edges_threshold = get_config_value(config, 'cognition', 'visual_edges_threshold', default=50.0, expected_type=(float, int), logger_instance=logger)
        self.brightness_threshold_high = get_config_value(config, 'cognition', 'brightness_threshold_high', default=200.0, expected_type=(float, int), logger_instance=logger)
        self.brightness_threshold_low = get_config_value(config, 'cognition', 'brightness_threshold_low', default=50.0, expected_type=(float, int), logger_instance=logger)

        # Cast thresholds to float to ensure correct comparison arithmetic later.
        # While expected_type checks type, explicit cast ensures float.
        self.audio_energy_threshold = float(self.audio_energy_threshold)
        self.visual_edges_threshold = float(self.visual_edges_threshold)
        self.brightness_threshold_high = float(self.brightness_threshold_high)
        self.brightness_threshold_low = float(self.brightness_threshold_low)


        logger.info(f"UnderstandingModule initialized. Audio Energy Threshold: {self.audio_energy_threshold}, Visual Edge Threshold: {self.visual_edges_threshold}, Brightness High Threshold: {self.brightness_threshold_high}, Brightness Low Threshold: {self.brightness_threshold_low}")


    # ... (process and cleanup methods - same as before) ...

    def process(self, processed_inputs, learned_representation, relevant_memory_entries, current_concepts):
        """
        Gelen Representation, ilgili bellek girdileri, anlık Process çıktıları ve öğrenilmiş kavramları kullanarak anlama işlemini yapar.
        ... (Docstring aynı) ...
        """
        # Varsayılan anlama çıktısı dictionary'si
        understanding_signals = {
            'similarity_score': 0.0,
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.0,
            'most_similar_concept_id': None,
        }

        # Girdi geçerliliği kontrolleri.
        # check_input_not_none ve check_input_type artık src.core.utils'tan geliyor.
        # check_numpy_input da src.core.utils'tan geliyor.
        is_valid_memory_list = check_input_not_none(relevant_memory_entries, "relevant_memory_entries for UnderstandingModule", logger) and \
                               check_input_type(relevant_memory_entries, list, "relevant_memory_entries for UnderstandingModule", logger)

        is_valid_representation = check_input_not_none(learned_representation, "learned_representation for UnderstandingModule", logger) and \
                                  check_numpy_input(learned_representation, expected_dtype=np.number, expected_ndim=1, input_name="learned_representation for UnderstandingModule", logger_instance=logger)

        is_valid_processed_inputs = check_input_not_none(processed_inputs, "processed_inputs for UnderstandingModule", logger) and \
                                    isinstance(processed_inputs, dict) # check_input_type dict için yeterli değil, isinstance daha esnek


        is_valid_concepts_list = check_input_not_none(current_concepts, "current_concepts for UnderstandingModule", logger) and \
                                  check_input_type(current_concepts, list, "current_concepts for UnderstandingModule", logger)


        logger.debug(f"UnderstandingModule.process: Anlama işlemi başlatıldı. Girdi geçerliliği - Repr:{is_valid_representation}, Mem:{is_valid_memory_list}, Proc:{is_valid_processed_inputs}, Concepts:{is_valid_concepts_list}")

        # Representation geçerliyse normunu hesapla, aksi halde Representation tabanlı hesaplamaları atla.
        query_norm = 0.0
        if is_valid_representation:
             query_norm = np.linalg.norm(learned_representation)
             if query_norm < 1e-8:
                 logger.warning("UnderstandingModule.process: Learned representation near zero norm. Benzerlik ve Kavram tanıma atlanacak.")
                 is_valid_representation = False # Geçersiz gibi davran

        # Bu try bloğu, numpy veya indexleme gibi işlemlerden kaynaklanabilecek hataları yakalar.
        # src/core/utils'daki check_* fonksiyonları exception fırlatmadığı için, buradaki except
        # daha çok np.dot, np.linalg.norm, np.mean gibi işlemlerden kaynaklı hataları yakalayacaktır.
        try:
            # 1. Bellek Benzerlik Skoru Hesapla (Eğer Representation ve Bellek geçerliyse)
            max_memory_similarity = 0.0
            if is_valid_representation and is_valid_memory_list and relevant_memory_entries:
                for memory_entry in relevant_memory_entries:
                    stored_representation = memory_entry.get('representation')
                    # check_numpy_input artık src.core.utils'tan geliyor ve False dönüyor.
                    if stored_representation is not None and check_numpy_input(stored_representation, expected_dtype=np.number, expected_ndim=1, input_name="memory_entry['representation'] for UnderstandingModule", logger_instance=logger):
                         stored_norm = np.linalg.norm(stored_representation)
                         if stored_norm > 1e-8:
                              similarity = np.dot(learned_representation, stored_representation) / (query_norm * stored_norm)
                              if not np.isnan(similarity):
                                   max_memory_similarity = max(max_memory_similarity, similarity)
                understanding_signals['similarity_score'] = max_memory_similarity


            # 2. Öğrenilmiş Kavramlarla Benzerlik Hesapla (Eğer Representation ve Kavramlar geçerliyse)
            max_concept_similarity = 0.0
            most_similar_concept_id = None

            if is_valid_representation and is_valid_concepts_list and current_concepts:
                 for i, concept_rep in enumerate(current_concepts):
                      # check_numpy_input artık src.core.utils'tan geliyor ve False dönüyor.
                      if concept_rep is not None and check_numpy_input(concept_rep, expected_dtype=np.number, expected_ndim=1, input_name=f"current_concepts[{i}] for UnderstandingModule", logger_instance=logger):
                           concept_norm = np.linalg.norm(concept_rep)
                           if concept_norm > 1e-8:
                                similarity = np.dot(learned_representation, concept_rep) / (query_norm * concept_norm)
                                if not np.isnan(similarity):
                                     if similarity > max_concept_similarity:
                                          max_concept_similarity = similarity
                                          most_similar_concept_id = i

                 understanding_signals['max_concept_similarity'] = max_concept_similarity
                 understanding_signals['most_similar_concept_id'] = most_similar_concept_id


            # 3. Process Çıktıları Tabanlı Basit Anlama (Eğer Processed Inputs geçerliyse)
            # check_numpy_input False döndüğünde exception fırlatmaz, bu doğru.
            # Indexleme veya np.mean gibi işlemlerden exception gelebilir.
            if is_valid_processed_inputs:
                 # Ses Enerjisi Kontrolü
                 audio_features = processed_inputs.get('audio')
                 # check_numpy_input dtype kontrolü zaten yapıyor.
                 if isinstance(audio_features, np.ndarray) and \
                    check_numpy_input(audio_features, expected_dtype=np.number, expected_ndim=1, input_name="processed_inputs['audio']", logger_instance=logger) and \
                    audio_features.shape[0] >= 1: # size>0 kontrolü gibi audio_features.shape[0] >= 1
                      audio_energy = float(audio_features[0]) # Indexleme hatası olabilir (boş array)
                      if audio_energy >= self.audio_energy_threshold:
                           understanding_signals['high_audio_energy'] = True
                           logger.debug(f"UnderstandingModule.process: Yüksek ses enerjisi Algılandı: {audio_energy:.4f} >= Eşik {self.audio_energy_threshold:.4f}")


                 # Görsel Kenar Yoğunluğu Kontrolü
                 visual_features = processed_inputs.get('visual')
                 if isinstance(visual_features, dict):
                      edges_data = visual_features.get('edges')
                      # check_numpy_input dtype kontrolü zaten yapıyor.
                      if isinstance(edges_data, np.ndarray) and \
                         check_numpy_input(edges_data, expected_dtype=np.number, expected_ndim=(1,2), input_name="processed_inputs['visual']['edges']", logger_instance=logger) and \
                         edges_data.size > 0: # size > 0 kontrolü mean için önemli
                           visual_edges_mean = np.mean(edges_data) # mean boş arrayde hata verebilir
                           if visual_edges_mean >= self.visual_edges_threshold:
                                understanding_signals['high_visual_edges'] = True
                                # logger.debug(...)


                 # Görsel Parlaklık/Karanlık Controlü
                 if isinstance(visual_features, dict):
                      grayscale_data = visual_features.get('grayscale')
                      # check_numpy_input dtype kontrolü zaten yapıyor.
                      if isinstance(grayscale_data, np.ndarray) and \
                         check_numpy_input(grayscale_data, expected_dtype=np.number, expected_ndim=(1,2), input_name="processed_inputs['visual']['grayscale']", logger_instance=logger) and \
                         grayscale_data.size > 0: # size > 0 kontrolü mean için önemli
                           visual_brightness_mean = np.mean(grayscale_data) # mean boş arrayde hata verebilir

                           if visual_brightness_mean >= self.brightness_threshold_high:
                                understanding_signals['is_bright'] = True
                                # logger.debug(...)
                           elif visual_brightness_mean <= self.brightness_threshold_low:
                                understanding_signals['is_dark'] = True
                                # logger.debug(...)


            logger.debug(f"UnderstandingModule.process: Üretilen anlama sinyalleri: {understanding_signals}")

        except Exception as e:
            # Hesaplama veya girdi işleme sırasında beklenmedik bir hata olursa logla ve varsayılanları döndür.
            # check_* fonksiyonları False döndüğünde burası TETİKLENMEZ.
            # Sadece numpy işlemleri (norm, dot, mean) veya indexleme hatalarında tetiklenir.
            logger.error(f"UnderstandingModule.process: Anlama işlemi sırasında beklenmedik hata: {e}", exc_info=True)
            # Hata durumunda varsayılan anlama sinyallerini döndür
            return {
                'similarity_score': 0.0,
                'high_audio_energy': False,
                'high_visual_edges': False,
                'is_bright': False,
                'is_dark': False,
                'max_concept_similarity': 0.0,
                'most_similar_concept_id': None,
            }

        return understanding_signals

    # ... (cleanup metodu) ...
    def cleanup(self):
        """
        UnderstandingModule kaynaklarını temizler.

        Şimdilik özel bir kaynak kullanmadığı için temizleme adımı içermez.
        Gelecekte model temizliği veya bağlantı kapatma gerekebilir.
        module_loader.py bu metotu program sonlanırken çağırır (varsa).
        """
        logger.info("UnderstandingModule objesi temizleniyor.")
        pass