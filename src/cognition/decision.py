# src/cognition/decision.py
#
# Evo's decision making module.
# Makes an action decision using signals from the understanding module, memory entries, and internal state.

import logging
import numpy as np # np.floating için ve potansiyel diğer NumPy işlemleri için
import random

# Import utility functions
from src.core.utils import check_input_not_none 
# check_input_type artık bu modülde doğrudan kullanılmıyor gibi görünüyor.
from src.core.config_utils import get_config_value

logger = logging.getLogger(__name__)

class DecisionModule:
    """
    Evo's decision making capability class (Phase 3/4 implementation).

    Receives signals from the Understanding module, memory entries, and internal state (curiosity level).
    Applies priority logic to these signals to select an action decision.
    Updates its internal state (e.g., curiosity level) based on the decision made.
    """
    def __init__(self, config):
        """
        Initializes the DecisionModule.

        Reads configuration settings for decision thresholds and curiosity dynamics.

        Args:
            config (dict): Cognitive core configuration settings (full config dict).
                           DecisionModule will read its relevant section from this dict,
                           specifically settings under 'cognition'.
        """
        self.config = config 
        logger.info("DecisionModule initializing (Phase 3/4)...")

        # Config'den değerleri oku ve Python float'ına çevir.
        # np.floating, config dosyasından direkt NumPy float türü gelirse diye eklenmiştir,
        # ancak YAML'den okurken genelde standart Python türleri gelir. Yine de zararı olmaz.
        self.familiarity_threshold = float(get_config_value(config, 'cognition', 'familiarity_threshold', default=0.8, expected_type=(float, int, np.floating), logger_instance=logger))
        self.audio_energy_threshold = float(get_config_value(config, 'cognition', 'audio_energy_threshold', default=1000.0, expected_type=(float, int, np.floating), logger_instance=logger))
        self.visual_edges_threshold = float(get_config_value(config, 'cognition', 'visual_edges_threshold', default=50.0, expected_type=(float, int, np.floating), logger_instance=logger))
        self.brightness_threshold_high = float(get_config_value(config, 'cognition', 'brightness_threshold_high', default=200.0, expected_type=(float, int, np.floating), logger_instance=logger))
        self.brightness_threshold_low = float(get_config_value(config, 'cognition', 'brightness_threshold_low', default=50.0, expected_type=(float, int, np.floating), logger_instance=logger))
        self.concept_recognition_threshold = float(get_config_value(config, 'cognition', 'concept_recognition_threshold', default=0.85, expected_type=(float, int, np.floating), logger_instance=logger))
        self.curiosity_threshold = float(get_config_value(config, 'cognition', 'curiosity_threshold', default=5.0, expected_type=(float, int, np.floating), logger_instance=logger))
        self.curiosity_increment_new = float(get_config_value(config, 'cognition', 'curiosity_increment_new', default=1.0, expected_type=(float, int, np.floating), logger_instance=logger))
        self.curiosity_decrement_familiar = float(get_config_value(config, 'cognition', 'curiosity_decrement_familiar', default=0.5, expected_type=(float, int, np.floating), logger_instance=logger))
        self.curiosity_decay = float(get_config_value(config, 'cognition', 'curiosity_decay', default=0.1, expected_type=(float, int, np.floating), logger_instance=logger))

        # Eşik değerleri için basit kontroller
        if not (0.0 <= self.familiarity_threshold <= 1.0):
             logger.warning(f"DecisionModule: Config 'familiarity_threshold' ({self.familiarity_threshold}) out of [0,1] range. Using 0.8.")
             self.familiarity_threshold = 0.8
        if self.audio_energy_threshold < 0.0:
             logger.warning(f"DecisionModule: Config 'audio_energy_threshold' ({self.audio_energy_threshold}) is negative. Using 1000.0.")
             self.audio_energy_threshold = 1000.0
        if self.visual_edges_threshold < 0.0:
             logger.warning(f"DecisionModule: Config 'visual_edges_threshold' ({self.visual_edges_threshold}) is negative. Using 50.0.")
             self.visual_edges_threshold = 50.0
        if self.brightness_threshold_high < 0.0:
             logger.warning(f"DecisionModule: Config 'brightness_threshold_high' ({self.brightness_threshold_high}) is negative. Using 200.0.")
             self.brightness_threshold_high = 200.0
        if self.brightness_threshold_low < 0.0:
             logger.warning(f"DecisionModule: Config 'brightness_threshold_low' ({self.brightness_threshold_low}) is negative. Using 50.0.")
             self.brightness_threshold_low = 50.0
        if self.brightness_threshold_low >= self.brightness_threshold_high:
             logger.warning(f"DecisionModule: Config 'brightness_threshold_low' ({self.brightness_threshold_low}) >= 'high' ({self.brightness_threshold_high}). Using defaults 50.0, 200.0.")
             self.brightness_threshold_low = 50.0
             self.brightness_threshold_high = 200.0
        if not (0.0 <= self.concept_recognition_threshold <= 1.0):
             logger.warning(f"DecisionModule: Config 'concept_recognition_threshold' ({self.concept_recognition_threshold}) out of [0,1] range. Using 0.85.")
             self.concept_recognition_threshold = 0.85
        if self.curiosity_threshold < 0.0:
             logger.warning(f"DecisionModule: Config 'curiosity_threshold' ({self.curiosity_threshold}) is negative. Using 5.0.")
             self.curiosity_threshold = 5.0
        if self.curiosity_increment_new < 0.0:
             logger.warning(f"DecisionModule: Config 'curiosity_increment_new' ({self.curiosity_increment_new}) is negative. Using 1.0.")
             self.curiosity_increment_new = 1.0
        if self.curiosity_decrement_familiar < 0.0:
             logger.warning(f"DecisionModule: Config 'curiosity_decrement_familiar' ({self.curiosity_decrement_familiar}) is negative. Using 0.5.")
             self.curiosity_decrement_familiar = 0.5
        if self.curiosity_decay < 0.0:
             logger.warning(f"DecisionModule: Config 'curiosity_decay' ({self.curiosity_decay}) is negative. Using 0.1.")
             self.curiosity_decay = 0.1

        self.curiosity_level = 0.0 

        logger.info(f"DecisionModule initialized. Familiarity Threshold: {self.familiarity_threshold:.2f}, Concept Rec Threshold: {self.concept_recognition_threshold:.2f}, Curiosity Threshold: {self.curiosity_threshold:.2f}")
        logger.debug(f"Curiosity - Increment: {self.curiosity_increment_new:.2f}, Decrement: {self.curiosity_decrement_familiar:.2f}, Decay: {self.curiosity_decay:.2f}")

    def decide(self, understanding_signals, relevant_memory_entries, current_concepts):
        if not check_input_not_none(understanding_signals, "understanding_signals for DecisionModule", logger):
             # Merak seviyesi güncellenmeli (finally bloğunda)
             decision = None # Bu, finally bloğunun doğru çalışmasını sağlar
             # return None # Erken return yerine finally'ye gitmeli
        elif not isinstance(understanding_signals, dict):
             logger.error(f"DecisionModule.decide: understanding_signals has unexpected type: {type(understanding_signals)}. Dictionary or None expected.")
             decision = None
             # return None
        else: # understanding_signals geçerli bir dict veya None (None durumu yukarıda halledildi)
            decision = None 

            similarity_score_input = understanding_signals.get('similarity_score', 0.0)
            high_audio_energy = understanding_signals.get('high_audio_energy', False)
            high_visual_edges = understanding_signals.get('high_visual_edges', False)
            is_bright = understanding_signals.get('is_bright', False)
            is_dark = understanding_signals.get('is_dark', False)
            max_concept_similarity_input = understanding_signals.get('max_concept_similarity', 0.0)
            most_similar_concept_id = understanding_signals.get('most_similar_concept_id', None)
            
            # Gelen değerleri Python float'ına çevir
            similarity_score = 0.0
            if isinstance(similarity_score_input, (int, float, np.floating)): 
                similarity_score = float(similarity_score_input)
            else: # None veya beklenmedik tür
                if similarity_score_input is not None: # Sadece None değilse uyar
                    logger.warning(f"DecisionModule.decide: 'similarity_score' received non-convertible value {similarity_score_input} (type: {type(similarity_score_input)}). Defaulting to 0.0.")
                # Zaten 0.0 olarak başlatıldı

            max_concept_similarity = 0.0
            if isinstance(max_concept_similarity_input, (int, float, np.floating)):
                max_concept_similarity = float(max_concept_similarity_input)
            else: # None veya beklenmedik tür
                if max_concept_similarity_input is not None: # Sadece None değilse uyar
                    logger.warning(f"DecisionModule.decide: 'max_concept_similarity' received non-convertible value {max_concept_similarity_input} (type: {type(max_concept_similarity_input)}). Defaulting to 0.0.")
                # Zaten 0.0 olarak başlatıldı

            logger.debug(f"DecisionModule.decide: Signals - Sim:{similarity_score:.4f}, Audio:{high_audio_energy}, Visual:{high_visual_edges}, Bright:{is_bright}, Dark:{is_dark}, ConceptSim:{max_concept_similarity:.4f}, ConceptID:{most_similar_concept_id}. Curiosity: {self.curiosity_level:.2f}.")

            is_fundamentally_familiar = (similarity_score >= self.familiarity_threshold)

            try:
                # Öncelik sırasına göre karar verme mantığı
                if self.curiosity_level >= self.curiosity_threshold:
                    decision_options = ["explore_surroundings", "make_a_random_sound", "focus_on_new_detail"]
                    decision = random.choice(decision_options)
                    logger.debug(f"Decision: '{decision}' (Curiosity high: {self.curiosity_level:.2f} >= {self.curiosity_threshold:.2f})")
                elif isinstance(high_audio_energy, bool) and high_audio_energy:
                    decision = "react_to_loud_sound"
                    logger.debug(f"Decision: '{decision}' (High audio energy detected)")
                elif isinstance(high_visual_edges, bool) and high_visual_edges:
                    decision = "examine_complex_visual"
                    logger.debug(f"Decision: '{decision}' (High visual edge density detected)")
                elif isinstance(is_bright, bool) and is_bright:
                    decision = "acknowledge_bright_light"
                    logger.debug(f"Decision: '{decision}' (Environment detected as bright)")
                elif isinstance(is_dark, bool) and is_dark:
                    decision = "acknowledge_darkness"
                    logger.debug(f"Decision: '{decision}' (Environment detected as dark)")
                elif max_concept_similarity >= self.concept_recognition_threshold and most_similar_concept_id is not None:
                    # Kavram ID'sinin türü önemli (int, str vb.)
                    decision = f"interact_with_concept_{most_similar_concept_id}"
                    logger.debug(f"Decision: '{decision}' (Concept recognized: Sim {max_concept_similarity:.4f} >= Thres {self.concept_recognition_threshold:.4f}, ID: {most_similar_concept_id})")
                elif is_fundamentally_familiar:
                    decision = "observe_familiar_input"
                    logger.debug(f"Decision: '{decision}' (Memory similarity high: {similarity_score:.4f} >= Thres {self.familiarity_threshold:.4f})")
                
                if decision is None: # Eğer yukarıdaki koşulların hiçbiri karşılanmadıysa
                    decision = "perceive_new_stimulus" # Daha genel bir "yeni" kararı
                    logger.debug(f"Decision: '{decision}' (Default for new/unprioritized input)")

            except Exception as e: # Karar verme mantığı içinde bir hata olursa
                logger.error(f"DecisionModule.decide: Error during decision logic: {e}", exc_info=True)
                decision = None # Hata durumunda karar None olur

        # `finally` bloğu, `try` bloğundan bir `return` olsa bile çalışır.
        # `decide` metodunun sonunda merak seviyesini her zaman güncelle.
        try:
            curiosity_change = 0.0
            if decision == "perceive_new_stimulus" or decision == "explore_surroundings" or decision == "focus_on_new_detail": # "Yeni" veya "keşif" ile ilgili kararlar
                curiosity_change = self.curiosity_increment_new
            elif decision == "observe_familiar_input" or (isinstance(decision, str) and decision.startswith("interact_with_concept_")): # "Tanıdık" veya "kavram" ile ilgili kararlar
                curiosity_change = -self.curiosity_decrement_familiar # Negatif olacak
            
            self.curiosity_level += curiosity_change
            self.curiosity_level -= self.curiosity_decay # Her döngüde sabit azalma
            self.curiosity_level = max(0.0, self.curiosity_level) # Merak seviyesi negatif olamaz

            if decision is not None:
                 logger.debug(f"Updated Curiosity Level to: {self.curiosity_level:.2f} (Decision: '{decision}')")
            else: # Eğer karar None ise (örn: understanding_signals None geldi veya hata oluştu)
                 logger.debug(f"Decision was None or understanding failed. Curiosity Level updated to: {self.curiosity_level:.2f}")
        
        except Exception as e:
            logger.error(f"DecisionModule: Error updating curiosity level: {e}", exc_info=True)
            # Merak seviyesi güncellenemese bile, alınan kararı döndür.

        return decision

    def cleanup(self):
        """
        Cleans up DecisionModule resources.
        Currently, this module does not use specific resources that require explicit cleanup.
        """
        logger.info("DecisionModule object cleaning up.")
        # Gelecekte, eğer dosya veya veritabanı bağlantıları gibi kaynaklar kullanılırsa,
        # burada kapatılmaları/temizlenmeleri gerekir.
        pass