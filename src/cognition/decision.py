# src/cognition/decision.py
#
# Evo'nın karar alma modülünü temsil eder.
# Anlama modülünden gelen sinyalleri, bellek girdilerini ve içsel durumu kullanarak bir eylem kararı alır.

import logging
import numpy as np
import random

# Yardımcı fonksiyonları import et (girdi kontrolleri ve config için)
# check_* fonksiyonları src/core/utils'tan gelir
# get_config_value src/core/config_utils'tan gelir
# Bu importların başarılı olduğu varsayılır (config_utils workaround sonrası).
from src.core.utils import check_input_not_none, check_input_type
from src.core.config_utils import get_config_value

logger = logging.getLogger(__name__)

class DecisionModule:
    """
    Evo'nın denetimsiz öğrenme (kavram keşfi) yeteneğini sağlayan sınıf (Faz 4 implementasyonu).
    ... (Docstring aynı) ...
    """
    def __init__(self, config):
        """
        DecisionModule'ü başlatır.
        ... (Docstring aynı) ...
        """
        self.config = config
        logger.info("DecisionModule initializing (Phase 3/4)...")

        # Get thresholds and curiosity settings from config using get_config_value.
        # Corrected: Use default= keyword format for all calls.
        # Based on config, these settings are under the 'cognition' key, not under a 'decision' sub-key.
        self.familiarity_threshold = get_config_value(config, 'cognition', 'familiarity_threshold', default=0.8, expected_type=(float, int), logger_instance=logger)
        self.audio_energy_threshold = get_config_value(config, 'cognition', 'audio_energy_threshold', default=1000.0, expected_type=(float, int), logger_instance=logger)
        self.visual_edges_threshold = get_config_value(config, 'cognition', 'visual_edges_threshold', default=50.0, expected_type=(float, int), logger_instance=logger)
        self.brightness_threshold_high = get_config_value(config, 'cognition', 'brightness_threshold_high', default=200.0, expected_type=(float, int), logger_instance=logger)
        self.brightness_threshold_low = get_config_value(config, 'cognition', 'brightness_threshold_low', default=50.0, expected_type=(float, int), logger_instance=logger)
        self.concept_recognition_threshold = get_config_value(config, 'cognition', 'concept_recognition_threshold', default=0.85, expected_type=(float, int), logger_instance=logger)
        self.curiosity_threshold = get_config_value(config, 'cognition', 'curiosity_threshold', default=5.0, expected_type=(float, int), logger_instance=logger)
        self.curiosity_increment_new = get_config_value(config, 'cognition', 'curiosity_increment_new', default=1.0, expected_type=(float, int), logger_instance=logger)
        self.curiosity_decrement_familiar = get_config_value(config, 'cognition', 'curiosity_decrement_familiar', default=0.5, expected_type=(float, int), logger_instance=logger)
        self.curiosity_decay = get_config_value(config, 'cognition', 'curiosity_decay', default=0.1, expected_type=(float, int), logger_instance=logger)


        # Simple value checks for thresholds (0.0-1.0 range for similarity, non-negative for others)
        # get_config_value should return correct defaults now. Perform our own range checks on the values obtained.
        # Cast to float for comparison robustness.
        if not (0.0 <= float(self.familiarity_threshold) <= 1.0):
             logger.warning(f"DecisionModule: Config 'familiarity_threshold' out of expected range ({self.familiarity_threshold}). Using default 0.8.")
             self.familiarity_threshold = 0.8
        if float(self.audio_energy_threshold) < 0.0:
             logger.warning(f"DecisionModule: Config 'audio_energy_threshold' is negative ({self.audio_energy_threshold}). Using default 1000.0.")
             self.audio_energy_threshold = 1000.0
        if float(self.visual_edges_threshold) < 0.0:
             logger.warning(f"DecisionModule: Config 'visual_edges_threshold' is negative ({self.visual_edges_threshold}). Using default 50.0.")
             self.visual_edges_threshold = 50.0
        # Brightness threshold check: Must be positive and low < high
        if float(self.brightness_threshold_high) < 0.0:
             logger.warning(f"DecisionModule: Config 'brightness_threshold_high' is negative ({self.brightness_threshold_high}). Using default 200.0.")
             self.brightness_threshold_high = 200.0
        if float(self.brightness_threshold_low) < 0.0:
             logger.warning(f"DecisionModule: Config 'brightness_threshold_low' is negative ({self.brightness_threshold_low}). Using default 50.0.")
             self.brightness_threshold_low = 50.0
        # After previous checks, check the current values
        if float(self.brightness_threshold_low) >= float(self.brightness_threshold_high):
             logger.warning(f"DecisionModule: Config 'brightness_threshold_low' ({self.brightness_threshold_low}) is greater than or equal to 'brightness_threshold_high' ({self.brightness_threshold_high}). Using defaults 50.0 and 200.0.")
             self.brightness_threshold_low = 50.0
             self.brightness_threshold_high = 200.0 # Both low and high reset to ensure low < high.

        if not (0.0 <= float(self.concept_recognition_threshold) <= 1.0):
             logger.warning(f"DecisionModule: Config 'concept_recognition_threshold' out of expected range ({self.concept_recognition_threshold}). Expected between 0.0 and 1.0. Using default 0.85.")
             self.concept_recognition_threshold = 0.85
        # Curiosity threshold and update amounts must not be negative
        if float(self.curiosity_threshold) < 0.0:
             logger.warning(f"DecisionModule: Config 'curiosity_threshold' is negative ({self.curiosity_threshold}). Using default 5.0.")
             self.curiosity_threshold = 5.0
        if float(self.curiosity_increment_new) < 0.0:
             logger.warning(f"DecisionModule: Config 'curiosity_increment_new' is negative ({self.curiosity_increment_new}). Using default 1.0.")
             self.curiosity_increment_new = 1.0
        if float(self.curiosity_decrement_familiar) < 0.0:
             logger.warning(f"DecisionModule: Config 'curiosity_decrement_familiar' is negative ({self.curiosity_decrement_familiar}). Using default 0.5.")
             self.curiosity_decrement_familiar = 0.5
        # Curiosity decay must not be negative
        if float(self.curiosity_decay) < 0.0:
             logger.warning(f"DecisionModule: Config 'curiosity_decay' is negative ({self.curiosity_decay}). Using default 0.1.")
             self.curiosity_decay = 0.1


        # Internal state variables (for now, just curiosity level)
        self.curiosity_level = 0.0 # Curiosity level (starts at 0.0).

        logger.info(f"DecisionModule initialized. Familiarity Threshold: {self.familiarity_threshold}, Audio Threshold: {self.audio_energy_threshold}, Visual Threshold: {self.visual_edges_threshold}, Brightness High Threshold: {self.brightness_threshold_high}, Brightness Low Threshold: {self.brightness_threshold_low}, Concept Recognition Threshold: {self.concept_recognition_threshold}, Curiosity Threshold: {self.curiosity_threshold}")
        logger.debug(f"DecisionModule: Curiosity Increment (New): {self.curiosity_increment_new}, Decrement (Familiar): {self.curiosity_decrement_familiar}, Decay: {self.curiosity_decay}")

    # ... (decide and cleanup methods - same as before) ...

    def decide(self, understanding_signals, relevant_memory_entries, internal_state=None):
        """
        Anlama sinyallerine ve içsel duruma (curiosity_level) göre bir eylem kararı alır.
        ... (Docstring aynı) ...
        """
        # Girdi kontrolleri. understanding_signals'ın geçerli bir dictionary mi?
        if not check_input_not_none(understanding_signals, input_name="understanding_signals for DecisionModule", logger_instance=logger):
             logger.debug("DecisionModule.decide: understanding_signals None. Karar alınamıyor.")
             # Merak seviyesi None durumda güncellenmez.
             return None

        if not isinstance(understanding_signals, dict):
             logger.error(f"DecisionModule.decide: understanding_signals beklenmeyen tipte: {type(understanding_signals)}. Dict veya None bekleniyordu. Karar alınamıyor.")
             # Merak seviyesi dict olmayan durumda güncellenmez.
             return None


        decision = None # Alınan kararı tutacak değişken.

        # Anlama sinyallerini dictionary'den güvenle al. Anahtarlar yoksa default değerler kullanılır.
        # get() metodu anahtar yoksa None veya belirtilen default değeri döndürür.
        # Bu değerlerin tipi get_config_value ile alınanlar gibi garanti değildir, bu yüzden karar mantığında tip kontrolü yapalım.
        # get() metodu ile alınan değerlerin None veya yanlış tipte olma ihtimaline karşı karar mantığı sağlam olmalı.
        similarity_score = understanding_signals.get('similarity_score', 0.0)
        high_audio_energy = understanding_signals.get('high_audio_energy', False)
        high_visual_edges = understanding_signals.get('high_visual_edges', False)
        is_bright = understanding_signals.get('is_bright', False)
        is_dark = understanding_signals.get('is_dark', False)
        max_concept_similarity = understanding_signals.get('max_concept_similarity', 0.0)
        most_similar_concept_id = understanding_signals.get('most_similar_concept_id', None)


        # Gelen sinyallerin tiplerini doğrulayalım veya karar mantığında defansif olalım.
        # Bool beklenenler (high_audio_energy, high_visual_edges, is_bright, is_dark) get() ile False dönerse sorun olmaz.
        # Sayısal beklenenler (similarity_score, max_concept_similarity) için karar mantığında kontrol yapalım.
        # most_similar_concept_id None veya int olabilir.

        logger.debug(f"DecisionModule.decide: Anlama sinyalleri alindi - Sim:{similarity_score:.4f}, Audio:{high_audio_energy}, Visual:{high_visual_edges}, Bright:{is_bright}, Dark:{is_dark}, ConceptSim:{max_concept_similarity:.4f}, ConceptID:{most_similar_concept_id}. Mevcut Merak: {self.curiosity_level:.2f}. Karar veriliyor.")

        # Hangi temel bellek/benzerlik durumu oluştuğunu belirle (merak güncellemesi için de ipucu verir)
        # Bu, process sinyalleri ve kavram tanıma OVERRIDE etmeden önceki temel durumdur.
        # similarity_score sayısal ve eşiğe eşit veya üstünde ise tanıdık.
        is_fundamentally_familiar = False
        if isinstance(similarity_score, (int, float)) and similarity_score >= self.familiarity_threshold:
             is_fundamentally_familiar = True

        is_fundamentally_new = not is_fundamentally_familiar # Temel durum ya tanıdıktır ya yenidir.


        try:
            # Karar Alma Mantığı (Faz 3/4): Öncelikli Mantık
            # Öncelik Sırası: Merak > Ses > Görsel Kenar > Parlaklık/Karanlık > Kavram Tanıma > Bellek Tanıdıklığı > Varsayılan (Yeni)

            # 1. Merak Eşiği Aşıldı mı? (En yüksek öncelik)
            # self.curiosity_level'ın sayısal olduğunu kontrol et.
            if isinstance(self.curiosity_level, (int, float)) and float(self.curiosity_level) >= float(self.curiosity_threshold):
                 # Merak eşiği aşıldysa rastgele bir keşif/sinyal kararı ver
                 decision = random.choice(["explore_randomly", "make_noise"])
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Merak eşiği ({self.curiosity_level:.2f} >= {self.curiosity_threshold:.2f}) aşıldı.")

            # 2. Yüksek ses enerjisi var mı? (İkinci öncelik)
            elif isinstance(high_audio_energy, bool) and high_audio_energy: # high_audio_energy bool mu kontrol et
                 decision = "sound_detected"
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Yüksek ses enerjisi algılandı.")

            # 3. Yüksek görsel kenar yoğunluğu var mı? (Üçüncü öncelik)
            elif isinstance(high_visual_edges, bool) and high_visual_edges: # high_visual_edges bool mu kontrol et
                 decision = "complex_visual_detected"
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Yüksek görsel kenar yoğunluğu algılandı.")

            # 4. Ortam Parlak mı veya Karanlık mı? (Dördüncü öncelik)
            elif isinstance(is_bright, bool) and is_bright: # is_bright bool mu kontrol et
                 decision = "bright_light_detected"
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Ortam parlak algılandı.")

            elif isinstance(is_dark, bool) and is_dark: # is_dark bool mu kontrol et
                 decision = "dark_environment_detected"
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Ortam karanlık algılandı.")

            # 5. Kavram Tanındı mı? (Beşinci öncelik)
            # Kavram benzerlik skoru sayısal mı, ID None değil mi ve benzerlik eşiği aşıyor mu?
            elif isinstance(max_concept_similarity, (int, float)) and float(max_concept_similarity) >= float(self.concept_recognition_threshold) and most_similar_concept_id is not None:
                 # most_similar_concept_id'nin int olduğundan emin olalım (f-string içinde int() kullanılıyordu)
                 if isinstance(most_similar_concept_id, int):
                      decision = f"recognized_concept_{most_similar_concept_id}"
                      logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Kavram tanındı (Benzerlik: {max_concept_similarity:.4f} >= Eşik {self.concept_recognition_threshold:.4f}, ID: {most_similar_concept_id}).")
                 else:
                      # ID geçerli tipte değilse kavram tanıma kararı verilmez, bir sonraki önceliğe düşer.
                      logger.debug(f"DecisionModule.decide: Kavram tanıma benzerliği yüksek ({max_concept_similarity:.4f}) ancak ConceptID geçerli int değil ({type(most_similar_concept_id)}). Kavram tanıma atlandi.")


            # 6. Bellek benzerlik skoru eşiği aşıyor mu? (Altıncı öncelik)
            # is_fundamentally_familiar değişkeni en başta doğru hesaplandı.
            # Bu kontrol, ancak yukarıdaki Process/Kavram tanıma sinyalleri yoksa yapılır.
            elif is_fundamentally_familiar:
                 decision = "familiar_input_detected"
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Bellek benzerlik skoru ({similarity_score:.4f}) >= Eşik ({self.familiarity_threshold:.4f}).")


            # 7. Hiçbir öncelikli koşul sağlanmazsa (Varsayılan)
            # Bu durumda temel durum is_fundamentally_new olmalıdır.
            # else: # is_fundamentally_new'e düşer.
            #     decision = "new_input_detected"
            #     logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Hiçbir öncelikli durum algılanamadı (Temel Durum: Yeni).")

            # Daha net bir fallback: Eğer yukarıdaki elif zinciri decision'ı hala None bırakmışsa
            if decision is None:
                 decision = "new_input_detected" # Varsayılan karar
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Hiçbir öncelikli durum algılanamadı.")


            # Gelecekte daha karmaşık mantıklar:
            # - Başka anlama çıktılarından gelen bilgileri karara dahil etme.
            # - Birden fazla kriteri birleştirme (örn: hem benzerlik yüksek hem de anlama çıktısı spesifik bir nesne algıladı).
            # - Başka içsel durumları karara dahil etme (boredom, hunger vb.).
            # - Farklı karar türleri (örn: "hareket et", "ses çıkar").

        except Exception as e:
            # Karar alma işlemi sırasında beklenmedik bir hata olursa logla.
            logger.error(f"DecisionModule.decide: Karar alma sırasında beklenmedik hata: {e}", exc_info=True)
            # Merak seviyesi hata durumunda güncellenmez (finally bloğunda kontrol ediliyor).
            return None # Hata durumunda None döndür.

        finally:
            # --- Merak Seviyesini Güncelle ---
            # Sadece Karar başarılı bir şekilde belirlendiyse (decision is not None) merakı güncelle.
            # curiosity_level'ın sayısal olduğundan emin ol.
            if decision is not None and isinstance(self.curiosity_level, (int, float)):
                try:
                    curiosity_before_decay = float(self.curiosity_level)

                    # Merak seviyesini ALINAN karara göre artır/azalt
                    # Process sinyalleri veya Merak eşiği kararları (explore_randomly, make_noise) merak seviyesini sadece decay ettirir.
                    # Sadece "new" veya "familiar/recognized" kararları inc/dec uygular.
                    # Dikkat: Kararlar string karşılaştırmasıyla kontrol ediliyor.
                    if decision == "new_input_detected" or decision == "new_input_detected_fallback":
                         curiosity_before_decay += self.curiosity_increment_new
                         logger.debug(f"DecisionModule: Merak artışı ({self.curiosity_increment_new:.2f}) kararı: '{decision}'.")
                    elif decision == "familiar_input_detected" or (isinstance(decision, str) and decision.startswith("recognized_concept_")):
                         curiosity_before_decay -= self.curiosity_decrement_familiar
                         logger.debug(f"DecisionModule: Merak azalışı ({self.curiosity_decrement_familiar:.2f}) kararı: '{decision}'.")
                    else:
                         # Ses, Görsel Kenar, Parlak/Karanlık, Explore/Noise kararları. Merakı artırmaz/azaltmaz, sadece decay olur.
                         # curiosity_before_decay zaten güncellenmedi.
                         logger.debug(f"DecisionModule: Merak değişimi yok (sadece decay). Karar: '{decision}'.")


                    # Her döngü adımında merak seviyesini azalt (decay).
                    self.curiosity_level = max(0.0, curiosity_before_decay - float(self.curiosity_decay)) # Decay uygula

                    # Güncel merak seviyesini logla.
                    logger.debug(f"DecisionModule: Güncel Merak Seviyesi: {self.curiosity_level:.2f}")

                except Exception as e:
                     logger.error(f"DecisionModule: Merak seviyesi güncellenirken beklenmedik hata: {e}", exc_info=True)
            # else: decision None idi, merak güncellenmedi.


        return decision # Alınan karar stringi veya None döndürülür.

    def cleanup(self):
        """
        DecisionModule kaynaklarını temizler.
        ... (Docstring aynı) ...
        """
        logger.info("DecisionModule objesi temizleniyor.")
        # TODO: İçsel durumun kaydedilmesi buraya gelebilir (gelecek TODO).
        pass