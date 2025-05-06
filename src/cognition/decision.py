# src/cognition/decision.py
#
# Evo'nın karar alma modülünü temsil eder.
# Anlama modülünden gelen sinyalleri, bellek girdilerini ve içsel durumu kullanarak bir eylem kararı alır.

import logging
import numpy as np # Sayısal işlemler ve type checking için
import random # Merak kararları için

# Yardımcı fonksiyonları import et (girdi kontrolleri ve config için)
from src.core.utils import check_input_not_none, check_input_type, get_config_value # <<< Utils importları


# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

class DecisionModule:
    """
    Evo'nın karar alma yeteneğini sağlayan sınıf (Faz 3/4 implementasyonu).

    UnderstandingModule'den gelen anlama sinyallerini (dictionary), bellek girdilerini
    ve içsel durumu (curiosity_level - merak seviyesi) alır. Bu bilgilere dayanarak,
    MotorControl modülüne iletilecek bir eylem kararı alır.
    Mevcut implementasyon: Process çıktısı (enerji/kenar/parlaklık), merak seviyesi,
    öğrenilmiş kavram tanıma VE bellek benzerlik skorına dayalı öncelikli bir karar verme mantığı uygular.
    Gelecekte daha karmaşık karar ağaçları, kural tabanlı sistemler veya
    öğrenilmiş karar modelleri implement edilecektir.
    """
    def __init__(self, config):
        """
        DecisionModule'ü başlatır.

        Args:
            config (dict): Karar alma modülü yapılandırma ayarları.
                           'familiarity_threshold': Bellek benzerliğine dayalı "tanıdık" eşiği (float, varsayılan 0.8).
                           'audio_energy_threshold': Yüksek ses enerjisine dayalı karar eşiği (float, varsayılan 1000.0).
                           'visual_edges_threshold': Yüksek görsel kenar yoğunluğuna dayalı karar eşiği (float, varsayılan 50.0).
                           'brightness_threshold_high': Parlak ortam algılama eşiği (float, varsayılan 200.0).
                           'brightness_threshold_low': Karanlık ortam algılama eşiği (float, varsayılan 50.0).
                           'concept_recognition_threshold': Kavram tanıma için Representation'ın kavram temsilcisine
                                                           en az benzemesi gereken benzerlik eşiği (float, varsayılan 0.85).
                           'curiosity_threshold': Merak seviyesinin keşif/sinyal kararı için ulaşması gereken eşik (float, varsayılan 5.0).
                           'curiosity_increment_new': "Yeni" input algılandığında merak seviyesi artış miktarı (float, varsayılan 1.0).
                           'curiosity_decrement_familiar': "Tanıdık" input algılandığında merak seviyesi azalış miktarı (float, varsayılan 0.5).
                           'curiosity_decay': Her döngü adımında merak seviyesinin otomatik azalışı (float, varsayılan 0.1).
                           Gelecekte karar kuralları, eşikleri veya model yolları gelebilir.
        """
        self.config = config
        logger.info("DecisionModule başlatılıyor (Faz 3/4)...")

        # Yapılandırmadan eşikleri ve merak ayarlarını alırken get_config_value kullan.
        self.familiarity_threshold = get_config_value(config, 'familiarity_threshold', 0.8, expected_type=(float, int), logger_instance=logger)
        self.audio_energy_threshold = get_config_value(config, 'audio_energy_threshold', 1000.0, expected_type=(float, int), logger_instance=logger)
        self.visual_edges_threshold = get_config_value(config, 'visual_edges_threshold', 50.0, expected_type=(float, int), logger_instance=logger)
        self.brightness_threshold_high = get_config_value(config, 'brightness_threshold_high', 200.0, expected_type=(float, int), logger_instance=logger)
        self.brightness_threshold_low = get_config_value(config, 'brightness_threshold_low', 50.0, expected_type=(float, int), logger_instance=logger)
        self.concept_recognition_threshold = get_config_value(config, 'concept_recognition_threshold', 0.85, expected_type=(float, int), logger_instance=logger)
        # Merak ayarları - İngilizce isimler kullanıldı.
        self.curiosity_threshold = get_config_value(config, 'curiosity_threshold', 5.0, expected_type=(float, int), logger_instance=logger)
        self.curiosity_increment_new = get_config_value(config, 'curiosity_increment_new', 1.0, expected_type=(float, int), logger_instance=logger)
        self.curiosity_decrement_familiar = get_config_value(config, 'curiosity_decrement_familiar', 0.5, expected_type=(float, int), logger_instance=logger)
        self.curiosity_decay = get_config_value(config, 'curiosity_decay', 0.1, expected_type=(float, int), logger_instance=logger)


        # Eşik değerleri için basit değer kontrolü (0.0-1.0 arası benzerlik, negatif olmamalı diğerleri)
        if not (0.0 <= self.familiarity_threshold <= 1.0):
             logger.warning(f"DecisionModule: Konfig 'familiarity_threshold' beklenmeyen aralıkta ({self.familiarity_threshold}). Varsayılan 0.8 kullanılıyor.")
             self.familiarity_threshold = 0.8
        if self.audio_energy_threshold < 0.0:
             logger.warning(f"DecisionModule: Konfig 'audio_energy_threshold' negatif ({self.audio_energy_threshold}). Varsayılan 1000.0 kullanılıyor.")
             self.audio_energy_threshold = 1000.0
        if self.visual_edges_threshold < 0.0:
             logger.warning(f"DecisionModule: Konfig 'visual_edges_threshold' negatif ({self.visual_edges_threshold}). Varsayılan 50.0 kullanılıyor.")
             self.visual_edges_threshold = 50.0
        if self.brightness_threshold_high < 0.0:
             logger.warning(f"DecisionModule: Konfig 'brightness_threshold_high' negatif ({self.brightness_threshold_high}). Varsayılan 200.0 kullanılıyor.")
             self.brightness_threshold_high = 200.0
        if self.brightness_threshold_low < 0.0:
             logger.warning(f"DecisionModule: Konfig 'brightness_threshold_low' negatif ({self.brightness_threshold_low}). Varsayılan 50.0 kullanılıyor.")
             self.brightness_threshold_low = 50.0
        if self.brightness_threshold_low >= self.brightness_threshold_high:
             logger.warning(f"DecisionModule: Konfig 'brightness_threshold_low' ({self.brightness_threshold_low}) 'brightness_threshold_high'dan ({self.brightness_threshold_high}) büyük veya eşit. Eşikleri kontrol edin.")
        if not (0.0 <= self.concept_recognition_threshold <= 1.0):
             logger.warning(f"DecisionModule: Konfig 'concept_recognition_threshold' beklenmeyen aralıkta ({self.concept_recognition_threshold}). 0.0 ile 1.0 arası bekleniyordu. Varsayılan 0.85 kullanılıyor.")
             self.concept_recognition_threshold = 0.85
        # Merak eşiği ve güncelleme miktarları negatif olmamalı
        if self.curiosity_threshold < 0.0:
             logger.warning(f"DecisionModule: Konfig 'curiosity_threshold' negatif ({self.curiosity_threshold}). Varsayılan 5.0 kullanılıyor.")
             self.curiosity_threshold = 5.0
        if self.curiosity_increment_new < 0.0:
             logger.warning(f"DecisionModule: Konfig 'curiosity_increment_new' negatif ({self.curiosity_increment_new}). Varsayılan 1.0 kullanılıyor.")
             self.curiosity_increment_new = 1.0
        if self.curiosity_decrement_familiar < 0.0:
             logger.warning(f"DecisionModule: Konfig 'curiosity_decrement_familiar' negatif ({self.curiosity_decrement_familiar}). Varsayılan 0.5 kullanılıyor.")
             self.curiosity_decrement_familiar = 0.5
        # Merak decay negatif olmamalı
        if self.curiosity_decay < 0.0:
             logger.warning(f"DecisionModule: Konfig 'curiosity_decay' negatif ({self.curiosity_decay}). Varsayılan 0.1 kullanılıyor.")
             self.curiosity_decay = 0.1


        # İçsel durum değişkenleri (şimdilik sadece merak seviyesi)
        self.curiosity_level = 0.0 # Merak seviyesi (0.0 ile baslar). Gelecekte kalıcı hale getirilebilir.
        # Diğer içsel durumlar gelecekte buraya eklenecek (örn: energy_level, boredom_level)

        logger.info(f"DecisionModule başlatıldı. Tanıdıklık Eşiği: {self.familiarity_threshold}, Ses Eşiği: {self.audio_energy_threshold}, Görsel Eşiği: {self.visual_edges_threshold}, Parlaklık Yüksek Eşiği: {self.brightness_threshold_high}, Parlaklık Düşük Eşiği: {self.brightness_threshold_low}, Kavram Tanıma Eşiği: {self.concept_recognition_threshold}, Merak Eşiği: {self.curiosity_threshold}")
        logger.debug(f"DecisionModule: Merak Artış (Yeni): {self.curiosity_increment_new}, Azalış (Tanıdık): {self.curiosity_decrement_familiar}, Decay: {self.curiosity_decay}")


    # relevant_memory_entries hala burada duruyor (CognitionCore'dan geliyor)
    # internal_state parametresi şimdilik bu sınıfın içindeki self.curiosity_level'i temsil etmiyor,
    # genel bir placeholder. self.curiosity_level doğrudan burada yönetiliyor.
    def decide(self, understanding_signals, relevant_memory_entries, internal_state=None):
        """
        Anlama sinyallerine ve içsel duruma (curiosity_level) göre bir eylem kararı alır.

        UnderstandingModule'den gelen anlama sinyallerini (dictionary) alır.
        Process çıktısı tabanlı flag'lere, merak seviyesine, öğrenilmiş kavram tanıma
        ve bellek benzerlik skorına dayalı öncelikli bir karar verme mantığı uygular.
        Merak seviyesini günceller.

        Args:
            understanding_signals (dict or None): Anlama modülünden gelen anlama sinyalleri dictionary'si.
                                                Beklenen format: {'similarity_score': float, 'high_audio_energy': bool, 'high_visual_edges': bool, 'is_bright': bool, 'is_dark': bool, 'max_concept_similarity': float, 'most_similar_concept_id': int or None} veya None.
                                                UnderstandingModule hata durumunda varsayılan dict döndürmeyi hedefler.
            relevant_memory_entries (list or None): Memory modülünden gelen ilgili bellek girdileri listesi.
                                            Bu metotta doğrudan karar için kullanılmıyor, ama parametre olarak geliyor.
                                            Gelecekte bağlamsal karar için kullanılabilir.
            internal_state (dict, optional): Evo'nın içsel durumu (örn: energy_level, boredom_level). GELECEKTE KULLANILACAK. Şu an bu metot kendi içsel merak seviyesini yönetiyor. Varsayılan None.

        Returns:
            str or None: Alınan karar stringi (örn: "sound_detected", "complex_visual_detected", "bright_light_detected", "dark_environment_detected", "recognized_concept_X", "familiar_input_detected", "new_input_detected", "explore_randomly", "make_noise")
                         veya girdi (understanding_signals) geçersizse ya da hata durumunda None.
        """
        # Girdi kontrolleri. understanding_signals'ın geçerli bir dictionary mi?
        if not check_input_not_none(understanding_signals, input_name="understanding_signals for DecisionModule", logger_instance=logger):
             logger.debug("DecisionModule.decide: understanding_signals None. Karar alınamıyor.")
             # Merak seviyesi None durumda güncellenmez (finally bloğunda ek kontrol eklendi).
             return None # Girdi None ise None döndür.

        if not isinstance(understanding_signals, dict):
             logger.error(f"DecisionModule.decide: understanding_signals beklenmeyen tipte: {type(understanding_signals)}. Dict veya None bekleniyordu. Karar alınamıyor.")
             # Merak seviyesi dict olmayan durumda güncellenmez.
             return None # Girdi dict değilse None döndür.

        # relevant_memory_entries ve internal_state şimdilik bu karar mantığında doğrudan kullanılmıyor.
        # if not check_input_type(relevant_memory_entries, list, ...): ...
        # if internal_state is None: logger.debug("DecisionModule.decide: internal_state None.")

        decision = None # Alınan kararı tutacak değişken.

        # Anlama sinyallerini dictionary'den güvenle al. Anahtarlar yoksa default değerler kullanılır.
        similarity_score = understanding_signals.get('similarity_score', 0.0)
        high_audio_energy = understanding_signals.get('high_audio_energy', False)
        high_visual_edges = understanding_signals.get('high_visual_edges', False)
        is_bright = understanding_signals.get('is_bright', False)
        is_dark = understanding_signals.get('is_dark', False)
        max_concept_similarity = understanding_signals.get('max_concept_similarity', 0.0) # Yeni sinyal
        most_similar_concept_id = understanding_signals.get('most_similar_concept_id', None) # Yeni sinyal


        logger.debug(f"DecisionModule.decide: Anlama sinyalleri alindi - Sim:{similarity_score:.4f}, Audio:{high_audio_energy}, Visual:{high_visual_edges}, Bright:{is_bright}, Dark:{is_dark}, ConceptSim:{max_concept_similarity:.4f}, ConceptID:{most_similar_concept_id}. Mevcut Merak: {self.curiosity_level:.2f}. Karar veriliyor.")

        # Hangi temel bellek/benzerlik durumu oluştuğunu belirle (merak güncellemesi için kullanılacak)
        # Bu, process sinyalleri ve kavram tanıma OVERRIDE etmeden önceki temel durumdur.
        is_fundamentally_familiar = False
        is_fundamentally_new = False
        # Similarity score'un sayısal olduğunu kontrol etmeden float() çağırmayalım.
        if np.isscalar(similarity_score) and isinstance(similarity_score, np.number) and float(similarity_score) >= self.familiarity_threshold: # <<< HATA DÜZELTME
             is_fundamentally_familiar = True
             # logger.debug("DecisionModule.decide: Temel durum: Tanıdık (Bellek Eşiği Aşıldı).")
        else:
             is_fundamentally_new = True
             # logger.debug("DecisionModule.decide: Temel durum: Yeni (Bellek Eşiği Altında veya Skor Geçersiz).")


        try:
            # Karar Alma Mantığı (Faz 3/4): Öncelikli Mantık
            # Öncelik Sırası (Örnek): Merak > Ses > Görsel Kenar > Parlaklık/Karanlık > Kavram Tanıma > Bellek Tanıdıklığı > Varsayılan (Yeni)

            # 1. Merak Eşiği Aşıldı mı? (En yüksek öncelik)
            # Merak seviyesinin sayısal olduğunu kontrol etmeden karşılaştırma yapmayalım.
            if np.isscalar(self.curiosity_level) and isinstance(self.curiosity_level, np.number) and float(self.curiosity_level) >= self.curiosity_threshold: # <<< HATA DÜZELTME
                 # Merak eşiği aşıldysa rastgele bir keşif/sinyal kararı ver
                 decision = random.choice(["explore_randomly", "make_noise"]) # İki seçenekten birini rastgele seç.
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Merak eşiği ({self.curiosity_level:.2f} >= {self.curiosity_threshold:.2f}) aşıldı.")

            # 2. Yüksek ses enerjisi var mı? (İkinci öncelik)
            elif high_audio_energy: # high_audio_energy UnderstandingModule'da bool olarak belirlendi.
                 decision = "sound_detected"
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Yüksek ses enerjisi algılandı.")

            # 3. Yüksek görsel kenar yoğunluğu var mı? (Üçüncü öncelik)
            elif high_visual_edges: # high_visual_edges UnderstandingModule'da bool olarak belirlendi.
                 decision = "complex_visual_detected"
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Yüksek görsel kenar yoğunluğu algılandı.")

            # 4. Ortam Parlak mı veya Karanlık mı? (Dördüncü öncelik)
            elif is_bright: # is_bright UnderstandingModule'da bool olarak belirlendi.
                 decision = "bright_light_detected"
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Ortam parlak algılandı.")

            elif is_dark: # is_dark UnderstandingModule'da bool olarak belirlendi.
                 decision = "dark_environment_detected"
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Ortam karanlık algılandı.")

            # 5. Kavram Tanındı mı? (Beşinci öncelik)
            # Kavram benzerlik skoru ve ID'sinin geçerli olduğunu ve eşiği aştığını kontrol et.
            elif np.isscalar(max_concept_similarity) and isinstance(max_concept_similarity, np.number) and float(max_concept_similarity) >= self.concept_recognition_threshold and most_similar_concept_id is not None: # <<< HATA DÜZELTME
                 # Kavram tanındıysa, kavram ID'sini içeren bir karar stringi üret.
                 decision = f"recognized_concept_{int(most_similar_concept_id)}" # ID int olmalı.
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Kavram tanındı (Benzerlik: {max_concept_similarity:.4f} >= Eşik {self.concept_recognition_threshold:.4f}, ID: {most_similar_concept_id}).")

            # 6. Bellek benzerlik skoru eşiği aşıyor mu? (Altıncı öncelik)
            # Bu kontrol, ancak yukarıdaki Process/Kavram tanıma sinyalleri yoksa yapılır.
            elif is_fundamentally_familiar: # float(similarity_score) >= self.familiarity_threshold
                 decision = "familiar_input_detected"
                 # logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Bellek benzerlik skoru ({similarity_score:.4f}) >= Eşik ({self.familiarity_threshold:.4f}).") # Log yukarıda temel durumda yapıldı.

            # 7. Hiçbir öncelikli koşul sağlanmazsa (Varsayılan)
            else: # is_fundamentally_new # float(similarity_score) < self.familiarity_threshold
                 decision = "new_input_detected"
                 # logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Hiçbir öncelikli durum algılanamadı (Temel Durum: Yeni).") # Log yukarıda temel durumda yapıldı.


            # Gelecekte daha karmaşık mantıklar:
            # - Başka anlama çıktılarından gelen bilgileri karara dahil etme.
            # - Birden fazla kriteri birleştirme (örn: hem benzerlik yüksek hem de anlama çıktısı spesifik bir nesne algıladı).
            # - Başka içsel durumları karara dahil etme (boredom, hunger vb.).
            # - Farklı karar türleri (örn: "hareket et", "ses çıkar").

        except Exception as e:
            # Karar alma işlemi sırasında beklenmedik bir hata olursa logla.
            logger.error(f"DecisionModule.decide: Karar alma sırasında beklenmedik hata: {e}", exc_info=True)
            # Merak seviyesi hata durumunda güncellenmez.
            return None # Hata durumunda None döndür.

        finally:
            # --- Merak Seviyesini Güncelle ---
            # Karar ne olursa olsun (hata durumu hariç), merak seviyesini güncelle.
            # Güncelleme, inputun temel 'Yeni' veya 'Tanıdık' durumuna göre yapılır.
            # curiosity_level'ın sayısal olduğunu kontrol et.
            if np.isscalar(self.curiosity_level) and isinstance(self.curiosity_level, np.number): # <<< HATA DÜZELTME
                try:
                    # Sadece Karar başarılı bir şekilde belirlendiyse (decision is not None) merakı güncelle.
                    # Bu, hata durumında merakın yanlış güncellenmesini önler.
                    if decision is not None:
                        # Karar stringine göre merakı güncelle
                        if decision == "new_input_detected": # Yeni input kararı verildiğinde merak artar
                             self.curiosity_level += self.curiosity_increment_new
                             #logger.debug(f"DecisionModule: Merak artışı ({self.curiosity_increment_new:.2f}). Karar 'Yeni'.") # Log DecisionModule başlangıcına taşındı
                        elif decision == "familiar_input_detected" or (isinstance(decision, str) and decision.startswith("recognized_concept_")): # Tanıdık veya Kavram tanıma kararı verildiğinde merak azalır
                             self.curiosity_level = max(0.0, self.curiosity_level - self.curiosity_decrement_familiar) # Merak negatif olmasın.
                             #logger.debug(f"DecisionModule: Merak azalışı ({self.curiosity_decrement_familiar:.2f}). Karar 'Tanıdık' veya 'Kavram'.")
                        # Ses, Görsel Kenar, Parlak/Karanlık kararları verildiğinde Merak ne olsun?
                        # Şu anki mantık temel duruma göre güncellemeye devam etsin.
                        # Alternatif: Eğer Process tabanlı veya Kavram tanıma kararı verilirse merak seviyesi DEĞİŞMESİN.
                        # Bu ikinci mantık daha mantıklı olabilir: Evo yeni bir şey algıladığında meraklanır, ama o şeyi tanıyınca veya kategorize edince merakı azalır.
                        # Yüksek ses veya detaylı görsel gibi Process özellikleri, başlı başına merakı artırmamalı veya azaltmamalı.
                        # Önceki mantığı (temel duruma göre güncelleme) koruyorum şimdilik, ama burası gelecekte gözden geçirilebilir.


                    # Her döngü adımında merak seviyesini azalt (decay).
                    self.curiosity_level = max(0.0, self.curiosity_level - self.curiosity_decay) # Merak negatif olmasın.
                    # logger.debug(f"DecisionModule: Merak decay ({self.curiosity_decay:.2f}).") # Log DecisionModule başlangıcına taşındı

                    # Merak seviyesinin üst sınırını belirle (opsiyonel, config'e eklenebilir)
                    # self.curiosity_level = min(self.max_merak_level, self.curiosity_level)

                    # Güncel merak seviyesini logla.
                    logger.debug(f"DecisionModule: Güncel Merak Seviyesi: {self.curiosity_level:.2f}")

                except Exception as e:
                     logger.error(f"DecisionModule: Merak seviyesi güncellenirken beklenmedik hata: {e}", exc_info=True)
            # else: DecisionModule init'te loglandı.


        return decision # Alınan karar stringi veya None döndürülür.

    def cleanup(self):
        """
        DecisionModule kaynaklarını temizler.

        Şimdilik özel bir kaynak kullanmadığı için temizleme adımı içermez.
        Gelecekte model temizliği veya kural seti temizliği gerekebilir.
        module_loader.py bu metotu program sonlanırken çağrır (varsa).
        """
        logger.info("DecisionModule objesi temizleniyor.")
        # İçsel durumun kaydedilmesi buraya gelebilir (gelecek TODO).
        # self._save_internal_state() # Gelecek TODO
        pass