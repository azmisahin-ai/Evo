# src/cognition/decision.py
#
# Evo'nın karar alma modülünü temsil eder.
# Anlama modülünden gelen sinyalleri ve içsel durumu kullanarak bir eylem kararı alır.

import logging
# import numpy as np # Gerekirse sayısal işlemler için (şimdilik gerekmiyor)

# Yardımcı fonksiyonları import et (girdi kontrolleri ve config için)
from src.core.utils import check_input_not_none, check_input_type, get_config_value # <<< Utils importları


# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

class DecisionModule:
    """
    Evo'nın karar alma yeteneğini sağlayan sınıf (Faz 3 implementasyonu).

    UnderstandingModule'den gelen anlama sinyallerini (dictionary) ve içsel durumu (gelecekte) alır.
    Bu bilgilere dayanarak, MotorControl modülüne iletilecek bir eylem kararı alır.
    Mevcut implementasyon: Process çıktısı (enerji/kenar) ve bellek benzerlik skoruna dayalı
    öncelikli bir karar verme mantığı uygular.
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
                           Gelecekte karar kuralları, eşikleri veya model yolları gelebilir.
        """
        self.config = config
        logger.info("DecisionModule başlatılıyor (Faz 3)...")

        # Yapılandırmadan eşikleri alırken get_config_value kullan.
        self.familiarity_threshold = get_config_value(config, 'familiarity_threshold', 0.8, expected_type=(float, int), logger_instance=logger)
        self.audio_energy_threshold = get_config_value(config, 'audio_energy_threshold', 1000.0, expected_type=(float, int), logger_instance=logger)
        self.visual_edges_threshold = get_config_value(config, 'visual_edges_threshold', 50.0, expected_type=(float, int), logger_instance=logger)


        # Eşik değerleri için basit değer kontrolü (0.0 ile 1.0 arası benzerlik, negatif olmamalı diğerleri)
        if not (0.0 <= self.familiarity_threshold <= 1.0):
             logger.warning(f"DecisionModule: Konfig 'familiarity_threshold' beklenmeyen aralıkta ({self.familiarity_threshold}). Varsayılan 0.8 kullanılıyor.")
             self.familiarity_threshold = 0.8
        if self.audio_energy_threshold < 0.0:
             logger.warning(f"DecisionModule: Konfig 'audio_energy_threshold' negatif ({self.audio_energy_threshold}). Varsayılan 1000.0 kullanılıyor.")
             self.audio_energy_threshold = 1000.0
        if self.visual_edges_threshold < 0.0:
             logger.warning(f"DecisionModule: Konfig 'visual_edges_threshold' negatif ({self.visual_edges_threshold}). Varsayılan 50.0 kullanılıyor.")
             self.visual_edges_threshold = 50.0


        logger.info(f"DecisionModule başlatıldı. Tanıdıklık Eşiği: {self.familiarity_threshold}, Ses Enerji Eşiği: {self.audio_energy_threshold}, Görsel Kenar Eşiği: {self.visual_edges_threshold}")


    # relevant_memory_entries hala burada duruyor (CognitionCore'dan geliyor)
    # internal_state gelecekte kullanılacak.
    def decide(self, understanding_signals, relevant_memory_entries, internal_state=None):
        """
        Anlama sinyallerine ve içsel duruma göre bir eylem kararı alır.

        Mevcut implementasyon: UnderstandingModule'den gelen anlama sinyalleri dictionary'sini
        alır ve Process çıktısı tabanlı flag'lere öncelik vererek veya
        bellek benzerlik skorunu eşikle karşılaştırarak karar verir.

        Args:
            understanding_signals (dict or None): Anlama modülünden gelen anlama sinyalleri dictionary'si.
                                                Beklenen format: {'similarity_score': float, 'high_audio_energy': bool, 'high_visual_edges': bool} veya None.
                                                UnderstandingModule hata durumunda varsayılan dict döndürmeyi hedefler.
            relevant_memory_entries (list): Memory modülünden gelen ilgili bellek girdileri listesi.
                                            Bu metotta doğrudan karar için kullanılmıyor, ama parametre olarak geliyor.
                                            Gelecekte bağlamsal karar için kullanılabilir.
            internal_state (dict, optional): Evo'nın içsel durumu (örn: enerji seviyesi, merak). Gelecekte kullanılacak. Varsayılan None.

        Returns:
            str or None: Alınan karar stringi (örn: "sound_detected", "complex_visual_detected", "familiar_input_detected", "new_input_detected")
                         veya girdi (understanding_signals) geçersizse ya da hata durumunda None.
        """
        # Girdi kontrolleri. understanding_signals'ın geçerli bir dictionary mi?
        # check_input_not_none ve check_input_type fonksiyonlarını kullanalım.
        # Eğer None ise veya dict değilse karar alamayız. UnderstandingModule varsayılan dict döndürdüğü için genellikle dict gelecek.
        if not check_input_not_none(understanding_signals, input_name="understanding_signals for DecisionModule", logger_instance=logger):
             logger.debug("DecisionModule.decide: understanding_signals None. Karar alınamıyor.")
             return None # Girdi None ise None döndür.

        if not check_input_type(understanding_signals, dict, input_name="understanding_signals for DecisionModule", logger_instance=logger):
             logger.error(f"DecisionModule.decide: understanding_signals beklenmeyen tipte: {type(understanding_signals)}. Dict veya None bekleniyordu. Karar alınamıyor.")
             return None # Girdi dict değilse None döndür.

        # relevant_memory_entries ve internal_state şimdilik bu karar mantığında doğrudan kullanılmıyor.
        # if not check_input_type(relevant_memory_entries, list, ...): ...
        # if internal_state is None: logger.debug("DecisionModule.decide: internal_state None.")

        decision = None # Alınan kararı tutacak değişken.

        # Anlama sinyallerini dictionary'den güvenle al. Anahtarlar yoksa default değerler (0.0, False) kullanılır.
        # UnderstandingModule varsayılan dict döndürdüğü için anahtarların var olması beklenir, ama sağlamlık için get() kullanmak iyi.
        similarity_score = understanding_signals.get('similarity_score', 0.0) # Default 0.0
        high_audio_energy = understanding_signals.get('high_audio_energy', False) # Default False
        high_visual_edges = understanding_signals.get('high_visual_edges', False) # Default False

        logger.debug(f"DecisionModule.decide: Anlama sinyalleri alindi - Sim:{similarity_score:.4f}, Audio:{high_audio_energy}, Visual:{high_visual_edges}. Karar veriliyor.")

        try:
            # Karar Alma Mantığı (Faz 3): Öncelikli Mantık
            # 1. Yüksek ses enerjisi var mı? (En yüksek öncelik)
            if high_audio_energy:
                 decision = "sound_detected"
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Yüksek ses enerjisi algılandı.")

            # 2. Yüksek görsel kenar yoğunluğu var mı? (İkinci öncelik)
            elif high_visual_edges:
                 decision = "complex_visual_detected"
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Yüksek görsel kenar yoğunluğu algılandı.")

            # 3. Bellek benzerlik skoru eşiği aşıyor mu? (Üçüncü öncelik)
            # Sim score float/int olmalı, init'te threshold da float/int yapıldı. float() ile çevirerek karşılaştır.
            elif float(similarity_score) >= self.familiarity_threshold:
                 decision = "familiar_input_detected"
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Bellek benzerlik skoru ({similarity_score:.4f}) >= Eşik ({self.familiarity_threshold:.4f}).")

            # 4. Hiçbir koşul sağlanmazsa (Varsayılan)
            else:
                 decision = "new_input_detected"
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Hiçbir öncelikli durum algılanamadı.")


            # Gelecekte daha karmaşık mantıklar:
            # - Başka anlama çıktılarından gelen bilgileri karara dahil etme.
            # - Birden fazla kriteri birleştirme (örn: hem benzerlik yüksek hem de anlama çıktısı spesifik bir nesne algıladı).
            # - İçsel durumu (merak, enerji) karara dahil etme.
            # - Farklı karar türleri (örn: "hareket et", "ses çıkar").

        except Exception as e:
            # Karar alma işlemi sırasında beklenmedik bir hata olursa logla.
            logger.error(f"DecisionModule.decide: Karar alma sırasında beklenmedik hata: {e}", exc_info=True)
            return None # Hata durumunda None döndür.

        return decision # Alınan karar stringi veya None döndürülür.

    def cleanup(self):
        """
        DecisionModule kaynaklarını temizler.

        Şimdilik özel bir kaynak kullanmadığı için temizleme adımı içermez.
        Gelecekte model temizliği veya kural seti temizliği gerekebilir.
        module_loader.py bu metodu program sonlanırken çağrır (varsa).
        """
        logger.info("DecisionModule objesi temizleniyor.")
        pass