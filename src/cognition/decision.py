# src/cognition/decision.py
#
# Evo'nın karar alma modülünü temsil eder.
# Anlama modülünden gelen sinyalleri, bellek girdilerini ve içsel durumu kullanarak bir eylem kararı alır.

import logging
import numpy as np # Sayısal işlemler ve type checking için

# Yardımcı fonksiyonları import et (girdi kontrolleri ve config için)
from src.core.utils import check_input_not_none, check_input_type, get_config_value # <<< Utils importları


# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

class DecisionModule:
    """
    Evo'nın karar alma yeteneğini sağlayan sınıf (Faz 3 implementasyonu).

    UnderstandingModule'den gelen anlama sinyallerini (dictionary), bellek girdilerini
    ve içsel durumu (şimdilik merak seviyesi) alır. Bu bilgilere dayanarak,
    MotorControl modülüne iletilecek bir eylem kararı alır.
    Mevcut implementasyon: Process çıktısı (enerji/kenar), bellek benzerlik skoru ve
    merak seviyesine dayalı öncelikli bir karar verme mantığı uygular.
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
                           'merak_threshold': Merak seviyesinin keşif/sinyal kararı için ulaşması gereken eşik (float, varsayılan 5.0).
                           'merak_increment_new': "Yeni" input algılandığında merak seviyesi artış miktarı (float, varsayılan 1.0).
                           'merak_decrement_familiar': "Tanıdık" input algılandığında merak seviyesi azalış miktarı (float, varsayılan 0.5).
                           'merak_decay': Her döngü adımında merak seviyesinin otomatik azalışı (float, varsayılan 0.1).
                           Gelecekte karar kuralları, eşikleri veya model yolları gelebilir.
        """
        self.config = config
        logger.info("DecisionModule başlatılıyor (Faz 3)...")

        # Yapılandırmadan eşikleri ve merak ayarlarını alırken get_config_value kullan.
        self.familiarity_threshold = get_config_value(config, 'familiarity_threshold', 0.8, expected_type=(float, int), logger_instance=logger)
        self.audio_energy_threshold = get_config_value(config, 'audio_energy_threshold', 1000.0, expected_type=(float, int), logger_instance=logger)
        self.visual_edges_threshold = get_config_value(config, 'visual_edges_threshold', 50.0, expected_type=(float, int), logger_instance=logger)
        self.merak_threshold = get_config_value(config, 'merak_threshold', 5.0, expected_type=(float, int), logger_instance=logger)
        self.merak_increment_new = get_config_value(config, 'merak_increment_new', 1.0, expected_type=(float, int), logger_instance=logger)
        self.merak_decrement_familiar = get_config_value(config, 'merak_decrement_familiar', 0.5, expected_type=(float, int), logger_instance=logger)
        self.merak_decay = get_config_value(config, 'merak_decay', 0.1, expected_type=(float, int), logger_instance=logger)


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
        # Merak eşiği ve güncelleme miktarları negatif olmamalı
        if self.merak_threshold < 0.0:
             logger.warning(f"DecisionModule: Konfig 'merak_threshold' negatif ({self.merak_threshold}). Varsayılan 5.0 kullanılıyor.")
             self.merak_threshold = 5.0
        if self.merak_increment_new < 0.0:
             logger.warning(f"DecisionModule: Konfig 'merak_increment_new' negatif ({self.merak_increment_new}). Varsayılan 1.0 kullanılıyor.")
             self.merak_increment_new = 1.0
        if self.merak_decrement_familiar < 0.0:
             logger.warning(f"DecisionModule: Konfig 'merak_decrement_familiar' negatif ({self.merak_decrement_familiar}). Varsayılan 0.5 kullanılıyor.")
             self.merak_decrement_familiar = 0.5
        # Merak decay negatif olmamalı
        if self.merak_decay < 0.0:
             logger.warning(f"DecisionModule: Konfig 'merak_decay' negatif ({self.merak_decay}). Varsayılan 0.1 kullanılıyor.")
             self.merak_decay = 0.1


        # İçsel durum değişkenleri
        self.merak_level = 0.0 # Merak seviyesi (0.0 ile başlar)
        # Diğer içsel durumlar gelecekte buraya eklenecek (örn: energy_level, boredom_level)

        logger.info(f"DecisionModule başlatıldı. Tanıdıklık Eşiği: {self.familiarity_threshold}, Ses Eşiği: {self.audio_energy_threshold}, Görsel Eşiği: {self.visual_edges_threshold}, Merak Eşiği: {self.merak_threshold}")
        logger.debug(f"DecisionModule: Merak Artış (Yeni): {self.merak_increment_new}, Azalış (Tanıdık): {self.merak_decrement_familiar}, Decay: {self.merak_decay}")


    # relevant_memory_entries hala burada duruyor (CognitionCore'dan geliyor)
    # internal_state parametresi şimdilik bu sınıfın içindeki self.merak_level'i temsil etmiyor,
    # genel bir placeholder. self.merak_level doğrudan burada yönetiliyor.
    def decide(self, understanding_signals, relevant_memory_entries, internal_state=None):
        """
        Anlama sinyallerine ve içsel duruma (merak seviyesi) göre bir eylem kararı alır.

        UnderstandingModule'den gelen anlama sinyalleri dictionary'sini alır.
        Process çıktısı tabanlı flag'lere, merak seviyesine ve bellek benzerlik skoruna dayalı
        öncelikli bir karar verme mantığı uygular. Merak seviyesini günceller.

        Args:
            understanding_signals (dict or None): Anlama modülünden gelen anlama sinyalleri dictionary'si.
                                                Beklenen format: {'similarity_score': float, 'high_audio_energy': bool, 'high_visual_edges': bool} veya None.
                                                UnderstandingModule hata durumunda varsayılan dict döndürmeyi hedefler.
            relevant_memory_entries (list or None): Memory modülünden gelen ilgili bellek girdileri listesi.
                                            Bu metotta doğrudan karar için kullanılmıyor, ama parametre olarak geliyor.
                                            Gelecekte bağlamsal karar için kullanılabilir.
            internal_state (dict, optional): Evo'nın içsel durumu (örn: enerji seviyesi, merak). GELECEKTE KULLANILACAK. Şu an bu metot kendi içsel merak seviyesini yönetiyor. Varsayılan None.

        Returns:
            str or None: Alınan karar stringi (örn: "sound_detected", "complex_visual_detected", "familiar_input_detected", "new_input_detected", "explore_randomly", "make_noise")
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
        similarity_score = understanding_signals.get('similarity_score', 0.0) # Default 0.0
        high_audio_energy = understanding_signals.get('high_audio_energy', False) # Default False
        high_visual_edges = understanding_signals.get('high_visual_edges', False) # Default False

        logger.debug(f"DecisionModule.decide: Anlama sinyalleri alindi - Sim:{similarity_score:.4f}, Audio:{high_audio_energy}, Visual:{high_visual_edges}. Mevcut Merak: {self.merak_level:.2f}. Karar veriliyor.")

        # Hangi temel bellek/benzerlik durumu oluştuğunu belirle (merak güncellemesi için kullanılacak)
        # Bu, process sinyalleri OVERRIDE etmeden önceki temel durumdur.
        is_fundamentally_familiar = False
        is_fundamentally_new = False
        if float(similarity_score) >= self.familiarity_threshold:
             is_fundamentally_familiar = True
             logger.debug("DecisionModule.decide: Temel durum: Tanıdık (Bellek Eşiği Aşıldı).")
        else:
             is_fundamentally_new = True
             logger.debug("DecisionModule.decide: Temel durum: Yeni (Bellek Eşiği Altında).")


        try:
            # Karar Alma Mantığı (Faz 3): Öncelikli Mantık
            # Öncelik Sırası: Merak > Ses > Görsel Kenar > Bellek Tanıdıklığı > Varsayılan (Yeni)

            # 1. Merak Eşiği Aşıldı mı? (En yüksek öncelik)
            if self.merak_level >= self.merak_threshold:
                 # Merak eşiği aşıldıysa rastgele bir keşif/sinyal kararı ver
                 decision = random.choice(["explore_randomly", "make_noise"]) # İki seçenekten birini rastgele seç.
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Merak eşiği ({self.merak_level:.2f} >= {self.merak_threshold:.2f}) aşıldı.")

            # 2. Yüksek ses enerjisi var mı? (İkinci öncelik)
            elif high_audio_energy:
                 decision = "sound_detected"
                 # high_audio_energy True ise eşik zaten UnderstandingModule'da aşıldı.
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Yüksek ses enerjisi algılandı.")

            # 3. Yüksek görsel kenar yoğunluğu var mı? (Üçüncü öncelik)
            elif high_visual_edges:
                 decision = "complex_visual_detected"
                 # high_visual_edges True ise eşik zaten UnderstandingModule'da aşıldı.
                 logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Yüksek görsel kenar yoğunluğu algılandı.")

            # 4. Bellek benzerlik skoru eşiği aşıyor mu? (Dördüncü öncelik)
            # Bu kontrol, ancak Process tabanlı sinyaller yoksa yapılır.
            # Sim score float/int olmalı, init'te threshold da float/int yapıldı. float() ile çevirerek karşılaştır.
            elif is_fundamentally_familiar: # float(similarity_score) >= self.familiarity_threshold
                 decision = "familiar_input_detected"
                 # logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Bellek benzerlik skoru ({similarity_score:.4f}) >= Eşik ({self.familiarity_threshold:.4f}).") # Log yukarıda temel durumda yapıldı.

            # 5. Hiçbir öncelikli koşul sağlanmazsa (Varsayılan)
            else: # is_fundamentally_new # float(similarity_score) < self.familiarity_threshold
                 decision = "new_input_detected"
                 # logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Hiçbir öncelikli durum algılanamadı (Temel Durum: Yeni).") # Log yukarıda temel durumda yapıldı.


            # Gelecekte daha karmaşık mantıklar:
            # - Başka anlama çıktılarından gelen bilgileri karara dahil etme.
            # - Birden fazla kriteri birleştirme.
            # - Başka içsel durumları karara dahil etme (boredom, hunger vb.).
            # - Farklı karar türleri (örn: "hareket et", "ses çıkar").

        except Exception as e:
            # Karar alma işlemi sırasında beklenmedik bir hata olursa logla.
            logger.error(f"DecisionModule.decide: Karar alma sırasında beklenmedik hata: {e}", exc_info=True)
            return None # Hata durumunda None döndür.

        finally:
            # --- Merak Seviyesini Güncelle ---
            # Karar ne olursa olsun (hata durumu hariç), merak seviyesini güncelle.
            # Güncelleme, inputun temel 'Yeni' veya 'Tanıdık' durumuna göre yapılır.
            try:
                if is_fundamentally_new: # Eğer input Process sinyalleri veya merak eşiği tarafından override edilmeseydi "Yeni" olacaktı.
                     self.merak_level += self.merak_increment_new
                     logger.debug(f"DecisionModule: Merak artışı ({self.merak_increment_new:.2f}). Temel durum 'Yeni'.")
                elif is_fundamentally_familiar: # Eğer input Process sinyalleri veya merak eşiği tarafından override edilmeseydi "Tanıdık" olacaktı.
                     self.merak_level = max(0.0, self.merak_level - self.merak_decrement_familiar) # Merak negatif olmasın.
                     logger.debug(f"DecisionModule: Merak azalışı ({self.merak_decrement_familiar:.2f}). Temel durum 'Tanıdık'.")
                # else: # Ne yeni ne tanıdık (örn: representation/memory invalid), merak değişmesin veya decay uygula.

                # Her döngü adımında merak seviyesini azalt (decay).
                self.merak_level = max(0.0, self.merak_level - self.merak_decay) # Merak negatif olmasın.
                # logger.debug(f"DecisionModule: Merak decay ({self.merak_decay:.2f}).")

                # Güncel merak seviyesini logla.
                logger.debug(f"DecisionModule: Güncel Merak Seviyesi: {self.merak_level:.2f}")

            except Exception as e:
                 logger.error(f"DecisionModule: Merak seviyesi güncellenirken beklenmedik hata: {e}", exc_info=True)

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
        pass