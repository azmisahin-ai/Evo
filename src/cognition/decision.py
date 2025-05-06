# src/cognition/decision.py
#
# Evo'nın karar alma modülünü temsil eder.
# Anlama modülünden gelen sonucu ve bellek girdilerini kullanarak bir eylem kararı alır.

import logging

import numpy as np
# import numpy as np # Gerekirse sayısal işlemler için (şimdilik gerekmiyor)

# Yardımcı fonksiyonları import et (girdi kontrolleri ve config için)
from src.core.utils import check_input_not_none, check_input_type, get_config_value # <<< Utils importları


# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

class DecisionModule:
    """
    Evo'nın karar alma yeteneğini sağlayan sınıf (Faz 3 implementasyonu).

    Anlama modülünden gelen anlama sonucunu (şimdilik en yüksek bellek benzerlik skoru)
    ve içsel durumu (gelecekte) alır. Config'ten okunan bir eşiğe göre
    ("familiarity_threshold") "tanıdık" veya "yeni" input kararı verir.
    Bu kararı MotorControl modülüne iletilmek üzere döndürür.
    Gelecekte daha karmaşık karar ağaçları, kural tabanlı sistemler veya
    öğrenilmiş karar modelleri implement edilecektir.
    """
    def __init__(self, config):
        """
        DecisionModule'ü başlatır.

        Args:
            config (dict): Karar alma modülü yapılandırma ayarları.
                           'familiarity_threshold': Girdiyi "tanıdık" kabul etmek için
                                                    gereken en düşük bellek benzerlik skoru (float, varsayılan 0.8).
                           Gelecekte karar kuralları, eşikleri veya model yolları gelebilir.
        """
        self.config = config
        logger.info("DecisionModule başlatılıyor (Faz 3)...")

        # Yapılandırmadan tanıdıklık eşiğini alırken get_config_value kullan.
        # Float veya int tipinde olması beklenir. Varsayılan 0.8.
        self.familiarity_threshold = get_config_value(config, 'familiarity_threshold', 0.8, expected_type=(float, int), logger_instance=logger)

        # Eşik değeri için basit değer kontrolü (0.0 ile 1.0 arası)
        if not (0.0 <= self.familiarity_threshold <= 1.0):
             logger.warning(f"DecisionModule: Konfigurasyonda tanıdıklık eşiği beklenmeyen aralıkta ({self.familiarity_threshold}). 0.0 ile 1.0 arası bekleniyordu. Varsayılan 0.8 kullanılıyor.")
             self.familiarity_threshold = 0.8 # Geçersiz aralıktaysa varsayılanı kullan.


        logger.info(f"DecisionModule başlatıldı. Tanıdıklık Eşiği: {self.familiarity_threshold}")


    def decide(self, understanding_result, relevant_memory_entries, internal_state=None):
        """
        Anlama sonucu (en yüksek bellek benzerlik skoru) ve içsel duruma göre bir eylem kararı alır.

        Mevcut implementasyon: UnderstandingModule'den gelen benzerlik skorunu alır
        ve bu skoru `familiarity_threshold` ile karşılaştırarak karar verir.

        Args:
            understanding_result (float or None): Anlama modülünden gelen en yüksek bellek benzerlik skoru
                                                  veya hata durumunda 0.0/None. Beklenen tip: float veya 0.0/None.
            relevant_memory_entries (list): Memory modülünden gelen ilgili bellek girdileri listesi.
                                            Bu metotta doğrudan karar için kullanılmıyor (Anlama modülü kullandı).
                                            Ancak gelecekte bağlamsal karar için kullanılabilir.
            internal_state (dict, optional): Evo'nın içsel durumu (örn: enerji seviyesi, merak). Gelecekte kullanılacak. Varsayılan None.

        Returns:
            str or None: Alınan karar ("familiar_input_detected", "new_input_detected")
                         veya girdi (understanding_result) geçersizse ya da hata durumunda None.
        """
        # Girdi kontrolleri. understanding_result'ın float veya None olup olmadığını kontrol et.
        # check_input_not_none ve check_input_type fonksiyonlarını kullanalım.
        # None olması veya float olmaması durumunda karar alamayız.
        # Eğer understanding_result 0.0 gelirse bu 'yeni' olarak yorumlanır, bu yüzden None kontrolü yeterli.
        if understanding_result is None:
             logger.debug("DecisionModule.decide: understanding_result None. Karar alınamıyor.")
             return None # Girdi None ise None döndür.

        # understanding_result float veya int gibi sayısal bir değer olmalı.
        if not np.isscalar(understanding_result) or not np.issubdtype(type(understanding_result), np.number):
             logger.error(f"DecisionModule.decide: understanding_result beklenmeyen tipte: {type(understanding_result)}. Sayısal (float/int) veya None bekleniyordu. Karar alınamıyor.")
             return None # Girdi sayısal değilse None döndür.

        # relevant_memory_entries ve internal_state şimdilik bu karar mantığında doğrudan kullanılmıyor.
        # if not check_input_type(relevant_memory_entries, list, ...): ...
        # if internal_state is None: logger.debug("DecisionModule.decide: internal_state None.")


        decision = None # Alınan kararı tutacak değişken.
        similarity_score = float(understanding_result) # Sayısal değeri float'a çevir.

        logger.debug(f"DecisionModule.decide: Anlama skoru alindi: {similarity_score:.4f}. Eşik: {self.familiarity_threshold:.4f}. Karar veriliyor.")

        try:
            # Karar Alma Mantığı (Faz 3):
            # Anlama skoru (en yüksek bellek benzerliği) belirlenen eşiğin üzerindeyse "tanıdık", aksi halde "yeni".

            if similarity_score >= self.familiarity_threshold:
                # Eğer benzerlik skoru eşiğe eşit veya üzerindeyse
                decision = "familiar_input_detected" # "Tanıdık input algılandı" kararı
                logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Skor ({similarity_score:.4f}) >= Eşik ({self.familiarity_threshold:.4f}).")
            else:
                # Eğer benzerlik skoru eşiğin altındaysa (veya 0.0 ise)
                decision = "new_input_detected" # "Yeni input algılandı" kararı
                logger.debug(f"DecisionModule.decide: Karar: '{decision}'. Skor ({similarity_score:.4f}) < Eşik ({self.familiarity_threshold:.4f}).")

            # Gelecekte daha karmaşık mantıklar:
            # - Başka anlama çıktılarından gelen bilgileri (plana göre Direct Processor Output gibi) karara dahil etme.
            # - Birden fazla kriteri birleştirme (örn: hem benzerlik yüksek hem de anlama çıktısı spesifik bir nesne algıladı).
            # - İçsel durumu (merak, enerji) karara dahil etme.
            # - Farklı karar türleri (örn: "hareket et", "ses çıkar").

        except Exception as e:
            # Karar alma işlemi sırasında beklenmedik bir hata olursa logla.
            logger.error(f"DecisionModule.decide: Karar alma sırasında beklenmedik hata: {e}", exc_info=True)
            return None # Hata durumunda None döndür.

        return decision # Alınan kararı döndür ("familiar_input_detected", "new_input_detected", veya None).

    def cleanup(self):
        """
        DecisionModule kaynaklarını temizler.

        Şimdilik özel bir kaynak kullanmadığı için temizleme adımı içermez.
        Gelecekte model temizliği veya kural seti temizliği gerekebilir.
        module_loader.py bu metodu program sonlanırken çağırır (varsa).
        """
        logger.info("DecisionModule objesi temizleniyor.")
        pass