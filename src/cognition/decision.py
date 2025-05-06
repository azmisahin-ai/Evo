# src/cognition/decision.py
#
# Evo'nın karar alma modülünü temsil eder.
# Anlama modülünden gelen sonucu ve bellek girdilerini kullanarak bir eylem kararı alır.

import logging
# import numpy as np # Gerekirse sayısal işlemler için (şimdilik gerekmiyor)

# Yardımcı fonksiyonları import et (girdi kontrolleri için)
from src.core.utils import check_input_not_none, check_input_type # <<< Yeni importlar


# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

class DecisionModule:
    """
    Evo'nın karar alma yeteneğini sağlayan sınıf (Faz 3 implementasyonu).

    Anlama modülünden gelen sonuçları, belleği (ilgili girdiler) ve içsel durumu (gelecekte) alır.
    Bu bilgilere dayanarak, MotorControl modülüne iletilecek bir eylem kararı alır.
    Şimdilik temel mantık: Bellekten ilgili anı bulunup bulunmamasına göre karar verir.
    Gelecekte daha karmaşık karar ağaçları, kural tabanlı sistemler veya
    öğrenilmiş karar modelleri implement edilecektir.
    """
    def __init__(self, config):
        """
        DecisionModule'ü başlatır.

        Args:
            config (dict): Karar alma modülü yapılandırma ayarları.
                           Gelecekte karar kuralları, eşikleri veya model yolları gelebilir.
        """
        self.config = config
        logger.info("DecisionModule başlatılıyor (Faz 3)...")
        # Modül başlatma mantığı buraya gelebilir (örn: karar kural seti yükleme)
        logger.info("DecisionModule başlatıldı.")


    def decide(self, understanding_result, relevant_memory_entries, internal_state=None):
        """
        Anlama sonucu, bellek girdileri ve içsel duruma göre bir eylem kararı alır.

        Mevcut implementasyon: Memory'den ilgili anı gelip gelmemesine göre
        "tanıdık" veya "yeni" input kararı verir.

        Args:
            understanding_result (any): Anlama modülünden gelen anlama işleminin sonucu. None olabilir.
                                        Şimdilik karar mantığında doğrudan kullanılmıyor, ama gelecekte kullanılacak.
            relevant_memory_entries (list): Memory modülünden gelen ilgili bellek girdileri listesi.
                                            Bellek boşsa veya sorgu sırasında hata oluştuysa boş liste `[]` olabilir.
                                            Bu listenin boş olup olmamasına göre karar alınır.
                                            Beklenen format: liste.
            internal_state (dict, optional): Evo'nın içsel durumu (örn: enerji seviyesi, merak). Gelecekte kullanılacak. Varsayılan None.

        Returns:
            str or None: Alınan karar (şimdilik "familiar_input_detected", "new_input_detected" stringleri)
                         veya girdi geçersizse ya da hata durumunda None.
        """
        # Girdi kontrolleri. relevant_memory_entries'in liste olup olmadığını kontrol et.
        # check_input_not_none ve check_input_type fonksiyonlarını kullanalım.
        # relevant_memory_entries'in None olması veya liste olmaması durumunda karar alamayız.
        if not check_input_not_none(relevant_memory_entries, input_name="relevant_memory_entries for DecisionModule", logger_instance=logger):
             logger.debug("DecisionModule.decide: relevant_memory_entries None. Karar alınamıyor.")
             return None # Girdi None ise None döndür.

        if not check_input_type(relevant_memory_entries, list, input_name="relevant_memory_entries for DecisionModule", logger_instance=logger):
             logger.error("DecisionModule.decide: relevant_memory_entries liste değil. Karar alınamıyor.")
             return None # Girdi liste değilse None döndür.

        # learned_representation ve understanding_result şimdilik karar mantığında doğrudan kullanılmıyor
        # ama varlıkları loglanabilir veya gelecekteki geliştirmeler için burada kontrol edilebilir.
        # if learned_representation is None: logger.debug("DecisionModule.decide: learned_representation None.")
        # if understanding_result is None: logger.debug("DecisionModule.decide: understanding_result None.")
        # if internal_state is None: logger.debug("DecisionModule.decide: internal_state None.")


        decision = None # Alınan kararı tutacak değişken.

        try:
            # Basit Karar Alma Mantığı (Faz 3 başlangıcı):
            # Memory'den gelen ilgili anı listesi (relevant_memory_entries) boş değilse,
            # inputu "tanıdık" olarak kabul et. Aksi takdirde "yeni" olarak kabul et.
            # Belleğin Representation benzerliğine göre doldurulduğunu varsayıyoruz.

            if relevant_memory_entries: # Eğer liste boş değilse (anı bulunduysa)
                decision = "familiar_input_detected" # "Tanıdık input algılandı" kararı
                logger.debug(f"DecisionModule.decide: Tanıdık input kararı alındı (Memory'de {len(relevant_memory_entries)} anı bulundu).")
            else: # Eğer liste boşsa (anı bulunamadıysa)
                decision = "new_input_detected" # "Yeni input algılandı" kararı
                logger.debug("DecisionModule.decide: Yeni input kararı alındı (Memory'de ilgili anı bulunamadı).")

            # Gelecekte daha karmaşık mantıklar:
            # - Anıların benzerlik skorlarına bakarak bir eşik belirleme.
            # - Anlama modülünden gelen sonucun içeriğini kullanma.
            # - İçsel durumu (merak seviyesi, uyarılma vb.) dikkate alma.
            # - Farklı karar türleri (örn: "hareket et", "ses çıkar", "belleğe kaydet").

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