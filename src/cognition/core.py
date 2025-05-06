# src/cognition/core.py
#
# Evo'nın bilişsel çekirdeğini temsil eder.
# Gelen temsilleri ve bellekteki anıları kullanarak dünyayı anlamaya çalışır ve bir eylem kararı alır.
# UnderstandingModule ve DecisionModule gibi alt modülleri koordine eder.

import logging # Loglama için.
# numpy, temsil ve bellek verileri için gerekli, ancak doğrudan core.py'de her yerde kullanılmıyor.
import numpy as np # check_numpy_input için gerekli

# Yardımcı fonksiyonları import et
from src.core.utils import check_input_not_none, check_numpy_input, check_input_type # <<< check_input_not_none, check_numpy_input, check_input_type import edildi

# Alt modül sınıflarını import et
from .understanding import UnderstandingModule
from .decision import DecisionModule


# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)


class CognitionCore:
    """
    Evo'nın bilişsel çekirdek sınıfı.

    Bilgi akışını (temsil, bellek) alır.
    Bu bilgileri Anlama modülüne ileterek anlama çıktısını alır (şimdilik benzerlik skoru).
    Anlama çıktısını ve bellek girdilerini Karar Alma modülüne ileterek bir karar alır.
    UnderstandingModule ve DecisionModule alt modüllerini koordine eder.
    Hata durumlarında işlemleri loglar ve programın çökmesini engeller.
    """
    def __init__(self, config):
        """
        CognitionCore modülünü başlatır.

        Alt modülleri (UnderstandingModule, DecisionModule) başlatır.
        Başlatma sırasında hata oluşursa alt modüllerin objeleri None kalabilir.

        Args:
            config (dict): Bilişsel çekirdek yapılandırma ayarları.
                           Alt modüllere ait ayarlar kendi adları altında beklenir
                           (örn: {'understanding': {...}, 'decision': {...}}).
        """
        self.config = config
        logger.info("Cognition modülü başlatılıyor...")

        self.understanding_module = None # Anlama modülü objesi.
        self.decision_module = None # Karar alma modülü objesi.

        # Alt modülleri başlatmayı dene. Başlatma hataları kendi içlerinde loglanır.
        # initialize_modules'daki hata yönetimi, alt modüllerin başlatılmasının
        # main loop'u durdurup durdurmayacağını kontrol eder (CognitionCore'un kendisi kritik olduğu için).
        try:
            # Anlama modülünü yapılandırmasından başlat
            understanding_config = config.get('understanding', {})
            # Alt modülün init metodunun hata durumunda None döndürmesi veya exception atması beklenir.
            self.understanding_module = UnderstandingModule(understanding_config)
            if self.understanding_module is None:
                 logger.error("CognitionCore: UnderstandingModule başlatılamadı.")

            # Karar alma modülünü yapılandırmasından başlat
            decision_config = config.get('decision', {})
            self.decision_module = DecisionModule(decision_config)
            if self.decision_module is None:
                 logger.error("CognitionCore: DecisionModule başlatılamadı.")

        except Exception as e:
             # Alt modül başlatma sırasında beklenmedik hata olursa
             # Bu hata initialize_modules tarafından yakalanır ve CognitionCore'u kritik olarak işaretler.
             logger.critical(f"CognitionCore: Alt modülleri başlatılırken hata oluştu: {e}", exc_info=True)
             # Hata durumunda alt modül objeleri None kalır.


        logger.info("Cognition modülü başlatıldı.")

    # run_evo.py bu metodu çağırıyor. processed_inputs da aslında geliyor ama şu anki tanımda yok.
    # TODO: processed_inputs'u da girdi olarak alacak şekilde güncelleyin: decide(self, processed_inputs, learned_representation, relevant_memory_entries)
    def decide(self, learned_representation, relevant_memory_entries):
        """
        Öğrenilmiş temsil ve ilgili bellek girdilerine dayanarak bir eylem kararı alır.

        Gelen representation ve bellek girdilerini önce anlama modülüne iletir,
        ardından anlama sonucu ve bellek girdilerini karar alma modülüne iletir
        ve karar alma modülünden gelen kararı döndürür.
        Alt modüller (understanding, decision) başlatılmamışsa veya işlem sırasında hata
        oluşursa None döndürür.

        Args:
            learned_representation (numpy.ndarray or None): RepresentationLearner'dan gelen en son öğrenilmiş temsil vektörü
                                                         veya işleme sırasında hata oluştuysa None.
                                                         Beklenen format: shape (D,), dtype sayısal, veya None.
            relevant_memory_entries (list): Memory modülünden gelen ilgili bellek girdileri listesi.
                                            Bellek boşsa veya sorgu sırasında hata oluştuysa boş liste `[]` olabilir.
                                            Beklenen format: liste.
            # processed_inputs (dict, optional): Processor modüllerinden gelen işlenmiş ham veriler. Gelecekte kullanılacak.


        Returns:
            str or None: Alınan karar (formatı gelecekte belirlenecek, örn: string veya dict)
                         veya karar alınamadıysa ya da hata durumunda None.
        """
        # Hata yönetimi: CognitionCore'un alt modülleri başlatılamamışsa işlem yapma.
        if self.understanding_module is None or self.decision_module is None:
            logger.error("CognitionCore.decide: Alt modüller (Understanding/Decision) başlatılmamış. Karar alınamıyor.")
            return None

        # Hata yönetimi: Girdilerin DecisionModule için geçerli formatta olup olmadığını kontrol et.
        # DecisionModule sadece understanding_result'ı (float/None) ve relevant_memory_entries'i (list/None) kullanıyor.
        # learned_representation ve relevant_memory_entries'in tip kontrolleri UnderstandingModule ve DecisionModule içinde zaten yapılıyor.
        # processed_inputs (gelecekte eklenecek) için de burada veya alt modüllerde kontrol yapılmalı.


        understanding_result = None # Anlama modülünden gelecek sonuç (benzerlik skoru).
        decision = None # Karar alma modülünden gelecek karar.

        try:
            # 1. Gelen temsil ve bellek girdilerini anlama modülüne ilet.
            # UnderstandingModule.process artık float (en yüksek benzerlik skoru) veya 0.0/None döndürür.
            understanding_result = self.understanding_module.process(
                learned_representation, # Learned Representation (Representation module'den)
                relevant_memory_entries # Memory'den gelen ilgili anılar
                # processed_inputs # Gelecekte buraya eklenecek.
            )
            # DEBUG logu: Anlama sonucu (benzerlik skoru)
            # understanding_result float, 0.0 veya None olabilir.
            if isinstance(understanding_result, float): # Eğer float döndüyse (0.0 dahil)
                 logger.debug(f"CognitionCore.decide: Anlama sonucu alindi (benzerlik skoru): {understanding_result:.4f}")
            elif understanding_result is not None: # Float değil ama None da değilse (Beklenmeyen durum)
                 logger.warning(f"CognitionCore.decide: Anlama sonucu beklenmeyen tipte: {type(understanding_result)}. Float veya None bekleniyordu.")
            else: # None ise
                 logger.debug("CognitionCore.decide: Anlama sonucu None.")


            # 2. Anlama sonucunu ve bellek girdilerini karar alma modülüne ilet.
            # DecisionModule.decide artık understanding_result'ı (float/None) ve relevant_memory_entries'i (list/None) bekler.
            decision = self.decision_module.decide(
                understanding_result, # Anlama modülünün çıktısı (benzerlik skoru)
                relevant_memory_entries # Bellek girdileri (list/None) - Gelecekte bağlamsal karar için
                # internal_state # Gelecekte eklenecek.
            )

            # DEBUG logu: Karar sonucu
            # DecisionModule.decide string veya None döndürür.
            if decision is not None:
                 if isinstance(decision, str):
                     logger.debug(f"CognitionCore.decide: Karar sonucu alindi (string): '{decision}'")
                 else: # String değilse (Beklenmeyen durum)
                     logger.warning(f"CognitionCore.decide: Karar sonucu beklenmeyen tipte: {type(decision)}. String veya None bekleniyordu.")
                     logger.debug(f"CognitionCore.decide: Karar sonucu: {repr(decision)}")
            else:
                 logger.debug("CognitionCore.decide: Karar sonucu None.")


        except Exception as e:
            # Alt modüllerin metotlarını çağırırken veya içlerinde (eğer yakalamadılarsa) beklenmedik hata olursa.
            logger.error(f"CognitionCore.decide: Anlama veya karar alma sırasında beklenmedik hata: {e}", exc_info=True)
            return None # Hata durumunda None döndür.

        # Başarılı durumda alınan kararı döndür.
        return decision

    def cleanup(self):
        """
        CognitionCore modülü kaynaklarını temizler.

        Alt modüllerin (UnderstandingModule, DecisionModule) cleanup metotlarını (varsa) çağırır.
        module_loader.py bu metotu program sonlanırken çağrır (varsa).
        """
        logger.info("Cognition modülü objesi siliniyor...")
        # Alt modüllerin cleanup metotlarını çağır (varsa).
        if self.understanding_module and hasattr(self.understanding_module, 'cleanup'):
             self.understanding_module.cleanup()
        if self.decision_module and hasattr(self.decision_module, 'cleanup'):
             self.decision_module.cleanup()

        logger.info("Cognition modülü objesi silindi.")