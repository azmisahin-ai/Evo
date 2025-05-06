# src/cognition/core.py
#
# Evo'nın bilişsel çekirdeğini temsil eder.
# Gelen temsilleri, bellek girdilerini ve işlenmiş anlık duyu özelliklerini kullanarak dünyayı anlamaya çalışır ve bir eylem kararı alır.
# UnderstandingModule ve DecisionModule gibi alt modülleri koordine eder.

import logging # Loglama için.
# numpy, temsil ve bellek verileri için gerekli, ancak doğrudan core.py'de her yerde kullanılmıyor.
import numpy as np # check_input_numpy için gerekli

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

    Bilgi akışını (işlenmiş girdiler, Representation, bellek) alır.
    Bu bilgileri Anlama modülüne ileterek anlama çıktısını alır (dictionary sinyaller).
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
             # Hata durumında alt modül objeleri None kalır.


        logger.info("Cognition modülü başlatıldı.")

    # run_evo.py bu metodu çağırıyor. processed_inputs'u da girdi olarak alacak şekilde güncelledik.
    def decide(self, processed_inputs, learned_representation, relevant_memory_entries):
        """
        İşlenmiş girdiler, öğrenilmiş temsil ve ilgili bellek girdilerine dayanarak bir eylem kararı alır.

        Gelen bilgileri önce anlama modülüne ileterek anlama çıktısını (dictionary sinyaller) alır.
        Anlama çıktısını ve bellek girdilerini Karar Alma modülüne ileterek bir karar alır.

        Args:
            processed_inputs (dict or None): Processor modülleriiden gelen işlenmiş ham veriler.
                                            Beklenen format: {'visual': dict, 'audio': np.ndarray} veya None/boş dict.
            learned_representation (numpy.ndarray or None): RepresentationLearner'dan gelen en son öğrenilmiş temsil vektörü
                                                         veya işleme sırasında hata oluştuysa None.
                                                         Beklenen format: shape (D,), dtype sayısal, veya None.
            relevant_memory_entries (list or None): Memory modülünden gelen ilgili bellek girdileri listesi.
                                            Bellek boşsa veya sorgu sırasında hata oluştuysa boş liste `[]` olabilir.
                                            Beklenen format: list or None.

        Returns:
            str or None: Alınan karar (string)
                         veya anlama/karar alma sırasında hata durumunda None.
        """
        # Hata yönetimi: CognitionCore'un alt modülleri başlatılamamışsa işlem yapma.
        if self.understanding_module is None or self.decision_module is None:
            logger.error("CognitionCore.decide: Alt modüller (Understanding/Decision) başlatılmamış. Karar alınamıyor.")
            return None

        # Girdilerin DecisionModule için geçerli formatta olup olmadığını DecisionModule kendi içinde kontrol ediyor.
        # processed_inputs'un UnderstandingModule için geçerli olup olmadığını UnderstandingModule kontrol ediyor.
        # learned_representation'ın UnderstandingModule için geçerli olup olmadığını UnderstandingModule kontrol ediyor.
        # relevant_memory_entries'in UnderstandingModule ve DecisionModule için geçerli olup olmadığını ilgili modüller kontrol ediyor.


        understanding_signals = None # Anlama modülünden gelecek sinyaller dictionary'si.
        decision = None # Karar alma modülünden gelecek karar.

        try:
            # 1. Gelen bilgileri anlama modülüne ilet.
            # UnderstandingModule.process dictionary sinyalleri döndürür.
            # processed_inputs, learned_representation, relevant_memory_entries argümanlarını UnderstandingModule'e iletiyoruz.
            understanding_signals = self.understanding_module.process(
                processed_inputs, # İşlenmiş anlık duyu girdileri
                learned_representation, # Learned Representation
                relevant_memory_entries # Memory'den gelen ilgili anılar
                # internal_state # Gelecekte buradan da iletilebilir, şimdilik DecisionModule kendi yönetiyor.
            )
            # DEBUG logu: Anlama sinyalleri dictionary'si (UnderstandingModule içinde loglanıyor)
            # if isinstance(understanding_signals, dict): ...


            # 2. Anlama çıktısını ve bellek girdilerini Karar alma modülüne ilet.
            # DecisionModule.decide understanding_signals (dict/None) ve relevant_memory_entries (list/None) bekler.
            # Kendi içsel durumunu (curiosity) DecisionModule yönetiyor.
            decision = self.decision_module.decide(
                understanding_signals, # Anlama modülünün çıktısı (dictionary sinyaller)
                relevant_memory_entries # Bellek girdileri (list/None) - Karar modülü bunu bağlamsal karar için kullanabilir
                # internal_state # Gelecekte buradan da iletilebilir.
            )

            # DEBUG logu: Karar sonucu (DecisionModule içinde loglanıyor)
            # if decision is not None: ...


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