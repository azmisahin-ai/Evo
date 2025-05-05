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
from .understanding import UnderstandingModule # <<< UnderstandingModule import edildi
from .decision import DecisionModule # <<< DecisionModule import edildi


# Bu modül için bir logger oluştur
# 'src.cognition.core' adında bir logger döndürür.
logger = logging.getLogger(__name__)


class CognitionCore:
    """
    Evo'nın bilişsel çekirdek sınıfı.

    Bilgi akışını (temsil, bellek) alır, anlama modülüne iletir,
    anlama sonucunu ve bellek girdilerini karar alma modülüne iletir
    ve karar alma modülünden gelen kararı döndürür.
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

        Returns:
            str or None: Alınan karar (formatı gelecekte belirlenecek, örn: string veya dict)
                         veya karar alınamadıysa ya da hata durumunda None.
        """
        # Hata yönetimi: CognitionCore'un alt modülleri başlatılamamışsa işlem yapma.
        if self.understanding_module is None or self.decision_module is None:
            logger.error("CognitionCore.decide: Alt modüller (Understanding/Decision) başlatılmamış. Karar alınamıyor.")
            return None

        # Hata yönetimi: Girdilerin beklenen tiplerde olup olmadığını kontrol et.
        # check_numpy_input ve check_input_type alt metotlarda da kullanılıyor, ama burada da kontrol edilebilir.
        # Burada daha çok None olup olmadığını kontrol etmek veya tipleri loglamak yeterli olabilir.
        # learned_representation: None veya numpy array (sayısal, 1D) beklenir.
        # relevant_memory_entries: list beklenir.

        # Representation tipi kontrolü (None veya beklenen numpy array)
        # check_numpy_input(learned_representation, expected_dtype=np.number, expected_ndim=1, input_name="learned_representation", logger_instance=logger) # Loglama utils içinde yapılıyor
        # Bellek girdileri tipi kontrolü (liste)
        # check_input_type(relevant_memory_entries, list, input_name="relevant_memory_entries", logger_instance=logger) # Loglama memory içinde yapılıyor.


        understanding_result = None # Anlama modülünden gelecek sonuç.
        decision = None # Karar alma modülünden gelecek karar.

        try:
            # 1. Gelen temsil ve bellek girdilerini anlama modülüne ilet.
            # Anlama modülünün process metodu None döndürebilir (hata veya anlam çıkmaması).
            understanding_result = self.understanding_module.process(
                learned_representation,
                relevant_memory_entries
            )
            # DEBUG logu: Anlama sonucu (None değilse)
            # if understanding_result is not None:
            #      logger.debug(f"CognitionCore.decide: Anlama sonucu alindi (Placeholder): {understanding_result}")


            # 2. Anlama sonucunu ve bellek girdilerini karar alma modülüne ilet.
            # Karar alma modülünün decide metodu None döndürebilir (hata veya karar alınmaması).
            decision = self.decision_module.decide(
                understanding_result, # Anlama modülünün çıktısı
                relevant_memory_entries # Bellek girdileri tekrar karar için kullanılabilir
                # İçsel durum (internal_state) gelecekte buraya eklenecek.
            )

            # DEBUG logu: Karar sonucu (None değilse)
            # if decision is not None: # Zaten None değilse buraya gelinir.
            #      logger.debug(f"CognitionCore.decide: Karar sonucu alindi (Placeholder): '{decision}'")


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
        module_loader.py bu metodu program sonlanırken çağırır (varsa).
        """
        logger.info("Cognition modülü objesi siliniyor...")
        # Alt modüllerin cleanup metotlarını çağır (varsa).
        if hasattr(self.understanding_module, 'cleanup'):
             self.understanding_module.cleanup()
        if hasattr(self.decision_module, 'cleanup'):
             self.decision_module.cleanup()

        logger.info("Cognition modülü objesi silindi.")