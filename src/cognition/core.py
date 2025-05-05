# src/cognition/core.py
#
# Evo'nın bilişsel çekirdeğini temsil eder.
# Gelen temsilleri ve bellekteki anıları kullanarak dünyayı anlamaya çalışır ve bir eylem kararı alır.

import logging # Loglama için.
# numpy, temsil ve bellek verileri için gerekli, ancak doğrudan core.py'de her yerde kullanılmıyor.
import numpy as np # check_numpy_input için gerekli

# Yardımcı fonksiyonları import et
from src.core.utils import check_input_not_none, check_numpy_input, check_input_type # <<< Yeni importlar


# Bu modül için bir logger oluştur
# 'src.cognition.core' adında bir logger döndürür.
logger = logging.getLogger(__name__)

# cognition/understanding.py ve cognition/decision.py dosyaları şu an sadece placeholder olabilir.
# Gelecekte bu sınıflar veya fonksiyonlar burada import edilecek ve kullanılacak.
# from .understanding import UnderstandingModule # Gelecek TODO
# from .decision import DecisionModule # Gelecek TODO


class CognitionCore:
    """
    Evo'nın bilişsel çekirdek sınıfı.

    RepresentationLearner'dan gelen öğrenilmiş temsilleri ve Memory modülünden
    gelen ilgili bellek girdilerini girdi olarak alır.
    Bu girdilere dayanarak bir eylem kararı (örn: 'interact', 'explore', 'rest') alır.
    Şimdilik çok basit placeholder karar mantığı içerir.
    Gelecekte daha karmaşık anlama, akıl yürütme ve karar alma mekanizmaları buraya eklenecek.
    """
    def __init__(self, config):
        """
        CognitionCore modülünü başlatır.

        Args:
            config (dict): Bilişsel çekirdek yapılandırma ayarları.
                           Gelecekte alt modüllerin (Understanding, Decision) veya
                           karar eşiklerinin ayarları buraya gelebilir.
        """
        self.config = config
        logger.info("Cognition modülü başlatılıyor...")
        # Alt modüllerin başlatılması buraya gelebilir (Gelecek TODO).
        # self.understanding_module = UnderstandingModule(config.get('understanding', {})) # Gelecek
        # self.decision_module = DecisionModule(config.get('decision', {})) # Gelecek
        logger.info("Cognition modülü başlatıldı.")

    def decide(self, learned_representation, relevant_memory_entries):
        """
        Öğrenilmiş temsil ve ilgili bellek girdilerine dayanarak bir eylem kararı alır.

        Gelen representation ve bellek girdilerini kullanarak Evo'nın ne yapacağına
        karar verir. Şimdilik çok basit placeholder mantığı.
        Girdiler None veya boş olabilir, bu durumları yönetir.
        Karar alma sırasında hata oluşursa None döndürür.

        Args:
            learned_representation (numpy.ndarray or None): RepresentationLearner'dan gelen en son öğrenilmiş temsil vektörü
                                                         veya işleme sırasında hata oluştuysa None.
                                                         Beklenen format: shape (D,), dtype sayısal, veya None.
            relevant_memory_entries (list): Memory modülünden gelen ilgili bellek girdileri listesi.
                                            Bellek boşsa veya sorgu sırasında hata oluştuysa boş liste `[]` olabilir.
                                            Beklenen format: liste.

        Returns:
            str or None: Alınan karar (şimdilik metin olarak temsil ediliyor, örn: 'processing_and_remembering')
                         veya karar alınamadıysa ya da hata durumunda None.
                         Gelecekte daha yapısal bir karar formatı beklenir (örn: {'action': 'move', 'params': {'direction': 'forward'}}).
        """
        # Hata yönetimi: Karar almak için temel girdilerin varlığı kontrolü.
        # Representation None olabilir veya bellek girdileri boş olabilir. Bu durumlar normaldir.
        # check_input_not_none fonksiyonu sadece None kontrolü yapar. Burada None olması hata değil.

        # Hata yönetimi: Girdilerin beklenen tiplerde olup olmadığını kontrol et.
        # learned_representation: None veya numpy array (sayısal, 1D) beklenir.
        # relevant_memory_entries: list beklenir.

        # Representation tipi kontrolü (None veya beklenen numpy array)
        if learned_representation is not None and not check_numpy_input(learned_representation, expected_dtype=np.number, expected_ndim=1, input_name="learned_representation", logger_instance=logger):
             logger.warning("CognitionCore.decide: Learned representation beklenmeyen tip veya formatta, yoksayılıyor.")
             learned_representation = None # Geçersizse None olarak ele al.

        # Bellek girdileri tipi kontrolü (liste)
        if not check_input_type(relevant_memory_entries, list, input_name="relevant_memory_entries", logger_instance=logger):
             logger.warning("CognitionCore.decide: Relevant memory entries beklenmeyen tipte, boş liste olarak ele alınıyor.")
             relevant_memory_entries = [] # Geçersizse boş liste olarak ele al.

        # Karar almak için yeterli bilgi var mı kontrolü (Temsil VEYA Bellek Girdisi).
        if learned_representation is None and not relevant_memory_entries:
             # Hem temsil yoksa hem de bellek girdisi yoksa karar alınamaz.
             logger.debug("CognitionCore.decide: Karar almak için yeterli girdi (temsil veya bellek) yok.")
             return None # Karar alınamıyorsa None döndür.

        decision = None # Alınan kararı tutacak değişken. Başlangıçta None.

        try:
            # Basit Placeholder Karar Mantığı:
            # Eğer en son bir temsil öğrenildiyse VEYA ilgili bellek girdileri varsa,
            # "işleme ve hatırlama" kararı al.
            if learned_representation is not None or relevant_memory_entries:
                decision = "processing_and_remembering" #Placeholder karar stringi.
            else:
                 # Bu durum, yukarıdaki kontrol nedeniyle buraya gelmemeli.
                 logger.debug("CognitionCore.decide: Girdi olmasına rağmen (placeholder mantığına göre) karar alınmadı.")
                 decision = None # Karar yoksa None döndür.


            # Gelecekte Kullanım Örneği (Alt Modüller):
            # # Anlama modülünü kullanarak girdiyi anlamlandır
            # processed_understanding = self.understanding_module.process(learned_representation, relevant_memory_entries)
            # # Karar modülünü kullanarak anlama ve içsel duruma göre karar al
            # decision = self.decision_module.decide_based_on(processed_understanding, self.internal_state)


            # DEBUG logu: Alınan karar (None değilse).
            # if decision is not None: # Zaten None değilse buraya gelinir.
            #      logger.debug(f"CognitionCore.decide: Bilişsel karar alındı (placeholder): '{decision}'")


        except Exception as e:
            # Karar alma işlemi sırasında beklenmedik bir hata oluşursa logla.
            # Örneğin, gelecekteki karmaşık karar algoritmalarında veya model çalıştırırken hata olabilir.
            logger.error(f"CognitionCore.decide: Karar alma sırasında beklenmedik hata: {e}", exc_info=True)
            return None # Hata durumunda None döndürerek main loop'un çökmesini engelle.

        # Başarılı durumda alınan kararı döndür.
        return decision

    def cleanup(self):
        """
        CognitionCore modülü kaynaklarını temizler.

        Şimdilik özel bir kaynak kullanmadığı için temizleme adımı içermez,
        sadece bilgilendirme logu içerir.
        Gelecekte alt modüllerin (Understanding, Decision) cleanup metotları varsa
        burada çağrılmaları gerekir.
        module_loader.py bu metodu program sonlanırken çağırır (varsa).
        """
        logger.info("Cognition modülü objesi siliniyor...")
        # Alt modüllerin cleanup metotlarını çağır (varsa) (Gelecek TODO).
        # if hasattr(self.understanding_module, 'cleanup'): self.understanding_module.cleanup()
        # if hasattr(self.decision_module, 'cleanup'): self.decision_module.cleanup()

        logger.info("Cognition modülü objesi silindi.")