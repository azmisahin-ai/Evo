# src/cognition/core.py
import logging

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

# cognition/understanding.py ve cognition/decision.py dosyaları şu an sadece placeholder olabilir
# Gelecekte bu sınıflar veya fonksiyonlar burada import edilecek
# from .understanding import UnderstandingModule # Gelecek
# from .decision import DecisionModule # Gelecek


class CognitionCore:
    """
    Evo'nın bilişsel çekirdeği. Temsilleri ve belleği kullanarak dünyayı anlamaya çalışır
    ve bir karar alır.
    """
    def __init__(self, config):
        self.config = config
        logger.info("Cognition modülü başlatılıyor...")
        # Alt modüllerin başlatılması buraya gelebilir (Understanding, Decision vb.)
        # self.understanding_module = UnderstandingModule(config.get('understanding', {})) # Gelecek
        # self.decision_module = DecisionModule(config.get('decision', {})) # Gelecek
        logger.info("Cognition modülü başlatıldı.")

    def decide(self, learned_representation, relevant_memory_entries):
        """
        Öğrenilmiş temsil ve ilgili bellek girdilerine dayanarak bir eylem kararı alır.
        Şimdilik çok basit placeholder mantığı.

        Args:
            learned_representation (numpy.ndarray or None): RepresentationLearner'dan gelen temsil.
            relevant_memory_entries (list): Memory modülünden gelen ilgili bellek girdileri listesi.

        Returns:
            str or None: Alınan karar (metin olarak temsil ediliyor) veya hata durumunda None.
                         Gelecekte daha yapısal bir karar formatı (örn: {'action': 'move', 'params': {'direction': 'forward'}})
        """
        # Temel hata yönetimi: Girdilerin varlığı kontrolü (Representation None olabilir, liste boş olabilir, sorun değil)
        # Sadece her ikisi de None/boş ise bir problem olabilir (gerçi bu durumda main loop decide'ı çağırmazdı).
        # Yine de iç kontrol ekleyelim:
        if learned_representation is None and not relevant_memory_entries:
             logger.debug("CognitionCore.decide: Karar almak için yeterli girdi (temsil veya bellek) yok.")
             return None # Karar alınamıyorsa None döndür

        decision = None # Alınan kararı tutacak değişken

        try:
            # Basit Placeholder Karar Mantığı:
            # Eğer yeni bir temsil veya ilgili bellek girdileri varsa, "işleme ve hatırlama" kararı al.
            # Gelecekte: input'un içeriğine, bellektekilerle ilişkisine, içsel duruma göre karmaşık karar.
            if learned_representation is not None or relevant_memory_entries:
                decision = "processing_and_remembering" #Placeholder: 'Çevreyi algılıyorum ve hatırlıyorum.' çıktısı için.
            else:
                 # Bu durum normalde yukarıdaki kontrol yüzünden oluşmaz ama yine de
                 logger.debug("CognitionCore.decide: Girdi olmasına rağmen karar alınmadı (placeholder mantığına göre).")
                 decision = None


            # Gelecekte:
            # processed_understanding = self.understanding_module.process(learned_representation, relevant_memory_entries) # Anlama modülü
            # decision = self.decision_module.decide_based_on(processed_understanding, internal_state) # Karar modülü


            # DEBUG logu: Alınan karar (None değilse)
            # if decision is not None:
            #      logger.debug(f"CognitionCore.decide: Bilişsel karar alındı (placeholder): {decision}")


        except Exception as e:
            # Karar alma sırasında beklenmedik hata
            logger.error(f"CognitionCore.decide: Karar alma sırasında beklenmedik hata: {e}", exc_info=True)
            return None # Hata durumunda None döndür

        return decision # Başarılı durumda kararı döndür

    def cleanup(self):
        """Kaynakları temizler (alt modülleri vb.)."""
        logger.info("Cognition modülü objesi siliniyor...")
        # Alt modüllerin cleanup metodunu çağır (varsa)
        # if hasattr(self.understanding_module, 'cleanup'): self.understanding_module.cleanup() # Gelecek
        # if hasattr(self.decision_module, 'cleanup'): self.decision_module.cleanup() # Gelecek
        logger.info("Cognition modülü objesi silindi.")