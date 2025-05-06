# src/cognition/understanding.py
#
# Evo'nın anlama modülünü temsil eder.
# İşlenmiş temsilleri ve bellekteki anıları kullanarak dünyayı anlamaya çalışır.
# Çeşitli anlama algoritmalarını içerecektir.

import logging
import numpy as np # Benzerlik hesaplaması için

# Yardımcı fonksiyonları import et (girdi kontrolleri için)
from src.core.utils import check_input_not_none, check_input_type, check_numpy_input # <<< Utils importları

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

class UnderstandingModule:
    """
    Evo'nın anlama yeteneğini sağlayan sınıf (Faz 3 implementasyonu).

    RepresentationLearner'dan gelen temsilleri ve Memory'den gelen ilgili anıları alır.
    Mevcut implementasyon: Gelen Representation ile ilgili anılar arasındaki en yüksek
    vektör benzerlik skorunu hesaplayarak ilkel bir "anlama" çıktısı (float) üretir.
    Gelecekte daha karmaşık anlama algoritmaları implement edilecektir.
    """
    def __init__(self, config):
        """
        UnderstandingModule'ü başlatır.

        Args:
            config (dict): Anlama modülü yapılandırma ayarları.
                           Gelecekte model yolları, anlama stratejileri gibi ayarlar gelebilir.
        """
        self.config = config
        logger.info("UnderstandingModule başlatılıyor (Faz 3)...")
        # Modül başlatma mantığı buraya gelebilir (örn: model yükleme)
        logger.info("UnderstandingModule başlatıldı.")

    def process(self, learned_representation, relevant_memory_entries):
        """
        Gelen Representation ve ilgili bellek girdilerini kullanarak anlama işlemini yapar.

        Mevcut implementasyon: Learned Representation ile ilgili anılar arasındaki
        en yüksek kosinüs benzerlik skorunu hesaplar ve döndürür.

        Args:
            learned_representation (numpy.ndarray or None): En son öğrenilmiş temsil vektörü.
                                                         Beklenen format: shape (D,), dtype sayısal, veya None.
            relevant_memory_entries (list): Memory modülünden gelen ilgili bellek girdileri listesi.
                                            Bu liste Representation benzerliğine göre sıralanmış olabilir
                                            (Memory.retrieve'e göre), ancak biz yine de doğruluğunu kontrol edeceğiz.
                                            Her öğe {'representation': np.ndarray, ...} formatında olmalıdır.
                                            Boş veya None olabilir.

        Returns:
            float or None: En yüksek kosinüs benzerlik skoru (0.0 ile 1.0 arası float).
                           Eğer Representation None/geçersizse, bellek boşsa/geçersizse
                           veya hesaplama sırasında hata olursa 0.0 döndürülür.
        """
        # Girdi kontrolleri.
        # Representation geçerli mi?
        if not check_input_not_none(learned_representation, input_name="learned_representation for UnderstandingModule", logger_instance=logger) or \
           not check_numpy_input(learned_representation, expected_dtype=np.number, expected_ndim=1, input_name="learned_representation for UnderstandingModule", logger_instance=logger):
             logger.debug("UnderstandingModule.process: Learned representation None veya geçersiz. Anlama yapılamıyor.")
             return 0.0 # Geçersiz representation inputu durumunda skor 0.0

        # Bellek girdileri geçerli bir liste mi?
        if not check_input_not_none(relevant_memory_entries, input_name="relevant_memory_entries for UnderstandingModule", logger_instance=logger) or \
           not check_input_type(relevant_memory_entries, list, input_name="relevant_memory_entries for UnderstandingModule", logger_instance=logger):
             logger.debug("UnderstandingModule.process: relevant_memory_entries None veya liste değil. Anlama yapılamıyor.")
             return 0.0 # Geçersiz bellek girdileri durumunda skor 0.0

        # Bellek listesi boş mu?
        if not relevant_memory_entries:
             logger.debug("UnderstandingModule.process: relevant_memory_entries boş liste. Anlama yapılamıyor.")
             return 0.0 # Bellek boşsa skor 0.0

        logger.debug(f"UnderstandingModule.process: Representation (Shape: {learned_representation.shape}) ve {len(relevant_memory_entries)} bellek girdisi alındı. Anlama yapılıyor.")

        max_similarity = 0.0 # Başlangıçta en yüksek benzerlik 0.0
        query_norm = np.linalg.norm(learned_representation)

        # Sorgu vektörü sıfır ise benzerlik hesaplanamaz.
        if query_norm < 1e-8: # Sıfıra yakınlığı kontrol et
             logger.warning("UnderstandingModule.process: Learned representation has near-zero norm. Cannot calculate similarity. Returning 0.0.")
             return 0.0

        try:
            # Bellekteki her anı ile sorgu arasındaki benzerliği hesapla ve en yükseği bul.
            for memory_entry in relevant_memory_entries:
                # Anıdaki representation'ı al ve geçerliliğini kontrol et.
                # Memory.retrieve zaten Representation vektörlerini içeren dict'ler döndürmeyi hedefler.
                # Burada sadece Representation'ın kendisinin geçerli numpy array olduğunu kontrol edelim.
                stored_representation = memory_entry.get('representation') # .get ile güvenli erişim

                if stored_representation is not None and isinstance(stored_representation, np.ndarray) and np.issubdtype(stored_representation.dtype, np.number) and stored_representation.ndim == 1:
                     stored_norm = np.linalg.norm(stored_representation)
                     if stored_norm > 1e-8: # Stored vektör sıfır ise bölme hatası olmaması için kontrol et.
                         # Kosinüs benzerliği hesapla.
                         similarity = np.dot(learned_representation, stored_representation) / (query_norm * stored_norm)
                         # Benzerlik değeri NaN (Not a Number) olabilir (örn: sıfıra bölme hatası veya geçersiz input).
                         # NaN değerler genellikle sıralamada sorun yaratır, bu yüzden kontrol edip yoksayalım.
                         if not np.isnan(similarity):
                            max_similarity = max(max_similarity, similarity) # En yüksek benzerliği güncelle
                         else:
                            logger.warning("UnderstandingModule.process: Calculated NaN similarity for a memory entry. Skipping.")
                     # else: logger.debug("UnderstandingModule.process: Stored representation has near-zero norm, skipping similarity calculation.")
                # else: logger.debug("UnderstandingModule.process: Stored entry does not contain a valid numeric 1D numpy array representation, skipping.")


            logger.debug(f"UnderstandingModule.process: En yüksek bellek benzerlik skoru hesaplandı: {max_similarity:.4f}")


        except Exception as e:
            # Hesaplama sırasında beklenmedik bir hata olursa logla.
            logger.error(f"UnderstandingModule.process: Anlama işlemi sırasında beklenmedik hata: {e}", exc_info=True)
            return 0.0 # Hata durumunda skor 0.0 döndür

        return max_similarity # Hesaplanan en yüksek benzerlik skorunu döndür.

    def cleanup(self):
        """
        UnderstandingModule kaynaklarını temizler.

        Şimdilik özel bir kaynak kullanmadığı için temizleme adımı içermez.
        Gelecekte model temizliği veya bağlantı kapatma gerekebilir.
        module_loader.py bu metodu program sonlanırken çağırır (varsa).
        """
        logger.info("UnderstandingModule objesi temizleniyor.")
        pass