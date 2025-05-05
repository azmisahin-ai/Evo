# src/memory/core.py
#
# Evo'nın temel bellek sistemini temsil eder.
# Öğrenilmiş temsilleri saklar ve gerektiğinde geri çağırır.

import numpy as np # Temsil vektörleri (numpy array) için.
import time # Anıların zaman damgası için.
import random # Placeholder retrieve (rastgele seçim) için.
import logging # Loglama için.

# Yardımcı fonksiyonları import et
from src.core.utils import check_input_not_none, check_numpy_input, get_config_value, check_input_type # <<< Yeni importlar


# Bu modül için bir logger oluştur
# 'src.memory.core' adında bir logger döndürür.
logger = logging.getLogger(__name__)

class Memory:
    """
    Evo'nın bellek sistemi sınıfı.

    RepresentationLearner'dan gelen öğrenilmiş temsil vektörlerini depolar.
    Basit bir kısa süreli bellek gibi çalışır (maksimum boyutu aşarsa en eski anıyı siler - FIFO).
    Bellekten ilgili anıları geri çağırma (retrieve) yeteneğine sahiptir (şimdilik placeholder).
    Hata durumlarında işlemleri loglar ve programın çökmesini engeller.
    """
    def __init__(self, config):
        """
        Memory modülünü başlatır.

        Args:
            config (dict): Bellek sistemi yapılandırma ayarları.
                           'max_memory_size': Bellekte saklanacak maksimum temsil sayısı (int, varsayılan 1000).
                           'num_retrieved_memories': retrieve metodunda varsayılan olarak geri çağrılacak anı sayısı (int, varsayılan 5).
                           Gelecekte kalıcı depolama ayarları (file_path vb.) buraya gelebilir.
        """
        self.config = config
        logger.info("Memory modülü başlatılıyor...")

        # Yapılandırmadan ayarları alırken get_config_value kullan
        self.max_memory_size = get_config_value(config, 'max_memory_size', 1000, expected_type=int, logger_instance=logger)
        # num_retrieved_memories int > 0 olmalı. get_config_value ile int kontrolü yapalım.
        self.num_retrieved_memories = get_config_value(config, 'num_retrieved_memories', 5, expected_type=int, logger_instance=logger)
        # Eğer num_retrieved_memories 0 veya negatif gelirse düzeltelim.
        if self.num_retrieved_memories < 0:
             logger.warning(f"Memory: Konfigurasyonda num_retrieved_memories negatif ({self.num_retrieved_memories}). Varsayılan 5 kullanılıyor.")
             self.num_retrieved_memories = 5


        # Bellek depolama yapısı: Anı öğelerinin listesi.
        # Her öğe bir sözlüktür: {'representation': numpy_array, 'metadata': dict, 'timestamp': float}
        self.memory_storage = []

        # Kalıcı bellekten yükleme mantığı buraya gelebilir (Gelecek TODO).
        # Örneğin, dosyadan veya veritabanından eski anıları yükleme.
        # self._load_from_storage() # Gelecek TODO


        logger.info(f"Memory modülü başlatıldı. Maksimum boyut: {self.max_memory_size}. Varsayılan geri çağrı sayısı: {self.num_retrieved_memories}")


    def store(self, representation, metadata=None):
        """
        Öğrenilmiş bir temsili (ve ilişkili metadatayı) belleğe kaydeder.

        Eğer representation None veya beklenmeyen numpy array tipinde ise kaydetme işlemini atlar.
        Bellek boyutu self.max_memory_size değerini aşarsa, en eski anıyı (listenin başındaki) siler (FIFO prensibi).
        Başarısızlık durumunda hatayı loglar.

        Args:
            representation (numpy.ndarray or None): Belleğe saklanacak temsil vektörü.
                                                    Genellikle RepresentationLearner'dan gelir.
                                                    Beklenen format: shape (D,), dtype sayısal.
            metadata (dict, optional): Temsille ilişkili ek bilgiler (örn: kaynak, zaman aralığı, durum vb.).
                                       Varsayılan None. None ise boş sözlük olarak saklanır.
                                       Beklenen tip: dict veya None.
        """
        # Hata yönetimi: Saklanacak representation None mu? check_input_not_none kullan.
        if not check_input_not_none(representation, input_name="representation", logger_instance=logger):
             return # None ise saklama atla.

        # Hata yönetimi: Representation'ın numpy array ve sayısal dtype olup olmadığını kontrol et.
        # expected_ndim=1 çünkü representation genellikle 1D vektördür.
        if not check_numpy_input(representation, expected_dtype=np.number, expected_ndim=1, input_name="representation", logger_instance=logger):
             return # Geçersiz tip, dtype veya boyut ise saklama atla.

        # Hata yönetimi: Metadata None veya dict mi? check_input_type kullan.
        if metadata is not None and not check_input_type(metadata, dict, input_name="metadata", logger_instance=logger):
             # Metadata None değil ama dict de değilse uyarı logla ve metadata'yı None yap.
             logger.warning("Memory.store: Metadata beklenmeyen tipte, yoksayılıyor.")
             metadata = None


        try:
            # Yeni bellek öğesi oluştur.
            # Representation, metadata (varsa) ve zaman damgasını içerir.
            memory_entry = {
                'representation': representation,
                'metadata': metadata if metadata is not None else {}, # metadata None ise boş sözlük sakla.
                'timestamp': time.time() # Kayıt zamanı (epoch zamanı float olarak).
            }

            # Bellek depolama listesine yeni öğeyi ekle (listenin sonuna).
            self.memory_storage.append(memory_entry)
            # DEBUG logu: Saklama işleminin başarıyla yapıldığı ve güncel boyut bilgisi.
            # logger.debug(f"Memory.store: Temsil başarıyla saklandı. Güncel hafıza boyutu: {len(self.memory_storage)}")


            # Maksimum bellek boyutu aşıldıysa en eski öğeyi sil (FIFO).
            if len(self.memory_storage) > self.max_memory_size:
                # max_memory_size 0 veya negatif ise bu kontrol gereksiz olabilir ama >= 0 varsayıyoruz.
                # Listenin başındaki (en eski) öğeyi çıkar.
                removed_entry = self.memory_storage.pop(0)
                # DEBUG logu: Silinen anı hakkında bilgi.
                # logger.debug(f"Memory.store: Maksimum boyut aşıldı ({self.max_memory_size}). En eski anı silindi (timestamp: {removed_entry['timestamp']:.2f}).")


        except Exception as e:
            # Saklama işlemi sırasında beklenmedik bir hata oluşursa logla.
            # Listeye ekleme veya listeden çıkarma sırasında oluşabilecek hatalar nadirdir.
            logger.error(f"Memory.store: Belleğe kaydetme sırasında beklenmedik hata: {e}", exc_info=True)
            # Hata durumunda programın çökmesini engelle, sadece logla ve devam et.


    def retrieve(self, query_representation, num_results=None):
        """
        Bellekten ilgili anıları geri çağırır.

        Şimdilik placeholder mantığı: query_representation parametresini kullanmaz.
        Bunun yerine bellekteki anıların bir alt kümesini (belirtilen sayıda rastgele anı)
        veya belleğin tamamını (num_results çok büyükse veya None ise) döndürür.
        Gelecekte 'query_representation' kullanılarak benzerlik (similarity) tabanlı arama yapılacak.
        Hata durumunda veya bellek boşsa boş liste döndürür.

        Args:
            query_representation (numpy.ndarray or None): Sorgu için kullanılan temsil vektörü.
                                                         Gelecekte anılarla benzerliğini ölçmek için kullanılacak.
                                                         Şimdilik kullanılmıyor, None veya numpy array olabilir.
            num_results (int, optional): Geri çağrılacak maksimum anı sayısı.
                                         Varsayılan self.num_retrieved_memories.
                                         None ise varsayılan kullanılır. Geçersiz int ise varsayılan veya 5 kullanılır.

        Returns:
            list: Geri çağrılan bellek öğelerinin listesi.
                  Her öğe bir sözlüktür: {'representation': numpy_array, 'metadata': dict, 'timestamp': float}.
                  Hata durumunda veya bellek boşsa boş liste `[]` döner.
        """
        # num_results için varsayılan değeri ayarla (eğer None olarak geldiyse).
        if num_results is None:
            num_results = self.num_retrieved_memories

        # Hata yönetimi: num_results'ın geçerli bir integer (>= 0) olup olmadığını kontrol et.
        # check_input_type kullan.
        if not check_input_type(num_results, int, input_name="num_results", logger_instance=logger) or num_results < 0:
             # Geçersiz değer ise uyarı logu ver.
             logger.warning(f"Memory.retrieve: Geçersiz num_results değeri veya tipi ({num_results}). Varsayılan ({self.num_retrieved_memories}) veya 5 kullanılacak.")
             # Geçersiz değer ise varsayılanı (kendi değişkeninden) veya mutlak bir varsayılanı (5) kullan.
             num_results = self.num_retrieved_memories if isinstance(self.num_retrieved_memories, int) and self.num_retrieved_memories >= 0 else 5

        # Hata yönetimi: Bellek boşsa (veya num_results 0 veya negatifse)
        if not self.memory_storage or num_results <= 0:
            # logger.debug("Memory.retrieve: Bellek boş veya istenen sonuç sayısı 0 veya negatif. Geri çağrılamadı.")
            return [] # Bellek boşsa veya sonuç sayısı 0/negatifse boş liste döndür.

        # query_representation şimdilik kullanılmıyor ama gelecekte buraya işlenecek (örn: anıların similarity score'unu hesaplamak için).
        # None olabilir veya numpy array olabilir. Gelecekte tip kontrolü burada yapılabilir.
        # if query_representation is not None and not check_numpy_input(query_representation, expected_dtype=np.number, expected_ndim=1, input_name="query_representation", logger_instance=logger):
        #      logger.warning(f"Memory.retrieve: Sorgu representation tipi beklenmiyor ({type(query_representation)}). numpy.ndarray veya None bekleniyordu. Sorgu dikkate alinmayacak.")


        retrieved_list = [] # Geri çağrılacak anıları tutacak liste.

        try:
            # Placeholder Geri Çağırma Mantığı:
            # Bellekteki anıların bir alt kümesini veya tamamını rastgele seçerek döndür.
            # random.sample fonksiyonu, bir listeden belirli sayıda eşsiz öğe seçer.
            # Seçilecek anı sayısı (actual_num_results), istenen sayı (num_results) ile bellekteki toplam anı sayısının minimumu olmalıdır.
            actual_num_results = min(num_results, len(self.memory_storage))
            if actual_num_results > 0: # Seçilecek en az bir anı varsa random.sample çağır.
                 retrieved_list = random.sample(self.memory_storage, actual_num_results)

            # DEBUG logu: Geri çağrılan anı sayısı.
            # logger.debug(f"Memory.retrieve: Hafızadan {len(retrieved_list)} girdi geri çağrıldı (placeholder).")
            # Bu log run_evo.py'de de var, çift loglamayı önlemek için birini yorum satırı yapabiliriz.


        except Exception as e:
            # Geri çağırma işlemi sırasında beklenmedik bir hata oluşursa logla.
            # random.sample gibi fonksiyonlarda veya gelecekteki daha karmaşık arama algoritmalarında hata olabilir.
            logger.error(f"Memory.retrieve: Bellekten geri çağırma sırasında beklenmedik hata: {e}", exc_info=True)
            return [] # Hata durumunda boş liste döndürerek main loop'un devam etmesini sağla.

        # Başarılı durumda geri çağrılan anı listesini döndür.
        return retrieved_list

    def cleanup(self):
        """
        Memory modülü kaynaklarını temizler.

        Şimdilik sadece bellekteki anı listesini temizler ve bilgilendirme logu içerir.
        Gelecekte kalıcı bellek depolamaya kaydetme (save) mantığı buraya gelebilir.
        module_loader.py bu metodu program sonlanırken çağırır (varsa).
        """
        logger.info("Memory modülü objesi siliniyor...")
        # Belleği kalıcı depolamaya kaydetme mantığı buraya gelebilir (Gelecek TODO).
        # Örneğin, anıları bir dosyaya kaydetme.
        # self._save_to_storage() # Gelecek TODO

        # Bellekteki anı listesini temizle.
        # Listeyi None yapmak veya boş bir liste atamak, objelerin garbage collection tarafından toplanmasına yardımcı olur.
        self.memory_storage = [] # Veya self.memory_storage = None

        logger.info("Memory modülü objesi silindi.")

# # Kalıcı depolama için yardımcı metotlar (Gelecek TODO)
# def _load_from_storage(self):
#     """Kalıcı depolamadan (dosya vb.) belleği yükler."""
#     pass # Implement edilecek

# def _save_to_storage(self):
#     """Belleği kalıcı depolamaya (dosya vb.) kaydeder."""
#     pass # Implement edilecek