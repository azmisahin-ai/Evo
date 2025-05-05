# src/memory/core.py
#
# Evo'nın temel bellek sistemini temsil eder.
# Öğrenilmiş temsilleri saklar ve gerektiğinde geri çağırır.
# Gelecekte episodik ve semantik bellek gibi alt modülleri koordine edecektir.

import numpy as np # Temsil vektörleri (numpy array) için.
import time # Anıların zaman damgası için.
import random # Placeholder retrieve (rastgele seçim) için.
import logging # Loglama için.

# Yardımcı fonksiyonları import et
from src.core.utils import check_input_not_none, check_numpy_input, get_config_value, check_input_type, run_safely, cleanup_safely # utils fonksiyonları kullanılmış

# Alt bellek modüllerini import et (Placeholder sınıflar)
from .episodic import EpisodicMemory # <<< Yeni import
from .semantic import SemanticMemory # <<< Yeni import


# Bu modül için bir logger oluştur
# 'src.memory.core' adında bir logger döndürür.
logger = logging.getLogger(__name__)

class Memory:
    """
    Evo'nın bellek sistemi ana sınıfı (Koordinatör/Yönetici).

    RepresentationLearner'dan gelen öğrenilmiş temsil vektörlerini alır.
    Bu temsilleri ve/veya ilgili bilgileri farklı bellek türlerine (core/working,
    episodik, semantik) yönlendirir ve yönetir.
    İstek üzerine ilgili anıları veya bilgileri farklı bellek türlerinden geri çağırır.
    Şimdilik temel list tabanlı depolama (core/working memory) implementasyonu içerir.
    Hata durumlarında işlemleri loglar ve programın çökmesini engeller.
    """
    def __init__(self, config):
        """
        Memory modülünü başlatır.

        Temel depolama yapısını (şimdilik liste) başlatır ve alt bellek modüllerini
        (EpisodicMemory, SemanticMemory) başlatmayı dener (gelecekte).

        Args:
            config (dict): Bellek sistemi yapılandırma ayarları.
                           'max_memory_size': Core/Working bellekte saklanacak maksimum temsil sayısı (int, varsayılan 1000).
                           'num_retrieved_memories': retrieve metodunda varsayılan olarak geri çağrılacak anı sayısı (int, varsayılan 5).
                           Alt modüllere ait ayarlar kendi adları altında beklenir
                           (örn: {'episodic': {...}, 'semantic': {...}}).
                           Gelecekte kalıcı depolama ayarları (file_path vb.) buraya gelebilir.
        """
        self.config = config
        logger.info("Memory modülü başlatılıyor...")

        # Yapılandırmadan ayarları alırken get_config_value kullan
        self.max_memory_size = get_config_value(config, 'max_memory_size', 1000, expected_type=int, logger_instance=logger)
        self.num_retrieved_memories = get_config_value(config, 'num_retrieved_memories', 5, expected_type=int, logger_instance=logger)
        if self.num_retrieved_memories < 0:
             logger.warning(f"Memory: Konfigurasyonda num_retrieved_memories negatif ({self.num_retrieved_memories}). Varsayılan 5 kullanılıyor.")
             self.num_retrieved_memories = 5


        # Bellek depolama yapıları. Şimdilik sadece temel list tabanlı (core/working memory).
        # Her öğe bir sözlüktür: {'representation': numpy_array, 'metadata': dict, 'timestamp': float}
        self.core_memory_storage = [] # Temel / Çalışma Belleği Depolama

        self.episodic_memory = None # Anısal bellek alt modülü objesi.
        self.semantic_memory = None # Kavramsal bellek alt modülü objesi.


        # Alt bellek modüllerini başlatmayı dene (Gelecek TODO).
        # Başlatma hataları kendi içlerinde veya _initialize_single_module gibi bir utility ile yönetilmeli.
        # Şu anki module_loader initiate_modules fonksiyonu bu sınıfın init'ini çağırıyor.
        # Alt modüllerin başlatılması initialize_modules içinde değil, burada (ana modül init içinde) olmalıdır.
        # Ancak alt modüllerin init hataları Memory modülünün kendisinin başlatılmasını (initialize_modules'da)
        # KRİTİK olarak işaretlememeli (eğer Memory koordinatör ise).

        # TODO: Alt bellek modüllerini burada başlatma mantığı eklenecek.
        # try:
        #     episodic_config = config.get('episodic', {})
        #     self.episodic_memory = EpisodicMemory(episodic_config)
        #     if self.episodic_memory is None: logger.error("Memory: EpisodicMemory başlatılamadı.")
        # except Exception as e: logger.error(f"Memory: EpisodicMemory başlatılırken hata: {e}", exc_info=True); self.episodic_memory = None

        # try:
        #     semantic_config = config.get('semantic', {})
        #     self.semantic_memory = SemanticMemory(semantic_config)
        #     if self.semantic_memory is None: logger.error("Memory: SemanticMemory başlatılamadı.")
        # except Exception as e: logger.error(f"Memory: SemanticMemory başlatılırken hata: {e}", exc_info=True); self.semantic_memory = None


        # Kalıcı bellekten yükleme mantığı buraya gelebilir (Gelecek TODO).
        # self._load_from_storage() # Gelecek TODO


        logger.info(f"Memory modülü başlatıldı. Maksimum Core Bellek boyutu: {self.max_memory_size}. Varsayılan geri çağrı sayısı: {self.num_retrieved_memories}")


    def store(self, representation, metadata=None):
        """
        Öğrenilmiş bir temsili (ve ilişkili metadatayı) belleğe kaydeder.

        Gelen representation ve metadatayı, hangi bellek türüne (core/working,
        episodik, semantik) uygun olduğuna karar vererek ilgili bellek yapısına
        veya alt modüle kaydeder.
        Şimdilik sadece temel list tabanlı core belleğe kaydeder.
        Eğer representation None veya beklenmeyen numpy array tipinde ise kaydetme işlemini atlar.
        Bellek boyutu self.max_memory_size değerini aşarsa, en eski anıyı (core bellekte) siler (FIFO prensibi).
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
            # TODO: Gelecekte: Gelen representation ve metadata'ya bakarak hangi bellek türüne (core/working, episodic, semantic) kaydedileceğine karar ver.
            # Örneğin, zaman damgası belirgin olan veya özel bağlama sahip olanlar episodik belleğe,
            # sık tekrar eden veya ilişkisel olanlar semantik belleğe gidebilir.
            # Şu an sadece temel list tabanlı core belleğe kaydediyoruz.

            # Basitçe temel (core) belleğe kaydet (FIFO)
            # Yeni bellek öğesi oluştur.
            memory_entry = {
                'representation': representation,
                'metadata': metadata if metadata is not None else {}, # metadata None ise boş sözlük sakla.
                'timestamp': time.time() # Kayıt zamanı (epoch zamanı float olarak).
            }

            # Core bellek depolama listesine yeni öğeyi ekle (listenin sonuna).
            self.core_memory_storage.append(memory_entry)
            # DEBUG logu: Saklama işleminin başarıyla yapıldığı ve güncel boyut bilgisi.
            # logger.debug(f"Memory.store: Temsil başarıyla core belleğe saklandı. Güncel boyutu: {len(self.core_memory_storage)}")


            # Maksimum core bellek boyutu aşıldıysa en eski öğeyi sil (FIFO).
            if len(self.core_memory_storage) > self.max_memory_size:
                # max_memory_size 0 veya negatif ise bu kontrol gereksiz olabilir ama >= 0 varsayıyoruz.
                # Listenin başındaki (en eski) öğeyi çıkar.
                removed_entry = self.core_memory_storage.pop(0)
                # DEBUG logu: Silinen anı hakkında bilgi.
                # logger.debug(f"Memory.store: Maksimum core bellek boyutu aşıldı ({self.max_memory_size}). En eski anı silindi (timestamp: {removed_entry['timestamp']:.2f}).")

            # TODO: Gelecekte: Eğer alt bellek modülleri başlatıldıysa, ilgili verileri onlara da kaydet.
            # if self.episodic_memory:
            #      self.episodic_memory.store_event(representation, memory_entry['timestamp'], context=memory_entry['metadata'])
            # if self.semantic_memory:
            #      self.semantic_memory.store_concept(representation, relations=...)


        except Exception as e:
            # Saklama işlemi sırasında beklenmedik bir hata oluşursa logla.
            logger.error(f"Memory.store: Belleğe kaydetme sırasında beklenmedik hata: {e}", exc_info=True)
            # Hata durumunda programın çökmesini engelle, sadece logla ve devam et.


    def retrieve(self, query_representation, num_results=None):
        """
        Bellekten ilgili anıları geri çağırır.

        Gelen sorgu (query_representation) ve parametrelere (num_results vb.)
        bakarak hangi bellek türlerinden (core/working, episodik, semantik)
        arama yapılacağına karar verir, ilgili alt modülleri çağırır ve sonuçları birleştirir.
        Şimdilik placeholder mantığı: query_representation parametresini kullanmaz.
        Bunun yerine temel (core) bellekteki anıların bir alt kümesini (belirtilen sayıda rastgele anı)
        veya belleğin tamamını (num_results çok büyükse veya None ise) döndürür.
        Gelecekte 'query_representation' kullanılarak farklı bellek türlerinden arama yapılacak.
        Hata durumunda veya bellek boşsa boş liste döndürür.

        Args:
            query_representation (numpy.ndarray or None): Sorgu için kullanılan temsil vektörü.
                                                         Gelecekte anılarla benzerliğini ölçmek için kullanılacak.
                                                         Şimdilik kullanılmıyor, None veya numpy array olabilir.
                                                         Beklenen format: shape (D,), dtype sayısal, veya None.
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

        # Hata yönetimi: query_representation tipi kontrolü (Gelecekte kullanılacağı zaman).
        # Şimdilik None olabilir veya numpy array olabilir.
        # if query_representation is not None and not check_numpy_input(query_representation, expected_dtype=np.number, expected_ndim=1, input_name="query_representation", logger_instance=logger):
        #      logger.warning(f"Memory.retrieve: Sorgu representation tipi beklenmiyor ({type(query_representation)}). numpy.ndarray veya None bekleniyordu. Sorgu dikkate alinmayacak.")
        #      # Hata vermeden devam et ama logla.


        # TODO: Gelecekte: Gelen query_representation ve diğer parametrelere bakarak hangi bellek türlerinden (core, episodic, semantic) arama yapılacağına karar ver.
        # Örneğin, query temsiline en benzer anıları core bellekten getir, belirli bir zamandaki olayları episodik bellekten getir,
        # belirli bir kavramla ilgili bilgiyi semantik bellekten getir.
        # Şu an sadece temel (core) bellekte arama yapıyoruz.

        retrieved_list = [] # Geri çağrılan anıları/bilgileri tutacak liste.

        try:
            # Placeholder Geri Çağırma Mantığı (Core Bellekten Rastgele Seçim):
            # Bellekteki anıların bir alt kümesini veya tamamını rastgele seçerek döndür.
            # Eğer core bellek boşsa veya istenen sonuç sayısı 0 veya negatifse, boş liste döndür.
            if not self.core_memory_storage or num_results <= 0:
                # logger.debug("Memory.retrieve: Core bellek boş veya istenen sonuç sayısı 0 veya negatif. Geri çağrılamadı.")
                return [] # Bellek boşsa veya sonuç sayısı 0/negatifse boş liste döndür.


            # random.sample fonksiyonu, bir listeden belirtilen sayıda eşsiz öğe seçer.
            # Seçilecek anı sayısı (actual_num_results), istenen sayı (num_results) ile core bellekteki toplam anı sayısının minimumu olmalıdır.
            actual_num_results = min(num_results, len(self.core_memory_storage))
            if actual_num_results > 0: # Seçilecek en az bir anı varsa random.sample çağır.
                 retrieved_list = random.sample(self.core_memory_storage, actual_num_results)

            # TODO: Gelecekte: Eğer alt bellek modülleri başlatıldıysa, onlardan da ilgili sonuçları al ve retrieved_list ile birleştir.
            # if self.episodic_memory:
            #      episodic_results = self.episodic_memory.retrieve_event(query_representation, time_range=...)
            #      retrieved_list.extend(episodic_results)
            # if self.semantic_memory:
            #      semantic_results = self.semantic_memory.retrieve_concept(query_representation, relation_filter=...)
            #      retrieved_list.extend(semantic_results)

            # TODO: Gelecekte: Farklı bellek türlerinden gelen sonuçları önceliklendir veya sırala.


            # DEBUG logu: Geri çağrılan anı sayısı (tüm bellek türlerinden gelenler).
            # logger.debug(f"Memory.retrieve: Hafızadan toplam {len(retrieved_list)} girdi geri çağrıldı (placeholder).")
            # Bu log run_evo.py'de de var, çift loglamayı önlemek için birini yorum satırı yapabiliriz.


        except Exception as e:
            # Geri çağırma işlemi sırasında beklenmedik bir hata oluşursa logla.
            # random.sample gibi fonksiyonlarda veya gelecekteki daha karmaşık arama algoritmalarında hata olabilir.
            logger.error(f"Memory.retrieve: Bellekten geri çağırma sırasında beklenmedik hata: {e}", exc_info=True)
            return [] # Hata durumında boş liste döndürerek main loop'un devam etmesini sağla.

        # Başarılı durumda geri çağrılan anı listesini döndür.
        return retrieved_list

    def cleanup(self):
        """
        Memory modülü kaynaklarını temizler.

        Temel (core) bellek listesini temizler ve alt bellek modüllerinin
        (EpisodicMemory, SemanticMemory) cleanup metotlarını (varsa) çağırır.
        Gelecekte kalıcı bellek depolamaya kaydetme (save) mantığı buraya gelebilir.
        module_loader.py bu metodu program sonlanırken çağırır (varsa).
        """
        logger.info("Memory modülü objesi siliniyor...")

        # Belleği kalıcı depolamaya kaydetme mantığı buraya gelebilir (Gelecek TODO).
        # self._save_to_storage() # Gelecek TODO

        # Temel (core) bellekteki anı listesini temizle.
        # Listeyi None yapmak veya boş bir liste atamak, objelerin garbage collection tarafından toplanmasına yardımcı olur.
        self.core_memory_storage = [] # Veya self.core_memory_storage = None
        logger.info("Memory: Core bellek temizlendi.")

        # Alt bellek modüllerinin cleanup metotlarını çağır (varsa).
        # cleanup_safely yardımcı fonksiyonunu kullanabiliriz.
        if self.episodic_memory:
             cleanup_safely(self,self.episodic_memory.cleanup, logger_instance=logger, error_message="Memory: EpisodicMemory temizlenirken hata")
        if self.semantic_memory:
             cleanup_safely(self,self.semantic_memory.cleanup, logger_instance=logger, error_message="Memory: SemanticMemory temizlenirken hata")


        logger.info("Memory modülü objesi silindi.")

# # Kalıcı depolama için yardımcı metotlar (Gelecek TODO)
# def _load_from_storage(self):
#     """Kalıcı depolamadan (dosya vb.) belleği yükler."""
#     pass # Implement edilecek

# def _save_to_storage(self):
#     """Belleği kalıcı depolamaya (dosya vb.) kaydeder."""
#     pass # Implement edilecek