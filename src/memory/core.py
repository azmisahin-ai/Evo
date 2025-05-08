# src/memory/core.py
#
# Evo'nın temel bellek sistemini temsil eder.
# Öğrenilmiş temsilleri saklar ve gerektiğinde geri çağırır.
# Belleğe dosya tabanlı kalıcılık kazandırır.
# Gelecekte episodik ve semantik bellek gibi alt modülleri koordine edecektir.

import numpy as np # For representation vectors (numpy array).
import time # For memory entry timestamps.
import random # For placeholder retrieve (random selection).
import logging # For logging.
import pickle # For saving and loading memory to file
import os # For filesystem operations (checking existence, creating directories)

# Import utility functions
# setup_logging will not be called here. isinstance is used instead of check_* functions.
from src.core.utils import run_safely, cleanup_safely # Only run_safely and cleanup_safely are used here
from src.core.config_utils import get_config_value # get_config_value is imported from here

# Import sub-memory modules (Placeholder classes)
# Commenting out imports for now if they are just placeholders and might cause import errors
# if the files/classes don't fully exist or are not intended to be tested/used yet.
# from .episodic import EpisodicMemory
# from .semantic import SemanticMemory


# Create a logger for this module
# Returns a logger named 'src.memory.core'.
# Logging level and handlers are configured externally (by conftest.py or the main run script).
logger = logging.getLogger(__name__)


class Memory:
    """
    Evo's primary memory system class (Coordinator/Manager).
    ... (Docstring same) ...
    """
    def __init__(self, config):
        """
        Initializes the Memory module.
        ... (Docstring same) ...
        """
        self.config = config
        logger.info("Memory module initializing...")

        # Get configuration settings using get_config_value
        # Pass logger_instance=logger to each call to ensure logs within get_config_value are visible.
        # Corrected: Use default= keyword format for all calls.
        # Based on config, these settings are under the 'memory' key.
        self.max_memory_size = get_config_value(config, 'memory', 'max_memory_size', default=1000, expected_type=int, logger_instance=logger)
        self.num_retrieved_memories = get_config_value(config, 'memory', 'num_retrieved_memories', default=5, expected_type=int, logger_instance=logger)
        # The representation dimension is under the 'representation' key in the main config.
        # The Memory module needs to know this dimension.
        # Get it from the main config's representation section.
        self.representation_dim = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)

        # memory_file_path is retrieved from config.
        self.memory_file_path = get_config_value(config, 'memory', 'memory_file_path', default='data/core_memory.pkl', expected_type=str, logger_instance=logger)


        # Check for negative values for num_retrieved_memories
        if self.num_retrieved_memories < 0:
             logger.warning(f"Memory: Invalid num_retrieved_memories config value ({self.num_retrieved_memories}). Using default 5.")
             self.num_retrieved_memories = 5

        # Check for negative values for max_memory_size
        if self.max_memory_size < 0:
             logger.warning(f"Memory: Invalid max_memory_size config value ({self.max_memory_size}). Using default 1000.")
             self.max_memory_size = 1000

        # Check for non-positive values for representation_dim
        if self.representation_dim <= 0:
             logger.warning(f"Memory: Invalid representation_dim config value ({self.representation_dim}). Expected a positive value. Using default 128.")
             self.representation_dim = 128


        # Log the memory_file_path obtained from config - important for DEBUG logs!
        # This log will clearly show the result of the get_config_value call.
        logger.info(f"Memory __init__: memory_file_path set from config: {self.memory_file_path}")


        # Memory storage structures. For now, just a simple list-based core/working memory.
        # Each element is a dictionary: {'representation': numpy_array, 'metadata': dict, 'timestamp': float}
        self.core_memory_storage = [] # Core / Working Memory Storage

        # Sub-memory module objects (if they exist)
        self.episodic_memory = None
        self.semantic_memory = None

        # Logic to load from persistent storage
        self._load_from_storage()


        # Try to initialize sub-memory modules (Future TODO).
        # TODO: Add logic here to initialize sub-memory modules.
        # try:
        #     episodic_config = config.get('episodic', {})
        #     # If EpisodicMemory class is imported and exists, initialize it
        #     if 'EpisodicMemory' in globals() and EpisodicMemory:
        #         self.episodic_memory = EpisodicMemory(episodic_config)
        #     if self.episodic_memory is None: logger.error("Memory: EpisodicMemory initialization failed.")
        # except Exception as e: logger.error(f"Memory: Error during EpisodicMemory initialization: {e}", exc_info=True); self.episodic_memory = None

        # try:
        #     semantic_config = config.get('semantic', {})
        #     # If SemanticMemory class is imported and exists, initialize it
        #     if 'SemanticMemory' in globals() and SemanticMemory:
        #         self.semantic_memory = SemanticMemory(semantic_config)
        #     if self.semantic_memory is None: logger.error("Memory: SemanticMemory initialization failed.")
        # except Exception as e: logger.error(f"Memory: Error during SemanticMemory initialization: {e}", exc_info=True); self.semantic_memory = None


        logger.info(f"Memory module initialized. Maximum Core Memory size: {self.max_memory_size}. Default retrieval count: {self.num_retrieved_memories}. Persistence file: {self.memory_file_path}. Loaded memories count: {len(self.core_memory_storage)}")


    # ... (_load_from_storage, _save_to_storage, store, retrieve, get_all_representations, cleanup methods - same as before) ...

    def _load_from_storage(self):
        """
        Belirtilen dosyadan bellek durumunu (core_memory_storage) yükler.
        Dosya yoksa, okunamıyorsa veya bozuksa, belleği boş başlatır.
        """
        # Yükleme işlemine başlandığını ve kullanılan path'i logla - DEBUG loglarını görmek önemli!
        logger.info(f"Memory._load_from_storage: Yükleme işlemi başlatılıyor. Kullanılan path: {self.memory_file_path}")


        # Dosya yolu geçerli bir string mi kontrol et
        if not isinstance(self.memory_file_path, str) or not self.memory_file_path:
             logger.warning("Memory._load_from_storage: Geçersiz veya boş bellek dosyası yolu belirtildi. Yükleme atlandi.")
             self.core_memory_storage = [] # Yüklenemediyse belleği boş başlat.
             return

        # Dosya mevcut mu kontrol et
        if not os.path.exists(self.memory_file_path):
            logger.info(f"Memory._load_from_storage: Bellek dosyası bulunamadı: {self.memory_file_path}. Bellek boş başlatılıyor.")
            self.core_memory_storage = [] # Dosya yoksa boş başlat.
            return

        # Dosyayı okumayı ve pickle ile yüklemeyi dene
        try:
            with open(self.memory_file_path, 'rb') as f: # 'rb' binary read mode
                # pickle.load ile dosyadan veriyi yükle
                # Güvenlik: Bilinmeyen veya güvenilmeyen kaynaklardan gelen pickle dosyalarını yüklemeyin.
                # Bu proje kapsamında kendi kaydettiğimiz dosyalar olduğu için güvendik.
                loaded_data = pickle.load(f)

            # Yüklenen verinin beklendiği gibi bir liste olup olmadığını kontrol et.
            # Daha sağlam bir kontrol, listedeki her öğenin {'representation': np.ndarray, 'metadata': dict, 'timestamp': float}
            # formatında olup olmadığını da içerebilir, ancak bu başlangıç için yeterli.
            if isinstance(loaded_data, list):
                # Yüklenen listedeki her öğenin minimum beklenen formatta olup olmadığını kontrol et
                # Bu, bozuk veya uyumsuz pickle dosyalarını yakalamaya yardımcı olur.
                valid_loaded_data = []
                for item in loaded_data:
                     if isinstance(item, dict) and 'representation' in item and 'metadata' in item and 'timestamp' in item:
                           # Representation formatını da kontrol edebiliriz (isteğe bağlı ama iyi)
                           rep = item['representation']
                           if isinstance(rep, np.ndarray) and rep.ndim == 1 and np.issubdtype(rep.dtype, np.number) and rep.shape[0] == self.representation_dim:
                                valid_loaded_data.append(item)
                           else:
                                logger.warning("Memory._load_from_storage: Geçersiz representation formatına sahip anı bulundu, yoksayılıyor.")
                     else:
                         logger.warning("Memory._load_from_storage: Beklenmeyen formata sahip anı bulundu, yoksayılıyor.")

                self.core_memory_storage = valid_loaded_data

                logger.info(f"Memory._load_from_storage: Bellek başarıyla yüklendi: {self.memory_file_path} ({len(self.core_memory_storage)} anı, {len(loaded_data)-len(valid_loaded_data)} geçersiz anı yoksayıldı).")

                # Yüklenen anı sayısı max_memory_size'ı aşıyorsa eski anıları sil (yükleme sonrası temizlik)
                if len(self.core_memory_storage) > self.max_memory_size:
                    logger.warning(f"Memory._load_from_storage: Yüklenen anı sayısı ({len(self.core_memory_storage)}) maksimum boyutu ({self.max_memory_size}) aşıyor. Eski anılar siliniyor.")
                    # Sadece son max_memory_size kadar anıyı tut.
                    # Negatif index slice güvenlidir.
                    self.core_memory_storage = self.core_memory_storage[-self.max_memory_size:]


            else:
                # Yüklenen veri liste formatında değilse
                logger.error(f"Memory._load_from_storage: Yüklenen bellek dosyası beklenmeyen formatta: {self.memory_file_path}. Liste bekleniyordu, geldi: {type(loaded_data)}. Bellek boş başlatılıyor.", exc_info=True)
                self.core_memory_storage = [] # Format yanlışsa boş başlat.

        except FileNotFoundError:
            # os.path.exists kontrolü yapıldı ama yine de yakalamak sağlamlık katabilir.
            logger.warning(f"Memory._load_from_storage: Bellek dosyası bulunamadı (yeniden kontrol sonrası): {self.memory_file_path}. Bellek boş başlatılıyor.")
            self.core_memory_storage = []

        except (pickle.UnpicklingError, EOFError, ImportError, IndexError) as e:
            # pickle yükleme sırasında oluşabilecek hatalar (bozuk dosya, uyumsuz pickle sürümü, vb.)
            logger.error(f"Memory._load_from_storage: Bellek dosyası yüklenirken pickle hatası oluştu: {self.memory_file_path}. Bellek boş başlatılıyor.", exc_info=True)
            self.core_memory_storage = [] # Yükleme hatası olursa boş başlat.

        except Exception as e:
            # Diğer tüm beklenmedik hatalar.
            logger.error(f"Memory._load_from_storage: Bellek dosyası yüklenirken beklenmedik hata: {self.memory_file_path}. Bellek boş başlatılıyor.", exc_info=True)
            self.core_memory_storage = []


    def _save_to_storage(self):
        """
        Mevcut bellek durumunu (core_memory_storage) belirtilen dosyaya kaydeder (pickle formatında).
        Bellek boşsa kaydetme işlemini atlar. Kaydetme sırasında hata oluşursa loglar.
        """
        # Bellek boşsa kaydetme.
        if not self.core_memory_storage:
            logger.info("Memory._save_to_storage: Core memory boş. Kaydetme atlandi.")
            return

        # Dosya yolu geçerli bir string mi ve boş değil mi kontrol et
        if not isinstance(self.memory_file_path, str) or not self.memory_file_path:
             logger.warning("Memory._save_to_storage: Geçersiz veya boş bellek dosyası yolu belirtildi. Kaydetme atlandi.")
             return

        # Dosya yolunun dizinini oluştur (eğer yoksa)
        save_dir = os.path.dirname(self.memory_file_path)
        # Eğer save_dir boş string değilse (örn: sadece dosya adı verilmişse dizin os.path.dirname sonucunda boş olur) ve dizin mevcut değilse oluştur.
        if save_dir and not os.path.exists(save_dir):
             try:
                  os.makedirs(save_dir, exist_ok=True) # exist_ok=True dizin zaten varsa hata vermez
                  logger.info(f"Memory._save_to_storage: Kaydetme dizini oluşturuldu: {save_dir}")
             except OSError as e:
                  logger.error(f"Memory._save_to_storage: Kaydetme dizini oluşturulurken hata: {save_dir}. Kaydetme atlandi.", exc_info=True)
                  return # Dizin oluşturulamazsa kaydetme.
             except Exception as e:
                  logger.error(f"Memory._save_to_storage: Kaydetme dizini oluşturulurken beklenmedik hata: {save_dir}. Kaydetme atlandi.", exc_info=True)
                  return # Dizin oluşturulamazsa kaydetme.


        # Dosyaya yazmayı ve pickle ile kaydetmeyi dene
        try:
            with open(self.memory_file_path, 'wb') as f: # 'wb' binary write mode
                # pickle.dump ile veriyi dosyaya kaydet
                # Güvenlik için kopyasını kaydetmek daha iyi olabilir, ancak Representationlar zaten numpy array, immutable sayılabilir.
                pickle.dump(self.core_memory_storage, f)
            logger.info(f"Memory._save_to_storage: Bellek başarıyla kaydedildi: {self.memory_file_path} ({len(self.core_memory_storage)} anı).")

        except (pickle.PicklingError, IOError, OSError) as e:
            # pickle kaydetme veya dosya yazma sırasında oluşabilecek hatalar
            logger.error(f"Memory._save_to_storage: Bellek dosyası kaydedilirken hata oluştu: {self.memory_file_path}.", exc_info=True)

        except Exception as e:
             # Diğer tüm beklenmedik hatalar.
             logger.error(f"Memory._save_to_storage: Bellek dosyası kaydedilirken beklenmedik hata: {self.memory_file_path}.", exc_info=True)


    def store(self, representation, metadata=None):
        """
        Öğrenilmiş bir temsili (ve ilişkili metadatasını) belleğe kaydeder.

        Gelen representation ve metadatayı, hangi bellek türüne (core/working,
        episodik, semantik) uygun olduğuna karar vererek ilgili bellek yapısına
        ve/veya alt modüle kaydeder.
        Şimdilik sadece temel list tabanlı core belleğe kaydeder.
        Eğer representation None veya beklenmeyen numpy array tipinde ise kaydetme işlemini atlar.
        Bellek boyutu self.max_memory_size değerini aşarsa, en eski anıyı (core bellekte) siler (FIFO prensibi).
        Başarısızlık durumunda hatayı loglar.

        Args:
            representation (numpy.ndarray or None): Belleğe saklanacak temsil vektörü.
                                                    Genellikle RepresentationLearner'dan gelir.
                                                    Beklenen format: shape (self.representation_dim,), dtype sayısal.
            metadata (dict, optional): Temsille ilişkili ek bilgiler (örn: kaynak, zaman aralığı, durum vb.).
                                       Varsayılan None. None ise boş sözlük olarak saklanır.
                                       Beklenen tip: dict veya None.
        """
        # Hata yönetimi: Saklanacak representation None mu?
        if representation is None:
             logger.debug("Memory.store: Representation input None. Saklama atlandi.")
             return # None ise saklama atla.

        # Hata yönetimi: Representation'ın numpy array, 1D ve sayısal dtype olup olmadığını kontrol et.
        # expected_ndim=1 çünkü representation genellikle 1D vektördür.
        # RepresentationLearner çıktısı float64 olduğu için dtype np.float64 veya np.number olabilir.
        if not isinstance(representation, np.ndarray) or representation.ndim != 1 or not np.issubdtype(representation.dtype, np.number):
            logger.error(f"Memory.store: Representation input numpy array değil veya yanlış dtype/boyut. Beklenen: numpy array (1D, sayısal), Geldi: {type(representation)}, ndim: {getattr(representation, 'ndim', 'N/A')}, dtype: {getattr(representation, 'dtype', 'N/A')}. Saklama atlandi.")
            return # Geçersiz tip, dtype veya boyut ise saklama atla.

        # Representation boyutunu kontrol et (config'deki representation_dim ile uyumlu olmalı)
        # Bu, RepresentationLearner çıktısının beklenen boyutta olduğunu doğrular.
        if representation.shape != (self.representation_dim,):
             logger.warning(f"Memory.store: Beklenmeyen representation boyutu ({representation.shape}) ile bellek eklenmeye çalışıldı. Beklenen: {(self.representation_dim,)}. Saklama atlandi.")
             return # Boyut uyuşmazsa saklama atla.

        # Hata yönetimi: Metadata None veya dict mi?
        if metadata is not None and not isinstance(metadata, dict):
             # Metadata None değil ama dict de değilse uyarı logla ve metadata'yı None yap.
             logger.warning(f"Memory.store: Metadata beklenmeyen tipte ({type(metadata)}), dict bekleniyordu. Yoksayılıyor.")
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
            logger.debug(f"Memory.store: Temsil başarıyla core belleğe saklandı. Güncel boyutu: {len(self.core_memory_storage)}")


            # Maksimum core bellek boyutu aşıldysa en eski öğeyi sil (FIFO).
            if len(self.core_memory_storage) > self.max_memory_size:
                # max_memory_size 0 veya negatif ise bu kontrol gereksiz olabilir ama >= 0 varsayıyoruz.
                # Listenin başındaki (en eski) öğeyi çıkar.
                # core_memory_storage boş değilse index 0 geçerlidir.
                removed_entry = self.core_memory_storage.pop(0)
                # DEBUG logu: Silinen anı hakkında bilgi.
                logger.debug(f"Memory.store: Maksimum core bellek boyutu aşıldı ({self.max_memory_size}). En eski anı silindi (timestamp: {removed_entry['timestamp']:.2f}).")

            # TODO: Gelecekte: Eğer alt bellek modülleri başlatıldıysa, ilgili verileri onlara da kaydet.
            # if self.episodic_memory and hasattr(self.episodic_memory, 'store_event'):
            #      # metadata'daki context bilgisi episodic memory için kullanılabilir.
            #      self.episodic_memory.store_event(representation, memory_entry['timestamp'], context=memory_entry['metadata'])
            # if self.semantic_memory and hasattr(self.semantic_memory, 'store_concept'):
            #      # representation ve relations (metadata'dan türetilebilir veya ayrı algılanabilir) semantic memory için kullanılabilir.
            #      self.semantic_memory.store_concept(representation, relations=...)


        except Exception as e:
            # Saklama işlemi sırasında beklenmedik bir hata oluşursa logla.
            logger.error(f"Memory.store: Belleğe kaydetme sırasında beklenmedik hata: {e}", exc_info=True)
            # Hata durumında programın çökmesini engelle, sadece logla ve devam et.


    def retrieve(self, query_representation, num_results=None):
        """
        Bellekten ilgili anıları geri çağırır.

        Gelen sorgu (query_representation) ile temel (core) bellekteki anıları
        (representation vektörleri üzerinden) vektör benzerliği hesaplayarak
        en ilgili olanları geri çağırır.
        query_representation None veya geçersiz ise boş liste döner.
        Hata durumunda veya bellek boşsa boş liste döndürür.

        Args:
            query_representation (numpy.ndarray or None): Sorgu için kullanılan temsil vektörü.
                                                         Genellikle RepresentationLearner'dan gelir.
                                                         Beklenen format: shape (self.representation_dim,), dtype sayısal, veya None.
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
        if not isinstance(num_results, int) or num_results < 0:
             logger.warning(f"Memory.retrieve: Geçersiz num_results değeri veya tipi ({num_results}). Varsayılan ({self.num_retrieved_memories}) veya 5 kullanılacak.")
             # self.num_retrieved_memories değeri de geçersiz olabilir, en güvenlisi sabit 5 kullanmak.
             fallback_num = self.num_retrieved_memories if isinstance(self.num_retrieved_memories, int) and self.num_retrieved_memories >= 0 else 5
             num_results = fallback_num

        # Gerçekten kaç sonuç çağırılacağını belirle (bellek boyutunu aşmasın)
        actual_num_results = min(num_results, len(self.core_memory_storage))

        retrieved_list = [] # Geri çağrılan anıları/bilgileri tutacak liste.

        # Sorgu representation'ın geçerli (None değil, numpy array, 1D, sayısal, doğru boyut) olup olmadığını kontrol et.
        valid_query = query_representation is not None \
                      and isinstance(query_representation, np.ndarray) \
                      and query_representation.ndim == 1 \
                      and np.issubdtype(query_representation.dtype, np.number) \
                      and query_representation.shape[0] == self.representation_dim # Boyut kontrolü eklendi


        if not self.core_memory_storage or actual_num_results <= 0:
            # Bellek boşsa veya istenen sonuç sayısı 0 veya negatifse, boş liste döndür.
            logger.debug("Memory.retrieve: Core memory empty or effective num_results non-positive. Returning empty list.")
            return []

        if not valid_query:
             # Sorgu geçersizse (None, yanlış tip, yanlış boyut vb.), benzerlik araması yapamayız. Boş liste döndür.
             logger.warning(f"Memory.retrieve: Geçersiz query representation input. Tip: {type(query_representation)}, ndim: {getattr(query_representation, 'ndim', 'N/A')}, dtype: {getattr(query_representation, 'dtype', 'N/A')}, shape: {getattr(query_representation, 'shape', 'N/A')}. Benzerlik araması atlandi.")
             return [] # Veya bu durumda rastgele çağırmaya dönebiliriz policy'e göre. Şimdilik boş dönelim.


        # --- Vektör Benzerliği ile Geri Çağırma Mantığı ---
        logger.debug(f"Memory.retrieve: Valid query representation provided (Shape: {query_representation.shape}, Dtype: {query_representation.dtype}). Performing similarity search.")
        similarities = []
        query_norm = np.linalg.norm(query_representation) # np.linalg.norm kullanıldı

        # Sorgu vektörü sıfır ise benzerlik hesaplanamaz.
        if query_norm < 1e-8: # Sıfıra yakınlığı kontrol et
             logger.warning("Memory.retrieve: Query representation has near-zero norm. Cannot calculate cosine similarity meaningfully. Returning empty list.")
             return []


        try:
            # Bellekteki her anı ile sorgu arasındaki benzerliği hesapla.
            for memory_entry in self.core_memory_storage:
                # Anıdaki representation'ı al ve geçerliliğini kontrol et.
                # Stored representation'ın geçerli bir sayısal 1D numpy array olduğunu doğrula.
                # ve boyutunun da doğru olduğunu kontrol et.
                if memory_entry is not None and isinstance(memory_entry, dict): # memory_entry dict mi kontrol et
                     stored_representation = memory_entry.get('representation') # .get ile güvenli erişim

                     if stored_representation is not None \
                        and isinstance(stored_representation, np.ndarray) \
                        and stored_representation.ndim == 1 \
                        and np.issubdtype(stored_representation.dtype, np.number) \
                        and stored_representation.shape[0] == self.representation_dim: # Boyut kontrolü eklendi

                          stored_norm = np.linalg.norm(stored_representation) # np.linalg.norm kullanıldı
                          if stored_norm > 1e-8: # Stored vektör sıfır ise bölme hatası olmaması için kontrol et.
                               # Kosinüs benzerliği hesapla: (dot product) / (norm1 * norm2)
                               similarity = np.dot(query_representation, stored_representation) / (query_norm * stored_norm)
                               if not np.isnan(similarity): # NaN benzerlik skorlarını yoksay.
                                    # Benzerlik skoru ile birlikte anıyı sakla.
                                    # Anının orijinal indeksini de saklamak faydalı olabilir, ancak şimdilik gerek yok.
                                    similarities.append((float(similarity), memory_entry)) # Similarity'yi float'a çevir
                               else:
                                    logger.debug("Memory.retrieve: Calculated NaN similarity, skipping entry.")
                          # else: logger.debug("Memory.retrieve: Stored rep near zero norm, skipping similarity.")
                     # else: logger.debug("Memory.retrieve: Invalid stored rep format/type/shape, skipping.")
                # else: logger.warning("Memory.retrieve: Core memory list element is not a dict, skipping.")


            # Benzerliklere göre azalan sırada sırala.
            # Eğer similarities listesi boşsa sort hata vermez, sadece bir şey yapmaz.
            similarities.sort(key=lambda item: item[0], reverse=True)

            # En yüksek benzerliğe sahip 'actual_num_results' kadar anıyı al.
            # Eğer similarities listesi istenen sayıdan azsa, sadece listedeki kadarını alır.
            retrieved_list = [item[1] for item in similarities[:actual_num_results]]

            logger.debug(f"Memory.retrieve: Found {len(similarities)} memories with valid representations for similarity check. Retrieved top {len(retrieved_list)} by similarity.")


            # TODO: Gelecekte: Eğer alt bellek modülleri başlatıldıysa, onlardan da ilgili sonuçları al ve retrieved_list ile birleştir.
            # if hasattr(self.episodic_memory, 'retrieve_event'):
            #      episodic_results = self.episodic_memory.retrieve_event(query_representation, ...)
            #      retrieved_list.extend(episodic_results)
            # if hasattr(self.semantic_memory, 'retrieve_concept'):
            #      semantic_results = self.semantic_memory.retrieve_concept(query_representation, ...)
            #      retrieved_list.extend(semantic_results)

            # TODO: Gelecekte: Farklı bellek türlerinden gelen sonuçları önceliklendir veya sırala (benzerlik, alaka düzeyi, zaman damgası vb.).

        except Exception as e:
            # Geri çağırma işlemi sırasında beklenmedik bir hata oluşursa logla.
            # Vektör operasyonları (np.dot, np.linalg.norm) hataları burada yakalanabilir.
            logger.error(f"Memory.retrieve: Bellekten geri çağırma sırasında beklenmedik hata: {e}", exc_info=True)
            return [] # Hata durumında boş liste döndürerek main loop'un devam etmesini sağla.

        # Başarılı durumda geri çağrılan anı listesini döndür.
        # logger.debug(f"Memory.retrieve: Geri çağrılan anı listesi boyutu: {len(retrieved_list)}") # run_evo.py'de loglanıyor
        return retrieved_list


    def get_all_representations(self):
        """
        Core bellekte depolanmış tüm Representation vektörlerinin bir listesini döndürür.
        LearningModule gibi dış modüller tarafından öğrenme için kullanılır.

        Returns:
            list: numpy arraylerden oluşan liste. Hata durumunda boş liste döner.
        """
        logger.debug("Memory.get_all_representations çağrıldı.")
        representations = []
        try:
            # Sadece geçerli numpy array Representationları içeren bir liste döndür.
            # core_memory_storage'daki her öğenin {'representation': np.ndarray, ...} formatında olduğu varsayılır.
            representations = [entry.get('representation')
                               for entry in self.core_memory_storage
                               if isinstance(entry, dict) # Öğenin sözlük olduğundan emin ol
                                  and entry.get('representation') is not None # Representation anahtarının None olmadığından emin ol
                                  and isinstance(entry.get('representation'), np.ndarray) # numpy array olduğundan emin ol
                                  and entry.get('representation').ndim == 1 # 1D vektör olduğundan emin ol
                                  and np.issubdtype(entry.get('representation').dtype, np.number) # Sayısal dtype olduğundan emin ol
                                  and entry.get('representation').shape[0] == self.representation_dim] # Boyutun doğru olduğundan emin ol


            logger.debug(f"Memory.get_all_representations: {len(representations)} geçerli Representation bulundu.")
            # Kopyasını döndürmek daha güvenli olabilir, ama basit unit test için referans dönmek yeterli.
            return representations

        except Exception as e:
            # İşlem sırasında hata oluşursa (örneğin, core_memory_storage beklenmeyen formatta ise)
            logger.error(f"Memory.get_all_representations sırasında hata oluştu: {e}", exc_info=True)
            return [] # Hata durumunda boş liste döndür.


    def cleanup(self):
        """
        Memory modülü kaynaklarını temizler.

        Temel (core) bellek listesini kalıcı depolamaya kaydeder
        ve alt bellek modüllerinin (EpisodicMemory, SemanticMemory)
        cleanup metotlarını (varsa) çağırır.
        module_loader.py bu metotu program sonlanırken çağrır (varsa).
        """
        logger.info("Memory modülü objesi siliniyor...")

        # Belleği kalıcı depolamaya kaydetme mantığı
        # Hata oluşursa cleanup_safely kullanmak daha sağlam olabilir.
        run_safely(self._save_to_storage, logger_instance=logger, error_message="Memory: _save_to_storage temizlenirken hata")

        # Temel (core) bellekteki anı listesini temizle (kaydedildikten sonra).
        # Listeyi None yapmak veya boş bir liste atamak, objelerin garbage collection tarafından toplanmasına yardımcı olur.
        self.core_memory_storage = [] # Veya self.core_memory_storage = None
        logger.info("Memory: Core bellek temizlendi (RAM).") # RAM'deki kopyanın temizlendiğini belirt.

        # Alt bellek modüllerinin cleanup metotlarını çağır (varsa).
        # cleanup_safely yardımcı fonksiyonunu kullanabiliriz.
        # cleanup_safely'ye sadece method referansı gönderilmelidir.
        # Alt modül objeleri None değilse ve cleanup metotları varsa çağır.
        # Alt modül sınıfları import edildiyse ve objeler oluşturulduysa cleanup çağrılır.
        # isinstance(self.episodic_memory, EpisodicMemory) kontrolü de yapılabilir.
        # TODO: Alt modül cleanup çağrıları buraya gelecek
        # if hasattr(self.episodic_memory, 'cleanup'):
        #      cleanup_safely(self.episodic_memory.cleanup, logger_instance=logger, error_message="Memory: EpisodicMemory temizlenirken hata")
        # if hasattr(self.semantic_memory, 'cleanup'):
        #      cleanup_safely(self.semantic_memory.cleanup, logger_instance=logger, error_message="Memory: SemanticMemory temizlenirken hata")


        logger.info("Memory modülü objesi silindi.")

# Pickle modülü importu eksikti, ekleyelim
import pickle