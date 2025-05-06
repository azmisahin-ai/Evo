# src/memory/core.py
#
# Evo'nın temel bellek sistemini temsil eder.
# Öğrenilmiş temsilleri saklar ve gerektiğinde geri çağırır.
# Belleğe dosya tabanlı kalıcılık kazandırır.
# Gelecekte episodik ve semantik bellek gibi alt modülleri koordine edecektir.

import numpy as np # Temsil vektörleri (numpy array) için.
import time # Anıların zaman damgası için.
import random # Placeholder retrieve (rastgele seçim) için.
import logging # Loglama için.
import pickle # Belleği dosyaya kaydetmek ve yüklemek için
import os # Dosya sistemi işlemleri için (kontrol etme, dizin oluşturma)

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
    Temel list tabanlı depolama (core/working memory) implementasyonu içerir
    ve bu belleğe dosya tabanlı kalıcılık (pickle) kazandırılmıştır.
    Hata durumlarında işlemleri loglar ve programın çökmesini engeller.
    """
    def __init__(self, config):
        """
        Memory modülünü başlatır.

        Temel depolama yapısını (şimdilik liste) başlatır, kalıcı bellekten yükler
        ve alt bellek modüllerini (EpisodicMemory, SemanticMemory) başlatmayı dener (gelecekte).

        Args:
            config (dict): Bellek sistemi yapılandırma ayarları.
                           'max_memory_size': Core/Working bellekte saklanacak maksimum temsil sayısı (int, varsayılan 1000).
                           'num_retrieved_memories': retrieve metodunda varsayılan olarak geri çağrılacak anı sayısı (int, varsayılan 5).
                           'memory_file_path': Belleğin kaydedileceği/yükleneceği dosya yolu (str, varsayılan 'data/core_memory.pkl').
                           Alt modüllere ait ayarlar kendi adları altında beklenir
                           (örn: {'episodic': {...}, 'semantic': {...}}).
        """
        self.config = config
        logger.info("Memory modülü başlatılıyor...")

        # Yapılandırmadan ayarları alırken get_config_value kullan
        self.max_memory_size = get_config_value(config, 'max_memory_size', 1000, expected_type=(float, int), logger_instance=logger)
        self.num_retrieved_memories = get_config_value(config, 'num_retrieved_memories', 5, expected_type=(float, int), logger_instance=logger)
        self.memory_file_path = get_config_value(config, 'memory_file_path', 'data/core_memory.pkl', expected_type=str, logger_instance=logger)

        # num_retrieved_memories için negatif değer kontrolü (get_config_value tipi kontrol ediyor ama değeri etmiyor)
        if not isinstance(self.num_retrieved_memories, int) or self.num_retrieved_memories < 0:
             logger.warning(f"Memory: Konfigürasyonda num_retrieved_memories geçersiz ({self.num_retrieved_memories}). Varsayılan 5 kullanılıyor.")
             self.num_retrieved_memories = 5

        # max_memory_size için negatif veya float değer kontrolü
        if not isinstance(self.max_memory_size, int) or self.max_memory_size < 0:
             logger.warning(f"Memory: Konfigürasyonda max_memory_size geçersiz ({self.max_memory_size}). Varsayılan 1000 kullanılıyor.")
             self.max_memory_size = 1000


        # Bellek depolama yapıları. Şimdilik sadece temel list tabanlı (core/working memory).
        # Her öğe bir sözlüktür: {'representation': numpy_array, 'metadata': dict, 'timestamp': float}
        self.core_memory_storage = [] # Temel / Çalışma Belleği Depolama

        self.episodic_memory = None # Anısal bellek alt modülü objesi.
        self.semantic_memory = None # Kavramsal bellek alt modülü objesi.


        # Kalıcı bellekten yükleme mantığı
        self._load_from_storage()


        # Alt bellek modüllerini başlatmayı dene (Gelecek TODO).
        # Başlatma hataları kendi içlerinde veya _initialize_single_module gibi bir utility ile yönetilmeli.
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


        logger.info(f"Memory modülü başlatıldı. Maksimum Core Bellek boyutu: {self.max_memory_size}. Varsayılan geri çağrı sayısı: {self.num_retrieved_memories}. Kalıcılık dosyası: {self.memory_file_path}. Yüklenen anı sayısı: {len(self.core_memory_storage)}")


    def _load_from_storage(self):
        """
        Belirtilen dosyadan bellek durumunu (core_memory_storage) yükler.
        Dosya yoksa, okunamıyorsa veya bozuksa, belleği boş başlatır.
        """
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
                loaded_data = pickle.load(f)

            # Yüklenen verinin beklendiği gibi bir liste olup olmadığını kontrol et.
            # Daha sağlam bir kontrol, listedeki her öğenin {'representation': np.ndarray, 'metadata': dict, 'timestamp': float}
            # formatında olup olmadığını da içerebilir, ancak bu başlangıç için yeterli.
            if isinstance(loaded_data, list):
                self.core_memory_storage = loaded_data
                logger.info(f"Memory._load_from_storage: Bellek başarıyla yüklendi: {self.memory_file_path} ({len(self.core_memory_storage)} anı).")
                # Yüklenen anı sayısı max_memory_size'ı aşıyorsa eski anıları sil (yükleme sonrası temizlik)
                if len(self.core_memory_storage) > self.max_memory_size:
                    logger.warning(f"Memory._load_from_storage: Yüklenen anı sayısı ({len(self.core_memory_storage)}) maksimum boyutu ({self.max_memory_size}) aşıyor. Eski anılar siliniyor.")
                    self.core_memory_storage = self.core_memory_storage[-self.max_memory_size:] # Sadece son max_memory_size kadar anıyı tut.

            else:
                # Yüklenen veri liste formatında değilse
                logger.error(f"Memory._load_from_storage: Yüklenen bellek dosyası beklenmeyen formatta: {self.memory_file_path}. Liste bekleniyordu, geldi: {type(loaded_data)}. Bellek boş başlatılıyor.", exc_info=True)
                self.core_memory_storage = [] # Format yanlışsa boş başlat.

        except FileNotFoundError:
            # os.path.exists kontrolü yapıldı ama yine de yakalamak sağlamlık katabilir.
            logger.warning(f"Memory._load_from_storage: Bellek dosyası bulunamadı (yeniden kontrol sonrası): {self.memory_file_path}. Bellek boş başlatılıyor.")
            self.core_memory_storage = []

        except (pickle.UnpicklingError, EOFError, ImportError, IndexError, Exception) as e:
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
        if save_dir and not os.path.exists(save_dir): # Boş string kontrolü ekledim.
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

        except (pickle.PicklingError, IOError, OSError, Exception) as e:
            # pickle kaydetme veya dosya yazma sırasında oluşabilecek hatalar
            logger.error(f"Memory._save_to_storage: Bellek dosyası kaydedilirken hata oluştu: {self.memory_file_path}.", exc_info=True)

        except Exception as e:
             # Diğer tüm beklenmedik hatalar.
             logger.error(f"Memory._save_to_storage: Bellek dosyası kaydedilirken beklenmedik hata: {self.memory_file_path}.", exc_info=True)


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
        if not check_input_not_none(representation, input_name="representation for Memory.store", logger_instance=logger):
             logger.debug("Memory.store: Representation input None. Saklama atlandi.")
             return # None ise saklama atla.

        # Hata yönetimi: Representation'ın numpy array ve sayısal dtype olup olmadığını kontrol et.
        # expected_ndim=1 çünkü representation genellikle 1D vektördür.
        if not check_numpy_input(representation, expected_dtype=np.number, expected_ndim=1, input_name="representation for Memory.store", logger_instance=logger):
             logger.error("Memory.store: Representation input numpy array değil veya yanlış dtype/boyut. Saklama atlandi.") # check_numpy_input kendi içinde loglar.
             return # Geçersiz tip, dtype veya boyut ise saklama atla.

        # Hata yönetimi: Metadata None veya dict mi? check_input_type kullan.
        if metadata is not None and not check_input_type(metadata, dict, input_name="metadata for Memory.store", logger_instance=logger):
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
            logger.debug(f"Memory.store: Temsil başarıyla core belleğe saklandı. Güncel boyutu: {len(self.core_memory_storage)}")


            # Maksimum core bellek boyutu aşıldysa en eski öğeyi sil (FIFO).
            if len(self.core_memory_storage) > self.max_memory_size:
                # max_memory_size 0 veya negatif ise bu kontrol gereksiz olabilir ama >= 0 varsayıyoruz.
                # Listenin başındaki (en eski) öğeyi çıkar.
                removed_entry = self.core_memory_storage.pop(0)
                # DEBUG logu: Silinen anı hakkında bilgi.
                logger.debug(f"Memory.store: Maksimum core bellek boyutu aşıldı ({self.max_memory_size}). En eski anı silindi (timestamp: {removed_entry['timestamp']:.2f}).")

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

        Gelen sorgu (query_representation) ile temel (core) bellekteki anıları
        (representation vektörleri üzerinden) vektör benzerliği hesaplayarak
        en ilgili olanları geri çağırır.
        query_representation None veya geçersiz ise rastgele anı çağırma (eski placeholder)
        mantığına geri döner.
        Hata durumunda veya bellek boşsa boş liste döndürür.

        Args:
            query_representation (numpy.ndarray or None): Sorgu için kullanılan temsil vektörü.
                                                         Genellikle RepresentationLearner'dan gelir.
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
        if not isinstance(num_results, int) or num_results < 0:
             logger.warning(f"Memory.retrieve: Geçersiz num_results değeri veya tipi ({num_results}). Varsayılan ({self.num_retrieved_memories}) veya 5 kullanılacak.")
             num_results = self.num_retrieved_memories if isinstance(self.num_retrieved_memories, int) and self.num_retrieved_memories >= 0 else 5
        actual_num_results = min(num_results, len(self.core_memory_storage)) # Listenin boyutunu aşmasın

        retrieved_list = [] # Geri çağrılan anıları/bilgileri tutacak liste.

        # Sorgu representation'ın geçerli (None değil, numpy array, 1D, sayısal) olup olmadığını kontrol et.
        valid_query = query_representation is not None and check_numpy_input(query_representation, expected_dtype=np.number, expected_ndim=1, input_name="query_representation for Memory.retrieve", logger_instance=logger)


        if not self.core_memory_storage or actual_num_results <= 0:
            # Bellek boşsa veya istenen sonuç sayısı 0 veya negatifse, boş liste döndür.
            logger.debug("Memory.retrieve: Core memory empty or effective num_results non-positive. Returning empty list.")
            return []

        try:
            if valid_query:
                # --- Vektör Benzerliği ile Geri Çağırma Mantığı ---
                logger.debug(f"Memory.retrieve: Valid query representation provided (Shape: {query_representation.shape}). Performing similarity search.")
                similarities = []
                query_norm = np.linalg.norm(query_representation)

                # Sorgu vektörü sıfır ise benzerlik hesaplanamaz.
                if query_norm < 1e-8: # Sıfıra yakınlığı kontrol et
                     logger.warning("Memory.retrieve: Query representation has near-zero norm. Cannot calculate cosine similarity meaningfully. Returning empty list.")
                     return [] # Veya bu durumda rastgele çağırmaya dönebiliriz policy'e göre. Şimdilik boş dönelim.

                # Bellekteki her anı ile sorgu arasındaki benzerliği hesapla.
                for memory_entry in self.core_memory_storage:
                    # Anıdaki representation'ı al ve geçerliliğini kontrol et.
                    # Stored representation'ın geçerli bir sayısal 1D numpy array olduğunu doğrula.
                    if memory_entry is not None and isinstance(memory_entry, dict): # memory_entry dict mi kontrol et
                         stored_representation = memory_entry.get('representation') # .get ile güvenli erişim
                         # np.issubtype yerine isinstance ve np.number kullanımı (Hata düzeltme).
                         if stored_representation is not None and isinstance(stored_representation, np.ndarray) and isinstance(stored_representation.dtype, np.number) and stored_representation.ndim == 1: # <<< HATA DÜZELTME
                              stored_norm = np.linalg.norm(stored_representation)
                              if stored_norm > 1e-8: # Stored vektör sıfır ise bölme hatası olmaması için kontrol et.
                                   # Kosinüs benzerliği hesapla: (dot product) / (norm1 * norm2)
                                   similarity = np.dot(query_representation, stored_representation) / (query_norm * stored_norm)
                                   if not np.isnan(similarity):
                                        similarities.append((similarity, memory_entry))
                              # else: logger.debug("Memory.retrieve: Stored rep near zero norm, skipping similarity.")
                         # else: logger.debug("Memory.retrieve: Invalid stored rep, skipping.")
                    # else: logger.warning("Memory.retrieve: Listedeki öğe dict değil, yoksayılıyor.")


                # Benzerliklere göre azalan sırada sırala.
                # Eğer similarities listesi boşsa sort hata vermez, sadece bir şey yapmaz.
                similarities.sort(key=lambda item: item[0], reverse=True)

                # En yüksek benzerliğe sahip 'actual_num_results' kadar anıyı al.
                # Eğer similarities listesi istenen sayıdan azsa, sadece listedeki kadarını alır.
                retrieved_list = [item[1] for item in similarities[:actual_num_results]]

                logger.debug(f"Memory.retrieve: Found {len(similarities)} memories with valid representations for similarity check. Retrieved top {len(retrieved_list)} by similarity.")

            else:
                # --- Placeholder Rastgele Geri Çağırma Mantığı (Query geçersizse) ---
                # Eğer sorgu representation None veya geçersiz ise, eski rastgele çağırma mantığına geri dön.
                logger.debug("Memory.retrieve: Query representation is None or invalid. Falling back to random retrieval.")
                if actual_num_results > 0: # Hala çağrılacak anı varsa rastgele seç.
                    # core_memory_storage listesinin her öğesinin dict olduğundan emin ol (yükleme/kaydetme mantığına göre olmalı).
                    valid_memories_for_random = [entry for entry in self.core_memory_storage if isinstance(entry, dict)]
                    if valid_memories_for_random:
                         retrieved_list = random.sample(valid_memories_for_random, min(actual_num_results, len(valid_memories_for_random))) # Geçerli olanlardan seç
                    # else: retrieved_list boş kalır.
                logger.debug(f"Memory.retrieve: Retrieved {len(retrieved_list)} memories randomly.")


            # TODO: Gelecekte: Eğer alt bellek modülleri başlatıldıysa, onlardan da ilgili sonuçları al ve retrieved_list ile birleştir.
            # if self.episodic_memory:
            #      episodic_results = self.episodic_memory.retrieve_event(query_representation, time_range=...)
            #      retrieved_list.extend(episodic_results)
            # if self.semantic_memory:
            #      semantic_results = self.semantic_memory.retrieve_concept(query_representation, relation_filter=...)
            #      retrieved_list.extend(semantic_results)

            # TODO: Gelecekte: Farklı bellek türlerinden gelen sonuçları önceliklendir veya sırala.


        except Exception as e:
            # Geri çağırma işlemi sırasında beklenmedik bir hata olursa logla.
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
            list: numpy arraylerden oluşan liste.
        """
        # Sadece geçerli numpy array Representationları içeren bir liste döndür.
        # core_memory_storage'daki her öğenin {'representation': np.ndarray, ...} formatında olduğu varsayılır.
        valid_representations = [entry.get('representation') for entry in self.core_memory_storage if isinstance(entry, dict) and entry.get('representation') is not None and isinstance(entry.get('representation'), np.ndarray) and np.issubtype(entry.get('representation').dtype, np.number) and entry.get('representation').ndim == 1]
        logger.debug(f"Memory.get_all_representations: {len(valid_representations)} geçerli Representation döndürülüyor.")
        return valid_representations # Kopyasını döndürmek daha güvenli olabilir: [rep.copy() for rep in valid_representations]


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
        self._save_to_storage()

        # Temel (core) bellek listesini temizle (kaydedildikten sonra).
        # Listeyi None yapmak veya boş bir liste atamak, objelerin garbage collection tarafından toplanmasına yardımcı olur.
        self.core_memory_storage = [] # Veya self.core_memory_storage = None
        logger.info("Memory: Core bellek temizlendi (RAM).") # RAM'deki kopyanın temizlendiğini belirt.

        # Alt bellek modüllerinin cleanup metotlarını çağır (varsa).
        # cleanup_safely yardımcı fonksiyonunu kullanabiliriz.
        # cleanup_safely'ye sadece method referansı gönderilmelidir.
        if self.episodic_memory and hasattr(self.episodic_memory, 'cleanup'):
             cleanup_safely(self.episodic_memory.cleanup, logger_instance=logger, error_message="Memory: EpisodicMemory temizlenirken hata")
        if self.semantic_memory and hasattr(self.semantic_memory, 'cleanup'):
             cleanup_safely(self.semantic_memory.cleanup, logger_instance=logger, error_message="Memory: SemanticMemory temizlenirken hata")


        logger.info("Memory modülü objesi silindi.")