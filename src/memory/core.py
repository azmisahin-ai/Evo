# src/memory/core.py
import numpy as np
import time # Anı zaman damgası için
import random # Placeholder retrieve için
import logging # Loglama için

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

class Memory:
    """
    Evo'nın bellek sistemi. Öğrenilmiş temsilleri saklar ve gerektiğinde geri çağırır.
    Şimdilik basit bir liste tabanlı geçici hafıza implementasyonu.
    Gelecekte kalıcı depolama, indeksleme ve daha gelişmiş geri çağırma gelecek.
    """
    def __init__(self, config):
        self.config = config
        self.max_memory_size = config.get('max_memory_size', 1000) # Saklanacak maksimum temsil sayısı
        self.num_retrieved_memories = config.get('num_retrieved_memories', 5) # Her çağrıda kaç tane döndürülecek

        self.memory_storage = [] # Bellek öğeleri listesi [{representation: ndarray, metadata: dict, timestamp: float}]

        logger.info("Memory modülü başlatılıyor...")
        # Kalıcı bellekten yükleme (gelecekte)
        # self._load_from_storage() # Gelecek TODO

        logger.info(f"Memory modülü başlatıldı. Maksimum boyut: {self.max_memory_size}")


    def store(self, representation, metadata=None):
        """
        Öğrenilmiş bir temsili (ve ilişkili metadatayı) belleğe kaydeder.
        Maksimum boyutu aşarsa en eski anıyı siler (FIFO - First-In, First-Out).

        Args:
            representation (numpy.ndarray): Saklanacak temsil vektörü.
            metadata (dict, optional): Temsille ilişkili ek bilgiler (örn: kaynak, zaman aralığı vb.). Varsayılan None.
        """
        # Temel hata yönetimi: Girdi tipi kontrolü
        if representation is None:
             logger.debug("Memory.store: Saklanacak representation None.")
             return # None gelirse saklama
        if not isinstance(representation, np.ndarray):
             logger.error(f"Memory.store: Beklenmeyen representation tipi: {type(representation)}. numpy.ndarray bekleniyordu.")
             return # Geçersiz tipse saklama

        try:
            # Yeni bellek öğesi oluştur
            memory_entry = {
                'representation': representation,
                'metadata': metadata if metadata is not None else {}, # Metadata yoksa boş sözlük
                'timestamp': time.time() # Kayıt zamanı
            }

            # Belleğe ekle
            self.memory_storage.append(memory_entry)
            # logger.debug(f"Memory.store: Temsil başarıyla saklandı. Güncel hafıza boyutu: {len(self.memory_storage)}")


            # Maksimum boyutu aştıysa en eski öğeyi sil
            if len(self.memory_storage) > self.max_memory_size:
                # En eski öğe listede ilk sıradadır (FIFO)
                removed_entry = self.memory_storage.pop(0)
                # logger.debug(f"Memory.store: Maksimum boyut aşıldı, en eski anı silindi (timestamp: {removed_entry['timestamp']:.2f}).")


        except Exception as e:
            # Saklama sırasında beklenmedik hata
            logger.error(f"Memory.store: Belleğe kaydetme sırasında beklenmedik hata: {e}", exc_info=True)
            # Hata durumunda liste bozulabilir, bu kritik olabilir.
            # Şimdilik sadece logla ve devam et, ama gelecekte daha sağlam olmalı.


    def retrieve(self, query_representation, num_results=None):
        """
        Bellekten ilgili anıları geri çağırır.
        Şimdilik sadece bellekteki tüm anıları veya belirtilen sayıda rastgele anıyı döndürür.
        Gelecekte 'query_representation' kullanılarak benzerlik tabanlı arama yapılacak.

        Args:
            query_representation (numpy.ndarray or None): Sorgu için kullanılan temsil vektörü (şimdilik kullanılmıyor).
            num_results (int, optional): Geri çağrılacak maksimum anı sayısı. Varsayılan self.num_retrieved_memories.

        Returns:
            list: İlgili bellek öğelerinin listesi. Hata durumunda veya anı yoksa boş liste [].
        """
        # num_results için varsayılan değeri ayarla
        if num_results is None:
            num_results = self.num_retrieved_memories

        # Temel hata yönetimi: Bellek boşsa veya num_results geçersizse
        if not self.memory_storage:
            # logger.debug("Memory.retrieve: Bellek boş. Geri çağrılamadı.")
            return [] # Bellek boşsa boş liste döndür

        if not isinstance(num_results, int) or num_results < 0:
             logger.warning(f"Memory.retrieve: Geçersiz num_results değeri: {num_results}. Varsayılan ({self.num_retrieved_memories}) veya 0 kullanılacak.")
             num_results = self.num_retrieved_memories if isinstance(self.num_retrieved_memories, int) and self.num_retrieved_memories >= 0 else 5
             if num_results <= 0: return [] # num_results hala geçersizse boş liste

        # query_representation şimdilik kullanılmıyor ama gelecekte buraya işlenecek.
        # if query_representation is not None and not isinstance(query_representation, np.ndarray):
        #      logger.warning(f"Memory.retrieve: Sorgu representation tipi beklenmiyor: {type(query_representation)}. numpy.ndarray veya None bekleniyordu.")
        #      # Hata vermeden devam et ama logla.

        retrieved_list = [] # Geri çağrılacak anıları tutacak liste

        try:
            # Şimdilik placeholder: Bellekteki anıların bir alt kümesini veya tamamını döndür
            # Rastgele seçim yapalım (retrieve mantığı geliştikçe burası değişecek)
            actual_num_results = min(num_results, len(self.memory_storage)) # Bellek boyutunu aşmasın
            if actual_num_results > 0:
                 retrieved_list = random.sample(self.memory_storage, actual_num_results)

            # DEBUG logu: Geri çağrılan anı sayısı
            # logger.debug(f"Memory.retrieve: Hafızadan {len(retrieved_list)} girdi geri çağrıldı (placeholder).")


        except Exception as e:
            # Geri çağırma sırasında beklenmedik hata (örn: random.sample hatası)
            logger.error(f"Memory.retrieve: Bellekten geri çağırma sırasında beklenmedik hata: {e}", exc_info=True)
            return [] # Hata durumunda boş liste döndür

        return retrieved_list # Başarılı durumda listeyi döndür

    def cleanup(self):
        """Kaynakları temizler (kalıcı bellek kaydetme vb.)."""
        logger.info("Memory modülü objesi siliniyor...")
        # Belleği kalıcı depolamaya kaydet (gelecekte)
        # self._save_to_storage() # Gelecek TODO
        # Şimdilik sadece listeyi temizleyelim (objeler garbage collection ile silinir)
        self.memory_storage = []
        logger.info("Memory modülü objesi silindi.")