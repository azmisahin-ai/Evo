# src/memory/core.py

import logging
import numpy as np
import time # Zaman damgası için
import random # Rastgele seçim için

class Memory:
    """
    Evo'nun temel hafıza birimini temsil eder.
    Öğrenilen temsilleri (representation) saklar ve gerektiğinde geri çağırır.
    """
    def __init__(self, config=None):
        logging.info("Memory modülü başlatılıyor...")
        self.config = config if config is not None else {}

        # Hafıza için basit bir depolama yapısı (şimdilik geçici - RAM üzerinde)
        # Gelecekte veritabanı veya dosya sistemi kullanılacak
        # Her girdi bir sözlük olacak: {'timestamp': ..., 'representation': ..., 'metadata': {...}}
        self._storage = []

        # Hafıza boyutu veya saklama limitleri burada tanımlanabilir
        self.max_size = self.config.get('max_memory_size', 1000) # Saklanacak maksimum temsil sayısı

        logging.info(f"Memory modülü başlatıldı. Maksimum boyut: {self.max_size}")

    def store(self, representation, metadata=None):
        """
        Öğrenilmiş bir temsil vektörünü ve opsiyonel metadata'yı hafızaya kaydeder.
        """
        if representation is None:
            # logging.debug("Memory: Saklanacak temsil verisi yok.")
            return # None temsil gelirse saklama

        if not isinstance(representation, np.ndarray):
             logging.warning(f"Memory: Saklanacak veri beklenmeyen formatta (NumPy array bekleniyordu, geldi: {type(representation)}). Saklama atlandı.")
             return # NumPy array değilse saklama

        # logging.debug(f"Memory: Temsil saklaniyor. Shape: {representation.shape}, Dtype: {representation.dtype}")

        try:
            memory_entry = {
                'timestamp': time.time(),
                'representation': representation,
                'metadata': metadata if metadata is not None else {} # İsteğe bağlı metadata
            }
            self._storage.append(memory_entry)

            # Eğer hafıza maksimum boyutu aştıysa, en eski girdiyi sil (basit FIFO)
            if len(self._storage) > self.max_size:
                oldest_entry = self._storage.pop(0) # Listenin başından sil
                # logging.debug(f"Memory: Maksimum boyuta ulaşıldı, en eski girdi silindi (Timestamp: {oldest_entry['timestamp']:.2f}).")

            logging.debug(f"Memory: Temsil başarıyla saklandı. Güncel hafıza boyutu: {len(self._storage)}")

        except Exception as e:
            logging.error(f"Memory store sırasında hata oluştu: {e}", exc_info=True)


    def retrieve(self, query_representation, num_results=1):
        """
        Verilen bir temsil vektörüne (query) en çok benzeyen temsilleri hafızadan geri çağırır.
        Şimdilik basitçe hafızadaki temsillerden 'num_results' kadarını döndürüyor (rastgele veya en yeniler).
        Gelecekte vektör benzerliği araması implement edilecek.
        """
        if query_representation is None:
            # logging.debug("Memory: Geri çağrılacak sorgu temsili yok.")
            return [] # Sorgu yoksa boş liste döndür

        # if not isinstance(query_representation, np.ndarray):
        #      logging.warning(f"Memory: Sorgu verisi beklenmeyen formatta (NumPy array bekleniyordu, geldi: {type(query_representation)}). Geri çağırma atlandı.")
        #      return [] # NumPy array değilse geri çağırma (Şimdilik sorgu vector kullanılmıyor)


        if not self._storage:
            # logging.debug("Memory: Hafıza boş, geri çağrılacak bir şey yok.")
            return [] # Hafıza boşsa boş liste döndür

        try:
            # --- Gerçek Geri Çağırma/Arama Mantığı Buraya Gelecek (Faz 2 ve sonrası) ---
            # Örnek: Vektör karşılaştırma (cosine similarity vb.), indeksleme (Faiss, Annoy vb.)
            # relevant_memories = self._find_similar(query_representation, num_results)

            # Şimdilik, hafızadaki temsillerden num_results kadarını seçme (basit placeholder)
            available_entries = self._storage
            num_to_retrieve = min(num_results, len(available_entries))

            if num_to_retrieve == 0:
                 # logging.debug("Memory: Geri çağrılacak yeterli girdi yok.")
                 return []

            # Basit placeholder: En yeni N girdiyi döndür
            # retrieved_entries = available_entries[-num_to_retrieve:]

            # Basit placeholder: Rastgele N girdi döndür
            retrieved_entries = random.sample(available_entries, num_to_retrieve)


            logging.debug(f"Memory: Hafızadan {num_to_retrieve} girdi geri çağrıldı (placeholder).")

            # Geri çağrılan girdilerin tamamını (timestamp, representation, metadata) döndür
            return retrieved_entries

        except Exception as e:
            logging.error(f"Memory retrieve sırasında hata oluştu: {e}", exc_info=True)
            return [] # Hata durumunda boş liste döndür


    # Örnek arama metodu (şimdilik kullanılmayacak)
    # def _find_similar(self, query, num_results):
    #     # Burada vektör arama algoritması çalışacak
    #     # return [best_match1, best_match2, ...]
    #     pass

    def get_current_size(self):
        """Hafızadaki mevcut girdi sayısını döndürür."""
        return len(self._storage)

    def __del__(self):
        """
        Nesne silindiğinde kaynakları temizler.
        """
        # Geçici bellek (liste) için özel bir temizlik gerekmez, Python halleder.
        # Kalıcı depolama (veritabanı bağlantısı vb.) kapatılmalıdır.
        logging.info("Memory modülü objesi silindi.")

# Modülü bağımsız test etmek için örnek kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print("Memory modülü test ediliyor...")

    memory = Memory({'max_memory_size': 5}) # Küçük bir hafıza boyutu test et
    representation_dim = 128 # Temsil boyutu

    # Sahte temsil vektörleri oluştur ve sakla
    print("\nHafızaya temsiller saklanıyor...")
    dummy_reps = [np.random.rand(representation_dim).astype(np.float32) for _ in range(10)] # 10 temsil
    for i, rep in enumerate(dummy_reps):
        memory.store(rep, metadata={'index': i}) # metadata ile sakla
        time.sleep(0.01) # Zaman damgaları farklı olsun diye kısa bekleme
    print(f"Saklama tamamlandı. Mevcut hafıza boyutu: {memory.get_current_size()}")


    # Geri çağırma testi (placeholder)
    print("\nHafızadan geri çağırma denemesi (placeholder)...")
    # query_vector olarak herhangi bir vektör gönderebiliriz, şu an mantığı kullanmıyor
    query_vector = np.random.rand(representation_dim).astype(np.float32)
    num_to_retrieve = 3
    retrieved_memories = memory.retrieve(query_vector, num_results=num_to_retrieve)

    print(f"Geri çağrılan girdi sayısı: {len(retrieved_memories)}")
    if retrieved_memories:
        print(f"İlk geri çağrılan girdinin yapısı (timestamp, representation.shape, metadata):")
        for i, entry in enumerate(retrieved_memories):
             # Geri çağrılan her entry bir sözlük
             print(f"  Girdi {i+1}: Timestamp={entry.get('timestamp', 'N/A'):.2f}, Shape={entry['representation'].shape if isinstance(entry.get('representation'), np.ndarray) else 'N/A'}, Metadata={entry.get('metadata', {})}")

    else:
        print("Hafızadan bir şey geri çağrılamadı.")

    # None girdi ile geri çağırma testi
    print("\nNone girdi (sorgu temsili için) ile geri çağırma testi...")
    retrieved_none = memory.retrieve(None)
    print(f"None girdi ile geri çağrılan girdi sayısı: {len(retrieved_none)}")
    if not retrieved_none:
        print("Doğru şekilde boş liste döndü.")


    print("\nMemory modülü testi bitti.")