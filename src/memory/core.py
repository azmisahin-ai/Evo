# src/memory/core.py

import logging
import numpy as np
import time # Zaman damgası için

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
        self._storage = [] # Temsilleri saklamak için basit bir liste

        # Hafıza boyutu veya saklama limitleri burada tanımlanabilir
        self.max_size = self.config.get('max_memory_size', 1000) # Saklanacak maksimum temsil sayısı

        logging.info(f"Memory modülü başlatıldı. Maksimum boyut: {self.max_size}")

    def store(self, representation):
        """
        Öğrenilmiş bir temsil vektörünü hafızaya kaydeder.
        """
        if representation is None:
            # logging.debug("Memory: Saklanacak temsil verisi yok.")
            return # None temsil gelirse saklama

        if not isinstance(representation, np.ndarray):
             logging.warning(f"Memory: Saklanacak veri beklenmeyen formatta (NumPy array bekleniyordu, geldi: {type(representation)}). Saklama atlandı.")
             return # NumPy array değilse saklama

        # logging.debug(f"Memory: Temsil saklaniyor. Shape: {representation.shape}, Dtype: {representation.dtype}")

        try:
            # Temsili zaman damgası ile birlikte sakla (örnek)
            memory_entry = {
                'timestamp': time.time(),
                'representation': representation
            }
            self._storage.append(memory_entry)

            # Eğer hafıza maksimum boyutu aştıysa, en eski girdiyi sil (basit FIFO)
            if len(self._storage) > self.max_size:
                oldest_entry = self._storage.pop(0) # Listenin başından sil
                # logging.debug(f"Memory: Maksimum boyuta ulaşıldı, en eski girdi silindi (Timestamp: {oldest_entry['timestamp']}).")

            logging.debug(f"Memory: Temsil başarıyla saklandı. Güncel hafıza boyutu: {len(self._storage)}")

        except Exception as e:
            logging.error(f"Memory store sırasında hata oluştu: {e}", exc_info=True)


    def retrieve(self, query_representation, num_results=1):
        """
        Verilen bir temsil vektörüne (query) en çok benzeyen temsilleri hafızadan geri çağırır.
        Şimdilik basitçe hafızadaki tüm temsilleri döndürüyor gibi yapacak (veya boş liste).
        """
        if query_representation is None:
            # logging.debug("Memory: Geri çağrılacak sorgu temsili yok.")
            return [] # Sorgu yoksa boş liste döndür

        if not isinstance(query_representation, np.ndarray):
             logging.warning(f"Memory: Sorgu verisi beklenmeyen formatta (NumPy array bekleniyordu, geldi: {type(query_representation)}). Geri çağırma atlandı.")
             return [] # NumPy array değilse geri çağırma

        # logging.debug(f"Memory: Temsil geri çağrılıyor (placeholder). Sorgu Shape: {query_representation.shape}")


        # --- Gerçek Geri Çağırma/Arama Mantığı Buraya Gelecek (Faz 2 ve sonrası) ---
        # Örnek: Vektör karşılaştırma (cosine similarity vb.), indeksleme (Faiss, Annoy vb.)
        # relevant_memories = self._find_similar(query_representation, num_results)

        # Şimdilik, hafızada bir şeyler varsa basitçe birkaç tanesini döndürelim (örnek)
        if self._storage:
             # İlk birkaç girdiyi döndürelim (örnek placeholder)
             # logging.debug(f"Memory: Hafızadan {min(num_results, len(self._storage))} girdi döndürülüyor (placeholder).")
             # Sadece temsil vektörlerini döndürelim
             placeholder_results = [entry['representation'] for entry in self._storage[:min(num_results, len(self._storage))]]
             return placeholder_results
        else:
            # logging.debug("Memory: Hafıza boş, geri çağrılacak bir şey yok.")
            return [] # Hafıza boşsa boş liste döndür


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

    # Sahte temsil vektörleri oluştur
    dummy_representation_1 = np.random.rand(128).astype(np.float32)
    dummy_representation_2 = np.random.rand(128).astype(np.float32)
    dummy_representation_3 = np.random.rand(128).astype(np.float32)

    # Saklama testi
    print("\nHafızaya temsiller saklanıyor...")
    memory.store(dummy_representation_1)
    memory.store(dummy_representation_2)
    print(f"Mevcut hafıza boyutu: {memory.get_current_size()}")
    memory.store(dummy_representation_3)
    print(f"Mevcut hafıza boyutu: {memory.get_current_size()}")

    # Fazla saklama testi (limit aşıyor)
    print("\nHafıza limitini aşacak kadar saklama denemesi...")
    for i in range(10):
        memory.store(np.random.rand(128).astype(np.float32))
    print(f"Mevcut hafıza boyutu (limit sonrası): {memory.get_current_size()}") # Maksimum boyutta kalmalı


    # Geri çağırma testi
    print("\nHafızadan geri çağırma denemesi (placeholder)...")
    # query_vector olarak herhangi bir vektör gönderebiliriz, şu an mantığı kullanmıyor
    query_vector = np.random.rand(128).astype(np.float32)
    retrieved_memories = memory.retrieve(query_vector, num_results=2)

    print(f"Geri çağrılan girdi sayısı: {len(retrieved_memories)}")
    if retrieved_memories:
        print(f"İlk geri çağrılan temsilin boyutu: {retrieved_memories[0].shape}")
    else:
        print("Hafızadan bir şey geri çağrılamadı.")

    # None girdi ile geri çağırma testi
    print("\nNone girdi ile geri çağırma testi...")
    retrieved_none = memory.retrieve(None)
    print(f"None girdi ile geri çağrılan girdi sayısı: {len(retrieved_none)}")


    print("\nMemory modülü testi bitti.")