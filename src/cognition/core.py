# src/cognition/core.py

import logging
import time
import numpy as np

class CognitionCore:
    """
    Evo'nun çekirdek bilişsel işlem birimini temsil eder.
    Öğrenilmiş temsilleri ve hafıza girdilerini kullanarak anlama ve karar alma süreçlerini yürütür.
    """
    def __init__(self, config=None):
        logging.info("Cognition modülü başlatılıyor...")
        self.config = config if config is not None else {}

        # Bilişsel modeller veya ayarlar burada yüklenebilir/tanımlanabilir
        # Örneğin: self.decision_threshold = self.config.get('decision_threshold', 0.5)

        logging.info("Cognition modülü başlatıldı.")

    def decide(self, representation, memory_entries):
        """
        Öğrenilmiş temsil vektörünü ve ilgili hafıza girdilerini alır,
        basit bir bilişsel işlem yaparak bir karar veya anlama çıktısı döndürür.

        representation: representation modülünden gelen NumPy array (veya None)
        memory_entries: memory modülünden gelen liste (entry sözlüklerinden oluşur, boş olabilir)
        """
        # logging.debug(f"Cognition: Karar alma isleniyor. Representation var mi: {representation is not None}, Hafiza girdisi sayisi: {len(memory_entries) if memory_entries else 0}")

        # --- Gerçek Anlama ve Karar Alma Mantığı Buraya Gelecek (Faz 3 ve sonrası) ---
        # Örnek: Temsilin belirli bir kalıba benzeyip benzemediğini kontrol et
        # Örnek: Hafızadaki bir girdiyle mevcut temsili karşılaştır
        # Örnek: Basit kural tabanlı mantık uygula

        # Şimdilik basit bir placeholder karar: Eğer temsil veya hafıza girdisi varsa 'aktif' bir sinyal döndür
        # Gelecekte, bu çıktı daha karmaşık bir "karar vektörü" veya "anlama durumu" olabilir.

        decision_output = None # Varsayılan çıktı

        if representation is not None or (memory_entries and len(memory_entries) > 0):
            # Eğer herhangi bir girdi (temsil veya hafıza) mevcutsa, basit bir "uyanıklık" veya "aktiflik" sinyali üret
            # decision_output = "something_detected" # Örnek string karar
            # decision_output = 1.0 # Örnek sayısal sinyal

            # Daha temsili bir çıktı: Girdi olup olmadığını belirten bir bool veya string
            if representation is not None and memory_entries:
                 decision_output = "processing_and_remembering"
            elif representation is not None:
                 decision_output = "processing_new_input"
            elif memory_entries:
                 decision_output = "recalling_memory"
            else: # Bu duruma normalde düşmemeli yukarıdaki if'ten dolayı
                 decision_output = "idle"

        else:
            # Ne temsil ne de hafıza girdisi varsa (çok nadir olmalı eğer sürekli girdi varsa)
             decision_output = "no_input"


        # logging.debug(f"Cognition: Karar alindi (placeholder). Output: {decision_output}")

        return decision_output # Karar çıktısını döndür

    # Gelecekte kullanılacak, daha karmaşık bilişsel metotlar
    # def understand(self, representation, memory_entries):
    #     # Temsilleri ve hafızayı kullanarak yüksek seviye anlam çıkarma
    #     pass
    # def plan(self, current_state, goal, memory):
    #      # Gelecekteki eylemleri planlama
    #      pass


    def __del__(self):
        """
        Nesne silindiğinde kaynakları temizler.
        """
        logging.info("Cognition modülü objesi silindi.")

# Modülü bağımsız test etmek için örnek kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print("Cognition modülü test ediliyor...")

    cognition = CognitionCore()
    representation_dim = 128

    # Sahte girdiler oluştur
    dummy_representation = np.random.rand(representation_dim).astype(np.float32)
    dummy_memory_entry_1 = {'timestamp': time.time(), 'representation': np.random.rand(representation_dim).astype(np.float32), 'metadata': {}}
    dummy_memory_entry_2 = {'timestamp': time.time(), 'representation': np.random.rand(representation_dim).astype(np.float32), 'metadata': {}}
    dummy_memory_entries = [dummy_memory_entry_1, dummy_memory_entry_2]

    # Temsil ve hafıza girdisi varken test et
    print("\nTemsil ve hafiza girdisi varken karar testi:")
    decision_both = cognition.decide(dummy_representation, dummy_memory_entries)
    print(f"Alinan karar: {decision_both}")
    if decision_both == "processing_and_remembering":
         print("Karar doğru görünüyor (iki girdi de var).")

    # Sadece temsil varken test et
    print("\nSadece temsil girdisi varken karar testi:")
    decision_rep_only = cognition.decide(dummy_representation, [])
    print(f"Alinan karar: {decision_rep_only}")
    if decision_rep_only == "processing_new_input":
         print("Karar doğru görünüyor (sadece temsil var).")

    # Sadece hafıza varken test et
    print("\nSadece hafiza girdisi varken karar testi:")
    decision_mem_only = cognition.decide(None, dummy_memory_entries)
    print(f"Alinan karar: {decision_mem_only}")
    if decision_mem_only == "recalling_memory":
         print("Karar doğru görünüyor (sadece hafiza var).")


    # Ne temsil ne hafıza varken test et
    print("\nHic girdi yokken karar testi:")
    decision_none = cognition.decide(None, [])
    print(f"Alinan karar: {decision_none}")
    if decision_none == "no_input":
         print("Karar doğru görünüyor (hic girdi yok).")


    print("\nCognition modülü testi bitti.")