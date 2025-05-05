# src/representation/models.py

import logging
import numpy as np
# Kendi temel sinir ağı bileşenlerimizi import et
from src.core.nn_components import Dense

class RepresentationLearner:
    """
    Evo'nın temsil öğrenme birimini temsil eder.
    İşlenmiş duyusal veriden (processing çıktısı) içsel temsiller öğrenir.
    Temel olarak Dense katmanları kullanarak bir model kurar.
    """
    def __init__(self, config=None):
        logging.info("RepresentationLearner başlatılıyor...")
        self.config = config if config is not None else {}

        # Temsil vektörünün boyutu
        self.representation_dim = self.config.get('representation_dim', 128)

        # --- Model Mimarisi Tanımı (Faz 1) ---
        # Girdi boyutu, config'den alınır. Processing çıktılarının boyutuna göre manuel ayarlanmalı.
        # Vision Processor varsayılan çıktı 64x64 gri -> 64*64 = 4096 boyutlu düzleştirilmiş vektör
        # Audio Processor varsayılan çıktı enerji -> 1 boyutlu float
        # Toplam girdi boyutu = 4096 + 1 = 4097
        self.input_dim = self.config.get('input_dim', 4097) # Varsayılan (64*64 + 1)

        # Basit bir Dense katmanı tanımlayalım: Giriş boyutundan representation_dim boyutuna
        try:
             self.dense_layer_1 = Dense(input_size=self.input_dim, output_size=self.representation_dim, activation='relu')
             # İsteğe bağlı olarak başka katmanlar da eklenebilir
             # self.dense_layer_2 = Dense(input_size=self.representation_dim, output_size=self.representation_dim)
             self.is_initialized = True # Başlatma başarılı
        except Exception as e:
             logging.critical(f"RepresentationLearner: Dense katmanı başlatılırken kritik hata oluştu: {e}", exc_info=True)
             self.is_initialized = False # Başlatma başarısız
             self.dense_layer_1 = None


        logging.info(f"RepresentationLearner başlatıldı. Girdi boyutu: {self.input_dim}, Temsil boyutu: {self.representation_dim}. Başlatma Başarılı: {self.is_initialized}")

    def learn(self, processed_inputs):
        """
        İşlenmiş duyusal verileri içeren bir sözlük (processed_inputs) alır
        ve onlardan birleşik, öğrenilmiş bir temsil vektörü (NumPy array) döndürür.
        processed_inputs: {'visual': NumPy array (gri, yeniden boyutlandırılmış), 'audio': float (enerji)}
        """
        if not self.is_initialized or self.dense_layer_1 is None:
             logging.error("RepresentationLearner: Modül tam başlatılmamış. Öğrenme atlandı.")
             return None

        visual_data = processed_inputs.get('visual')
        audio_data = processed_inputs.get('audio')

        if visual_data is None and audio_data is None:
            logging.debug("RepresentationLearner: İşlenecek işlenmiş veri yok (görsel ve ses ikisi de None).") # DEBUG logu
            return None # Veri yoksa None döndür

        logging.debug(f"RepresentationLearner: İşlenmiş veriler alindi. Visual var: {visual_data is not None}, Audio var: {audio_data is not None}") # DEBUG logu


        # --- Veriyi Model Girdisine Hazırlama ---
        # İşlenmiş görsel veriyi düzleştir (flatten)
        visual_flat = None
        if visual_data is not None and isinstance(visual_data, np.ndarray):
             visual_flat = visual_data.flatten()
             logging.debug(f"RepresentationLearner: Görsel veri düzleştirildi. Shape: {visual_flat.shape}") # DEBUG logu
        elif visual_data is not None:
             logging.warning(f"RepresentationLearner: Görsel veri beklenmeyen formatta (NumPy array bekleniyordu, geldi: {type(visual_data)}). Düzleştirme atlandı.")


        # İşlenmiş işitsel veriyi (float) bir NumPy array'e dönüştür
        audio_vec = None
        if audio_data is not None and isinstance(audio_data, (float, np.float32, np.float64)):
             audio_vec = np.array([audio_data], dtype=np.float32) # Tek elemanlı vektör yap
             logging.debug(f"RepresentationLearner: Ses verisi vektör yapıldı. Shape: {audio_vec.shape}") # DEBUG logu
        elif audio_data is not None and isinstance(audio_data, np.ndarray):
             # Eğer audio_data zaten array ise, onu da düzleştir
             audio_vec = audio_data.flatten()
             logging.debug(f"RepresentationLearner: Ses array verisi düzleştirildi. Shape: {audio_vec.shape}") # DEBUG logu

        elif audio_data is not None:
             logging.warning(f"RepresentationLearner: Ses verisi beklenmeyen formatta (float veya NumPy array bekleniyordu, geldi: {type(audio_data)}). Vektöre çevirme atlandı.")


        # Görsel ve işitsel vektörleri birleştir
        combined_input = None
        if visual_flat is not None and audio_vec is not None:
             # İkisini birleştir (concatenate)
             try:
                  combined_input = np.concatenate((visual_flat, audio_vec))
                  logging.debug(f"RepresentationLearner: Görsel ve ses verisi birleştirildi. Shape: {combined_input.shape}") # DEBUG logu
             except ValueError as e:
                  logging.error(f"RepresentationLearner: Görsel ve ses verisi birleştirilirken boyut hatası: {e}. Görsel Shape: {visual_flat.shape}, Ses Shape: {audio_vec.shape}", exc_info=True)
                  return None # Birleştirme hatası


        elif visual_flat is not None: # Sadece görsel varsa
             combined_input = visual_flat
             logging.debug(f"RepresentationLearner: Sadece görsel veri kullanılıyor. Shape: {combined_input.shape}") # DEBUG logu

        elif audio_vec is not None: # Sadece ses varsa
             combined_input = audio_vec
             logging.debug(f"RepresentationLearner: Sadece ses verisi kullanılıyor. Shape: {combined_input.shape}") # DEBUG logu


        if combined_input is None or combined_input.size == 0:
             logging.debug("RepresentationLearner: Model için geçerli girdi oluşturulamadı.") # DEBUG logu
             return None

        # Girdi boyutunun beklenen input_dim ile aynı olduğundan emin ol
        if combined_input.shape[-1] != self.input_dim:
            logging.error(f"RepresentationLearner: Birleştirilmiş girdi boyutu beklenen input_dim ({self.input_dim}) ile uyuşmuyor ({combined_input.shape[-1]}).")
            # Bu durumda model çalışmaz, None döndür.
            return None


        # --- Modeli Çalıştırma (İleri Yayılım) ---
        learned_representation = None
        try:
            # Birleştirilmiş girdiyi Dense katmanından geçir
            # Dense katmanı tek bir örnek bekliyor şimdilik: (input_dim,) shape
            # combined_input zaten bu shape'te olmalı eğer batch işlenmiyorsa.
            learned_representation = self.dense_layer_1.forward(combined_input)

            # Eğer başka katmanlar olsaydı, buradan devam ederdi:
            # hidden_output = self.dense_layer_2.forward(learned_representation)
            # final_representation = hidden_output # Veya son katmanın çıktısı


            if learned_representation is not None:
                 # Öğrenilmiş temsilin boyutunu kontrol et
                 if learned_representation.shape[-1] != self.representation_dim:
                      logging.error(f"RepresentationLearner: Öğrenilmiş temsil boyutu beklenen representation_dim ({self.representation_dim}) ile uyuşmuyor ({learned_representation.shape[-1]}).")
                      # Hata durumunda None döndür
                      return None

                 logging.debug(f"RepresentationLearner: Temsil başarıyla öğrenildi. Output Shape: {learned_representation.shape}, Dtype: {learned_representation.dtype}") # DEBUG logu
            # else: logging.debug("RepresentationLearner: Model çikti üretmedi (None döndürdü).") # DEBUG logu


            return learned_representation # Öğrenilmiş temsil vektörü (NumPy array veya None) döndür

        except Exception as e:
            logging.error(f"RepresentationLearner model forward sırasında hata oluştu: {e}", exc_info=True)
            return None


    def __del__(self):
        """
        Nesne silindiğinde kaynakları temizler.
        Model katmanları (Dense objeleri) otomatik silinecektir.
        """
        logging.info("RepresentationLearner objesi silindi.")

# Modülü bağımsız test etmek için örnek kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print("RepresentationLearner test ediliyor (Dense katman kullanarak)...")

    # Test config: input_dim = 64*64 (4096) + 1 = 4097
    # output_width=64, output_height=64 varsayılan processing çıktısı
    test_config = {
        'input_dim': 4096 + 1, # Vision (64*64) + Audio (1)
        'representation_dim': 128 # Öğrenilecek temsil boyutu
    }
    learner = RepresentationLearner(test_config)

    if learner.is_initialized:
        # Sahte işlenmiş girdiler oluştur
        # VisionProcessor çıktısı: gri tonlama 64x64 NumPy array (dtype uint8)
        dummy_processed_visual = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
        # AudioProcessor çıktısı: enerji float değeri (dtype float32/64)
        dummy_processed_audio = np.random.rand(1)[0].astype(np.float32) # Rastgele enerji değeri


        # Hem görsel hem ses girdisi ile dene
        print("\nHem görsel hem ses girdisi ile RepresentationLearner testi:")
        processed_inputs_both = {'visual': dummy_processed_visual, 'audio': dummy_processed_audio}
        representation_both = learner.learn(processed_inputs_both)
        if representation_both is not None:
            print(f"Öğrenilmiş temsil alındı. Shape: {representation_both.shape}, Dtype: {representation_both.dtype}")
            if representation_both.shape == (test_config['representation_dim'],): # Belirlenen representation_dim ile aynı boyutta olmalı
                 print("Temsil boyutu doğru.")
        else:
            print("Temsil öğrenilemedi (beklenmeyen durum).")


        # Sadece görsel girdi ile dene (AudioProcessor None döndürdüyse)
        print("\nSadece görsel girdi ile RepresentationLearner testi:")
        # Bu durumda birleştirilmiş girdi boyutu 4096 olacak, ancak model 4097 bekliyor. Hata beklenebilir.
        processed_inputs_visual = {'visual': dummy_processed_visual, 'audio': None} # Audio None gönder
        representation_visual = learner.learn(processed_inputs_visual)
        if representation_visual is not None:
            print(f"Öğrenilmiş temsil alındı (sadece görsel). Shape: {representation_visual.shape}, Dtype: {representation_visual.dtype}")
        else:
            print("Temsil öğrenilemedi (sadece görsel) (Beklenen: input_dim uyuşmazlığı olabilir).")


        # Sadece ses girdi ile dene (VisionProcessor None döndürdüyse)
        print("\nSadece ses girdi ile RepresentationLearner testi:")
        # Bu durumda birleştirilmiş girdi boyutu 1 olacak, ancak model 4097 bekliyor. Hata beklenebilir.
        processed_inputs_audio = {'visual': None, 'audio': dummy_processed_audio} # Visual None gönder
        representation_audio = learner.learn(processed_inputs_audio)
        if representation_audio is not None:
            print(f"Öğrenilmiş temsil alındı (sadece ses). Shape: {representation_audio.shape}, Dtype: {representation_audio.dtype}")
        else:
            print("Temsil öğrenilemedi (sadece ses) (Beklenen: input_dim uyuşmazlığı olabilir).")


        # None girdi ile dene
        print("\nNone girdi ile RepresentationLearner testi:")
        representation_none = learner.learn(None)
        if representation_none is None:
            print("None girdi ile öğrenme sonucu doğru şekilde None döndü.")
        else:
             print("None girdi ile öğrenme sonucu None dönmedi (beklenmeyen durum).")

        # Boş sözlük ile dene
        print("\nBoş sözlük girdisi ile RepresentationLearner testi:")
        representation_empty = learner.learn({})
        if representation_empty is None:
            print("Boş sözlük girdisi ile öğrenme sonucu doğru şekilde None döndü.")
        else:
             print("Boş sözlük girdisi ile öğrenme sonucu None dönmedi (beklenmeyen durum).")

    else:
        print("\nRepresentationLearner başlatılamadığı için testler atlandı.")


    print("\nRepresentationLearner testi bitti.")