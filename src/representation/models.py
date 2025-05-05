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
        # Girdi boyutu ne olacak? Vision Processor çıktısı (output_width * output_height) + Audio Processor çıktısı (şimdilik 1 - enerji).
        # Bu boyutlar config'den veya Processing modüllerinden alınmalı.
        # Şimdilik sabit varsayalım veya config'e ekleyelim.
        # Vision Processor varsayılan çıktı 64x64 gri -> 64*64 = 4096 boyutlu düzleştirilmiş vektör
        # Audio Processor varsayılan çıktı enerji -> 1 boyutlu float
        # Toplam girdi boyutu = 4096 + 1 = 4097
        # Bu girdi boyutunu RepresentationLearner config'ine ekleyelim.
        self.input_dim = self.config.get('input_dim', 4097) # Varsayılan (64*64 + 1)

        # Basit bir Dense katmanı tanımlayalım: Giriş boyutundan representation_dim boyutuna
        self.dense_layer_1 = Dense(input_size=self.input_dim, output_size=self.representation_dim, activation='relu')
        # İsteğe bağlı olarak başka katmanlar da eklenebilir
        # self.dense_layer_2 = Dense(input_size=self.representation_dim, output_size=self.representation_dim)


        logging.info(f"RepresentationLearner başlatıldı. Girdi boyutu: {self.input_dim}, Temsil boyutu: {self.representation_dim}")

    def learn(self, processed_inputs):
        """
        İşlenmiş duyusal verileri içeren bir sözlük (processed_inputs) alır
        ve onlardan birleşik, öğrenilmiş bir temsil vektörü (NumPy array) döndürür.
        processed_inputs: {'visual': NumPy array (gri, yeniden boyutlandırılmış), 'audio': float (enerji)}
        """
        visual_data = processed_inputs.get('visual')
        audio_data = processed_inputs.get('audio')

        if visual_data is None and audio_data is None:
            # logging.debug("RepresentationLearner: İşlenecek işlenmiş veri yok (görsel ve ses ikisi de None).")
            return None # Veri yoksa None döndür

        # --- Veriyi Model Girdisine Hazırlama ---
        # İşlenmiş görsel veriyi düzleştir (flatten)
        visual_flat = None
        if visual_data is not None and isinstance(visual_data, np.ndarray):
             visual_flat = visual_data.flatten()
             # logging.debug(f"Görsel veri düzleştirildi. Shape: {visual_flat.shape}")
        # else: logging.debug("Görsel veri None veya array değil, düzleştirme atlandı.")


        # İşlenmiş işitsel veriyi (float) bir NumPy array'e dönüştür
        audio_vec = None
        if audio_data is not None and isinstance(audio_data, (float, np.float32, np.float64)):
             audio_vec = np.array([audio_data], dtype=np.float32) # Tek elemanlı vektör yap
             # logging.debug(f"Ses verisi vektör yapıldı. Shape: {audio_vec.shape}")
        elif audio_data is not None and isinstance(audio_data, np.ndarray):
            # Eğer audio_data zaten array ise, onu da düzleştir
             audio_vec = audio_data.flatten()
             # logging.debug(f"Ses array verisi düzleştirildi. Shape: {audio_vec.shape}")

        # else: logging.debug("Ses verisi None, float veya array değil.")


        # Görsel ve işitsel vektörleri birleştir
        combined_input = None
        if visual_flat is not None and audio_vec is not None:
             # İkisini birleştir (concatenate)
             try:
                  combined_input = np.concatenate((visual_flat, audio_vec))
                  # logging.debug(f"Görsel ve ses verisi birleştirildi. Shape: {combined_input.shape}")
             except ValueError as e:
                  logging.error(f"Görsel ve ses verisi birleştirilirken boyut hatası: {e}. Görsel Shape: {visual_flat.shape}, Ses Shape: {audio_vec.shape}", exc_info=True)
                  # Eğer boyutlar uyuşmuyorsa birleştirme yapılamaz.
                  # Bu, Processing modüllerinin çıktı boyutları ile RepresentationLearner'ın beklediği input_dim arasında bir tutarsızlık olduğunu gösterebilir.
                  # Şimdilik hata loglayıp None döndürelim.
                  return None

        elif visual_flat is not None: # Sadece görsel varsa
             combined_input = visual_flat
             # logging.debug(f"Sadece görsel veri kullanılıyor. Shape: {combined_input.shape}")

        elif audio_vec is not None: # Sadece ses varsa
             combined_input = audio_vec
             # logging.debug(f"Sadece ses verisi kullanılıyor. Shape: {combined_input.shape}")


        if combined_input is None or combined_input.size == 0:
             # logging.debug("RepresentationLearner: Model için geçerli girdi oluşturulamadı.")
             return None


        # Girdi boyutunun beklenen input_dim ile aynı olduğundan emin ol
        if combined_input.shape[-1] != self.input_dim:
            logging.error(f"RepresentationLearner: Birleştirilmiş girdi boyutu beklenen input_dim ile uyuşmuyor. Beklenen: {self.input_dim}, Geldi: {combined_input.shape[-1]}.")
            # Bu durumda model çalışmaz, None döndür.
            return None


        # --- Modeli Çalıştırma (İleri Yayılım) ---
        learned_representation = None
        try:
            # Birleştirilmiş girdiyi Dense katmanından geçir
            learned_representation = self.dense_layer_1.forward(combined_input)

            # Eğer başka katmanlar olsaydı, buradan devam ederdi:
            # hidden_output = self.dense_layer_2.forward(learned_representation)
            # final_representation = hidden_output # Veya son katmanın çıktısı


            if learned_representation is not None:
                 # Öğrenilmiş temsilin boyutunu kontrol et
                 if learned_representation.shape[-1] != self.representation_dim:
                      logging.error(f"RepresentationLearner: Öğrenilmiş temsil boyutu beklenen representation_dim ile uyuşmuyor. Beklenen: {self.representation_dim}, Geldi: {learned_representation.shape[-1]}.")
                      # Hata durumunda None döndür
                      return None

                 logging.debug(f"RepresentationLearner: Temsil öğrenildi. Output Shape: {learned_representation.shape}, Dtype: {learned_representation.dtype}")
            # else: logging.debug("RepresentationLearner: Model çikti üretmedi (None döndürdü).")


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
        'input_dim': 4097, # 64x64 gri + 1 ses enerjisi
        'representation_dim': 128 # Öğrenilecek temsil boyutu
    }
    learner = RepresentationLearner(test_config)

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
    # Bu durumda input_dim 4096 olmalıydı, ancak model 4097 bekliyor. Şu an hata verecek.
    # Gelecekte RepresentationLearner farklı girdi kombinasyonlarını yönetmeli.
    # Şimdilik bu test başarısız olacaktır input_dim uyuşmazlığı nedeniyle.
    test_config_visual_only = {
        'input_dim': 4096, # Sadece görsel girdi boyutu
        'representation_dim': 128
    }
    learner_visual_only = None
    representation_visual = None
    try:
        learner_visual_only = RepresentationLearner(test_config_visual_only) # Yeni learner objesi
        processed_inputs_visual = {'visual': dummy_processed_visual, 'audio': None} # Audio None gönder
        representation_visual = learner_visual_only.learn(processed_inputs_visual)
        if representation_visual is not None:
            print(f"Öğrenilmiş temsil alındı (sadece görsel). Shape: {representation_visual.shape}, Dtype: {representation_visual.dtype}")
        else:
            print("Temsil öğrenilemedi (sadece görsel) (Beklenen: input_dim uyuşmazlığı olabilir).")
    except Exception as e:
         logging.exception("Sadece görsel test sırasında hata oluştu:")
         print("Sadece görsel test sırasında hata oluştu (beklenen).")
    finally:
        if learner_visual_only: del learner_visual_only # Objeyi sil

    # Sadece ses girdi ile dene (VisionProcessor None döndürdüyse)
    print("\nSadece ses girdi ile RepresentationLearner testi:")
    # Bu durumda input_dim 1 olmalıydı, ancak model 4097 bekliyor. Şu an hata verecek.
    test_config_audio_only = {
        'input_dim': 1, # Sadece ses girdi boyutu
        'representation_dim': 128
    }
    learner_audio_only = None
    representation_audio = None
    try:
        learner_audio_only = RepresentationLearner(test_config_audio_only) # Yeni learner objesi
        processed_inputs_audio = {'visual': None, 'audio': dummy_processed_audio} # Visual None gönder
        representation_audio = learner_audio_only.learn(processed_inputs_audio)
        if representation_audio is not None:
            print(f"Öğrenilmiş temsil alındı (sadece ses). Shape: {representation_audio.shape}, Dtype: {representation_audio.dtype}")
        else:
            print("Temsil öğrenilemedi (sadece ses) (Beklenen: input_dim uyuşmazlığı olabilir).")
    except Exception as e:
         logging.exception("Sadece ses testi sırasında hata oluştu:")
         print("Sadece ses testi sırasında hata oluştu (beklenen).")
    finally:
        if learner_audio_only: del learner_audio_only # Objeyi sil


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


    print("\nRepresentationLearner testi bitti.")