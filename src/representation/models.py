# src/representation/models.py
import numpy as np
import logging

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

# Basit bir Dense (Tam Bağlantılı) Katman sınıfı
class Dense:
    """
    Tek bir tam bağlantılı katman implementasyonu.
    ReLU aktivasyonunu destekler.
    """
    def __init__(self, input_dim, output_dim, activation='relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        logger.info(f"Dense katmanı başlatılıyor: Input={input_dim}, Output={output_dim}, Activation={activation}")

        # Ağırlıkları ve bias'ları rastgele başlat
        # He başlatma veya Xavier başlatma gibi daha iyi yöntemler kullanılabilir
        # Şimdilik basit random normal dağılım kullanalım
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros(output_dim)
        logger.info("Dense katmanı başlatıldı.")

    def forward(self, inputs):
        """
        Katmanın ileri geçişini hesaplar.
        """
        # Temel hata yönetimi: Girdi tipi veya boyutu kontrolü
        if not isinstance(inputs, np.ndarray):
             logger.error(f"Dense.forward: Beklenmeyen girdi tipi: {type(inputs)}. numpy.ndarray bekleniyordu.")
             return None
        # Input boyutu kontrolü (son boyut input_dim ile eşleşmeli)
        if inputs.shape[-1] != self.input_dim:
             logger.error(f"Dense.forward: Beklenmeyen girdi boyutu: {inputs.shape}. Son boyut {self.input_dim} bekleniyordu.")
             # DEBUG logu olarak girdi shape'ini de basabiliriz
             # logger.debug(f"Dense.forward: Input shape: {inputs.shape}, expected last dim: {self.input_dim}")
             return None


        try:
            # Matris çarpımı (girdiler batch halinde olabilir, (Batch, input_dim) veya (input_dim,) )
            output = np.dot(inputs, self.weights) + self.bias

            # Aktivasyon fonksiyonu
            if self.activation == 'relu':
                output = np.maximum(0, output) # ReLU: max(0, x)
            # Diğer aktivasyon fonksiyonları buraya eklenebilir (sigmoid, tanh vb.)
            # elif self.activation == 'sigmoid':
            #     output = 1 / (1 + np.exp(-output)) # Dikkat: exp over/underflow olabilir
            # elif self.activation == 'tanh':
            #     output = np.tanh(output)

        except Exception as e:
             # İleri geçiş sırasında beklenmedik hata (örn: np.dot hatası)
             logger.error(f"Dense.forward: İleri geçiş sırasında beklenmedik hata: {e}", exc_info=True)
             return None # Hata durumunda None döndür


        return output

    def cleanup(self):
        """Kaynakları temizler (şimdilik gerek yok, placeholder)."""
        logger.info(f"Dense katmanı objesi silindi: Input={self.input_dim}, Output={self.output_dim}")
        pass # Genellikle numpy arrayleri için explicit temizlik gerekmez


# Basit bir Representation Learner sınıfı
class RepresentationLearner:
    """
    İşlenmiş duyusal veriden içsel temsiller (latent vektörler) öğrenir.
    Şimdilik çok basit bir tam bağlantılı katman kullanıyor (Autoencoder prensibinin encoding kısmı gibi).
    Gelecekte daha karmaşık modeller (CNN, RNN, Transformer temelli) buraya gelecek.
    """
    def __init__(self, config):
        self.config = config
        self.input_dim = config.get('input_dim', 4097) # Görsel (64*64) + Ses (1) varsayılıyor
        self.representation_dim = config.get('representation_dim', 128)

        logger.info("RepresentationLearner başlatılıyor...")

        # Basit bir model iskeleti (örneğin 1 dense katman)
        # Gelecekte daha karmaşık katmanlar ve mimariler buraya gelecek
        # Bu tek katman şimdilik sadece bir dönüşüm yapıyor, "öğrenme" (ağırlık güncelleme) yok.
        # Gerçek bir öğrenici için fit/train metodu ve optimizer/loss fonksiyonları gerekir.
        self.layer1 = Dense(self.input_dim, self.representation_dim, activation='relu')

        # Başlatma başarılı bayrağı (layer1'in başarıyla başlatıldığını kontrol et)
        self.is_initialized = (self.layer1 is not None)

        logger.info(f"RepresentationLearner başlatıldı. Girdi boyutu: {self.input_dim}, Temsil boyutu: {self.representation_dim}. Başlatma Başarılı: {self.is_initialized}")


    def learn(self, processed_inputs):
        """
        İşlenmiş duyusal girdiden bir temsil öğrenir/çıkarır.
        Şimdilik sadece girdiyi bir dense katmandan geçirir.

        Args:
            processed_inputs (dict): İşlenmiş duyu verileri (örn: {'visual': ..., 'audio': ...})

        Returns:
            numpy.ndarray or None: Öğrenilmiş temsil vektörü veya hata durumunda None.
        """
        # Temel hata yönetimi: Modül başlatılamadıysa veya girdi boşsa
        if not self.is_initialized:
            logger.error("RepresentationLearner: Modül başlatılmamış. Temsil öğrenilemiyor.")
            return None
        if not processed_inputs:
            logger.debug("RepresentationLearner: İşlenmiş girdi boş. Temsil öğrenilemiyor.")
            return None

        # Girdiyi model için uygun formata getir (şimdilik görseli düzleştir + ses enerjisini ekle)
        # Bu kısım Processing katmanı daha karmaşık özellikler çıkardıkça değişecek.
        # Current: visual (H,W) uint8 -> flatten (H*W,) float32
        #          audio (float) -> (1,) float32
        #          birleştir (H*W + 1,) float32
        input_vector = []
        try:
            # Görsel veriyi işle
            visual_data = processed_inputs.get('visual')
            if visual_data is not None and isinstance(visual_data, np.ndarray):
                 # Görsel veriyi düzleştir ve float'a çevir
                 # uint8'den float'a çevirirken normalize etmek iyi bir fikir olabilir (örn: / 255.0)
                 # Şimdilik sadece float'a çevirelim
                 flattened_visual = visual_data.flatten().astype(np.float32)
                 # logger.debug(f"RepresentationLearner: Görsel veri düzleştirildi. Shape: {flattened_visual.shape}")
                 input_vector.append(flattened_visual)
            # else: logger.debug("RepresentationLearner: Görsel girdi None veya geçersiz.")


            # İşitsel veriyi işle
            audio_data = processed_inputs.get('audio')
            # AudioProcessor şimdilik tek bir float değer döndürüyor (enerji)
            if audio_data is not None and isinstance(audio_data, (int, float, np.number)): # Sayısal tip kontrolü
                # Tek float değeri numpy vektörüne çevir
                audio_vector = np.array([audio_data], dtype=np.float32)
                # logger.debug(f"RepresentationLearner: Ses verisi float'tan vektör yapıldı. Shape: {audio_vector.shape}")
                input_vector.append(audio_vector)
            # else: logger.debug("RepresentationLearner: Ses girdisi None veya geçersiz.")


            # Tüm girdi vektörlerini birleştir
            if not input_vector: # Hiç geçerli girdi yoksa
                 logger.debug("RepresentationLearner: İşlenmiş girdilerden geçerli veri çıkarılamadı. Temsil öğrenilemiyor.")
                 return None

            combined_input = np.concatenate(input_vector)

            # Girdi boyutu kontrolü (beklenen input_dim ile eşleşmeli)
            # Bu kontrol Dense katmanı içinde de yapılıyor ama burada da yapmak hata kaynağını anlamayı kolaylaştırır.
            if combined_input.shape[0] != self.input_dim:
                 logger.error(f"RepresentationLearner: Birleştirilmiş girdi boyutu config'teki input_dim ile eşleşmiyor: {combined_input.shape[0]} != {self.input_dim}. Config'i kontrol edin.")
                 return None

            # DEBUG logu: Birleştirilmiş girdi detayları
            # logger.debug(f"RepresentationLearner: Görsel ve ses verisi birleştirildi. Shape: {combined_input.shape}, Dtype: {combined_input.dtype}")


            # Modeli çalıştır (ileri geçiş)
            representation = self.layer1.forward(combined_input)

            if representation is None: # Dense katmanı hata döndürdüyse
                 logger.error("RepresentationLearner: Dense katmanindan None döndü, ileri geçiş başarısız.")
                 return None

            # DEBUG logu: Temsil çıktı detayları
            # logger.debug(f"RepresentationLearner: Temsil başarıyla öğrenildi. Output Shape: {representation.shape}, Dtype: {representation.dtype}")


            return representation

        except Exception as e:
            # Genel hata yönetimi: Öğrenme (ileri geçiş) sırasında hata
            logger.error(f"RepresentationLearner.learn: Temsil öğrenme sırasında beklenmedik hata: {e}", exc_info=True)
            return None # Hata durumunda None döndür


    def cleanup(self):
        """Kaynakları temizler (model katmanlarını vb.)."""
        logger.info("RepresentationLearner objesi siliniyor...")
        # İçindeki katmanların cleanup metodunu çağır (varsa)
        if hasattr(self.layer1, 'cleanup'):
             self.layer1.cleanup()
        logger.info("RepresentationLearner objesi silindi.")