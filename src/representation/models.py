# src/representation/models.py
#
# İşlenmiş duyusal veriden içsel temsiller (latent vektörler) öğrenir veya çıkarır.
# Temel sinir ağı katmanlarını implement eder.

import numpy as np # Sayısal işlemler ve arrayler için.
import logging # Loglama için.

# Yardımcı fonksiyonları import et
from src.core.utils import check_input_not_none, check_numpy_input, get_config_value # <<< Yeni importlar


# Bu modül için bir logger oluştur
# 'src.representation.models' adında bir logger döndürür.
logger = logging.getLogger(__name__)

# Basit bir Dense (Tam Bağlantılı) Katman sınıfı
class Dense:
    """
    Tek bir tam bağlantılı (fully connected) katman implementasyonu.

    Ağırlık ve bias parametrelerini içerir.
    ReLU aktivasyon fonksiyonunu destekler.
    İleri geçiş (forward pass) hesaplamasını yapar.
    """
    def __init__(self, input_dim, output_dim, activation='relu'):
        """
        Dense katmanını başlatır.

        Ağırlıkları ve bias'ları rastgele başlatır.

        Args:
            input_dim (int): Girdi özelliğinin boyutu.
            output_dim (int): Çıktı özelliğinin boyutu.
            activation (str, optional): Kullanılacak aktivasyon fonksiyonunun adı ('relu'). Varsayılan 'relu'.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        logger.info(f"Dense katmanı başlatılıyor: Input={input_dim}, Output={output_dim}, Activation={activation}")

        # Ağırlıkları ve bias'ları rastgele başlat.
        # Daha iyi başlangıç yöntemleri (He, Xavier) kullanılabilir (Gelecek TODO).
        # np.random.randn normal dağılımdan örnekler çeker.
        # Çok küçük değerlerle başlamak (0.01 çarpanı) genellikle iyidir.
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros(output_dim) # Bias'ları sıfır olarak başlat.
        logger.info("Dense katmanı başlatıldı.")

    def forward(self, inputs):
        """
        Katmanın ileri geçişini (forward pass) hesaplar.

        Girdi vektörünü veya matrisini ağırlık matrisi ile çarpar, bias ekler ve
        aktivasyon fonksiyonunu uygular.
        Girdi None veya yanlış tip/boyutta ise None döndürür.
        Hesaplama sırasında hata oluşursa None döndürür.

        Args:
            inputs (numpy.ndarray or None): Katmanın girdi verisi.
                                            Beklenen format: shape (..., input_dim), dtype sayısal.
                                            ... kısmı batch boyutu olabilir.

        Returns:
            numpy.ndarray or None: Katmanın çıktı verisi veya hata durumunda None.
                                   shape (..., output_dim), dtype sayısal.
        """
        # Hata yönetimi: Girdi None mu? check_input_not_none kullan.
        if not check_input_not_none(inputs, input_name="dense_inputs", logger_instance=logger):
             return None # Girdi None ise None döndür.

        # Hata yönetimi: Girdinin numpy array ve sayısal bir dtype olup olmadığını kontrol et.
        # expected_ndim=None çünkü girdi tek bir vektör veya bir batch olabilir (ndim >= 1).
        # shape kontrolü ayrıca yapılacak.
        if not check_numpy_input(inputs, expected_dtype=np.number, expected_ndim=None, input_name="dense_inputs", logger_instance=logger):
             return None # Geçersiz tip veya dtype ise None döndür.

        # Hata yönetimi: Girdi boyutunun (son boyutun) input_dim ile eşleşip eşleşmediğini kontrol et.
        # inputs.shape[-1] son boyutu verir.
        if inputs.shape[-1] != self.input_dim:
             logger.error(f"Dense.forward: Beklenmeyen girdi boyutu: {inputs.shape}. Son boyut ({inputs.shape[-1]}) beklenen input_dim ({self.input_dim}) ile eşleşmiyor.")
             return None # Boyut uymuyorsa None döndür.


        # DEBUG logu: Girdi detayları.
        # logger.debug(f"Dense.forward: Girdi alindi. Shape: {inputs.shape}, Dtype: {inputs.dtype}")


        output = None # Çıktıyı tutacak değişken.

        try:
            # Matris çarpımı (girdiler tek bir vektör (input_dim,) veya batch halinde ((Batch, input_dim)))
            # np.dot bu iki durumu da doğru yönetir.
            output = np.dot(inputs, self.weights) + self.bias

            # Aktivasyon fonksiyonu.
            # Şimdilik sadece 'relu' destekleniyor.
            if self.activation == 'relu':
                # ReLU: output = max(0, output)
                output = np.maximum(0, output)
            # Diğer aktivasyon fonksiyonları (sigmoid, tanh vb.) buraya eklenebilir (Gelecek TODO).
            # elif self.activation == 'sigmoid':
            #     output = 1 / (1 + np.exp(-output)) # Dikkat: exp fonksiyonunda over/underflow olabilir.
            # elif self.activation == 'tanh':
            #     output = np.tanh(output)
            # Bilinmeyen aktivasyon adı için uyarı/hata logu eklenebilir.


        except Exception as e:
             # İleri geçiş hesaplamaları sırasında beklenmedik bir hata oluşursa (örn: np.dot, aktivasyon fonksiyonu hatası).
             logger.error(f"Dense.forward: İleri geçiş sırasında beklenmedik hata: {e}", exc_info=True)
             return None # Hata durumunda None döndür.


        # Başarılı durumda hesaplanan çıktıyı döndür.
        # logger.debug(f"Dense.forward: İleri geçiş tamamlandı. Çıktı Shape: {output.shape}, Dtype: {output.dtype}")
        return output

    def cleanup(self):
        """
        Dense katmanı kaynaklarını temizler.

        Şimdilik özel bir kaynak kullanmadığı için temizleme adımı içermez,
        sadece bilgilendirme logu içerir.
        Genellikle numpy arrayleri için explicit temizlik gerekmez.
        """
        # Bilgilendirme logu.
        logger.info(f"Dense katmanı objesi silindi: Input={self.input_dim}, Output={self.output_dim}")
        pass


# Basit bir Representation Learner sınıfı
class RepresentationLearner:
    """
    İşlenmiş duyusal veriden içsel temsiller (latent vektörler) öğrenir veya çıkarır.

    Processing modüllerinden gelen işlenmiş duyu verilerini girdi olarak alır.
    Bu verileri kullanarak daha düşük boyutlu, anlamlı bir temsil vektörü oluşturur.
    Şimdilik çok basit bir tam bağlantılı katman kullanıyor (bir Autoencoder prensibinin encoding kısmı gibi).
    Gelecekte daha karmaşık modeller (CNN, RNN, Transformer temelli) buraya gelecek.
    Modül başlatılamazsa veya temsil öğrenme sırasında hata oluşursa None döndürür.
    """
    def __init__(self, config):
        """
        RepresentationLearner modülünü başlatır.

        Representation modelinin (şimdilik Dense katman) yapısını ayarlar.
        input_dim ve representation_dim yapılandırmadan alınır.

        Args:
            config (dict): RepresentationLearner yapılandırma ayarları.
                           'input_dim': Modelin beklediği girdi boyutu (int, varsayılan 4097,örn 64*64 görsel + 1 ses).
                           'representation_dim': Modelin üreteceği temsil vektörünün boyutu (int, varsayılan 128).
                           Gelecekte farklı model türleri veya katman ayarları buraya gelebilir.
        """
        self.config = config
        logger.info("RepresentationLearner başlatılıyor...")

        # Yapılandırmadan input ve representation boyutlarını alırken get_config_value kullan.
        self.input_dim = get_config_value(config, 'input_dim', 4097, expected_type=int, logger_instance=logger)
        self.representation_dim = get_config_value(config, 'representation_dim', 128, expected_type=int, logger_instance=logger)

        self.layer1 = None # Modelin ilk katmanı (şimdilik tek katman).
        self.is_initialized = False # Modülün başarıyla başlatılıp başlatılmadığını tutar.

        try:
            # Basit bir model iskeleti oluştur (örneğin 1 dense katman).
            # Gelecekte daha karmaşık katmanlar ve mimariler buraya gelecek (CNN, GRU vb.).
            # Bu tek katman şimdilik sadece bir lineer dönüşüm yapıyor, "öğrenme" (ağırlık güncelleme) yok.
            # Gerçek bir öğrenici için fit/train metodu ve optimizer/loss fonksiyonları gerekir (Gelecek TODO).
            self.layer1 = Dense(self.input_dim, self.representation_dim, activation='relu')

            # Başlatma başarılı bayrağı: Eğer layer1 objesi başarıyla oluşturulduysa True yap.
            # Dense katmanı başlatılırken hata olursa None dönebilir (kendi hata yönetimi).
            self.is_initialized = (self.layer1 is not None)

        except Exception as e:
            # Model katmanlarını başlatma sırasında beklenmedik bir hata olursa.
            # Bu hata init sırasında kritik kabul edilebilir (eğer model yoksa Represent iş yapamaz).
            logger.critical(f"RepresentationLearner başlatılırken kritik hata oluştu: {e}", exc_info=True)
            self.is_initialized = False # Hata durumunda başlatılamadı olarak işaretle.
            self.layer1 = None # Hata durumunda katmanı None yap.


        logger.info(f"RepresentationLearner başlatıldı. Girdi boyutu: {self.input_dim}, Temsil boyutu: {self.representation_dim}. Başlatma Başarılı: {self.is_initialized}")


    def learn(self, processed_inputs):
        """
        İşlenmiş duyusal girdiden bir temsil öğrenir veya çıkarır.

        Processing modüllerinden gelen işlenmiş duyusal verileri alır.
        Bu verileri birleştirir ve Representation modelinden (şimdilik Dense katman)
        geçirerek temsil vektörünü hesaplar.
        Modül başlatılmamışsa, girdi boşsa veya işleme/ileri geçiş sırasında
        hata oluşursa None döndürür.

        Args:
            processed_inputs (dict): İşlenmiş duyu verileri sözlüğü.
                                     Örn: {'visual': numpy_array or None, 'audio': float or None}.
                                     Genellikle Process modüllerinden gelir.

        Returns:
            numpy.ndarray or None: Öğrenilmiş temsil vektörü (shape (representation_dim,), dtype sayısal)
                                   veya hata durumunda ya da temsil çıkarılamazsa None.
        """
        # Hata yönetimi: Modül başlatılamamışsa işlem yapma.
        if not self.is_initialized or self.layer1 is None:
            logger.error("RepresentationLearner: Modül başlatılmamış veya model katmanı yok. Temsil öğrenilemiyor.")
            return None # Başlatılamadıysa None döndür.

        # Hata yönetimi: İşlenmiş girdi sözlüğü boşsa işlem yapma.
        # processed_inputs'ın None olması initialize_modules'da yönetiliyor, burada dict olması beklenir.
        if not processed_inputs: # Sözlük boşsa
            logger.debug("RepresentationLearner: İşlenmiş girdi sözlüğü boş. Temsil öğrenilemiyor.")
            return None # Girdi yoksa None döndür.


        input_vector = [] # Birleştirilecek girdi parçalarını tutacak liste.

        try:
            # İşlenmiş duyusal verileri alıp birleştirerek modelin beklediği tek bir girdi vektörü oluştur.
            # Bu kısım, farklı modalitelerden (görsel, işitsel) gelen işlenmiş veriyi
            # nasıl birleştirip model için tek bir girdi formatına getireceğimizi belirler.
            # Şu anki format: görsel (64x64 gri uint8) + işitsel (1 float enerji).
            # Bunları düzleştirip birleştiriyoruz: (64*64 + 1,) float32 vektör.

            # Görsel veriyi işle ve birleştirilecek listeye ekle
            visual_data = processed_inputs.get('visual')
            # check_numpy_input burada kullanılabilir ama Processing modülü zaten uint8 2D/3D numpy array döndürmeyi hedefler.
            # Burada daha çok None olup olmadığını ve temel array tipini kontrol edelim.
            if visual_data is not None and isinstance(visual_data, np.ndarray):
                 # Görsel veriyi düzleştirip (flatten) float'a çevir.
                 # uint8 değerleri 0-255 arasıdır. Float'a çevirirken 0-1 arasına normalizasyon (/ 255.0)
                 # yapmak model eğitimi için iyi bir pratik olabilir (Gelecek TODO).
                 flattened_visual = visual_data.flatten().astype(np.float32)
                 # logger.debug(f"RepresentationLearner: Görsel veri düzleştirildi. Shape: {flattened_visual.shape}")
                 input_vector.append(flattened_visual)
            # else: logger.debug("RepresentationLearner: Görsel girdi None veya geçersiz, birleştirme atlandi.")


            # İşitsel veriyi işle ve birleştirilecek listeye ekle
            audio_data = processed_inputs.get('audio')
            # AudioProcessor şimdilik tek bir float değer döndürüyor (enerji).
            # Gelen verinin sayısal bir değer olup olmadığını kontrol et.
            if audio_data is not None and np.isscalar(audio_data) and np.issubdtype(type(audio_data), np.number):
                # Tek float değeri numpy vektörüne çevir (shape (1,)).
                audio_vector = np.array([audio_data], dtype=np.float32)
                # logger.debug(f"RepresentationLearner: Ses verisi float'tan vektör yapıldı. Shape: {audio_vector.shape}")
                input_vector.append(audio_vector)
            # else: logger.debug("RepresentationLearner: Ses girdisi None veya geçersiz, birleştirme atlandi.")


            # Tüm girdi parçalarını tek bir numpy vektöründe birleştir (concatenate).
            if not input_vector: # Eğer hiçbir geçerli girdi parçası eklenmediyse
                 logger.debug("RepresentationLearner: İşlenmiş girdilerden geçerli veri çıkarılamadı veya hiçbiri mevcut değil. Temsil öğrenilemiyor.")
                 return None # Temsil öğrenilemez, None döndür.

            # np.concatenate, listedeki arrayleri tek bir arrayde birleştirir.
            combined_input = np.concatenate(input_vector)

            # Girdi boyutu kontrolü: Birleştirilmiş girdi boyutu (shape[0]) config'teki input_dim ile eşleşmeli.
            # Bu kontrol Dense katmanı içinde de yapılıyor (forward metodu), ama burada da yapmak
            # temsil öğrenme adımındaki veri hazırlığı sorunlarını anlamayı kolaylaştırır.
            if combined_input.shape[0] != self.input_dim:
                 logger.error(f"RepresentationLearner: Birleştirilmiş girdi boyutu config'teki input_dim ile eşleşmiyor: {combined_input.shape[0]} != {self.input_dim}. Lütfen config dosyasını ve Processing çıktı boyutlarını kontrol edin.")
                 return None # Boyut uymuyorsa temsil öğrenilemez, None döndür.

            # DEBUG logu: Birleştirilmiş girdi detayları.
            # logger.debug(f"RepresentationLearner: Görsel ve ses verisi birleştirildi. Shape: {combined_input.shape}, Dtype: {combined_input.dtype}")


            # Representation modelini çalıştır (ileri geçiş - forward pass).
            # Representation modelimiz şimdilik sadece self.layer1 (Dense katmanı).
            # Dense katmanının forward metodu girdi None ise veya hata olursa None döndürür.
            representation = self.layer1.forward(combined_input)

            # Eğer modelden None döndüyse (model içinde hata olduysa)
            if representation is None:
                 logger.error("RepresentationLearner: Representation modelinden (Dense katmanından) None döndü, ileri geçiş başarısız.")
                 return None # Hata durumunda None döndür.

            # DEBUG logu: Temsil çıktı detayları.
            # if representation is not None: # Zaten None değilse buraya gelinir.
            #     logger.debug(f"RepresentationLearner: Temsil başarıyla öğrenildi. Output Shape: {representation.shape}, Dtype: {representation.dtype}")


            # Başarılı durumda öğrenilen temsil vektörünü döndür.
            return representation

        except Exception as e:
            # Genel hata yönetimi: İşlem (veri hazırlığı, birleştirme, model çalıştırma) sırasında beklenmedik hata olursa logla.
            logger.error(f"RepresentationLearner.learn: Temsil öğrenme sırasında beklenmedik hata: {e}", exc_info=True)
            return None # Hata durumunda None döndür.


    def cleanup(self):
        """
        RepresentationLearner modülü kaynaklarını temizler (model katmanlarını vb.).

        İçindeki model katmanlarının cleanup metotlarını (varsa) çağırır.
        module_loader.py bu metodu program sonlanırken çağırır (varsa).
        """
        logger.info("RepresentationLearner objesi siliniyor...")
        # İçindeki model katmanlarının cleanup metodunu çağır (varsa)
        # Şu an sadece layer1 (Dense) var.
        if hasattr(self.layer1, 'cleanup'):
             self.layer1.cleanup()

        logger.info("RepresentationLearner objesi silindi.")