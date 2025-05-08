# src/representation/models.py
#
# İşlenmiş duyusal veriden içsel temsiller (latent vektörler) öğrenir veya çıkarır.
# Temel sinir ağı katmanlarını implement eder.
# Evo'nın Faz 1'deki temsil öğrenme yeteneklerinin bir parçasıdır.

import numpy as np # Sayısal işlemler ve arrayler için.
import logging # Loglama için.

# Yardımcı fonksiyonları import et
from src.core.config_utils import get_config_value
from src.core.utils import check_input_not_none, check_numpy_input # <<< Utils importları


# Bu modül için bir logger oluştur
# 'src.representation.models' adında bir logger döndürür.
logger = logging.getLogger(__name__)

# Basit bir Dense (Tam Bağlantılı) Katman sınıfı
# TODO: İleride src/core/nn_components.py'deki versiyonu merkezi olarak kullanmak daha temiz olacaktır.
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
        # Logger, RepresentationLearner tarafından zaten başlatılmış olmalı, burada tekrar getirmek yerine
        # doğrudan module-level logger'ı kullanabiliriz.
        # Logging burada RepresentationLearner'ın logger'ını kullanmak yerine kendi logger'ını kullanıyor gibi görünüyor.
        # RepresentationLearner init'te logger'ı pass etmiyor. Refactoring hedefi.
        logger.info(f"Dense katmanı başlatılıyor: Input={input_dim}, Output={output_dim}, Activation={activation}")

        # Ağırlıkları ve bias'ları rastgele başlat.
        # Daha iyi başlangıç yöntemleri (He, Xavier) kullanılabilir (Gelecek TODO).
        # np.random.randn normal dağılımdan örnekler çeker.
        # Çok küçük değerlerle başlamak (0.01 çarpanı) genellikle iyidir.
        # Eğer PyTorch kullanılıyorsa, ağırlıkların PyTorch tensörleri olması ve GPU'ya taşınması gerekebilir.
        # Şimdilik NumPy ile CPU üzerinde kalıyoruz.
        limit = np.sqrt(1. / input_dim) # Basit başlatma ölçeği (fan-in)
        self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.bias = np.zeros(output_dim) # Bias'ı genellikle sıfır başlatmak yaygındır.

        logger.info("Dense katmanı başlatıldı.")

    # ... (forward and cleanup methods - same as before) ...

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
             logger.debug("Dense.forward: Girdi None. None döndürülüyor.") # None girdisi hata değil, bilgilendirme.
             return None # Girdi None ise None döndür.

        # Hata yönetimi: Girdinin numpy array ve sayısal bir dtype olup olmadığını kontrol et.
        # expected_ndim=None çünkü girdi tek bir vektör veya bir batch olabilir (ndim >= 1).
        # shape kontrolü ayrıca yapılacak.
        if not check_numpy_input(inputs, expected_dtype=np.number, expected_ndim=None, input_name="dense_inputs", logger_instance=logger):
             logger.error("Dense.forward: Girdi numpy array değil veya sayısal değil. None döndürülüyor.") # check_numpy_input zaten kendi içinde loglar.
             return None # Geçersiz tip veya dtype ise None döndür.

        # Hata yönetimi: Girdi boyutunun (son boyutun) input_dim ile eşleşip eşleşmediğini kontrol et.
        # inputs.shape[-1] son boyutu verir.
        if inputs.shape[-1] != self.input_dim:
             logger.error(f"Dense.forward: Beklenmeyen girdi boyutu: {inputs.shape}. Son boyut ({inputs.shape[-1]}) beklenen input_dim ({self.input_dim}) ile eşleşmiyor. None döndürülüyor.")
             return None # Boyut uymuyorsa None döndür.


        # DEBUG logu: Girdi detayları.
        logger.debug(f"Dense.forward: Girdi alindi. Shape: {inputs.shape}, Dtype: {inputs.dtype}. Hesaplama yapılıyor.")


        output = None # Çıktıyı tutacak değişken.

        try:
            # Matris çarpımı (girdiler tek bir vektör (input_dim,) veya batch halinde ((Batch, input_dim)))
            # np.dot bu iki durumu da doğru yönetir.
            linear_output = np.dot(inputs, self.weights) + self.bias

            # Aktivasyon fonksiyonunu uygula (varsa)
            if self.activation == 'relu':
                # ReLU: output = max(0, output)
                output = np.maximum(0, linear_output) # ReLU aktivasyonu
            # TODO: Diğer aktivasyon fonksiyonları eklenecek (sigmoid, tanh vb.)
            # elif self.activation == 'sigmoid':
            #      output = 1 / (1 + np.exp(-linear_output)) # Dikkat: exp fonksiyonunda over/underflow olabilir.
            # elif self.activation == 'tanh':
            #      output = np.tanh(linear_output)
            # Bilinmeyen aktivasyon adı için uyarı/hata logu eklenebilir.
            elif self.activation is None: # None aktivasyon
                 output = linear_output
            else:
                 logger.warning(f"Dense.forward: Bilinmeyen aktivasyon fonksiyonu: '{self.activation}'. Lineer aktivasyon kullanılıyor.")
                 output = linear_output


        except Exception as e:
             # İleri geçiş hesaplamaları sırasında beklenmedik bir hata oluşursa (örn: np.dot, aktivasyon fonksiyonu hatası).
             logger.error(f"Dense.forward: İleri geçiş sırasında beklenmedik hata: {e}", exc_info=True)
             return None # Hata durumunda None döndür.


        # Başarılı durumda hesaplanan çıktıyı döndür.
        logger.debug(f"Dense.forward: İleri geçiş tamamlandı. Çıktı Shape: {output.shape}, Dtype: {output.dtype}.")
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

    Processing modüllerinden gelen işlenmiş duyu verilerini (sözlük formatında) alır.
    Bu verileri birleştirerek birleşik bir girdi vektörü oluşturur.
    Girdi vektörünü bir Encoder katmanından (Dense) geçirerek düşük boyutlu,
    anlamlı bir temsil vektörü (latent) oluşturur.
    Bir Decoder katmanı (Dense) da içerir (Autoencoder prensibi için),
    bu katman latent vektörden orijinal girdi boyutunda bir rekonstrüksiyon üretir.
    Gelecekte daha karmaşık modeller (CNN, RNN, Transformer temelli) buraya gelecek.
    Modül başlatılamazsa veya temsil öğrenme/çıkarma sırasında hata oluşursa None döndürür.
    """
    def __init__(self, config):
        """
        RepresentationLearner modülünü başlatır.

        Representation modelinin (şimdilik Encoder ve Decoder Dense katmanları) yapısını ayarlar.
        input_dim ve representation_dim yapılandırmadan alınır.

        Args:
            config (dict): RepresentationLearner yapılandırma ayarları.
                           'input_dim': Encoder'ın beklediği toplam girdi boyutu (int, varsayılan 8194).
                                         Bu boyut, learn metodunda birleştirilen tüm özelliklerin toplam boyutuna eşit olmalıdır.
                                         Örn: (64x64 gri görsel) + (64x64 kenar haritası) + (enerji, centroid) = 4096 + 4096 + 2 = 8194.
                           'representation_dim': Encoder'ın üreteceği ve Decoder'ın alacağı temsil vektörünün boyutu (int, varsayılan 128).
                           Gelecekte farklı model türleri veya katman ayarları buraya gelebilir.
        """
        self.config = config
        logger.info("RepresentationLearner başlatılıyor...")

        # Yapılandırmadan input ve representation boyutlarını alırken get_config_value kullan.
        # Düzeltme: get_config_value çağrılarını default=keyword formatına çevir.
        # Config'e göre bu ayarlar 'representation' anahtarı altında.
        self.input_dim = get_config_value(config, 'representation', 'input_dim', default=8194, expected_type=int, logger_instance=logger)
        self.representation_dim = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)

        self.encoder = None # Encoder katmanı (şimdilik Dense).
        self.decoder = None # Decoder katmanı (şimdilik Dense).
        self.is_initialized = False # Modülün başarıyla başlatılıp başlatılmadığını tutar.

        # TODO: Gelecekte: Farklı modalitelerden gelen girdilerin boyutlarını buradan config'ten alıp,
        # TODO: input_dim değerinin bu boyutların toplamına eşit olduğunu doğrulayabiliriz.
        # Processor çıktı boyutları RepresentationLearner için config'ten alınıyor, bu doğru.
        # AudioProcessor çıktı boyutu da burada alınıp kullanılabilir.
        visual_config = config.get('processors', {}).get('vision', {})
        audio_config = config.get('processors', {}).get('audio', {})
        visual_gray_size = visual_config.get('output_width', 64) * visual_config.get('output_height', 64)
        visual_edges_size = visual_config.get('output_width', 64) * visual_config.get('output_height', 64) # Genellikle aynı boyut
        audio_features_dim = audio_config.get('output_dim', 2)
        expected_input_dim_calc = visual_gray_size + visual_edges_size + audio_features_dim

        if self.input_dim != expected_input_dim_calc:
             logger.warning(f"RepresentationLearner: Config'teki input_dim ({self.input_dim}) beklenen hesaplanmış değer ({expected_input_dim_calc}) ile eşleşmiyor. Lütfen config dosyasını kontrol edin. Hesaplanan boyut Processing çıktı boyutlarına göre belirlenir.")
             # İsteğe bağlı: Bu durumda self.input_dim'i hesaplanan değere set edilebilir
             # self.input_dim = expected_input_dim_calc


        try:
            # Encoder katmanı oluştur (Girdi boyutu: self.input_dim, Çıktı boyutu: self.representation_dim).
            # Genellikle encoder'ın son katmanında aktivasyon olmaz veya linear olur (latent uzay).
            # Dense sınıfına None aktivasyon seçeneği ekledim. Burada None aktivasyon kullanalım.
            self.encoder = Dense(self.input_dim, self.representation_dim, activation=None) # Latent uzayda aktivasyon yok

            # Decoder katmanı oluştur (Girdi boyutu: self.representation_dim, Çıktı boyutu: self.input_dim).
            # Decoder çıktısı orijinal girdi boyutunda bir rekonstrüksiyon olmalı.
            # Decoder'ın son katmanında çıktının temsil ettiği veri türüne uygun aktivasyon olabilir (örn: görsel için sigmoid/tanh, sayısal için linear).
            # Şimdilik yine None aktivasyon kullanalım.
            self.decoder = Dense(self.representation_dim, self.input_dim, activation=None) # Rekonstruksiyon çıktısı


            # Başlatma başarılı bayrağı: Encoder ve Decoder objeleri başarıyla oluşturulduysa True yap.
            self.is_initialized = (self.encoder is not None and self.decoder is not None)

        except Exception as e:
            # Model katmanlarını başlatma sırasında beklenmedik bir hata olursa.
            # Bu hata init sırasında kritik kabul edilebilir (eğer model yoksa Represent iş yapamaz).
            logger.critical(f"RepresentationLearner başlatılırken kritik hata oluştu: {e}", exc_info=True)
            self.is_initialized = False # Hata durumunda başlatılamadı olarak işaretle.
            self.encoder = None
            self.decoder = None


        logger.info(f"RepresentationLearner başlatıldı. Girdi boyutu: {self.input_dim}, Temsil boyutu: {self.representation_dim}. Başlatma Başarılı: {self.is_initialized}")


    def learn(self, processed_inputs):
        """
        İşlenmiş duyusal girdiden bir temsil (latent vektör) çıkarır (Encoder forward pass).

        Processing modüllerinden gelen işlenmiş duyusal verileri (sözlük formatında) alır.
        Bu verileri birleştirerek birleşik bir girdi vektörü oluşturur.
        Girdi vektörünü Encoder katmanından geçirerek temsil vektörünü hesaplar.
        Modül başlatılmamışsa, girdi boşsa veya işleme/ileri geçiş sırasında
        hata oluşursa None döndürür.

        Args:
            processed_inputs (dict): İşlenmiş duyu verileri sözlüğü.
                                     Örn: {'visual': dict or None, 'audio': numpy.ndarray or None}.
                                     Genellikle Process modüllerinden gelir.

        Returns:
            numpy.ndarray or None: Öğrenilmiş temsil vektörü (shape (representation_dim,), dtype sayısal)
                                   veya hata durumunda ya da temsil çıkarılamazsa None.
        """
        # Hata yönetimi: Modül başlatılamamışsa işlem yapma.
        if not self.is_initialized or self.encoder is None: # Decoder yoksa bile Encoder çalışabilir policy'si olabilir.
             logger.error("RepresentationLearner.learn: Modül başlatılmamış veya Encoder katmanı yok. Temsil öğrenilemiyor.")
             return None # Başlatılamadıysa None döndür.

        # Hata yönetimi: İşlenmiş girdi sözlüğü None veya boşsa işlem yapma.
        # check_input_not_none fonksiyonunu processed_inputs sözlüğü için kullanalım.
        if not check_input_not_none(processed_inputs, input_name="processed_inputs for RepresentationLearner", logger_instance=logger):
             logger.debug("RepresentationLearner.learn: İşlenmiş girdi sözlüğü None. Temsil öğrenilemiyor.")
             return None # Girdi None ise None döndür.

        if not processed_inputs: # Sözlük boşsa
            logger.debug("RepresentationLearner.learn: İşlenmiş girdi sözlüğü boş. Temsil öğrenilemiyor.")
            return None # Girdi yoksa None döndür.


        input_vector_parts = [] # Birleştirilecek girdi parçalarını tutacak liste.

        try:
            # İşlenmiş duyusal verileri alıp birleştirerek modelin beklediği tek bir girdi vektörü oluştur.
            # Bu kısım, farklı modalitelerden (görsel, işitsel) gelen işlenmiş veriyi
            # nasıl birleştirip model için tek bir girdi formatına getireceğimizi belirler.
            # Beklenen format:
            # visual: VisionProcessor'dan gelen dict {'grayscale': np.ndarray (64x64 uint8), 'edges': np.ndarray (64x64 uint8)}
            # audio: AudioProcessor'dan gelen np.ndarray (shape (output_dim,), dtype float32) - [energy, spectral_centroid]
            # Amacımız RepresentationLearner'ın input_dim (8194) ile eşleşen bir vektör oluşturmak.
            # 8194 = (64*64 grayscale) + (64*64 edges) + (AudioProcessor output_dim)

            # Görsel veriyi işle ve birleştirilecek listeye ekle
            visual_processed = processed_inputs.get('visual')
            # İşlenmiş görsel verinin bir sözlük olup olmadığını kontrol et.
            if isinstance(visual_processed, dict):
                 # Sözlüğün içindeki 'grayscale' ve 'edges' anahtarlarını kontrol et ve değerlerinin numpy array olduğunu doğrula.
                 grayscale_data = visual_processed.get('grayscale')
                 edges_data = visual_processed.get('edges')

                 # Grayscale veriyi işle
                 # check_numpy_input artık expected_shape_0 argümanını kabul etmiyor.
                 if grayscale_data is not None and check_numpy_input(grayscale_data, expected_dtype=np.uint8, expected_ndim=2, input_name="processed_inputs['visual']['grayscale']", logger_instance=logger):
                      # Görsel veriyi düzleştirip (flatten) float32'ye çevir. Normalizasyon isteğe bağlı (Gelecek TODO).
                      flattened_grayscale = grayscale_data.flatten().astype(np.float32)
                      # TODO: Normalize flattened_grayscale to 0-1 range? flattened_grayscale = flattened_grayscale / 255.0
                      input_vector_parts.append(flattened_grayscale)
                      logger.debug(f"RepresentationLearner.learn: Görsel veri (grayscale) düzleştirildi. Shape: {flattened_grayscale.shape}")
                 else:
                      logger.warning("RepresentationLearner.learn: İşlenmiş görsel input sözlüğünde 'grayscale' geçerli numpy array değil veya boyutu yanlış. Yoksayılıyor.")

                 # Edges veriyi işle
                 # check_numpy_input artık expected_shape_0 argümanını kabul etmiyor.
                 if edges_data is not None and check_numpy_input(edges_data, expected_dtype=np.uint8, expected_ndim=2, input_name="processed_inputs['visual']['edges']", logger_instance=logger):
                      # Kenar haritasını düzleştirip (flatten) float32'ye çevir. 0-255 değerleri 0 veya 255'tir.
                      flattened_edges = edges_data.flatten().astype(np.float32)
                      # TODO: Normalize flattened_edges? flattened_edges = flattened_edges / 255.0 # Veya sadece 0/1 ikili değerler olarak mı kalsın?
                      input_vector_parts.append(flattened_edges)
                      logger.debug(f"RepresentationLearner.learn: Görsel veri (edges) düzleştirildi. Shape: {flattened_edges.shape}")
                 else:
                      logger.warning("RepresentationLearner.learn: İşlenmiş görsel input sözlüğünde 'edges' geçerli numpy array değil veya boyutu yanlış. Yoksayılıyor.")

            # elif visual_processed is not None: # Dict değil ama None da değilse
            #      logger.warning(f"RepresentationLearner.learn: İşlenmiş görsel input beklenmeyen tipte ({type(visual_processed)}). dict bekleniyordu. Yoksayılıyor.")
            # else: # visual_processed is None
            #      logger.debug("RepresentationLearner.learn: İşlenmiş görsel input None. Yoksayılıyor.")


            # İşitsel veriyi işle ve birleştirilecek listeye ekle
            audio_processed = processed_inputs.get('audio')
            # AudioProcessor artık bir numpy array (shape (output_dim,), dtype float32) veya None döndürmeli.
            # Gelen verinin numpy array ve doğru boyutta/dtype olduğunu kontrol et.
            # Beklenen dtype np.number (veya np.float32). Beklenen ndim 1. Beklenen shape[0] = AudioProcessor config'teki output_dim (2).
            # check_numpy_input artık expected_shape_0 argümanını kabul etmiyor. Shape kontrolünü manuel yapalım.
            expected_audio_dim = self.config.get('processors', {}).get('audio', {}).get('output_dim', 2) # Config'ten beklenen boyutu al.
            if audio_processed is not None and check_numpy_input(audio_processed, expected_dtype=np.number, expected_ndim=1, input_name="processed_inputs['audio']", logger_instance=logger):
                 # check_numpy_input temel array/dtype/ndim kontrolünü yaptı. Şimdi spesifik shape[0] kontrolünü yapalım.
                 if audio_processed.shape[0] == expected_audio_dim:
                      # Ses özellik array'ini doğrudan ekle (zaten 1D array). Dtype float32 olmalıydı AudioProcessor'da.
                      audio_features_array = audio_processed.astype(np.float32) # Ensure float32
                      # TODO: Normalize audio_features_array?
                      input_vector_parts.append(audio_features_array)
                      logger.debug(f"RepresentationLearner.learn: Ses verisi array'i eklendi. Shape: {audio_features_array.shape}")
                 else:
                      # check_numpy_input geçerli dese bile, beklenen ilk boyut uymuyor.
                      logger.warning(f"RepresentationLearner.learn: İşlenmiş ses input beklenmeyen boyutta. Beklenen shape ({expected_audio_dim},), gelen shape {audio_processed.shape}. Yoksayılıyor.")


            # elif audio_processed is not None: # Geçersiz format (numpy array değil veya yanlış dtype)
            #      logger.warning(f"RepresentationLearner.learn: İşlenmiş ses input beklenmeyen formatta ({type(audio_processed)}, shape {getattr(audio_processed, 'shape', 'N/A')}). 1D numpy array bekleniyordu. Yoksayılıyor.")
            # else: # audio_processed is None
            #      logger.debug("RepresentationLearner.learn: İşlenmiş ses input None. Yoksayılıyor.")


            # Tüm geçerli girdi parçalarını tek bir numpy vektöründe birleştir (concatenate).
            # Eğer grayscale, edges ve audio'dan en az biri geçerliyse devam et.
            if not input_vector_parts: # Eğer hiçbir geçerli girdi parçası (grayscale, edges, audio_features) eklenmediyse
                 logger.debug("RepresentationLearner.learn: İşlenmiş girdilerden (grayscale, edges, audio) geçerli veri çıkarılamadı veya hiçbiri mevcut değil. Temsil öğrenilemiyor.")
                 return None # Temsil öğrenilemez, None döndür.

            # np.concatenate, listedeki arrayleri tek bir arrayde birleştirir. axis=0 default olarak ilk boyutta birleştirir.
            combined_input = np.concatenate(input_vector_parts, axis=0)

            # Girdi boyutu kontrolü: Birleştirilmiş girdi boyutu (shape[0]) config'teki input_dim ile eşleşmeli.
            # Bu kontrol Dense katmanı içinde de yapılıyor (forward metodu), ama burada da yapmak
            # temsil öğrenme adımındaki veri hazırlığı sorunlarını anlamayı kolaylaştırır.
            # input_dim'i config'ten alıyoruz (default 8194). Birleştirilen verinin boyutu da bu olmalı.
            if combined_input.shape[0] != self.input_dim:
                 # Hata mesajını daha açıklayıcı yapalım.
                 logger.error(f"RepresentationLearner.learn: Birleştirilmiş girdi boyutu ({combined_input.shape[0]}) config'teki input_dim ({self.input_dim}) ile eşleşmiyor. Lütfen config dosyasındaki 'representation' input_dim ayarını ve Processing modüllerinin çıktı format/boyutlarını (şu an VisionProcessor'dan 'grayscale' 64x64, 'edges' 64x64 ve AudioProcessor'dan 2 özellik bekleniyor) kontrol edin.")
                 # Bu noktada boyut uyuşmazlığı ciddi bir konfigürasyon/veri akışı hatasıdır. None döndürüyoruz.
                 return None # Boyut uymuyorsa temsil öğrenilemez, None döndür.

            # DEBUG logu: Birleştirilmiş girdi detayları.
            logger.debug(f"RepresentationLearner.learn: Görsel ve ses verisi birleştirildi. Shape: {combined_input.shape}, Dtype: {combined_input.dtype}.")


            # Representation modelini çalıştır (encoder ileri geçiş - forward pass).
            # Representation modelimizin encoder'ı self.encoder (Dense katmanı).
            # Encoder katmanının forward metodu girdi None ise veya hata olursa None döndürür.
            representation = self.encoder.forward(combined_input)

            # Eğer modelden None döndüyse (encoder içinde hata olduysa)
            if representation is None:
                 logger.error("RepresentationLearner.learn: Encoder modelinden None döndü, ileri geçiş başarısız. Temsil öğrenilemedi.")
                 return None # Hata durumında None döndür.

            # DEBUG logu: Temsil çıktı detayları.
            # if representation is not None: # Zaten None değilse buraya gelinir.
            logger.debug(f"RepresentationLearner.learn: Temsil başarıyla öğrenildi. Output Shape: {representation.shape}, Dtype: {representation.dtype}.")


            # Başarılı durumda öğrenilen temsil vektörünü döndür.
            return representation

        except Exception as e:
            # Genel hata yönetimi: İşlem (veri hazırlığı, birleştirme, model çalıştırma) sırasında beklenmedik hata olursa logla.
            logger.error(f"RepresentationLearner.learn: Temsil öğrenme sırasında beklenmedik hata: {e}", exc_info=True)
            return None # Hata durumunda None döndür.


    def decode(self, latent_vector):
        """
        Bir latent vektörden orijinal girdi boyutunda bir rekonstrüksiyon üretir (Decoder forward pass).

        Bu metot, Autoencoder prensibinin decoder kısmını simüle eder.
        Run_evo.py'nin ana döngüsünde şu an kullanılmayacak.

        Args:
            latent_vector (numpy.ndarray or None): Decoder girdisi olan latent vektör (representation_dim boyutunda)
                                                  veya None.

        Returns:
            numpy.ndarray or None: Rekonstrukksiyon çıktısı (shape (input_dim,), dtype sayısal)
                                   veya hata durumunda ya da girdi None ise None.
        """
        # Hata yönetimi: Modül başlatılamamışsa veya Decoder yoksa işlem yapma.
        if not self.is_initialized or self.decoder is None:
             logger.error("RepresentationLearner.decode: Modül başlatılmamış veya Decoder katmanı yok. Rekonstrüksiyon yapılamıyor.")
             return None

        # Hata yönetimi: Latent vektör None mu? check_input_not_none kullan.
        if not check_input_not_none(latent_vector, input_name="latent_vector for RepresentationLearner.decode", logger_instance=logger):
             logger.debug("RepresentationLearner.decode: Girdi latent_vector None. None döndürülüyor.")
             return None

        # Hata yönetimi: Latent vektör numpy array ve doğru boyutta/dtype mı?
        # representation_dim boyutunda 1D numpy array beklenir.
        if not check_numpy_input(latent_vector, expected_dtype=np.number, expected_ndim=1, input_name="latent_vector for RepresentationLearner.decode", logger_instance=logger):
             # check_numpy_input temel kontrolü yaptı. Şimdi spesifik shape[0] kontrolünü yapalım.
             if not (isinstance(latent_vector, np.ndarray) and latent_vector.shape[0] == self.representation_dim):
                   logger.error(f"RepresentationLearner.decode: Girdi latent_vector yanlış formatta. Beklenen shape ({self.representation_dim},), ndim 1, dtype sayısal. Gelen shape {getattr(latent_vector, 'shape', 'N/A')}, ndim {getattr(latent_vector, 'ndim', 'N/A')}, dtype {getattr(latent_vector, 'dtype', 'N/A')}. None döndürülüyor.")
                   return None
             # check_numpy_input zaten kendi içinde hata logladıysa, burada sadece None dönelim.
             return None


        logger.debug(f"RepresentationLearner.decode: Latent vektör alindi. Shape: {latent_vector.shape}, Dtype: {latent_vector.dtype}. Rekonstruksiyon yapılıyor.")

        reconstruction = None

        try:
            # Decoder katmanını çalıştır (ileri geçiş - forward pass).
            # Decoder katmanının forward metodu girdi None ise veya hata olursa None döndürür.
            reconstruction = self.decoder.forward(latent_vector)

            # Eğer modelden None döndüyse (decoder içinde hata olduysa)
            if reconstruction is None:
                 logger.error("RepresentationLearner.decode: Decoder modelinden None döndü, ileri geçiş başarısız. Rekonstrüksiyon yapılamadı.")
                 return None # Hata durumında None döndür.

            # DEBUG logu: Rekonstrüksiyon çıktı detayları.
            logger.debug(f"RepresentationLearner.decode: Rekonstrukksiyon tamamlandı. Output Shape: {reconstruction.shape}, Dtype: {reconstruction.dtype}.")


            # Başarılı durumda rekonstrüksiyon çıktısını döndür.
            return reconstruction

        except Exception as e:
             # Genel hata yönetimi: İşlem sırasında beklenmedik hata olursa logla.
             logger.error(f"RepresentationLearner.decode: Rekonstrukksiyon sırasında beklenmedik hata: {e}", exc_info=True)
             return None # Hata durumunda None döndür.


    def cleanup(self):
        """
        RepresentationLearner modülü kaynaklarını temizler (model katmanlarını vb.).

        İçindeki model katmanlarının cleanup metotlarını (varsa) çağırır.
        module_loader.py bu metotu program sonlanırken çağırır (varsa).
        """
        logger.info("RepresentationLearner objesi siliniyor...")
        # Encoder ve Decoder katmanlarının cleanup metotlarını çağır (varsa)
        if hasattr(self.encoder, 'cleanup'):
             self.encoder.cleanup()
        if hasattr(self.decoder, 'cleanup'):
             self.decoder.cleanup()

        logger.info("RepresentationLearner objesi silindi.")

# Modülü bağımsız test etmek için örnek kullanım (isteğe bağlı, geliştirme sürecinde faydalı olabilir)
if __name__ == '__main__':
    # Test için temel loglama yapılandırması
    import sys
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("RepresentationLearner test ediliyor...")

    # Sahte bir config sözlüğü oluştur
    test_config = {
        'input_dim': 8194, # 64*64 (grayscale) + 64*64 (edges) + 2 (audio features)
        'representation_dim': 128,
        'processors': { # Processor çıktı boyutlarını taklit etmek için config'e ekleyelim (gerçekte Representation bunları doğrudan kullanmamalı, ama test için belirtebiliriz)
             'vision': {'output_width': 64, 'output_height': 64},
             'audio': {'output_dim': 2}
        }
    }

    # RepresentationLearner objesini başlat
    try:
        learner = RepresentationLearner(test_config)
        print("\nRepresentationLearner objesi başarıyla başlatıldı.")
        print(f"Beklenen girdi boyutu (input_dim): {learner.input_dim}")
        print(f"Beklenen temsil boyutu (representation_dim): {learner.representation_dim}")


        # Sahte işlenmiş girdi sözlüğü oluştur (VisionProcessor ve AudioProcessor çıktılarını taklit ederek)
        # VisionProcessor çıktısı: {'grayscale': 64x64 uint8 array, 'edges': 64x64 uint8 array}
        dummy_processed_visual_gray = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
        dummy_processed_visual_edges = np.random.randint(0, 2, size=(64, 64), dtype=np.uint8) * 255 # 0 veya 255
        dummy_processed_visual_dict = {
            'grayscale': dummy_processed_visual_gray,
            'edges': dummy_processed_visual_edges
        }
        # AudioProcessor çıktısı: np.array([energy, spectral_centroid], dtype=float32) - shape (2,)
        dummy_processed_audio_features = np.array([np.random.rand(), np.random.rand() * 10000], dtype=np.float32) # Centroid daha büyük değerler alabilir


        dummy_processed_inputs = {
            'visual': dummy_processed_visual_dict,
            'audio': dummy_processed_audio_features
        }

        print(f"\nSahte işlenmiş girdi sözlüğü oluşturuldu: {list(dummy_processed_inputs.keys())}")
        if isinstance(dummy_processed_inputs.get('visual'), dict):
             print(f"  Görsel (dict): Keys: {list(dummy_processed_inputs['visual'].keys())}")
             if dummy_processed_inputs['visual'].get('grayscale') is not None:
                  print(f"    'grayscale' Shape: {dummy_processed_inputs['visual']['grayscale'].shape}, Dtype: {dummy_processed_inputs['visual']['grayscale'].dtype}")
             if dummy_processed_inputs['visual'].get('edges') is not None:
                  print(f"    'edges' Shape: {dummy_processed_inputs['visual']['edges'].shape}, Dtype: {dummy_processed_inputs['visual']['edges'].dtype}")
        if isinstance(dummy_processed_inputs.get('audio'), np.ndarray):
             print(f"  Ses (array): Shape: {dummy_processed_inputs['audio'].shape}, Dtype: {dummy_processed_inputs['audio'].dtype}")
             if dummy_processed_inputs['audio'].shape[0] > 1:
                  print(f"    Değerler (Enerji, Centroid): {dummy_processed_inputs['audio']}")


        # learn metodunu test et (geçerli girdi ile)
        print("\nlearn metodu test ediliyor (geçerli girdi ile):")
        learned_representation = learner.learn(dummy_processed_inputs)

        # Representation çıktısını kontrol et
        if learned_representation is not None: # None dönmediyse başarılı demektir.
            print("learn metodu başarıyla çalıştı. Öğrenilmiş temsil:")
            if isinstance(learned_representation, np.ndarray):
                 print(f"  Temsil: numpy.ndarray, Shape: {learned_representation.shape}, Dtype: {learned_representation.dtype}")
                 # Beklenen çıktı boyutunu kontrol et
                 if learned_representation.shape == (test_config['representation_dim'],):
                      print(f"  Boyutlar beklenen ({test_config['representation_dim']},) ile eşleşiyor.")
                 else:
                       print(f"  UYARI: Boyutlar beklenmiyor ({learned_representation.shape} vs ({test_config['representation_dim']},)).")
            else:
                 print(f"  Temsil: Beklenmeyen tip: {type(learned_representation)}")

        else:
             print("learn metodu None döndürdü (işlem başarısız veya girdi geçersiz).")


        # decode metodunu test et (latent vektör girdisi ile)
        print("\ndecode metodu test ediliyor (latent vektör ile):")
        # learn metodu çalıştıysa çıktısını kullan, yoksa sahte bir latent vektör oluştur
        test_latent_vector = learned_representation if learned_representation is not None else np.random.rand(test_config['representation_dim']).astype(np.float32)
        if test_latent_vector is not None and isinstance(test_latent_vector, np.ndarray) and test_latent_vector.shape == (test_config['representation_dim'],):
            print(f"Decoder girdisi (latent): Shape: {test_latent_vector.shape}, Dtype: {test_latent_vector.dtype}")
            reconstruction = learner.decode(test_latent_vector)

            # Rekonstrüksiyon çıktısını kontrol et
            if reconstruction is not None:
                print("decode metodu başarıyla çalıştı. Rekonstrüksiyon çıktısı:")
                if isinstance(reconstruction, np.ndarray):
                     print(f"  Rekonstrüksiyon: numpy.ndarray, Shape: {reconstruction.shape}, Dtype: {reconstruction.dtype}")
                     # Beklenen çıktı boyutunu (orijinal girdi boyutu) kontrol et
                     if reconstruction.shape == (test_config['input_dim'],):
                          print(f"  Boyutlar beklenen ({test_config['input_dim']},) ile eşleşiyor.")
                     else:
                           print(f"  UYARI: Boyutlar beklenmiyor ({reconstruction.shape} vs ({test_config['input_dim']},)).")
                else:
                     print(f"  Rekonstrüksiyon: Beklenmeyen tip: {type(reconstruction)}")
            else:
                 print("decode metodu None döndürdü (işlem başarısız veya girdi geçersiz).")
        else:
             print("decode metodu için geçerli latent vektör girdisi yok veya RepresentationLearner başlatılamadı.")


        # Geçersiz girdi (boş dict) ile learn test et
        print("\nlearn metodu test ediliyor (boş dict girdi ile):")
        learned_representation_empty = learner.learn({})
        if learned_representation_empty is None:
             print("Boş dict girdi ile learn metodu doğru şekilde None döndürdü.")
        else:
             print("Boş dict girdi ile learn metodu None döndürmedi (beklenmeyen durum).")

        # Geçersiz girdi (None) ile learn test et
        print("\nlearn metodu test ediliyor (None girdi ile):")
        learned_representation_none = learner.learn(None)
        if learned_representation_none is None:
             print("None girdi ile learn metodu doğru şekilde None döndürdü.")
        else:
             print("None girdi ile learn metodu None döndürmedi (beklenmeyen durum).")

        # Geçersiz girdi (yanlış boyutlu array) ile decode test et
        print("\ndecode metodu test ediliyor (yanlış boyut latent ile):")
        invalid_latent = np.random.rand(test_config['representation_dim'] + 10).astype(np.float32) # Yanlış boyut
        reconstruction_invalid = learner.decode(invalid_latent)
        if reconstruction_invalid is None:
             print("Yanlış boyut latent girdi ile decode metodu doğru şekilde None döndürdü.")
        else:
             print("Yanlış boyut latent girdi ile decode metodu None döndürmedi (beklenmeyen durum).")


    except Exception as e:
        print(f"\nTest sırasında beklenmedik hata oluştu: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup metodunu çağır
        if 'learner' in locals() and hasattr(learner, 'cleanup'):
             print("\nRepresentationLearner cleanup çağrılıyor...")
             learner.cleanup()

    print("\nRepresentationLearner testi bitti.")