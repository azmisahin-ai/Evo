# src/representation/models.py
#
# Learns or extracts internal representations (latent vectors) from processed sensory data.
# Implements basic neural network layers.
# Part of Evo's Phase 1 representation learning capabilities.

import numpy as np # For numerical operations and arrays.
import logging # For logging.

# Import utility functions
from src.core.config_utils import get_config_value
from src.core.utils import check_input_not_none, check_numpy_input # <<< Utils imports


# Create a logger for this module
# Returns a logger named 'src.representation.models'.
logger = logging.getLogger(__name__)

# A simple Dense (Fully Connected) Layer class
# TODO: In the future, using the central version from src/core/nn_components.py would be cleaner.
class Dense:
    """
    A basic implementation of a dense (fully connected) layer.

    Contains weight and bias parameters.
    Supports ReLU activation function.
    Performs the forward pass calculation.
    """
    def __init__(self, input_dim, output_dim, activation='relu'):
        """
        Initializes the Dense layer.

        Initializes weights and biases randomly.

        Args:
            input_dim (int): The dimension of the input feature.
            output_dim (int): The dimension of the output feature.
            activation (str, optional): The name of the activation function to use ('relu'). Defaults to 'relu'.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        # Create a logger for this specific class instance or use module logger.
        # Currently using the module logger, which is fine.
        logger.info(f"Dense layer initializing: Input={input_dim}, Output={output_dim}, Activation={activation}")

        # Initialize weights and biases.
        # Better initialization methods (He, Xavier) could be used (Future TODO).
        # Using np.random.uniform for a simple initialization.
        # Scaling by sqrt(1. / input_dim) (fan-in) is a common heuristic.
        # If using PyTorch, weights would be PyTorch tensors and could be moved to GPU.
        # For now, staying on CPU with NumPy.
        limit = np.sqrt(1. / input_dim) # Simple initialization scale (fan-in)
        self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.bias = np.zeros(output_dim) # Biases are commonly initialized to zero.

        logger.info("Dense layer initialized.")

    def forward(self, inputs):
        """
        Computes the forward pass of the layer.

        Multiplies the input vector or matrix by the weight matrix, adds the bias, and
        applies the activation function.
        Returns None if the input is None or has the wrong type/dimensions.
        Returns None if an error occurs during calculation.

        Args:
            inputs (numpy.ndarray or None): The input data for the layer.
                                            Expected format: shape (..., input_dim), numerical dtype.
                                            ... can be a batch dimension.

        Returns:
            numpy.ndarray or None: The output data of the layer, or None on error.
                                   shape (..., output_dim), numerical dtype.
        """
        # Error handling: Is input None? Use check_input_not_none.
        if not check_input_not_none(inputs, input_name="dense_inputs", logger_instance=logger):
             logger.debug("Dense.forward: Input is None. Returning None.") # None input is not an error, just informational.
             return None # Return None if input is None.

        # Error handling: Check if the input is a numpy array and has a numerical dtype.
        # expected_ndim=None because input can be a single vector or a batch (ndim >= 1).
        # Shape check will be done separately.
        if not check_numpy_input(inputs, expected_dtype=np.number, expected_ndim=None, input_name="dense_inputs", logger_instance=logger):
             logger.error("Dense.forward: Input is not a numpy array or is not numerical. Returning None.") # check_numpy_input already logs internally.
             return None # If type or dtype is invalid, return None.

        # Error handling: Check if the input dimension (last dimension) matches the input_dim.
        # inputs.shape[-1] gives the last dimension.
        if inputs.shape[-1] != self.input_dim:
             logger.error(f"Dense.forward: Unexpected input dimension: {inputs.shape}. The last dimension ({inputs.shape[-1]}) does not match the expected input_dim ({self.input_dim}). Returning None.")
             return None # If dimensions don't match, return None.


        # DEBUG log: Input details.
        logger.debug(f"Dense.forward: Input received. Shape: {inputs.shape}, Dtype: {inputs.dtype}. Performing calculation.")


        output = None # Variable to hold the output.

        try:
            # Linear transformation: output = input_data @ weights + bias
            # For a single example (input_dim,) * weights (input_dim, output_dim) -> (output_dim,)
            # For a batch ((batch_size, input_dim)) * weights (input_dim, output_dim) -> (batch_size, output_dim)
            # np.dot handles both cases correctly.
            linear_output = np.dot(inputs, self.weights) + self.bias

            # Apply activation function (if any)
            if self.activation == 'relu':
                # ReLU: output = max(0, output)
                output = np.maximum(0, linear_output) # ReLU activation
            # TODO: Other activation functions will be added (sigmoid, tanh, etc.)
            # elif self.activation == 'sigmoid':
            #      output = 1 / (1 + np.exp(-linear_output)) # Be careful with potential over/underflow in exp.
            # elif self.activation == 'tanh':
            #      output = np.tanh(linear_output)
            # Add a warning/error log for unknown activation names.
            elif self.activation is None: # None activation
                 output = linear_output
            else:
                 logger.warning(f"Dense.forward: Unknown activation function: '{self.activation}'. Using linear activation.")
                 output = linear_output


        except Exception as e:
             # Catch any unexpected error during the forward pass calculations (e.g., np.dot, activation function error).
             logger.error(f"Dense.forward: Unexpected error during forward pass: {e}", exc_info=True)
             return None # Return None in case of error.


        # Return the calculated output on success.
        logger.debug(f"Dense.forward: Forward pass completed. Output Shape: {output.shape}, Dtype: {output.dtype}.")
        return output

    def cleanup(self):
        """
        Cleans up Dense layer resources.

        Currently, this layer does not use specific resources (files, connections, etc.)
        and does not require a cleanup step beyond basic object deletion.
        Includes an informational log.
        NumPy arrays generally do not require explicit cleanup.
        """
        # Informational log.
        logger.info(f"Dense layer object cleaning up: Input={self.input_dim}, Output={self.output_dim}")
        pass


# A simple Representation Learner class
class RepresentationLearner:
    """
    Learns or extracts internal representations (latent vectors) from processed sensory data.

    Receives processed sensory data (in a dictionary format) from Processing modules.
    Combines this data into a unified input vector.
    Passes the input vector through an Encoder layer (Dense) to create a low-dimensional,
    meaningful representation vector (latent).
    Also includes a Decoder layer (Dense) for Autoencoder principle, which produces a
    reconstruction in the original input dimensions from the latent vector.
    More complex models (CNN, RNN, Transformer-based) will be added here in the future.
    Returns None if the module fails to initialize or if an error occurs during
    representation learning/extraction.
    """
    def __init__(self, config):
        """
        Initializes the RepresentationLearner module.

        Sets up the structure of the representation model (currently Encoder and Decoder Dense layers).
        input_dim and representation_dim are obtained from config.

        Args:
            config (dict): RepresentationLearner configuration settings (full config dict).
                           Settings for this module are read from the 'representation' section,
                           and processor dimensions from the 'processors' section.
        """
        self.config = config # RepresentationLearner receives the full config
        logger.info("RepresentationLearner initializing...")

        # Get input and representation dimensions from config using get_config_value.
        # Corrected: Use default= keyword format for all calls.
        # Based on config, these settings are under the 'representation' key.
        self.input_dim = get_config_value(config, 'representation', 'input_dim', default=8194, expected_type=int, logger_instance=logger)
        self.representation_dim = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)

        self.encoder = None # Encoder layer (currently Dense).
        self.decoder = None # Decoder layer (currently Dense).
        self.is_initialized = False # Tracks if the module was initialized successfully.

        # TODO: In the future: Get input dimensions from different modalities from config,
        # TODO: and verify that the input_dim value matches the sum of these dimensions.
        # Processor output dimensions are used by RepresentationLearner for calculating the expected input_dim.
        # Get processor output dimensions using get_config_value for consistency.
        visual_config = config.get('processors', {}).get('vision', {})
        audio_config = config.get('processors', {}).get('audio', {})
        # Use get_config_value to retrieve values from the nested config structures.
        visual_out_width = get_config_value(visual_config, 'output_width', default=64, expected_type=int, logger_instance=logger) # Use visual_config here
        visual_out_height = get_config_value(visual_config, 'output_height', default=64, expected_type=int, logger_instance=logger) # Use visual_config here
        audio_features_dim = get_config_value(audio_config, 'output_dim', default=2, expected_type=int, logger_instance=logger) # Use audio_config here

        visual_gray_size = visual_out_width * visual_out_height
        visual_edges_size = visual_out_width * visual_out_height # Assume same dimensions for simplicity
        expected_input_dim_calc = visual_gray_size + visual_edges_size + audio_features_dim

        if self.input_dim != expected_input_dim_calc:
             logger.warning(f"RepresentationLearner: Configured input_dim ({self.input_dim}) does not match calculated expected value ({expected_input_dim_calc}). Please check the 'representation.input_dim' setting in the config file and the output dimensions of the Processors ('processors.vision.output_width/height', 'processors.audio.output_dim'). Calculated dimension is based on Processor output dimensions.")
             # Optional: Could set self.input_dim to the calculated value in this case.
             # self.input_dim = expected_input_dim_calc


        try:
            # Create the Encoder layer (Input dimension: self.input_dim, Output dimension: self.representation_dim).
            # The last layer of an encoder typically has no activation or linear activation (latent space).
            # Added None activation option to Dense class. Using None activation here.
            self.encoder = Dense(self.input_dim, self.representation_dim, activation=None) # No activation in latent space

            # Create the Decoder layer (Input dimension: self.representation_dim, Output dimension: self.input_dim).
            # The decoder output should be a reconstruction of the original input dimensions.
            # The last layer of the decoder might have an activation suitable for the output data type (e.g., sigmoid/tanh for images, linear for numerical).
            # For now, using None activation again.
            self.decoder = Dense(self.representation_dim, self.input_dim, activation=None) # Reconstruction output


            # Initialization successful flag: Set to True if Encoder and Decoder objects were created successfully.
            self.is_initialized = (self.encoder is not None and self.decoder is not None)

        except Exception as e:
            # Catch any unexpected error during model layer initialization.
            # This error can be considered critical during initialization (Represent module cannot function without a model).
            logger.critical(f"RepresentationLearner initialization failed critically: {e}", exc_info=True)
            self.is_initialized = False # Mark as not initialized in case of error.
            self.encoder = None
            self.decoder = None


        logger.info(f"RepresentationLearner initialized. Input dimension: {self.input_dim}, Representation dimension: {self.representation_dim}. Initialization Successful: {self.is_initialized}")


    # ... (learn, decode, cleanup methods - same as before) ...


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