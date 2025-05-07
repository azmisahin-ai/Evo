# src/cognition/understanding.py
#
# Evo'nın anlama modülünü temsil eder.
# İşlenmiş temsilleri, bellek girdilerini, anlık duyu özelliklerini ve öğrenilmiş kavramları kullanarak dünyayı anlamaya çalışır.

import logging
import numpy as np # Benzerlik hesaplaması, array işlemleri ve ortalama için

# Yardımcı fonksiyonları import et (girdi kontrolleri src/core/utils'tan, config src/core/config_utils'tan)
try:
    from src.core.utils import check_input_not_none, check_input_type, check_numpy_input # get_config_value BURADAN GELMEZ ARTIK
    from src.core.config_utils import get_config_value # get_config_value ARTIK BURADAN GELİYOR (nested versiyon)
except ImportError as e:
    logging.critical(f"Temel yardımcı modüller import edilemedi: {e}. Lütfen src/core/utils.py ve src/core/config_utils.py dosyalarının mevcut olduğundan ve PYTHONPATH'in doğru ayarlandığından emin olun.")
    # Programın devam etmesi için placeholder fonksiyonlar (debug/geliştirme amaçlı)
    def get_config_value(config, *keys, default=None, expected_type=None, logger_instance=None):
         print(f"PLACEHOLDER: get_config_value called for {keys}. Returning default: {default}")
         return default
    def check_input_not_none(input_data, input_name="Girdi", logger_instance=None):
         print(f"PLACEHOLDER: check_input_not_none called for {input_name}. Input is None: {input_data is None}")
         return input_data is not None
    def check_input_type(input_data, expected_type, input_name="Girdi", logger_instance=None):
         is_correct = isinstance(input_data, expected_type)
         print(f"PLACEHOLDER: check_input_type called for {input_name}. Expected {expected_type}, got {type(input_data)}. Correct: {is_correct}")
         return is_correct
    def check_numpy_input(input_data, expected_dtype=None, expected_ndim=None, input_name="Girdi", logger_instance=None):
         is_correct = isinstance(input_data, np.ndarray) # Basit kontrol
         print(f"PLACEHOLDER: check_numpy_input called for {input_name}. Input is ndarray: {is_correct}")
         # Detaylı numpy kontrolünü placeholder'da yapmak zor, testlerde gerçek utils'in kullanılmasını sağlayın.
         # Eğer import hatası alıyorsanız test ortamı kurulumunu kontrol edin.
         if is_correct and expected_dtype is not None and not np.issubdtype(input_data.dtype, expected_dtype):
              is_correct = False
              print(f"PLACEHOLDER: check_numpy_input dtype mismatch for {input_name}. Expected {expected_dtype}, got {input_data.dtype}")
         if is_correct and expected_ndim is not None and input_data.ndim != expected_ndim:
              is_correct = False
              print(f"PLACEHOLDER: check_numpy_input ndim mismatch for {input_name}. Expected {expected_ndim}, got {input_data.ndim}")
         return is_correct


# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

class UnderstandingModule:
    """
    Evo'nın anlama yeteneğini sağlayan sınıf (Faz 3/4 implementasyonu).

    RepresentationLearner'dan gelen Representation'ı, Memory'den gelen ilgili anıları,
    anlık Process çıktılarını (düşük seviyeli özellikler) ve LearningModule'den gelen
    öğrenilmiş kavram temsilcilerini alır. Bu girdileri kullanarak ilkel bir
    "anlama" çıktısı (şimdilik bir dictionary) üretir.
    Mevcut implementasyon: En yüksek bellek benzerlik skorunu, anlık duyu özelliklerine
    dayalı boolean flag'leri VE öğrenilmiş kavram temsilcilerine olan en yüksek benzerliği
    hesaplar.
    Gelecekte daha karmaşık anlama algoritmaları implement edilecektir.
    """
    def __init__(self, config):
        """
        UnderstandingModule'ü başlatır.

        Args:
            config (dict): Anlama modülü yapılandırma ayarları.
                           'audio_energy_threshold': Yüksek ses enerjisi algılama eşiği (float, varsayılan 1000.0).
                           'visual_edges_threshold': Yüksek görsel kenar yoğunluğu algılama eşiği (float, varsayılan 50.0).
                           'brightness_threshold_high': Parlak ortam algılama eşiği (float, varsayılan 200.0).
                           'brightness_threshold_low': Karanlık ortam algılama eşiği (float, varsayılan 50.0).
                           Gelecekte model yolları, anlama stratejileri gibi ayarlar gelebilir.
        """
        self.config = config
        logger.info("UnderstandingModule başlatılıyor (Faz 3/4)...")

        # get_config_value artık *keys alıyor, tek anahtar için şöyle kullanılır:
        self.audio_energy_threshold = get_config_value(config, 'audio_energy_threshold', default=1000.0, expected_type=(float, int), logger_instance=logger)
        self.visual_edges_threshold = get_config_value(config, 'visual_edges_threshold', default=50.0, expected_type=(float, int), logger_instance=logger)
        self.brightness_threshold_high = get_config_value(config, 'brightness_threshold_high', default=200.0, expected_type=(float, int), logger_instance=logger)
        self.brightness_threshold_low = get_config_value(config, 'brightness_threshold_low', default=50.0, expected_type=(float, int), logger_instance=logger)

        # get_config_value int tipini de float'a çevirmeli (ConfigUtils versiyonu çevirmiyor),
        # bu yüzden değerleri float'a çevirelim emin olmak için.
        self.audio_energy_threshold = float(self.audio_energy_threshold)
        self.visual_edges_threshold = float(self.visual_edges_threshold)
        self.brightness_threshold_high = float(self.brightness_threshold_high)
        self.brightness_threshold_low = float(self.brightness_threshold_low)


        logger.info(f"UnderstandingModule başlatıldı. Ses Enerji Eşiği: {self.audio_energy_threshold}, Görsel Kenar Eşiği: {self.visual_edges_threshold}, Parlaklık Yüksek Eşiği: {self.brightness_threshold_high}, Parlaklık Düşük Eşiği: {self.brightness_threshold_low}")


    def process(self, processed_inputs, learned_representation, relevant_memory_entries, current_concepts):
        """
        Gelen Representation, ilgili bellek girdileri, anlık Process çıktıları ve öğrenilmiş kavramları kullanarak anlama işlemini yapar.
        ... (Docstring aynı) ...
        """
        # Varsayılan anlama çıktısı dictionary'si
        understanding_signals = {
            'similarity_score': 0.0,
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.0,
            'most_similar_concept_id': None,
        }

        # Girdi geçerliliği kontrolleri.
        # check_input_not_none ve check_input_type artık src.core.utils'tan geliyor.
        # check_numpy_input da src.core.utils'tan geliyor.
        is_valid_memory_list = check_input_not_none(relevant_memory_entries, "relevant_memory_entries for UnderstandingModule", logger) and \
                               check_input_type(relevant_memory_entries, list, "relevant_memory_entries for UnderstandingModule", logger)

        is_valid_representation = check_input_not_none(learned_representation, "learned_representation for UnderstandingModule", logger) and \
                                  check_numpy_input(learned_representation, expected_dtype=np.number, expected_ndim=1, input_name="learned_representation for UnderstandingModule", logger_instance=logger)

        is_valid_processed_inputs = check_input_not_none(processed_inputs, "processed_inputs for UnderstandingModule", logger) and \
                                    isinstance(processed_inputs, dict) # check_input_type dict için yeterli değil, isinstance daha esnek


        is_valid_concepts_list = check_input_not_none(current_concepts, "current_concepts for UnderstandingModule", logger) and \
                                  check_input_type(current_concepts, list, "current_concepts for UnderstandingModule", logger)


        logger.debug(f"UnderstandingModule.process: Anlama işlemi başlatıldı. Girdi geçerliliği - Repr:{is_valid_representation}, Mem:{is_valid_memory_list}, Proc:{is_valid_processed_inputs}, Concepts:{is_valid_concepts_list}")

        # Representation geçerliyse normunu hesapla, aksi halde Representation tabanlı hesaplamaları atla.
        query_norm = 0.0
        if is_valid_representation:
             query_norm = np.linalg.norm(learned_representation)
             if query_norm < 1e-8:
                 logger.warning("UnderstandingModule.process: Learned representation near zero norm. Benzerlik ve Kavram tanıma atlanacak.")
                 is_valid_representation = False # Geçersiz gibi davran

        # Bu try bloğu, numpy veya indexleme gibi işlemlerden kaynaklanabilecek hataları yakalar.
        # src/core/utils'daki check_* fonksiyonları exception fırlatmadığı için, buradaki except
        # daha çok np.dot, np.linalg.norm, np.mean gibi işlemlerden kaynaklı hataları yakalayacaktır.
        try:
            # 1. Bellek Benzerlik Skoru Hesapla (Eğer Representation ve Bellek geçerliyse)
            max_memory_similarity = 0.0
            if is_valid_representation and is_valid_memory_list and relevant_memory_entries:
                for memory_entry in relevant_memory_entries:
                    stored_representation = memory_entry.get('representation')
                    # check_numpy_input artık src.core.utils'tan geliyor ve False dönüyor.
                    if stored_representation is not None and check_numpy_input(stored_representation, expected_dtype=np.number, expected_ndim=1, input_name="memory_entry['representation'] for UnderstandingModule", logger_instance=logger):
                         stored_norm = np.linalg.norm(stored_representation)
                         if stored_norm > 1e-8:
                              similarity = np.dot(learned_representation, stored_representation) / (query_norm * stored_norm)
                              if not np.isnan(similarity):
                                   max_memory_similarity = max(max_memory_similarity, similarity)
                understanding_signals['similarity_score'] = max_memory_similarity


            # 2. Öğrenilmiş Kavramlarla Benzerlik Hesapla (Eğer Representation ve Kavramlar geçerliyse)
            max_concept_similarity = 0.0
            most_similar_concept_id = None

            if is_valid_representation and is_valid_concepts_list and current_concepts:
                 for i, concept_rep in enumerate(current_concepts):
                      # check_numpy_input artık src.core.utils'tan geliyor ve False dönüyor.
                      if concept_rep is not None and check_numpy_input(concept_rep, expected_dtype=np.number, expected_ndim=1, input_name=f"current_concepts[{i}] for UnderstandingModule", logger_instance=logger):
                           concept_norm = np.linalg.norm(concept_rep)
                           if concept_norm > 1e-8:
                                similarity = np.dot(learned_representation, concept_rep) / (query_norm * concept_norm)
                                if not np.isnan(similarity):
                                     if similarity > max_concept_similarity:
                                          max_concept_similarity = similarity
                                          most_similar_concept_id = i

                 understanding_signals['max_concept_similarity'] = max_concept_similarity
                 understanding_signals['most_similar_concept_id'] = most_similar_concept_id


            # 3. Process Çıktıları Tabanlı Basit Anlama (Eğer Processed Inputs geçerliyse)
            # check_numpy_input False döndüğünde exception fırlatmaz, bu doğru.
            # Indexleme veya np.mean gibi işlemlerden exception gelebilir.
            if is_valid_processed_inputs:
                 # Ses Enerjisi Kontrolü
                 audio_features = processed_inputs.get('audio')
                 # check_numpy_input dtype kontrolü zaten yapıyor.
                 if isinstance(audio_features, np.ndarray) and \
                    check_numpy_input(audio_features, expected_dtype=np.number, expected_ndim=1, input_name="processed_inputs['audio']", logger_instance=logger) and \
                    audio_features.shape[0] >= 1: # size>0 kontrolü gibi audio_features.shape[0] >= 1
                      audio_energy = float(audio_features[0]) # Indexleme hatası olabilir (boş array)
                      if audio_energy >= self.audio_energy_threshold:
                           understanding_signals['high_audio_energy'] = True
                           logger.debug(f"UnderstandingModule.process: Yüksek ses enerjisi Algılandı: {audio_energy:.4f} >= Eşik {self.audio_energy_threshold:.4f}")


                 # Görsel Kenar Yoğunluğu Kontrolü
                 visual_features = processed_inputs.get('visual')
                 if isinstance(visual_features, dict):
                      edges_data = visual_features.get('edges')
                      # check_numpy_input dtype kontrolü zaten yapıyor.
                      if isinstance(edges_data, np.ndarray) and \
                         check_numpy_input(edges_data, expected_dtype=np.number, expected_ndim=(1,2), input_name="processed_inputs['visual']['edges']", logger_instance=logger) and \
                         edges_data.size > 0: # size > 0 kontrolü mean için önemli
                           visual_edges_mean = np.mean(edges_data) # mean boş arrayde hata verebilir
                           if visual_edges_mean >= self.visual_edges_threshold:
                                understanding_signals['high_visual_edges'] = True
                                # logger.debug(...)


                 # Görsel Parlaklık/Karanlık Controlü
                 if isinstance(visual_features, dict):
                      grayscale_data = visual_features.get('grayscale')
                      # check_numpy_input dtype kontrolü zaten yapıyor.
                      if isinstance(grayscale_data, np.ndarray) and \
                         check_numpy_input(grayscale_data, expected_dtype=np.number, expected_ndim=(1,2), input_name="processed_inputs['visual']['grayscale']", logger_instance=logger) and \
                         grayscale_data.size > 0: # size > 0 kontrolü mean için önemli
                           visual_brightness_mean = np.mean(grayscale_data) # mean boş arrayde hata verebilir

                           if visual_brightness_mean >= self.brightness_threshold_high:
                                understanding_signals['is_bright'] = True
                                # logger.debug(...)
                           elif visual_brightness_mean <= self.brightness_threshold_low:
                                understanding_signals['is_dark'] = True
                                # logger.debug(...)


            logger.debug(f"UnderstandingModule.process: Üretilen anlama sinyalleri: {understanding_signals}")

        except Exception as e:
            # Hesaplama veya girdi işleme sırasında beklenmedik bir hata olursa logla ve varsayılanları döndür.
            # check_* fonksiyonları False döndüğünde burası TETİKLENMEZ.
            # Sadece numpy işlemleri (norm, dot, mean) veya indexleme hatalarında tetiklenir.
            logger.error(f"UnderstandingModule.process: Anlama işlemi sırasında beklenmedik hata: {e}", exc_info=True)
            # Hata durumunda varsayılan anlama sinyallerini döndür
            return {
                'similarity_score': 0.0,
                'high_audio_energy': False,
                'high_visual_edges': False,
                'is_bright': False,
                'is_dark': False,
                'max_concept_similarity': 0.0,
                'most_similar_concept_id': None,
            }

        return understanding_signals

    # ... (cleanup metodu) ...
    def cleanup(self):
        """
        UnderstandingModule kaynaklarını temizler.

        Şimdilik özel bir kaynak kullanmadığı için temizleme adımı içermez.
        Gelecekte model temizliği veya bağlantı kapatma gerekebilir.
        module_loader.py bu metotu program sonlanırken çağırır (varsa).
        """
        logger.info("UnderstandingModule objesi temizleniyor.")
        pass