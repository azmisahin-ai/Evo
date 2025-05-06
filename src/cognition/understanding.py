# src/cognition/understanding.py
#
# Evo'nın anlama modülünü temsil eder.
# İşlenmiş temsilleri, bellek girdilerini, anlık duyu özelliklerini ve öğrenilmiş kavramları kullanarak dünyayı anlamaya çalışır.

import logging
import numpy as np # Benzerlik hesaplaması, array işlemleri ve ortalama için

# Yardımcı fonksiyonları import et (girdi kontrolleri ve config için)
from src.core.utils import check_input_not_none, check_input_type, check_numpy_input, get_config_value # <<< Utils importları

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
                           # 'concept_recognition_threshold': Kavram tanıma için Representation'ın kavram temsilcisine
                           #                                en az benzemesi gereken benzerlik eşiği (float, varsayılan 0.85).
                           # Bu eşik DecisionModule'e taşındı.
                           Gelecekte model yolları, anlama stratejileri gibi ayarlar gelebilir.
        """
        self.config = config
        logger.info("UnderstandingModule başlatılıyor (Faz 3/4)...")

        # Yapılandırmadan Process özellik eşiklerini ve kavram tanıma eşiğini alırken get_config_value kullan.
        self.audio_energy_threshold = get_config_value(config, 'audio_energy_threshold', 1000.0, expected_type=(float, int), logger_instance=logger)
        self.visual_edges_threshold = get_config_value(config, 'visual_edges_threshold', 50.0, expected_type=(float, int), logger_instance=logger)
        self.brightness_threshold_high = get_config_value(config, 'brightness_threshold_high', 200.0, expected_type=(float, int), logger_instance=logger)
        self.brightness_threshold_low = get_config_value(config, 'brightness_threshold_low', 50.0, expected_type=(float, int), logger_instance=logger)
        # Concept recognition threshold DecisionModule'e taşındı.
        # self.concept_recognition_threshold = get_config_value(config, 'concept_recognition_threshold', 0.85, expected_type=(float, int), logger_instance=logger)


        # Eşik değerleri için basit değer kontrolü.
        # Daha önce yapıldı, burada tekrar etmeye gerek yok, get_config_value yeterli.


        logger.info(f"UnderstandingModule başlatıldı. Ses Enerji Eşiği: {self.audio_energy_threshold}, Görsel Kenar Eşiği: {self.visual_edges_threshold}, Parlaklık Yüksek Eşiği: {self.brightness_threshold_high}, Parlaklık Düşük Eşiği: {self.brightness_threshold_low}")


    # processed_inputs, learned_representation, relevant_memory_entries ve current_concepts argümanlarını alıyor.
    def process(self, processed_inputs, learned_representation, relevant_memory_entries, current_concepts):
        """
        Gelen Representation, ilgili bellek girdileri, anlık Process çıktıları ve öğrenilmiş kavramları kullanarak anlama işlemini yapar.

        Mevcut implementasyon: En yüksek bellek benzerlik skorunu, anlık duyu özelliklerine
        dayalı boolean flag'leri VE öğrenilmiş kavram temsilcilerine olan en yüksek benzerliği
        hesaplar. Sonuçları bir dictionary olarak döndürür.

        Args:
            processed_inputs (dict or None): Processor modülllerinden gelen işlenmiş ham veriler.
                                            Beklenen format: {'visual': dict, 'audio': np.ndarray} veya None/boş dict.
            learned_representation (numpy.ndarray or None): En son öğrenilmiş temsil vektörü.
                                                         Beklenen format: shape (D,), dtype sayısal, veya None.
            relevant_memory_entries (list or None): Memory modülünden gelen ilgili bellek girdileri listesi.
                                            Her öğe {'representation': np.ndarray, ...} formatında olmalıdır.
                                            Boş veya None olabilir.
            current_concepts (list or None): LearningModule'den gelen öğrenilmiş kavram temsilcileri listesi.
                                             Her öğe numpy array shape (D,) olmalıdır. Boş veya None olabilir.

        Returns:
            dict: Anlama sinyallerini içeren bir dictionary.
                  'similarity_score': float (0.0-1.0 arası, en yüksek bellek benzerliği)
                  'high_audio_energy': bool (Ses enerji eşiği aşıldı mı?)
                  'high_visual_edges': bool (Görsel kenar eşiği aşıldı mı?)
                  'is_bright': bool (Ortalama parlaklık yüksek eşiği aşıldı mı?)
                  'is_dark': bool (Ortalama parlaklık düşük eşiğin altında mı?)
                  'max_concept_similarity': float (0.0-1.0 arası, en yüksek kavram temsilcisi benzerliği) - YENİ
                  'most_similar_concept_id': int or None (En benzer kavramın ID'si) - YENİ
                  Hata durumunda veya geçerli girdi yoksa varsayılan değerlerle döndürülür (örn: {'similarity_score': 0.0, 'high_audio_energy': False, 'high_visual_edges': False, 'is_bright': False, 'is_dark': False, 'max_concept_similarity': 0.0, 'most_similar_concept_id': None}).
        """
        # Varsayılan anlama çıktısı dictionary'si
        understanding_signals = {
            'similarity_score': 0.0,
            'high_audio_energy': False,
            'high_visual_edges': False,
            'is_bright': False,
            'is_dark': False,
            'max_concept_similarity': 0.0,       # Yeni sinyal
            'most_similar_concept_id': None,     # Yeni sinyal
        }

        # Girdi geçerliliği kontrolleri.
        # relevant_memory_entries, learned_representation, processed_inputs, current_concepts
        is_valid_memory_list = relevant_memory_entries is not None and check_input_type(relevant_memory_entries, list, input_name="relevant_memory_entries for UnderstandingModule", logger_instance=logger)
        # Learned representation'ın numpy array ve sayısal 1D olduğunu kontrol et.
        is_valid_representation = learned_representation is not None and check_numpy_input(learned_representation, expected_dtype=np.number, expected_ndim=1, input_name="learned_representation for UnderstandingModule", logger_instance=logger)
        is_valid_processed_inputs = processed_inputs is not None and isinstance(processed_inputs, dict)
        # Current concepts'ın liste mi ve içindekilerin numpy array mi olduğunu kontrol et.
        is_valid_concepts_list = current_concepts is not None and check_input_type(current_concepts, list, input_name="current_concepts for UnderstandingModule", logger_instance=logger)

        logger.debug(f"UnderstandingModule.process: Anlama işlemi başlatıldı. Girdi geçerliliği - Repr:{is_valid_representation}, Mem:{is_valid_memory_list}, Proc:{is_valid_processed_inputs}, Concepts:{is_valid_concepts_list}")

        # Representation geçerliyse normunu hesapla, aksi halde Representation tabanlı hesaplamaları atla.
        query_norm = 0.0
        if is_valid_representation:
             query_norm = np.linalg.norm(learned_representation)
             if query_norm < 1e-8:
                 logger.warning("UnderstandingModule.process: Learned representation near zero norm. Benzerlik ve Kavram tanıma atlanacak.")
                 is_valid_representation = False # Geçersiz gibi davran

        try:
            # 1. Bellek Benzerlik Skoru Hesapla (Eğer Representation ve Bellek geçerliyse)
            max_memory_similarity = 0.0
            if is_valid_representation and is_valid_memory_list and relevant_memory_entries: # Bellek listesi de boş olmamalı
                for memory_entry in relevant_memory_entries:
                    stored_representation = memory_entry.get('representation')
                    # Stored representation'ın geçerli bir sayısal 1D numpy array olduğunu doğrula.
                    # np.issubtype yerine isinstance ve np.number kullanımı (Hata düzeltme).
                    if stored_representation is not None and isinstance(stored_representation, np.ndarray) and isinstance(stored_representation.dtype, np.number) and stored_representation.ndim == 1:
                         stored_norm = np.linalg.norm(stored_representation)
                         if stored_norm > 1e-8:
                              similarity = np.dot(learned_representation, stored_representation) / (query_norm * stored_norm)
                              if not np.isnan(similarity):
                                   max_memory_similarity = max(max_memory_similarity, similarity)

            understanding_signals['similarity_score'] = max_memory_similarity
            # logger.debug(f"UnderstandingModule.process: En yüksek bellek benzerlik skoru: {max_memory_similarity:.4f}") # DecisionModule logluyor


            # 2. Öğrenilmiş Kavramlarla Benzerlik Hesapla (Eğer Representation ve Kavramlar geçerliyse)
            max_concept_similarity = 0.0
            most_similar_concept_id = None

            if is_valid_representation and is_valid_concepts_list and current_concepts: # Kavram listesi de boş olmamalı
                 for i, concept_rep in enumerate(current_concepts):
                      # Kavram temsilcisinin geçerli olduğundan emin ol.
                      # np.issubtype yerine isinstance ve np.number kullanımı (Hata düzeltme).
                      if concept_rep is not None and isinstance(concept_rep, np.ndarray) and isinstance(concept_rep.dtype, np.number) and concept_rep.ndim == 1:
                           concept_norm = np.linalg.norm(concept_rep)
                           if concept_norm > 1e-8:
                                similarity = np.dot(learned_representation, concept_rep) / (query_norm * concept_norm)
                                if not np.isnan(similarity):
                                     if similarity > max_concept_similarity: # En yüksek benzerliği ve ilgili ID'yi güncelle
                                          max_concept_similarity = similarity
                                          most_similar_concept_id = i # Kavram ID'si listenin indeksi
                           # else: logger.debug("UM.process: Concept rep ID {i} near zero norm, skipping similarity.")
                      # else: logger.warning(f"UM.process: Invalid concept rep ID {i}, skipping.")

            understanding_signals['max_concept_similarity'] = max_concept_similarity
            understanding_signals['most_similar_concept_id'] = most_similar_concept_id
            # logger.debug(f"UnderstandingModule.process: En yüksek kavram benzerlik skoru: {max_concept_similarity:.4f} (ID: {most_similar_concept_id})") # DecisionModule logluyor


            # 3. Process Çıktıları Tabanlı Basit Anlama (Eğer Processed Inputs geçerliyse)
            if is_valid_processed_inputs:
                 # Ses Enerjisi Kontrolü
                 audio_features = processed_inputs.get('audio') # AudioProcessor'dan (2,) float32 numpy array bekleniyor veya None.
                 # check_numpy_input ile genel kontrol, sonra shape[0] >= 1 ve sayısal dtype kontrolü.
                 # np.issubtype yerine isinstance ve np.number kullanımı (Hata düzeltme).
                 if isinstance(audio_features, np.ndarray) and check_numpy_input(audio_features, expected_dtype=np.number, expected_ndim=1, input_name="processed_inputs['audio']", logger_instance=logger) and audio_features.shape[0] >= 1 and isinstance(audio_features.dtype, np.number):
                      audio_energy = float(audio_features[0])
                      if audio_energy >= self.audio_energy_threshold:
                           understanding_signals['high_audio_energy'] = True
                           logger.debug(f"UnderstandingModule.process: Yüksek ses enerjisi: {audio_energy:.4f} >= Eşik {self.audio_energy_threshold:.4f} (Algılandı: True)")
                      # else: logger.debug(f"UnderstandingModule.process: Ses enerjisi: {audio_energy:.4f} < Eşik {self.audio_energy_threshold:.4f} (Algılandı: False)")


                 # Görsel Kenar Yoğunluğu Kontrolü
                 visual_features = processed_inputs.get('visual')
                 if isinstance(visual_features, dict):
                      edges_data = visual_features.get('edges')
                      # check_numpy_input ile genel control, size > 0 ve sayısal dtype.
                      # np.issubtype yerine isinstance ve np.number kullanımı (Hata düzeltme).
                      if isinstance(edges_data, np.ndarray) and check_numpy_input(edges_data, expected_dtype=np.number, expected_ndim=(1,2), input_name="processed_inputs['visual']['edges']", logger_instance=logger) and edges_data.size > 0 and isinstance(edges_data.dtype, np.number):
                           visual_edges_mean = np.mean(edges_data)
                           if visual_edges_mean >= self.visual_edges_threshold:
                                understanding_signals['high_visual_edges'] = True
                                # logger.debug(f"UM.process: Yüksek görsel kenar yoğunluğu: {visual_edges_mean:.4f} >= Eşik {self.visual_edges_threshold:.4f} (Algılandı: True)")


                 # Görsel Parlaklık/Karanlık Kontrolü
                 if isinstance(visual_features, dict):
                      grayscale_data = visual_features.get('grayscale')
                      # check_numpy_input ile genel control, size > 0 ve sayısal dtype.
                      # np.issubtype yerine isinstance ve np.number kullanımı (Hata düzeltme).
                      if isinstance(grayscale_data, np.ndarray) and check_numpy_input(grayscale_data, expected_dtype=np.number, expected_ndim=(1,2), input_name="processed_inputs['visual']['grayscale']", logger_instance=logger) and grayscale_data.size > 0 and isinstance(grayscale_data.dtype, np.number):
                           visual_brightness_mean = np.mean(grayscale_data)

                           if visual_brightness_mean >= self.brightness_threshold_high:
                                understanding_signals['is_bright'] = True
                                # logger.debug(f"UM.process: Ortam Parlak: {visual_brightness_mean:.4f} >= Eşik {self.brightness_threshold_high:.4f} (Algılandı: True)")

                           elif visual_brightness_mean <= self.brightness_threshold_low:
                                understanding_signals['is_dark'] = True
                                # logger.debug(f"UM.process: Ortam Karanlık: {visual_brightness_mean:.4f} <= Eşik {self.brightness_threshold_low:.4f} (Algılandı: True)")


            logger.debug(f"UnderstandingModule.process: Üretilen anlama sinyalleri: {understanding_signals}")

        except Exception as e:
            # Hesaplama sırasında beklenmedik bir hata olursa logla.
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

        return understanding_signals # Hesaplanan anlama sinyallerini döndür.

    def cleanup(self):
        """
        UnderstandingModule kaynaklarını temizler.

        Şimdilik özel bir kaynak kullanmadığı için temizleme adımı içermez.
        Gelecekte model temizliği veya bağlantı kapatma gerekebilir.
        module_loader.py bu metotu program sonlanırken çağrır (varsa).
        """
        logger.info("UnderstandingModule objesi temizleniyor.")
        pass