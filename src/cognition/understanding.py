# src/cognition/understanding.py
#
# Evo'nın anlama modülünü temsil eder.
# İşlenmiş temsilleri, bellek girdilerini ve anlık duyu özelliklerini kullanarak dünyayı anlamaya çalışır.

import logging
import numpy as np # Benzerlik hesaplaması ve array işlemleri için

# Yardımcı fonksiyonları import et (girdi kontrolleri için)
from src.core.utils import check_input_not_none, check_input_type, check_numpy_input, get_config_value # <<< Utils importları

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

class UnderstandingModule:
    """
    Evo'nın anlama yeteneğini sağlayan sınıf (Faz 3 implementasyonu).

    RepresentationLearner'dan gelen Representation'ı, Memory'den gelen ilgili anıları
    ve anlık Process çıktılarını (düşük seviyeli özellikler) alır.
    Bu girdileri kullanarak ilkel bir "anlama" çıktısı (şimdilik bir dictionary) üretir.
    Mevcut implementasyon: En yüksek bellek benzerlik skorunu hesaplar VE anlık
    ses enerjisi/görsel kenar yoğunluğuna dayalı basit boolean flag'ler üretir.
    Gelecekte daha karmaşık anlama algoritmaları implement edilecektir.
    """
    def __init__(self, config):
        """
        UnderstandingModule'ü başlatır.

        Args:
            config (dict): Anlama modülü yapılandırma ayarları.
                           'audio_energy_threshold': Yüksek ses enerjisi algılama eşiği (float, varsayılan 1000.0).
                           'visual_edges_threshold': Yüksek görsel kenar yoğunluğu algılama eşiği (float, varsayılan 50.0).
                           Gelecekte model yolları, anlama stratejileri gibi ayarlar gelebilir.
        """
        self.config = config
        logger.info("UnderstandingModule başlatılıyor (Faz 3)...")

        # Yapılandırmadan Processor özellik eşiklerini alırken get_config_value kullan.
        self.audio_energy_threshold = get_config_value(config, 'audio_energy_threshold', 1000.0, expected_type=(float, int), logger_instance=logger)
        self.visual_edges_threshold = get_config_value(config, 'visual_edges_threshold', 50.0, expected_type=(float, int), logger_instance=logger)

        # Eşik değerleri için basit değer kontrolü (negatif olmamalı)
        if self.audio_energy_threshold < 0.0:
             logger.warning(f"UnderstandingModule: Konfigurasyonda ses enerji eşiği negatif ({self.audio_energy_threshold}). Varsayılan 1000.0 kullanılıyor.")
             self.audio_energy_threshold = 1000.0
        if self.visual_edges_threshold < 0.0:
             logger.warning(f"UnderstandingModule: Konfigurasyonda görsel kenar eşiği negatif ({self.visual_edges_threshold}). Varsayılan 50.0 kullanılıyor.")
             self.visual_edges_threshold = 50.0

        logger.info(f"UnderstandingModule başlatıldı. Ses Enerji Eşiği: {self.audio_energy_threshold}, Görsel Kenar Eşiği: {self.visual_edges_threshold}")


    # run_evo.py ve CognitionCore.decide'ın processed_inputs'u iletmesi için parametre tanımına ekledik.
    def process(self, processed_inputs, learned_representation, relevant_memory_entries):
        """
        Gelen Representation, ilgili bellek girdileri ve anlık Process çıktılarını kullanarak anlama işlemini yapar.

        Mev mevcut implementasyon: Learned Representation ile ilgili anılar arasındaki en yüksek
        vektör benzerlik skorunu hesaplar VE anlık ses enerjisi/görsel kenar yoğunluğuna
        dayalı basit boolean flag'ler üretir. Sonuçları bir dictionary olarak döndürür.

        Args:
            processed_inputs (dict or None): Processor modüllerinden gelen işlenmiş ham veriler.
                                            Beklenen format: {'visual': dict, 'audio': np.ndarray} veya None/boş dict.
            learned_representation (numpy.ndarray or None): En son öğrenilmiş temsil vektörü.
                                                         Beklenen format: shape (D,), dtype sayısal, veya None.
            relevant_memory_entries (list or None): Memory modülünden gelen ilgili bellek girdileri listesi.
                                            Her öğe {'representation': np.ndarray, ...} formatında olmalıdır.
                                            Boş veya None olabilir.

        Returns:
            dict: Anlama sinyallerini içeren bir dictionary.
                  'similarity_score': float (0.0-1.0 arası, en yüksek bellek benzerliği)
                  'high_audio_energy': bool (Ses enerji eşiği aşıldı mı?)
                  'high_visual_edges': bool (Görsel kenar eşiği aşıldı mı?)
                  Hata durumunda veya geçerli girdi yoksa varsayılan değerlerle döndürülür (örn: {'similarity_score': 0.0, 'high_audio_energy': False, 'high_visual_edges': False}).
        """
        # Varsayılan anlama çıktısı dictionary'si
        understanding_signals = {
            'similarity_score': 0.0,
            'high_audio_energy': False,
            'high_visual_edges': False,
        }

        # Girdi kontrolleri.
        # relevant_memory_entries'in geçerli bir liste mi?
        is_valid_memory_list = relevant_memory_entries is not None and check_input_type(relevant_memory_entries, list, input_name="relevant_memory_entries for UnderstandingModule", logger_instance=logger)
        if not is_valid_memory_list:
             logger.debug("UnderstandingModule.process: relevant_memory_entries None veya liste değil.") # Devam ediyoruz, sadece bellek tabanlı anlama çalışmayacak.

        # Representation geçerli mi? (Benzerlik hesaplaması için)
        is_valid_representation = learned_representation is not None and check_numpy_input(learned_representation, expected_dtype=np.number, expected_ndim=1, input_name="learned_representation for UnderstandingModule", logger_instance=logger)
        if not is_valid_representation:
             logger.debug("UnderstandingModule.process: Learned representation None veya geçersiz. Benzerlik skoru hesaplanamayacak.") # Devam ediyoruz, sadece bellek tabanlı anlama çalışmayacak.

        # Processed inputs geçerli bir dictionary mi?
        is_valid_processed_inputs = processed_inputs is not None and isinstance(processed_inputs, dict)
        if not is_valid_processed_inputs:
             logger.debug("UnderstandingModule.process: Processed inputs None veya dict değil. Process tabanlı anlama çalışmayacak.") # Devam ediyoruz, sadece Process tabanlı anlama çalışmayacak.


        logger.debug(f"UnderstandingModule.process: Anlama işlemi başlatıldı. Girdi geçerliliği - Repr:{is_valid_representation}, Mem:{is_valid_memory_list}, Proc:{is_valid_processed_inputs}")

        try:
            # 1. Bellek Benzerlik Skoru Hesapla (Eğer Representation ve Bellek geçerliyse)
            max_similarity = 0.0
            if is_valid_representation and is_valid_memory_list and relevant_memory_entries: # Bellek listesi de boş olmamalı
                query_norm = np.linalg.norm(learned_representation)
                if query_norm > 1e-8: # Sorgu vektörü sıfır değilse
                    for memory_entry in relevant_memory_entries:
                        stored_representation = memory_entry.get('representation')
                        if stored_representation is not None and isinstance(stored_representation, np.ndarray) and np.issubdtype(stored_representation.dtype, np.number) and stored_representation.ndim == 1:
                            stored_norm = np.linalg.norm(stored_representation)
                            if stored_norm > 1e-8:
                                similarity = np.dot(learned_representation, stored_representation) / (query_norm * stored_norm)
                                if not np.isnan(similarity):
                                    max_similarity = max(max_similarity, similarity)
                            # else: logger.debug("UM.process: Stored rep near zero norm, skipping.")
                        # else: logger.debug("UM.process: Invalid stored rep, skipping.")
                # else: logger.warning("UM.process: Query rep near zero norm, cannot calc similarity.")

            understanding_signals['similarity_score'] = max_similarity
            logger.debug(f"UnderstandingModule.process: En yüksek bellek benzerlik skoru: {max_similarity:.4f}")


            # 2. Process Çıktıları Tabanlı Basit Anlama (Eğer Processed Inputs geçerliyse)
            if is_valid_processed_inputs:
                 # Ses Enerjisi Kontrolü
                 audio_features = processed_inputs.get('audio') # AudioProcessor'dan (2,) float32 numpy array bekleniyor veya None.
                 if isinstance(audio_features, np.ndarray) and audio_features.ndim == 1 and audio_features.shape[0] >= 1 and np.issubdtype(audio_features.dtype, np.number):
                      # İlk elemanın (Enerji) sayısal ve eşik üzerinde olup olmadığını kontrol et.
                      audio_energy = float(audio_features[0]) # Enerji float/int olabilir, float'a çevir.
                      if audio_energy >= self.audio_energy_threshold:
                           understanding_signals['high_audio_energy'] = True
                           logger.debug(f"UnderstandingModule.process: Yüksek ses enerjisi algılandı: {audio_energy:.4f} >= Eşik {self.audio_energy_threshold:.4f}")
                      # else: logger.debug(f"UnderstandingModule.process: Ses enerjisi eşik altında: {audio_energy:.4f} < Eşik {self.audio_energy_threshold:.4f}")
                 # else: logger.debug("UnderstandingModule.process: Geçersiz audio features inputu.")


                 # Görsel Kenar Yoğunluğu Kontrolü
                 visual_features = processed_inputs.get('visual') # VisionProcessor'dan dict {'grayscale': array, 'edges': array} bekleniyor veya None/boş dict.
                 if isinstance(visual_features, dict):
                      edges_data = visual_features.get('edges') # 'edges' array'ini al.
                      if isinstance(edges_data, np.ndarray) and edges_data.ndim >= 1 and np.issubdtype(edges_data.dtype, np.number):
                           # Kenar array'inin ortalama piksel değerini hesapla (0-255 arası değerler)
                           # Boş array gelme ihtimaline karşı size kontrolü ekle.
                           if edges_data.size > 0:
                                visual_edges_mean = np.mean(edges_data)
                                if visual_edges_mean >= self.visual_edges_threshold:
                                     understanding_signals['high_visual_edges'] = True
                                     logger.debug(f"UnderstandingModule.process: Yüksek görsel kenar yoğunluğu algılandı: {visual_edges_mean:.4f} >= Eşik {self.visual_edges_threshold:.4f}")
                                # else: logger.debug(f"UnderstandingModule.process: Görsel kenar yoğunluğu eşik altında: {visual_edges_mean:.4f} < Eşik {self.visual_edges_threshold:.4f}")
                           # else: logger.debug("UnderstandingModule.process: Boş edges array, görsel kenar yoğunluğu hesaplanamadi.")
                      # else: logger.debug("UnderstandingModule.process: Geçersiz edges inputu.")
                 # else: logger.debug("UnderstandingModule.process: Geçersiz visual features inputu.")


            logger.debug(f"UnderstandingModule.process: Anlama sinyalleri üretildi: {understanding_signals}")

        except Exception as e:
            # Hesaplama sırasında beklenmedik bir hata olursa logla.
            logger.error(f"UnderstandingModule.process: Anlama işlemi sırasında beklenmedik hata: {e}", exc_info=True)
            # Hata durumunda varsayılan anlama sinyallerini döndür
            return {
                'similarity_score': 0.0,
                'high_audio_energy': False,
                'high_visual_edges': False,
            }

        return understanding_signals # Hesaplanan anlama sinyallerini döndür.

    def cleanup(self):
        """
        UnderstandingModule kaynaklarını temizler.

        Şimdilik özel bir kaynak kullanmadığı için temizleme adımı içermez.
        Gelecekte model temizliği veya bağlantı kapatma gerekebilir.
        module_loader.py bu metodu program sonlanırken çağırır (varsa).
        """
        logger.info("UnderstandingModule objesi temizleniyor.")
        pass