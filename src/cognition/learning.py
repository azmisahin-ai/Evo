# src/cognition/learning.py
#
# Evo'nın denetimsiz öğrenme (kavram keşfi) modülünü temsil eder.
# Bellekteki Representation vektörlerini analiz ederek temel desenleri/kümeleri (kavramları) keşfeder.

import logging
import numpy as np # Vektör işlemleri ve benzerlik hesaplaması için

# Yardımcı fonksiyonları import et (girdi kontrolleri ve config için)
from src.core.utils import check_input_not_none, check_input_type, check_numpy_input, get_config_value # <<< Utils importları

# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)

class LearningModule:
    """
    Evo'nın denetimsiz öğrenme (kavram keşfi) yeteneğini sağlayan sınıf (Faz 4 implementasyonu).

    Bellekteki Representation vektörlerini alır ve basit bir kümeleme benzeri mantık uygulayarak
    temel kavramları (desen gruplarını) keşfeder ve öğrenir. Keşfedilen kavramlar,
    kavram temsilcisi vektörleri (centroidler gibi) olarak saklanır.
    "From scratch" prensibiyle, karmaşık kütüphaneler olmadan basit bir yaklaşım kullanılır.
    """
    def __init__(self, config):
        """
        LearningModule'ü başlatır.

        Args:
            config (dict): Öğrenme modülü yapılandırma ayarları.
                           'new_concept_threshold': Bir Representation vektörünün yeni bir kavram
                                                    sayılması için mevcut kavram temsilcilerinden
                                                    en az benzemesi gereken benzerlik eşiği (float, varsayılan 0.7).
                                                    Bu değer 1.0'a yakınsa, sadece çok farklı olanlar yeni kavram olur.
                                                    0.0'a yakınsa, çoğu şey yeni kavram olur.
                           'representation_dim': İşlenecek Representation vektörlerinin boyutu (int).
                                                  Bu, RepresentationLearner'ın çıktısı ile aynı olmalıdır.
                                                  Config'ten veya RepresentationLearner'dan alınabilir.
                                                  Burada RepresentationLearner'ın output_dim'ine göre belirleneceğini varsayalım.
                                                  Şimdilik config'ten alalım.
        """
        self.config = config
        logger.info("LearningModule başlatılıyor (Faz 4)...")

        # Yapılandırmadan eşikleri alırken get_config_value kullan.
        self.new_concept_threshold = get_config_value(config, 'new_concept_threshold', 0.7, expected_type=(float, int), logger_instance=logger)
        # Representation boyutunu config'ten al (RepresentationLearner ile uyumlu olmalı).
        # Eğer config'te yoksa, varsayılan RepresentationLearner boyutunu (128) kullanalım.
        self.representation_dim = get_config_value(config, 'representation_dim', 128, expected_type=int, logger_instance=logger)


        # Eşik değeri için basit değer kontrolü (0.0 ile 1.0 arası)
        if not (0.0 <= self.new_concept_threshold <= 1.0):
             logger.warning(f"LearningModule: Konfig 'new_concept_threshold' beklenmeyen aralıkta ({self.new_concept_threshold}). 0.0 ile 1.0 arası bekleniyordu. Varsayılan 0.7 kullanılıyor.")
             self.new_concept_threshold = 0.7

        # Representation boyutu pozitif mi kontrol et.
        if self.representation_dim <= 0:
             logger.error(f"LearningModule: Konfig 'representation_dim' geçersiz ({self.representation_dim}). Pozitif bir değer bekleniyordu. Başlatma başarısız olabilir.")
             # Başlatma sırasında kritik hata vermiyoruz, sadece logluyoruz.


        # Öğrenilen kavramların temsilcilerini (vektörlerini) saklayacak liste.
        # Her kavramın bir ID'si (listenin indeksi) ve temsilci vektörü olacak.
        self.concept_representatives = [] # Liste elementleri: numpy array (shape (representation_dim,))

        logger.info(f"LearningModule başlatıldı. Yeni Kavram Eşiği: {self.new_concept_threshold}, Representation Boyutu: {self.representation_dim}. Öğrenilen Kavram Sayısı: {len(self.concept_representatives)}")


    def learn_concepts(self, representation_list):
        """
        Verilen Representation vektör listesini kullanarak yeni kavramları keşfeder veya mevcutları günceller.

        "From scratch" basit implementasyon: Listedeki her Representation vektörünü alır.
        Eğer bu vektör, mevcut kavram temsilcilerinden hiçbirine yeterince benzemiyorsa
        (benzerlik self.new_concept_threshold eşiğinin altındaysa), bu vektörü
        yeni bir kavram temsilcisi olarak ekler. Mevcut temsilcileri güncellemez.

        Args:
            representation_list (list or None): Memory'den gelen Representation vektörlerinin listesi.
                                                Her öğe Representation vektörü (numpy array shape (D,)) veya None olabilir.
                                                Liste boş veya None olabilir.

        Returns:
            list: Güncellenmiş kavram temsilcileri listesi (numpy array'ler).
                  Hata durumunda mevcut listeyi döndürür.
        """
        # Girdi kontrolü. representation_list'in geçerli bir liste mi?
        if not check_input_not_none(representation_list, input_name="representation_list for LearningModule", logger_instance=logger):
             logger.debug("LearningModule.learn_concepts: representation_list None. Kavram öğrenme atlandi.")
             return self.concept_representatives # Geçersiz girdi ise mevcut listeyi döndür.

        if not isinstance(representation_list, list): # List olduğundan emin ol
             logger.error(f"LearningModule.learn_concepts: representation_list liste değil ({type(representation_list)}). Kavram öğrenme atlandi.")
             return self.concept_representatives # Geçersiz tip ise mevcut listeyi döndür.

        if not representation_list:
             logger.debug("LearningModule.learn_concepts: representation_list boş liste. Kavram öğrenme atlandi.")
             return self.concept_representatives # Boş liste ise mevcut listeyi döndür.


        logger.debug(f"LearningModule.learn_concepts: {len(representation_list)} representation vektörü öğrenme için alındı. Mevcut {len(self.concept_representatives)} kavram var.")

        try:
            # Gelen her Representation vektörünü işle.
            for rep_vector in representation_list:
                # Representation vektörünün geçerli (numpy array, 1D, sayısal, doğru boyut) olduğundan emin ol.
                # check_numpy_input ile genel kontrol, sonra spesifik boyut kontrolü.
                if rep_vector is not None and check_numpy_input(rep_vector, expected_dtype=np.number, expected_ndim=1, input_name="item in representation_list", logger_instance=logger):
                    if rep_vector.shape[0] == self.representation_dim:
                        # Vektörün normalizasyonunu hesapla (benzerlik için).
                        rep_norm = np.linalg.norm(rep_vector)

                        # Eğer vektör sıfır değilse (anlamlıysa)
                        if rep_norm > 1e-8:
                            # Bu vektörün mevcut kavram temsilcilerine olan en yüksek benzerliğini bul.
                            max_similarity_to_concepts = 0.0

                            if self.concept_representatives: # Öğrenilmiş kavram varsa benzerlik hesapla.
                                for concept_rep in self.concept_representatives:
                                     # Kavram temsilcisinin de geçerli olduğundan emin ol (teorik olarak her zaman olmalı).
                                     concept_norm = np.linalg.norm(concept_rep)
                                     if concept_norm > 1e-8:
                                          similarity = np.dot(rep_vector, concept_rep) / (rep_norm * concept_norm)
                                          if not np.isnan(similarity):
                                               max_similarity_to_concepts = max(max_similarity_to_concepts, similarity)
                                     # else: logger.debug("LearningModule.learn_concepts: Concept rep near zero norm, skipping similarity.")

                                # logger.debug(f"LearningModule.learn_concepts: Vektör için en yüksek kavram benzerliği: {max_similarity_to_concepts:.4f}")

                            # Eğer en yüksek benzerlik eşiğin altındaysa, yeni kavram olarak ekle.
                            if max_similarity_to_concepts < self.new_concept_threshold:
                                # Yeni kavram temsilcisi olarak mevcut Representation vektörünü ekle.
                                self.concept_representatives.append(rep_vector) # Numpy array'in kendisini kopyalayarak saklamak daha güvenli olabilir. .copy()
                                logger.info(f"LearningModule.learn_concepts: Yeni kavram keşfedildi! Toplam kavram: {len(self.concept_representatives)}")
                                logger.debug(f"LearningModule.learn_concepts: Yeni kavram vektörü shape: {rep_vector.shape}, dtype: {rep_vector.dtype}. En yüksek mevcut benzerlik: {max_similarity_to_concepts:.4f} < Eşik {self.new_concept_threshold:.4f}")
                            # else: logger.debug(f"LearningModule.learn_concepts: Vektör yeterince tanıdık (benzerlik {max_similarity_to_concepts:.4f} >= eşik {self.new_concept_threshold:.4f}). Yeni kavram eklenmedi.")

                        # else: logger.debug("LearningModule.learn_concepts: Representation vektörü sıfır norm, işlenmedi.")
                    # else: logger.warning(f"LearningModule.learn_concepts: Representation vektörü yanlış boyut ({rep_vector.shape[0]} vs {self.representation_dim}), işlenmedi.")
                # else: logger.warning("LearningModule.learn_concepts: Listedeki öğe geçerli Representation vektörü değil, işlenmedi.")


            logger.debug(f"LearningModule.learn_concepts: Kavram öğrenme tamamlandı. Güncel kavram sayısı: {len(self.concept_representatives)}")


        except Exception as e:
            # Öğrenme işlemi sırasında beklenmedik bir hata olursa logla.
            logger.error(f"LearningModule.learn_concepts: Kavram öğrenme sırasında beklenmedik hata: {e}", exc_info=True)
            # Hata durumunda mevcut kavram listesini döndür.

        return self.concept_representatives # Güncellenmiş kavram temsilcileri listesini döndür.


    def get_concepts(self):
        """
        Öğrenilmiş kavram temsilcileri listesini döndürür.

        Returns:
            list: Öğrenilmiş kavram temsilcileri numpy array listesi.
        """
        # Güvenlik için listenin bir kopyasını döndürmek daha iyi olabilir. .copy()
        return self.concept_representatives # Listeyi referans olarak döndürür.


    def cleanup(self):
        """
        LearningModule kaynaklarını temizler.

        Öğrenilen kavramlar listesini temizler.
        Gelecekte öğrenilmiş model parametrelerini kaydetme/temizleme gerekebilir.
        module_loader.py bu metotu program sonlanırken çağırır (varsa).
        """
        logger.info("LearningModule objesi siliniyor...")
        # Öğrenilen kavramları kaydetme mantığı buraya gelebilir (gelecek TODO).
        # self._save_concepts() # Gelecek TODO

        # Öğrenilen kavramlar listesini temizle.
        self.concept_representatives = [] # Veya None
        logger.info("LearningModule: Öğrenilen kavramlar temizlendi.")
        pass