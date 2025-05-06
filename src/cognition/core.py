# src/cognition/core.py
#
# Evo'nın bilişsel çekirdeğini temsil eder.
# Gelen temsilleri, bellek girdilerini, işlenmiş anlık duyu özelliklerini kullanarak dünyayı anlamaya çalışır, kavramları öğrenir ve bir eylem kararı alır.
# UnderstandingModule, DecisionModule ve LearningModule alt modüllerini koordine eder.

import logging # Loglama için.
# numpy gerekli (parametre tipi için)
import numpy as np
import random # LearningModule için örneklem almak için

# Yardımcı fonksiyonları import et
from src.core.utils import check_input_not_none, check_numpy_input, check_input_type, get_config_value # <<< Utils importları

# Alt modül sınıflarını import et
from .understanding import UnderstandingModule
from .decision import DecisionModule
from .learning import LearningModule # LearningModule'ü import et


# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)


class CognitionCore:
    """
    Evo'nın bilişsel çekirdek sınıfı.

    Bilgi akışını (işlenmiş girdiler, Representation, bellek) alır.
    Bu bilgileri Anlama modülüne ileterek anlama çıktısını alır (dictionary sinyaller).
    Anlama çıktısını ve bellek girdilerini Karar Alma modülüne ileterek bir karar alır.
    Bellekteki Representationları kullanarak periyodik olarak Kavramları (LearningModule) öğrenir.
    UnderstandingModule, DecisionModule ve LearningModule alt modüllerini koordine eder.
    Hata durumlarında işlemleri loglar ve programın çökmesini engeller.
    """
    def __init__(self, config, module_objects): # module_objects dictionary'sini init sırasında al.
        """
        CognitionCore modülünü başlatır.

        Alt modülleri (UnderstandingModule, DecisionModule, LearningModule) başlatır.
        Memory modülü referansını saklar.
        Başlatma sırasında hata oluşursa alt modüllerin objeleri None kalabilir.

        Args:
            config (dict): Bilişsel çekirdek yapılandırma ayarları.
                           Alt modüllere ait ayarlar kendi adları altında beklenir
                           (örn: {'understanding': {...}, 'decision': {...}, 'learning': {...}}).
                           'learning_frequency': LearningModule'ün kaç bilişsel döngü adımında bir tetikleneceği (int, varsayılan 100).
                           'learning_memory_sample_size': LearningModule için Memory'den alınacak Representation sayısı (int, varsayılan 50).
        """
        self.config = config
        logger.info("Cognition modülü başlatılıyor...")

        self.understanding_module = None # Anlama modülü objesi.
        self.decision_module = None # Karar alma modülü objesi.
        self.learning_module = None # Öğrenme modülü objesi.

        # Memory modülü referansını sakla.
        # module_objects geçerli bir dict ise içinden Memory objesini al. None olabilir.
        # run_evo tarafından initialize_modules çağrılırken module_objects dictionary'si CognitiveCore init'ine iletilecektir.
        self.memory_instance = module_objects.get('memories', {}).get('core_memory')
        if self.memory_instance is None:
             logger.warning("CognitionCore: Memory modülü referansı alınamadı. Öğrenme (LearningModule) ve bazı karar mekanizmaları (Bellek tabanlı) çalışmayabilir.")


        # Learning Module'ün çalışma sıklığı ve Memory'den alınacak Representation sayısı.
        # config'ten alırken get_config_value kullan.
        self.learning_frequency = get_config_value(config, 'learning_frequency', 100, expected_type=int, logger_instance=logger)
        self.learning_memory_sample_size = get_config_value(config, 'learning_memory_sample_size', 50, expected_type=int, logger_instance=logger)

        self._loop_counter = 0 # LearningModule'ü tetiklemek için döngü sayacı.


        # Alt modülleri başlatmayı dene. Başlatma hataları kendi içlerinde loglanır.
        try:
            # Anlama modülünü yapılandırmasından başlat
            understanding_config = config.get('understanding', {})
            self.understanding_module = UnderstandingModule(understanding_config)
            if self.understanding_module is None:
                 logger.error("CognitionCore: UnderstandingModule başlatılamadı.")

            # Karar alma modülünü yapılandırmasından başlat
            decision_config = config.get('decision', {})
            self.decision_module = DecisionModule(decision_config)
            if self.decision_module is None:
                 logger.error("CognitionCore: DecisionModule başlatılamadı.")

            # Öğrenme modülünü yapılandırmasından başlat
            learning_config = config.get('learning', {})
            # Learning config'te representation_dim yoksa, üstteki representation anahtarı altındaki boyutu almayı dene.
            representation_config = config.get('representation', {})
            # get() ile güvenli erişim.
            learning_config['representation_dim'] = learning_config.get('representation_dim', representation_config.get('representation_dim', 128)) # Varsayılanı RL'den al

            self.learning_module = LearningModule(learning_config)
            if self.learning_module is None:
                 logger.error("CognitionCore: LearningModule başlatılamadı.")


        except Exception as e:
             # Alt modül başlatma sırasında beklenmedik hata olursa
             logger.critical(f"CognitionCore: Alt modülleri başlatılırken hata oluştu: {e}", exc_info=True)
             # Hata durumında alt modül objeleri None kalır.


        # Learning sıklığı ve örneklem boyutu kontrolü (negatif veya sıfır olmamalı)
        if self.learning_frequency <= 0:
             logger.warning(f"CognitionCore: Konfig 'learning_frequency' geçersiz ({self.learning_frequency}). Varsayılan 100 kullanılıyor.")
             self.learning_frequency = 100
        if self.learning_memory_sample_size <= 0:
             logger.warning(f"CognitionCore: Konfig 'learning_memory_sample_size' geçersiz ({self.learning_memory_sample_size}). Varsayılan 50 kullanılıyor.")
             self.learning_memory_sample_size = 50


        logger.info(f"Cognition modülü başlatıldı. Öğrenme Sıklığı: {self.learning_frequency} döngüde bir, Bellek Örneklem Boyutu: {self.learning_memory_sample_size}")


    # run_evo.py bu metodu çağırıyor. processed_inputs, learned_representation, relevant_memory_entries argümanlarını alıyor.
    def decide(self, processed_inputs, learned_representation, relevant_memory_entries):
        """
        İşlenmiş girdiler, öğrenilmiş temsil ve ilgili bellek girdilerine dayanarak bir eylem kararı alır.

        Gelen bilgileri önce anlama modülüne ileterek anlama çıktısını (dictionary sinyaller) alır.
        Anlama çıktısını ve bellek girdilerini Karar Alma modülüne ileterek bir karar alır.
        Periyodik olarak LearningModule'ü çağırarak kavramları öğrenir.

        Args:
            processed_inputs (dict or None): Processor modüllerinden gelen işlenmiş ham veriler.
                                            Beklenen format: {'visual': dict, 'audio': np.ndarray} veya None/boş dict.
            learned_representation (numpy.ndarray or None): RepresentationLearner'dan gelen en son öğrenilmiş temsil vektörü
                                                         veya işleme sırasında hata oluştuysa None.
                                                         Beklenen format: shape (D,), dtype sayısal, veya None.
            relevant_memory_entries (list or None): Memory modülünden gelen ilgili bellek girdileri listesi.
                                            Bellek boşsa veya sorgu sırasında hata oluştuysa boş liste `[]` olabilir.
                                            Beklenen format: list or None.

        Returns:
            str or None: Alınan karar (string)
                         veya anlama/karar alma sırasında hata durumunda None.
        """
        # Döngü sayacını artır.
        self._loop_counter += 1

        # LearningModule'ü tetikleme kontrolü. LearningModule ve Memory modülü varsa ve sıklık zamanı geldiyse.
        if self.learning_module is not None and self.memory_instance is not None and self._loop_counter % self.learning_frequency == 0:
             logger.info(f"CognitionCore: Öğrenme döngüsü tetiklendi (döngü #{self._loop_counter}).")
             try:
                  # Memory'den öğrenme için Representation örneklemi al.
                  # Memory modülünde get_all_representations metodu olmalı.
                  if hasattr(self.memory_instance, 'get_all_representations'):
                      # get_all_representations Representation vektörlerinin listesini döndürür.
                      all_memory_representations = self.memory_instance.get_all_representations()
                  else:
                      logger.warning("CognitionCore: Memory modülünde 'get_all_representations' metodu bulunamadı. LearningModule için Representation alınamadı.")
                      all_memory_representations = [] # Metot yoksa boş liste.

                  # Alınan Representation listesinin geçerli numpy array listesi olduğundan emin olalım.
                  # LearningModule'ün beklediği boyutta olanları alalım.
                  # issubtype yerine isinstance kullan (Hata düzeltme).
                  valid_representations_for_learning = [rep for rep in all_memory_representations if rep is not None and isinstance(rep, np.ndarray) and isinstance(rep.dtype, np.number) and rep.ndim == 1 and self.learning_module is not None and rep.shape[0] == self.learning_module.representation_dim]


                  if valid_representations_for_learning:
                       # Öğrenme için rastgele bir örneklem alalım (eğer bellek çok büyükse).
                       learning_sample = random.sample(valid_representations_for_learning, min(self.learning_memory_sample_size, len(valid_representations_for_learning)))
                       logger.debug(f"CognitionCore: Memory'den öğrenme için {len(learning_sample)} representation örneği alındı.")
                       # LearningModule'ü Representation örneklemi ile çağır.
                       self.learning_module.learn_concepts(learning_sample)
                  else:
                       logger.debug("CognitionCore: Memory'de öğrenme için yeterli geçerli Representation yok.")

             except Exception as e:
                  logger.error(f"CognitionCore: Öğrenme döngüsü sırasında bellekten veri alınırken veya LearningModule çağrılırken beklenmedik hata: {e}", exc_info=True)

        # CognitionCore'un alt modülleri başlatılamamışsa işlem yapma.
        if self.understanding_module is None or self.decision_module is None:
            logger.error("CognitionCore.decide: Alt modüller (Understanding/Decision) başlatılmamış. Karar alınamıyor.")
            return None

        # Anlama modülüne iletilecek güncel kavram temsilcileri listesini al.
        current_concepts = self.learning_module.get_concepts() if self.learning_module else []
        # Eğer LearningModule None ise veya get_concepts None/boş liste dönerse current_concepts boş liste olur.


        understanding_signals = None # Anlama modülünden gelecek sinyaller dictionary'si.
        decision = None # Karar alma modülünden gelecek karar.

        try:
            # 1. Gelen bilgileri anlama modülüne ilet.
            # UnderstandingModule.process dictionary sinyalleri döndürür.
            # processed_inputs, learned_representation, relevant_memory_entries VE current_concepts argümanlarını UnderstandingModule'e iletiyoruz.
            understanding_signals = self.understanding_module.process(
                processed_inputs, # İşlenmiş anlık duyu girdileri (dict/None)
                learned_representation, # Learned Representation (None/array)
                relevant_memory_entries, # Memory'den gelen ilgili anılar (list/None)
                current_concepts # LearningModule'den gelen güncel kavramlar (list) - YENİ
            )
            # DEBUG logu: Anlama sinyalleri dictionary'si (UnderstandingModule içinde loglanıyor)
            # if isinstance(understanding_signals, dict): ...


            # 2. Anlama çıktısını ve bellek girdilerini Karar alma modülüne ilet.
            # DecisionModule.decide understanding_signals (dict/None) ve relevant_memory_entries (list/None) bekler.
            # Kendi içsel durumunu (curiosity) DecisionModule yönetiyor. current_concepts burada kullanılmıyor.
            decision = self.decision_module.decide(
                understanding_signals, # Anlama modülünün çıktısı (dictionary sinyaller)
                relevant_memory_entries # Bellek girdileri (list/None) - Karar modülü bunu bağlamsal karar için kullanabilir
                # internal_state # Gelecekte buradan da iletilebilir.
                # current_concepts # Gelecekte DecisionModule doğrudan kullanabilir.
            )

            # DEBUG logu: Karar sonucu (DecisionModule içinde loglanıyor)
            # if decision is not None: ...


        except Exception as e:
            # Alt modüllerin metotlarını çağırırken veya içlerinde (eğer yakalamadılarsa) beklenmedik hata olursa.
            logger.error(f"CognitionCore.decide: Anlama veya karar alma sırasında beklenmedik hata: {e}", exc_info=True)
            return None # Hata durumunda None döndür.

        # Başarılı durumda alınan kararı döndür.
        return decision

    def cleanup(self):
        """
        CognitionCore modülü kaynaklarını temizler.

        Alt modülleri (UnderstandingModule, DecisionModule, LearningModule) cleanup metotlarını (varsa) çağırır.
        module_loader.py bu metotu program sonlanırken çağrır (varsa).
        """
        logger.info("Cognition modülü objesi siliniyor...")
        # Alt modüllerin cleanup metotlarını çağır (varsa).
        if self.understanding_module and hasattr(self.understanding_module, 'cleanup'):
             self.understanding_module.cleanup()
        if self.decision_module and hasattr(self.decision_module, 'cleanup'):
             self.decision_module.cleanup()
        if self.learning_module and hasattr(self.learning_module, 'cleanup'):
             self.learning_module.cleanup()


        logger.info("Cognition modülü objesi silindi.")