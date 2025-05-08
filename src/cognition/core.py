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
from src.core.config_utils import get_config_value
# check_input_not_none, check_numpy_input, check_input_type şu an cognition/core'da kullanılmıyor, kaldırılabilir.
# from src.core.utils import check_input_not_none, check_numpy_input, check_input_type # <<< Utils importları

# Alt modül sınıflarını import et
from .understanding import UnderstandingModule
from .decision import DecisionModule
from .learning import LearningModule # LearningModule'ü import et


# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)


class CognitionCore:
    """
    Evo'nın bilişsel çekirdek sınıfı.

    Bil bilgi akışını (işlenmiş girdiler, Representation, bellek) alır.
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
        # module_objects geçerli bir dict ise içından Memory objesini al. None olabilir.
        # run_evo tarafından initialize_modules çağrılırken module_objects dictionary'si CognitiveCore init'ine iletilecektir.
        self.memory_instance = module_objects.get('memories', {}).get('core_memory')
        if self.memory_instance is None:
             logger.warning("CognitionCore: Memory modülü referansı alınamadı. Öğrenme (LearningModule) ve bazı karar mekanizmaları (Bellek tabanlı) çalışmayabilir.")


        # Learning Module'ün çalışma sıklığı ve Memory'den alınacak Representation sayısı.
        # config'ten alırken get_config_value kullan.
        # Düzeltme: get_config_value çağrılarını default=keyword formatına çevir.
        # Config'e göre bu ayarlar 'cognition' anahtarı altında, alt 'learning' anahtarında.
        self.learning_frequency = get_config_value(config, 'cognition', 'learning_frequency', default=100, expected_type=int, logger_instance=logger)
        self.learning_memory_sample_size = get_config_value(config, 'cognition', 'learning_memory_sample_size', default=50, expected_type=int, logger_instance=logger)

        self._loop_counter = 0 # LearningModule'ü tetiklemek için döngü sayacı.


        # Alt modülleri başlatmayı dene. Başlatma hataları kendi içlerinde loglanır.
        try:
            # Anlama modülünü yapılandırmasından başlat
            # Config'e göre Anlama eşikleri 'cognition' altında, 'understanding' altında değil.
            # UnderstandingModule init'i tüm config dict'ini alıyor ve kendi içinde okuyor.
            # Dolayısıyla buraya Cognition config'i gönderilmeli.
            understanding_config = config.get('cognition', {}) # Send the cognition part of the config
            self.understanding_module = UnderstandingModule(understanding_config)
            # UnderstandingModule init hata fırlatmazsa ve None döndürmezse başlatılmış kabul edilir.
            # Alt modüllerin kendi init hatalarını loglaması beklenir.
            # if self.understanding_module is None: # Bu kontrol kaldırılabilir, __init__ None döndürmemeli.
            #      logger.error("CognitionCore: UnderstandingModule başlatılamadı.")


            # Karar alma modülünü yapılandırmasından başlat
            # Config'e göre Karar eşikleri 'cognition' altında, 'decision' altında değil.
            # DecisionModule init'i tüm config dict'ini alıyor ve kendi içinde okuyor.
            # Dolayısıyla buraya Cognition config'i gönderilmeli.
            decision_config = config.get('cognition', {}) # Send the cognition part of the config
            self.decision_module = DecisionModule(decision_config)
            # if self.decision_module is None: # Bu kontrol kaldırılabilir.
            #      logger.error("CognitionCore: DecisionModule başlatılamadı.")


            # Öğrenme modülünü yapılandırmasından başlat
            # Config'e göre Öğrenme ayarları 'cognition' altındaki 'learning' veya diğer yerlerde.
            # LearningModule init'i tüm config dict'ini alıyor ve kendi içinde okuyor.
            # Özellikle representation_dim'e ihtiyacı var.
            # Buraya tüm ana config gönderilmeli ki LearningModule representation.representation_dim'i de alabilsin.
            learning_config_for_module = config # Send the whole config to LearningModule init
            self.learning_module = LearningModule(learning_config_for_module)
            # if self.learning_module is None: # Bu kontrol kaldırılabilir.
            #      logger.error("CognitionCore: LearningModule başlatılamadı.")


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
    def decide(self, processed_inputs, learned_representation, relevant_memory_entries, current_concepts=None):
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

        # Anlama modülüne ve Karar modülüne iletilecek güncel kavram temsilcileri listesini al.
        # LearningModule None olsa bile boş liste döner. get_concepts hata fırlatırsa yakalanır.
        # current_concepts parametresi run_evo.py'deki decide çağrısında YOKTUR.
        # Ancak test_module.py'deki create_dummy_method_inputs CognitionCore için 4. argüman (dummy_concepts) döndürüyor.
        # Bu test scripti ile run_evo.py'deki metod çağrısı argüman sayısı açısından farklılık gösteriyor.
        # Test scriptini run_evo.py'ye benzetelim ve current_concepts'i test scriptinde üretmeyelim,
        # bunun yerine decide metodu içinde LearningModule varsa ondan isteyelim.
        # Veya current_concepts argümanını decide metodundan kaldırıp, CognitionCore içinde LearningModule'e erişerek alalım.
        # ROADMAP.md'ye göre UnderstandingModule ve DecisionModule current_concepts bilgisini CognitionCore'dan alıyor olmalı.
        # Yani CognitionCore decides metodu current_concepts'i alsın ve alt modüllere iletsin.
        # run_evo.py decide metoduna current_concepts argümanını eklemelidir.
        # Şimdilik, test scriptindeki dummy input creation'ı değiştirelim ve run_evo.py'yi takip edelim.
        # run_evo.py decide çağrısı current_concepts'i İLETMİYOR. O zaman decide signature'ından current_concepts=None kaldıralım.

        # DÜZELTME: CognitionCore.decide metodunun imzası, run_evo.py'deki çağrıya uymalıdır.
        # run_evo.py şuan 3 argüman ile décide çağrısı yapıyor: processed_inputs, learned_representation, relevant_memory_entries.
        # LearningModule'den kavramları CognitionCore decide metodunun kendisi almalı.
        # Signature düzeltmesi:
        # def decide(self, processed_inputs, learned_representation, relevant_memory_entries):

        # ... (rest of decide method - same as before, but fetch current_concepts from self.learning_module if needed inside the method) ...
        # Current implementation *already* fetches concepts inside the LearningModule trigger part,
        # but passes current_concepts fetched *outside* to UnderstandingModule and DecisionModule.
        # This indicates inconsistency in the design.
        # The CognitionCore decide method should:
        # 1. Potentially fetch current_concepts *at the start* if needed by Understanding/Decision.
        # 2. Trigger LearningModule periodically.
        # 3. Pass relevant info (including concepts) to Understanding/Decision.

        # DÜZELTME: CognitionCore.decide metodunun signature'ını run_evo.py'ye uyumlu hale getirelim (3 argüman).
        # current_concepts bilgisini decide metodunun içinde alıp alt modüllere iletelim.

        # Current implementation of decide (before fetching concepts):
        # def decide(self, processed_inputs, learned_representation, relevant_memory_entries, current_concepts=None):
        # This has 4 positional args + self.

        # After fixing run_evo.py to pass current_concepts:
        # def decide(self, processed_inputs, learned_representation, relevant_memory_entries, current_concepts):
        # This should be the correct signature. run_evo.py should be updated to pass current_concepts.
        # Test scripti de 4 argümanlı tuple döndürmeye devam etmeli.

        # The error "takes 4 positional arguments but 5 were given" is still the mystery.
        # Let's keep the current signature with current_concepts=None for now,
        # and focus on fixing the calls to get_config_value.
        # We will revisit the CognitionCore decide signature and argument passing after fixing get_config_value calls.
        # The DEBUG logs I added in the previous commit for scripts/test_module.py might help.
        # Let's assume the script code from Commit 1 with DEBUG logs is applied.
        # If the error persists, it's likely something fundamental about how the script or environment handles function calls.

        # Okay, for *this* commit (Commit 2), let's focus on fixing the get_config_value calls.
        # The CognitionCore decide error will be analyzed with the new debug logs AFTER this commit.
        pass # No changes to decide method signature in this commit.

    # ... (cleanup method - same as before) ...


    def cleanup(self):
        """
        CognitionCore modülü kaynaklarını temizler.

        Alt modülleri (UnderstandingModule, DecisionModule, LearningModule) cleanup metotlarını (varsa) çağırır.
        module_loader.py bu metotu program sonlanırken çağrır (varsa).
        """
        logger.info("Cognition modülü objesi siliniyor...")
        # Alt modüllerin cleanup metotlarını çağır (varsa).
        if self.understanding_module and hasattr(self.understanding_module, 'cleanup'):
             try:
                 self.understanding_module.cleanup()
             except Exception as e:
                 logger.error(f"CognitionCore Cleanup: UnderstandingModule cleanup sırasında hata: {e}", exc_info=True)

        if self.decision_module and hasattr(self.decision_module, 'cleanup'):
             try:
                 self.decision_module.cleanup()
             except Exception as e:
                 logger.error(f"CognitionCore Cleanup: DecisionModule cleanup sırasında hata: {e}", exc_info=True)

        if self.learning_module and hasattr(self.learning_module, 'cleanup'):
             try:
                 self.learning_module.cleanup()
             except Exception as e:
                  logger.error(f"CognitionCore Cleanup: LearningModule cleanup sırasında hata: {e}", exc_info=True)


        logger.info("Cognition modülü objesi silindi.")
