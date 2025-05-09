# src/cognition/core.py
# ... (importlar aynı) ...
import logging
import numpy as np
import random 

from src.core.config_utils import get_config_value
from .understanding import UnderstandingModule
from .decision import DecisionModule
from .learning import LearningModule 

logger = logging.getLogger(__name__)

class CognitionCore:
    def __init__(self, config, module_objects):
        self.config = config 
        logger.info("Cognition module initializing...")

        self.understanding_module = None; self.decision_module = None; self.learning_module = None
        
        self.memory_instance = module_objects.get('memories', {}).get('core_memory')
        if self.memory_instance is None:
             logger.warning("CognitionCore: Memory module reference not obtained. Learning may not function.")

        # Öğrenme ayarlarını cognition.learning altından al
        learning_cfg = config.get('cognition', {}).get('learning', {})
        self.learning_frequency = get_config_value(learning_cfg, 'learning_frequency', default=100, expected_type=int)
        self.learning_memory_sample_size = get_config_value(learning_cfg, 'learning_memory_sample_size', default=50, expected_type=int)
        self._loop_counter = 0

        try:
            self.understanding_module = UnderstandingModule(config)
            self.decision_module = DecisionModule(config)
            self.learning_module = LearningModule(config) # LearningModule kendi config'ini alır
        except Exception as e:
             logger.critical(f"CognitionCore: Error during sub-module initialization: {e}", exc_info=True)

        if self.learning_frequency <= 0: self.learning_frequency = 100; logger.warning("Invalid learning_frequency, using 100.")
        if self.learning_memory_sample_size <= 0: self.learning_memory_sample_size = 50; logger.warning("Invalid learning_memory_sample_size, using 50.")
        
        logger.info(f"Cognition module initialized. Learning Freq: {self.learning_frequency}, Mem Sample: {self.learning_memory_sample_size}")

    def decide(self, processed_inputs, learned_representation, relevant_memory_entries, current_concepts_ignored):
        # current_concepts_ignored: run_evo'dan gelen bu argümanı artık kullanmıyoruz,
        # çünkü LearningModule kavramları kendi içinde yönetiyor ve UnderstandingModule'a veriyor.
        self._loop_counter += 1

        if self.learning_module is not None and self.memory_instance is not None and \
           self._loop_counter % self.learning_frequency == 0:
             logger.info(f"CognitionCore: Learning cycle triggered (cycle #{self._loop_counter}).")
             try:
                  if hasattr(self.memory_instance, 'get_all_representations'):
                      # LTM'den veya WM'den örneklem al (Memory modülünün kendi mantığına göre)
                      # sample_size'ı LearningModule config'inden alalım
                      sample_size_for_learning = self.learning_memory_sample_size # Config'den alınmıştı
                      
                      # Öğrenme için temsilleri LTM'den almayı tercih et
                      representations_for_learning = self.memory_instance.get_all_representations(
                          from_ltm=True, 
                          sample_size=sample_size_for_learning
                      )
                      
                      # Eğer LTM'den yeterli gelmediyse veya LTM boşsa, WM'den de almayı dene (opsiyonel)
                      if not representations_for_learning or len(representations_for_learning) < sample_size_for_learning // 2:
                          logger.info("LTM provided insufficient/no samples, trying Working Memory for learning.")
                          wm_reps = self.memory_instance.get_all_representations(
                              from_ltm=False, 
                              sample_size=sample_size_for_learning - len(representations_for_learning)
                          )
                          representations_for_learning.extend(wm_reps)
                          # Tekrarları kaldır (önemli)
                          unique_reps_tuples = {tuple(rep.tolist()) for rep in representations_for_learning if rep is not None}
                          representations_for_learning = [np.array(t_rep) for t_rep in unique_reps_tuples]


                      if representations_for_learning:
                           logger.debug(f"CognitionCore: Providing {len(representations_for_learning)} unique representations to LearningModule.")
                           self.learning_module.learn_concepts(representations_for_learning)
                      else:
                           logger.debug("CognitionCore: No representations available from Memory for learning.")
                  else:
                      logger.warning("CognitionCore: Memory module missing 'get_all_representations'. Cannot learn.")
             except Exception as e:
                  logger.error(f"CognitionCore: Error during learning cycle: {e}", exc_info=True)

        if self.understanding_module is None or self.decision_module is None:
            logger.error("CognitionCore.decide: Critical sub-modules not initialized. Cannot decide.")
            return None

        # Güncel kavramları doğrudan LearningModule'den alalım
        current_concepts_data = []
        if self.learning_module:
            current_concepts_data = self.learning_module.get_concepts() # Bu artık [{'id':..., 'vector':...}, ...] listesi döndürür

        understanding_signals = None; decision = None
        try:
            understanding_signals = self.understanding_module.process(
                processed_inputs, 
                learned_representation, 
                relevant_memory_entries, 
                current_concepts_data # Güncellenmiş kavram listesi
            )
            decision = self.decision_module.decide(
                understanding_signals, 
                relevant_memory_entries, 
                current_concepts_data # Güncellenmiş kavram listesi
            )
        except Exception as e:
            logger.error(f"CognitionCore.decide: Error in understanding/decision: {e}", exc_info=True)
            return None

        if decision: logger.debug(f"CognitionCore.decide: Final decision: '{decision}'.")
        else: logger.debug("CognitionCore.decide: No specific decision.")
        return decision

    def cleanup(self):
        logger.info("CognitionCore cleaning up...")
        if self.understanding_module and hasattr(self.understanding_module, 'cleanup'): self.understanding_module.cleanup()
        if self.decision_module and hasattr(self.decision_module, 'cleanup'): self.decision_module.cleanup()
        if self.learning_module and hasattr(self.learning_module, 'cleanup'): self.learning_module.cleanup()
        logger.info("CognitionCore cleaned up.")