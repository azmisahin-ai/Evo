# src/memory/core.py
import numpy as np
import time
import logging
import pickle
import os
import random # LTM'den örnekleme için

from src.core.utils import run_safely, cleanup_safely
from src.core.config_utils import get_config_value

logger = logging.getLogger(__name__)

class Memory:
    def __init__(self, full_config):
        self.config = full_config.get('memory', {})
        self.representation_config = full_config.get('representation', {})
        logger.info("Memory module initializing...")

        # Çalışma Belleği Ayarları
        self.max_working_memory_size = get_config_value(self.config, 'max_working_memory_size', default=200, expected_type=int)
        self.num_retrieved_working_memories = get_config_value(self.config, 'num_retrieved_memories', default=5, expected_type=int) # Bu hala çalışma belleği için
        
        # Uzun Süreli Bellek (LTM) Ayarları
        self.enable_ltm_persistence = get_config_value(self.config, 'enable_ltm_persistence', default=True, expected_type=bool)
        self.ltm_file_path = get_config_value(self.config, 'ltm_file_path', default='data/long_term_memory.pkl', expected_type=str)
        self.max_ltm_size = get_config_value(self.config, 'max_ltm_size', default=5000, expected_type=int)

        self.representation_dim = get_config_value(self.representation_config, 'representation_dim', default=128, expected_type=int)

        # Değer kontrolleri
        if self.max_working_memory_size <= 0: self.max_working_memory_size = 200; logger.warning("Invalid max_working_memory_size, using 200.")
        if self.num_retrieved_working_memories < 0: self.num_retrieved_working_memories = 5; logger.warning("Invalid num_retrieved_working_memories, using 5.")
        if self.max_ltm_size < 0: self.max_ltm_size = 5000; logger.warning("Invalid max_ltm_size, using 5000 (0 for unlimited).")
        if self.representation_dim <= 0: self.representation_dim = 128; logger.warning("Invalid representation_dim, using 128.")

        self.working_memory = [] # Kısa süreli, aktif anılar
        self.long_term_memory = [] # Uzun süreli, daha kalıcı anılar

        if self.enable_ltm_persistence:
            self._load_ltm()
        
        logger.info(f"Memory initialized. WorkingMem Max: {self.max_working_memory_size}, LTM Max: {'Unlimited' if self.max_ltm_size == 0 else self.max_ltm_size}, LTM File: {self.ltm_file_path if self.enable_ltm_persistence else 'Disabled'}")
        logger.info(f"Initial LTM size: {len(self.long_term_memory)}")

    def _load_ltm(self):
        if not self.ltm_file_path or not os.path.exists(self.ltm_file_path):
            logger.info(f"LTM file not found: {self.ltm_file_path}. Initializing LTM as empty.")
            self.long_term_memory = []
            return
        try:
            with open(self.ltm_file_path, 'rb') as f:
                loaded_data = pickle.load(f)
            if isinstance(loaded_data, list):
                valid_loaded_data = []
                for item in loaded_data:
                    if isinstance(item, dict) and 'representation' in item and 'timestamp' in item: # metadata opsiyonel olabilir LTM'de
                        rep = item['representation']
                        if isinstance(rep, np.ndarray) and rep.ndim == 1 and np.issubdtype(rep.dtype, np.number) and rep.shape[0] == self.representation_dim:
                            valid_loaded_data.append(item)
                        else: logger.warning("LTM load: Invalid representation in entry, skipping.")
                    else: logger.warning("LTM load: Invalid entry format, skipping.")
                self.long_term_memory = valid_loaded_data
                logger.info(f"LTM loaded from {self.ltm_file_path} ({len(self.long_term_memory)} entries).")
                if self.max_ltm_size > 0 and len(self.long_term_memory) > self.max_ltm_size:
                    logger.info(f"LTM size ({len(self.long_term_memory)}) exceeds max ({self.max_ltm_size}). Trimming oldest.")
                    self.long_term_memory = self.long_term_memory[-self.max_ltm_size:]
            else:
                logger.error(f"LTM file {self.ltm_file_path} has unexpected format. Initializing empty LTM."); self.long_term_memory = []
        except Exception as e:
            logger.error(f"Error loading LTM from {self.ltm_file_path}: {e}", exc_info=True); self.long_term_memory = []

    def _save_ltm(self):
        if not self.enable_ltm_persistence or not self.ltm_file_path: return
        if not self.long_term_memory: logger.info("LTM is empty. Saving skipped."); return
        
        save_dir = os.path.dirname(self.ltm_file_path)
        if save_dir and not os.path.exists(save_dir):
            try: os.makedirs(save_dir, exist_ok=True)
            except OSError as e: logger.error(f"Error creating LTM save dir {save_dir}: {e}. Saving skipped."); return
        
        try:
            with open(self.ltm_file_path, 'wb') as f:
                pickle.dump(self.long_term_memory, f)
            logger.info(f"LTM saved to {self.ltm_file_path} ({len(self.long_term_memory)} entries).")
        except Exception as e:
            logger.error(f"Error saving LTM to {self.ltm_file_path}: {e}", exc_info=True)

    def store(self, representation, metadata=None):
        if representation is None or not (isinstance(representation, np.ndarray) and \
           representation.ndim == 1 and np.issubdtype(representation.dtype, np.number) and \
           representation.shape[0] == self.representation_dim):
            logger.warning(f"Memory.store: Invalid representation provided. Shape: {getattr(representation, 'shape', 'N/A')}. Skipping.")
            return

        timestamp = time.time()
        memory_entry = {'representation': representation, 'metadata': metadata or {}, 'timestamp': timestamp}
        
        # Çalışma Belleğine Ekle
        self.working_memory.append(memory_entry)
        logger.debug(f"Stored in Working Memory. Size: {len(self.working_memory)}")

        # Çalışma Belleği Boyut Kontrolü ve LTM'ye Transfer
        if len(self.working_memory) > self.max_working_memory_size:
            oldest_entry = self.working_memory.pop(0)
            logger.debug(f"Working Memory max size exceeded. Moved oldest to LTM consideration.")
            # En eski girdiyi LTM'ye ekle (LTM boyut kontrolü orada yapılır)
            self._add_to_ltm(oldest_entry)

    def _add_to_ltm(self, memory_entry):
        """Helper to add an entry to LTM, managing LTM size."""
        if not isinstance(memory_entry, dict) or 'representation' not in memory_entry:
            logger.warning("LTM Add: Invalid entry format for LTM. Skipping.")
            return

        self.long_term_memory.append(memory_entry)
        
        logger.debug(f"Added to LTM. LTM Size: {len(self.long_term_memory)}")
        if self.max_ltm_size > 0 and len(self.long_term_memory) > self.max_ltm_size:
            self.long_term_memory.pop(0) # FIFO for LTM if capped
            logger.debug(f"LTM max size exceeded. Removed oldest from LTM.")


    def retrieve(self, query_representation, num_results=None):
        """Öncelikle Çalışma Belleğinden, sonra gerekirse LTM'den arama yapar (şimdilik sadece WM)."""
        if num_results is None: num_results = self.num_retrieved_working_memories
        if not isinstance(num_results, int) or num_results < 0: num_results = self.num_retrieved_working_memories
        
        if not self.working_memory or num_results == 0:
            return []

        if query_representation is None or not (isinstance(query_representation, np.ndarray) and \
           query_representation.ndim == 1 and query_representation.shape[0] == self.representation_dim):
            logger.debug("Memory.retrieve: Invalid query representation. Retrieving most recent from working memory.")
            # Geçersiz sorgu durumunda en son eklenenleri döndür
            return self.working_memory[-min(num_results, len(self.working_memory)):]

        query_norm = np.linalg.norm(query_representation)
        if query_norm < 1e-9:
            logger.warning("Memory.retrieve: Query representation norm is near zero. Cosine similarity unreliable. Returning recents.");
            return self.working_memory[-min(num_results, len(self.working_memory)):]

        similarities = []
        # Şimdilik sadece çalışma belleğinden alıyoruz
        for entry in self.working_memory:
            stored_rep = entry['representation']
            stored_norm = np.linalg.norm(stored_rep)
            if stored_norm > 1e-9:
                similarity = np.dot(query_representation, stored_rep) / (query_norm * stored_norm)
                if not np.isnan(similarity):
                    similarities.append((float(similarity), entry))
        
        similarities.sort(key=lambda item: item[0], reverse=True)
        retrieved_list = [item[1] for item in similarities[:num_results]]
        logger.debug(f"Retrieved {len(retrieved_list)} entries from Working Memory by similarity.")
        return retrieved_list

    def get_all_representations(self, from_ltm=True, sample_size=None):
        """
        Kavram öğrenme için temsilleri alır. Öncelikli olarak LTM'den alır.
        Eğer LTM boşsa veya from_ltm=False ise çalışma belleğinden alır.
        """
        source_memory = self.long_term_memory if from_ltm and self.long_term_memory else self.working_memory
        source_name = "LTM" if from_ltm and self.long_term_memory else "Working Memory"

        if not source_memory:
            logger.debug(f"Memory.get_all_representations: {source_name} is empty.")
            return []
            
        representations = [
            entry['representation'] for entry in source_memory 
            if isinstance(entry, dict) and 'representation' in entry and \
               isinstance(entry['representation'], np.ndarray) and \
               entry['representation'].shape == (self.representation_dim,)
        ]
        
        if sample_size and len(representations) > sample_size:
            logger.debug(f"Sampling {sample_size} representations from {len(representations)} available in {source_name}.")
            return random.sample(representations, sample_size)
        
        logger.debug(f"Returning {len(representations)} representations from {source_name}.")
        return representations

    def cleanup(self):
        logger.info("Memory module cleaning up...")
        if self.enable_ltm_persistence:
            run_safely(self._save_ltm, logger_instance=logger, error_message="Memory: Error during LTM save on cleanup")
        self.working_memory = []
        self.long_term_memory = [] # RAM'i boşalt
        logger.info("Memory module cleaned up.")