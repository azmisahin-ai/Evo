# src/cognition/learning.py
import logging
import time
import numpy as np
import pickle # Kavramları kaydetmek/yüklemek için
import os

from src.core.utils import check_input_not_none, check_input_type, check_numpy_input
from src.core.config_utils import get_config_value

logger = logging.getLogger(__name__)

class LearningModule:
    def __init__(self, full_config):
        self.config = full_config.get('cognition', {}).get('learning', {})
        self.representation_config = full_config.get('representation', {})
        logger.info("LearningModule initializing...")

        self.new_concept_threshold = float(get_config_value(self.config, 'new_concept_threshold', default=0.75, expected_type=(float, int, np.floating)))
        self.representation_dim = get_config_value(self.representation_config, 'representation_dim', default=128, expected_type=int)
        
        self.enable_concept_persistence = get_config_value(self.config, 'enable_concept_persistence', default=True, expected_type=bool)
        self.concept_file_path = get_config_value(self.config, 'concept_file_path', default='data/concepts.pkl', expected_type=str)

        if not (0.0 <= self.new_concept_threshold <= 1.0):
             logger.warning(f"LearningModule: Invalid new_concept_threshold {self.new_concept_threshold}. Using 0.75.")
             self.new_concept_threshold = 0.75
        if self.representation_dim <= 0:
             logger.error(f"LearningModule: Invalid representation_dim {self.representation_dim}. Concept learning may fail.")
             # Bu durumda __init__ hata fırlatabilir veya is_initialized = False ayarlayabilir.

        self.concept_representatives = [] # [{'id': int, 'vector': np.ndarray, 'count': int, 'created_at': float}]
        self.next_concept_id = 0

        if self.enable_concept_persistence:
            self._load_concepts()

        logger.info(f"LearningModule initialized. NewConceptThresh: {self.new_concept_threshold}, RepDim: {self.representation_dim}, Persistence: {self.enable_concept_persistence} ({self.concept_file_path}), Initial Concepts: {len(self.concept_representatives)}")

    def _load_concepts(self):
        if not self.concept_file_path or not os.path.exists(self.concept_file_path):
            logger.info(f"Concept file not found: {self.concept_file_path}. Initializing with no concepts.")
            return
        try:
            with open(self.concept_file_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'concepts' in data and 'next_id' in data:
                loaded_concepts = data['concepts']
                valid_concepts = []
                max_id = -1
                for concept_data in loaded_concepts:
                    if isinstance(concept_data, dict) and 'id' in concept_data and 'vector' in concept_data and \
                       isinstance(concept_data['vector'], np.ndarray) and \
                       concept_data['vector'].shape == (self.representation_dim,):
                        valid_concepts.append(concept_data)
                        if concept_data['id'] > max_id: max_id = concept_data['id']
                    else:
                        logger.warning("Invalid concept data found in file. Skipping.")
                self.concept_representatives = valid_concepts
                self.next_concept_id = data.get('next_id', max_id + 1) # Kayıtlı next_id'yi veya max_id+1'i kullan
                logger.info(f"Concepts loaded from {self.concept_file_path}. Count: {len(self.concept_representatives)}, Next ID: {self.next_concept_id}")
            else:
                logger.error(f"Concept file {self.concept_file_path} has unexpected format. Initializing empty."); self._reset_concepts()
        except Exception as e:
            logger.error(f"Error loading concepts from {self.concept_file_path}: {e}", exc_info=True); self._reset_concepts()

    def _save_concepts(self):
        if not self.enable_concept_persistence or not self.concept_file_path: return
        if not self.concept_representatives and self.next_concept_id == 0 : # Kaydedecek bir şey yoksa
            logger.info("No concepts to save.")
            # Eğer dosya varsa ve içi boşsa silmek isteyebiliriz (opsiyonel)
            # if os.path.exists(self.concept_file_path): os.remove(self.concept_file_path)
            return

        save_dir = os.path.dirname(self.concept_file_path)
        if save_dir and not os.path.exists(save_dir):
            try: os.makedirs(save_dir, exist_ok=True)
            except OSError as e: logger.error(f"Error creating concept save dir {save_dir}: {e}. Saving skipped."); return
        
        try:
            with open(self.concept_file_path, 'wb') as f:
                pickle.dump({'concepts': self.concept_representatives, 'next_id': self.next_concept_id}, f)
            logger.info(f"Concepts saved to {self.concept_file_path}. Count: {len(self.concept_representatives)}, Next ID: {self.next_concept_id}")
        except Exception as e:
            logger.error(f"Error saving concepts to {self.concept_file_path}: {e}", exc_info=True)
            
    def _reset_concepts(self):
        self.concept_representatives = []
        self.next_concept_id = 0

    def learn_concepts(self, representation_list):
        if not representation_list or self.representation_dim <= 0:
            logger.debug(f"LM.learn_concepts: Empty list or invalid rep_dim ({self.representation_dim}). Skipping.")
            return self.get_concepts_vectors() 

        new_concepts_learned_this_cycle = 0
        for rep_vector in representation_list:
            if not (isinstance(rep_vector, np.ndarray) and rep_vector.shape == (self.representation_dim,) and np.issubdtype(rep_vector.dtype, np.number)):
                logger.warning(f"LM.learn_concepts: Invalid rep_vector format/type/shape. Skipping this vector."); continue

            rep_norm = np.linalg.norm(rep_vector)
            if rep_norm < 1e-9: logger.debug("LM.learn_concepts: Rep vector norm near zero. Skipping."); continue

            max_similarity = -1.0
            closest_concept_idx = -1

            for idx, concept_data in enumerate(self.concept_representatives):
                concept_vec = concept_data['vector']
                concept_norm = np.linalg.norm(concept_vec)
                if concept_norm > 1e-9:
                    similarity = np.dot(rep_vector, concept_vec) / (rep_norm * concept_norm)
                    similarity = np.clip(similarity, -1.0, 1.0) # Sayısal hatalara karşı
                    if similarity > max_similarity:
                        max_similarity = similarity
                        closest_concept_idx = idx
            
            # Eğer max_similarity eşiğin altındaysa YENİ kavram VEYA hiç kavram yoksa
            if max_similarity < self.new_concept_threshold or not self.concept_representatives:
                new_concept_data = {
                    'id': self.next_concept_id,
                    'vector': rep_vector.copy(), # Vektörün kopyasını sakla
                    'count': 1, # İlk kez görüldü
                    'created_at': time.time(),
                    'last_updated_at': time.time()
                }
                self.concept_representatives.append(new_concept_data)
                logger.info(f"LM: New concept ID {self.next_concept_id} discovered! MaxSim: {max_similarity:.4f} < Thresh: {self.new_concept_threshold:.2f}. Total concepts: {len(self.concept_representatives)}")
                self.next_concept_id += 1
                new_concepts_learned_this_cycle += 1
            elif closest_concept_idx != -1: # Mevcut bir kavrama benziyor
                # Kavramı güncelle (örn: sayacını artır, temsilcisini hareketli ortalama ile güncelle - opsiyonel)
                self.concept_representatives[closest_concept_idx]['count'] += 1
                self.concept_representatives[closest_concept_idx]['last_updated_at'] = time.time()
                # Basit hareketli ortalama ile kavram vektörünü güncelleme (opsiyonel)
                # old_vec = self.concept_representatives[closest_concept_idx]['vector']
                # n = self.concept_representatives[closest_concept_idx]['count']
                # self.concept_representatives[closest_concept_idx]['vector'] = (old_vec * (n-1) + rep_vector) / n
                logger.debug(f"LM: Rep similar to existing concept ID {self.concept_representatives[closest_concept_idx]['id']} (Sim: {max_similarity:.4f}). Count: {self.concept_representatives[closest_concept_idx]['count']}")

        if new_concepts_learned_this_cycle > 0 and self.enable_concept_persistence:
            self._save_concepts() # Yeni kavram öğrenildiyse kaydet
            
        return self.get_concepts_vectors() # Sadece vektör listesini döndür

    def get_concepts(self): # Artık tüm concept_data dict'lerini döndürür
        return self.concept_representatives[:] 
    
    def get_concepts_vectors(self): # Sadece vektörleri döndüren eski davranış
        return [cd['vector'] for cd in self.concept_representatives]

    def cleanup(self):
        logger.info("LearningModule cleaning up.")
        if self.enable_concept_persistence:
            self._save_concepts() # Kapanırken son bir kez kaydet
        self._reset_concepts()
        logger.info("LearningModule cleaned up.")