# src/cognition/understanding.py
# ... (importlar aynı) ...
import logging
import numpy as np

from src.core.utils import check_input_not_none, check_numpy_input # type check kaldırıldı
from src.core.config_utils import get_config_value

logger = logging.getLogger(__name__)

class UnderstandingModule:
    def __init__(self, full_config): # Tam config alır
        self.config = full_config.get('cognition', {}) 
        logger.info("UnderstandingModule initializing...")
        self.audio_energy_threshold = float(get_config_value(self.config, 'audio_energy_threshold', default=1000.0, expected_type=(float,int,np.floating)))
        self.visual_edges_threshold = float(get_config_value(self.config, 'visual_edges_threshold', default=0.1, expected_type=(float,int,np.floating))) # Normalize edilmiş kenarlar için
        self.brightness_threshold_high = float(get_config_value(self.config, 'brightness_threshold_high', default=200.0, expected_type=(float,int,np.floating)))
        self.brightness_threshold_low = float(get_config_value(self.config, 'brightness_threshold_low', default=50.0, expected_type=(float,int,np.floating)))
        logger.info(f"UnderstandingModule initialized. AudioT: {self.audio_energy_threshold}, EdgeT: {self.visual_edges_threshold}, BrightHighT: {self.brightness_threshold_high}, BrightLowT: {self.brightness_threshold_low}")

    def process(self, processed_inputs, learned_representation, relevant_memory_entries, current_concepts_data):
        # current_concepts_data: [{'id': int, 'vector': np.ndarray, ...}, ...]
        understanding_signals = {
            'similarity_score': 0.0, 'high_audio_energy': False, 'high_visual_edges': False,
            'is_bright': False, 'is_dark': False, 'max_concept_similarity': 0.0,
            'most_similar_concept_id': None,
        }

        is_valid_repr = learned_representation is not None and \
                        isinstance(learned_representation, np.ndarray) and \
                        learned_representation.ndim == 1 and \
                        np.issubdtype(learned_representation.dtype, np.number)
        
        query_norm = 0.0
        if is_valid_repr:
             query_norm = np.linalg.norm(learned_representation)
             if query_norm < 1e-9: is_valid_repr = False; logger.debug("UM: Query rep norm near zero.")
        
        try:
            # 1. Bellek Benzerliği
            if is_valid_repr and relevant_memory_entries and isinstance(relevant_memory_entries, list):
                max_mem_sim = 0.0
                for entry in relevant_memory_entries:
                    if isinstance(entry, dict):
                        stored_rep = entry.get('representation')
                        if isinstance(stored_rep, np.ndarray) and stored_rep.shape == learned_representation.shape:
                            stored_norm = np.linalg.norm(stored_rep)
                            if stored_norm > 1e-9:
                                sim = np.dot(learned_representation, stored_rep) / (query_norm * stored_norm)
                                if not np.isnan(sim): max_mem_sim = max(max_mem_sim, sim)
                understanding_signals['similarity_score'] = np.clip(max_mem_sim, 0.0, 1.0) # 0-1 arasına klipsle

            # 2. Kavram Benzerliği
            if is_valid_repr and current_concepts_data and isinstance(current_concepts_data, list):
                max_con_sim = 0.0
                best_con_id = None
                for concept_entry in current_concepts_data: # Artık dict listesi
                    if isinstance(concept_entry, dict) and 'vector' in concept_entry and 'id' in concept_entry:
                        concept_vec = concept_entry['vector']
                        if isinstance(concept_vec, np.ndarray) and concept_vec.shape == learned_representation.shape:
                            concept_norm = np.linalg.norm(concept_vec)
                            if concept_norm > 1e-9:
                                sim = np.dot(learned_representation, concept_vec) / (query_norm * concept_norm)
                                if not np.isnan(sim) and sim > max_con_sim: # Sadece daha büyükse al
                                    max_con_sim = sim
                                    best_con_id = concept_entry['id']
                understanding_signals['max_concept_similarity'] = np.clip(max_con_sim, 0.0, 1.0)
                understanding_signals['most_similar_concept_id'] = best_con_id
            
            # 3. Anlık Duyu Özellikleri
            if processed_inputs and isinstance(processed_inputs, dict):
                audio_feat = processed_inputs.get('audio') # Bu bir tensör/array olabilir
                if audio_feat is not None and hasattr(audio_feat, 'shape') and audio_feat.shape[0] >= 1:
                    # Eğer tensörse NumPy'a çevir (basitlik için CPU'da)
                    if hasattr(audio_feat, 'cpu') and hasattr(audio_feat, 'numpy'): audio_feat = audio_feat.cpu().numpy()
                    if isinstance(audio_feat, np.ndarray) and audio_feat.size > 0: # Boş olmadığından emin ol
                        try:
                            energy = float(audio_feat[0]) # İlk elemanın enerji olduğunu varsay
                            if energy >= self.audio_energy_threshold: understanding_signals['high_audio_energy'] = True
                        except (IndexError, TypeError): pass # Hata durumunda False kalır

                vis_feat = processed_inputs.get('visual') # Bu bir dict {'main_image': ..., 'edges': ...}
                if isinstance(vis_feat, dict):
                    edges_data = vis_feat.get('edges') # Tensör/array
                    if edges_data is not None and hasattr(edges_data, 'mean'): # NumPy veya PyTorch tensörü için
                         # Eğer tensörse NumPy'a çevir
                        if hasattr(edges_data, 'cpu') and hasattr(edges_data, 'numpy'): edges_data = edges_data.cpu().numpy()
                        if isinstance(edges_data, np.ndarray) and edges_data.size > 0: # Boş olmadığından emin ol
                            edge_density = np.mean(edges_data) # Kenarlar 0-1 normalize edilmiş olmalı
                            if edge_density >= self.visual_edges_threshold: understanding_signals['high_visual_edges'] = True
                    
                    main_img_data = vis_feat.get('main_image') # Tensör/array
                    if main_img_data is not None and hasattr(main_img_data, 'mean'):
                        if hasattr(main_img_data, 'cpu') and hasattr(main_img_data, 'numpy'): main_img_data = main_img_data.cpu().numpy()
                        if isinstance(main_img_data, np.ndarray) and main_img_data.size > 0:
                            brightness = np.mean(main_img_data) * 255 # 0-1 aralığından 0-255'e (varsayım)
                            if brightness >= self.brightness_threshold_high: understanding_signals['is_bright'] = True
                            elif brightness <= self.brightness_threshold_low: understanding_signals['is_dark'] = True
            
            logger.debug(f"UM Processed: SimScore:{understanding_signals['similarity_score']:.2f} ConceptSim:{understanding_signals['max_concept_similarity']:.2f} (ID:{understanding_signals['most_similar_concept_id']}) AudioE:{understanding_signals['high_audio_energy']} VisualE:{understanding_signals['high_visual_edges']}")

        except Exception as e:
            logger.error(f"UnderstandingModule.process error: {e}", exc_info=True)
            return { # Hata durumunda varsayılanları döndür
                'similarity_score': 0.0, 'high_audio_energy': False, 'high_visual_edges': False,
                'is_bright': False, 'is_dark': False, 'max_concept_similarity': 0.0,
                'most_similar_concept_id': None,
            }
        return understanding_signals

    def cleanup(self):
        logger.info("UnderstandingModule cleaning up.")