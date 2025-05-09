# src/run_evo.py
import logging
import time
import numpy as np
import torch
import cv2 

from src.core.logging_utils import setup_logging
from src.core.config_utils import load_config_from_yaml, get_config_value
from src.core.module_loader import initialize_modules, cleanup_modules
from src.core.compute_utils import initialize_compute_backend, get_backend, get_device, to_numpy

logger = None 
previous_latent_representation_debug = None # Artık latent temsiller için

show_camera_feed_active = False # Config'den okunacak
# show_processed_main_active = False # Config'den okunacak
# show_processed_edges_active = False # Config'den okunacak

def run_evo():
    global logger, previous_latent_representation_debug
    global show_camera_feed_active #, show_processed_main_active, show_processed_edges_active

    config_path = "config/main_config.yaml"
    config = load_config_from_yaml(config_path)

    setup_logging(config=config) 
    logger = logging.getLogger(__name__) 

    if not config: logging.critical(f"Evo: Config {config_path} failed. Exiting."); return

    initialize_compute_backend(config)
    current_backend = get_backend()
    current_device = get_device()

    debug_config = config.get('debug', {})
    show_camera_feed_active = get_config_value(debug_config, 'show_camera_feed', default=False, expected_type=bool)
    # Diğer show bayrakları da benzer şekilde okunabilir ama bu örnekte kullanmıyoruz
    # show_processed_main_active = get_config_value(debug_config, 'show_processed_main_image', default=False, expected_type=bool)
    # show_processed_edges_active = get_config_value(debug_config, 'show_processed_edges_image', default=False, expected_type=bool)
    
    logger.info(f"Display Settings - Raw Feed: {'ON' if show_camera_feed_active else 'OFF'}")
    logger.info(f"Evo awakening... Backend: {current_backend}, Device: {str(current_device)}")
    logger.info(f"Config loaded from: {config_path}")

    module_objects, can_run_main_loop = initialize_modules(config)

    if not can_run_main_loop:
        logger.critical("Evo: Core modules failed. Shutting down."); cleanup_modules(module_objects)
        if show_camera_feed_active: cv2.destroyAllWindows(); return

    sensors = module_objects.get('sensors', {}); processors = module_objects.get('processors', {})
    representers = module_objects.get('representers', {}); memories = module_objects.get('memories', {})
    cognition_modules = module_objects.get('cognition', {}); motor_control_modules = module_objects.get('motor_control', {})
    interaction_modules = module_objects.get('interaction', {})

    logger.info("Evo cognitive loop ready.")
    loop_interval_cfg = get_config_value(config, 'cognitive_loop_interval', default=0.1, expected_type=(float,int,np.floating))
    loop_interval = float(loop_interval_cfg) if isinstance(loop_interval_cfg, (float,int,np.floating)) and float(loop_interval_cfg) > 0 else 0.1
    num_mem_retrieve_cfg = get_config_value(config, 'memory', 'num_retrieved_memories', default=5, expected_type=(int,np.integer))
    num_memories_to_retrieve = int(num_mem_retrieve_cfg) if isinstance(num_mem_retrieve_cfg, (int,np.integer)) and int(num_mem_retrieve_cfg) >=0 else 5
    
    expected_input_order = config.get('representation', {}).get('_expected_input_order_', []) 
    if expected_input_order: logger.info(f"RUN_EVO: Expected input order for learner: {expected_input_order}")
    
    logger.info("RUN_EVO: RepresentationLearner (Autoencoder) will be used and trained.")

    try:
        loop_count = 0
        while True:
            loop_count += 1; start_time = time.time()
            raw_inputs = {}; current_raw_frame = None
            if sensors.get('vision'): 
                raw_inputs['visual'] = sensors['vision'].capture_frame()
                if show_camera_feed_active and raw_inputs.get('visual') is not None: current_raw_frame = raw_inputs['visual']
            if sensors.get('audio'): raw_inputs['audio'] = sensors['audio'].capture_chunk()

            if show_camera_feed_active and current_raw_frame is not None:
                try: cv2.imshow("Evo Camera Feed", current_raw_frame)
                except Exception as e: logger.error(f"Error displaying camera feed: {e}", exc_info=False)

            processed_inputs = {}
            if processors.get('vision') and raw_inputs.get('visual') is not None: 
                processed_inputs['visual'] = processors['vision'].process(raw_inputs['visual'])
            if processors.get('audio') and raw_inputs.get('audio') is not None: 
                processed_inputs['audio'] = processors['audio'].process(raw_inputs['audio'])
            
            combined_features_list = []; vis_out = processed_inputs.get('visual', {}); aud_out = processed_inputs.get('audio', {})
            has_valid_feature = False
            for key_full in expected_input_order:
                src, name = key_full.split('_', 1); data = None
                if src == "vision" and isinstance(vis_out, dict): data = vis_out.get(name)
                elif src == "audio": 
                    if isinstance(aud_out, dict) and name in aud_out : data = aud_out.get(name)
                    elif name == "audio_features" and not isinstance(aud_out, dict): data = aud_out
                if data is not None:
                    if isinstance(data, (torch.Tensor, np.ndarray)): combined_features_list.append(data.flatten()); has_valid_feature = True
                    else: logger.warning(f"RUN_EVO (L{loop_count}): Feat '{key_full}' type {type(data)} not Tensor/Array.")
            if not has_valid_feature and combined_features_list: combined_features_list.clear()

            learned_latent_rep_np = None; final_combined_input = None # Değişken adını netleştirelim
            if combined_features_list:
                if current_backend == "pytorch":
                    try:
                        target_device = torch.device("cpu") 
                        tensors = [d.to(target_device) if isinstance(d, torch.Tensor) else torch.from_numpy(d).float().to(target_device) for d in combined_features_list]
                        if tensors: final_combined_input = torch.cat(tensors)
                    except Exception as e: logger.error(f"RUN_EVO (L{loop_count}): Error concat PyTorch tensors: {e}", exc_info=True)
                elif current_backend == "numpy":
                    valid_arrays = [arr for arr in combined_features_list if isinstance(arr, np.ndarray)]
                    if valid_arrays: final_combined_input = np.concatenate(valid_arrays)
            
            if final_combined_input is not None and representers.get('main_learner'):
                output_latent_tensor = representers['main_learner'].learn(final_combined_input)
                if output_latent_tensor is not None: 
                    learned_latent_rep_np = to_numpy(output_latent_tensor)
                    if learned_latent_rep_np is not None:
                        logger.debug(f"RUN_EVO (L{loop_count}): Latent Rep (NP) Shape: {learned_latent_rep_np.shape}")
                        # Kosinüs benzerliği loglaması
                        if previous_latent_representation_debug is not None and \
                           previous_latent_representation_debug.shape == learned_latent_rep_np.shape and \
                           learned_latent_rep_np.size > 0 and previous_latent_representation_debug.size > 0 :
                            vec1 = previous_latent_representation_debug.flatten()
                            vec2 = learned_latent_rep_np.flatten()
                            dot_product = np.dot(vec1, vec2)
                            norm_prev = np.linalg.norm(vec1)
                            norm_curr = np.linalg.norm(vec2)
                            if norm_prev > 1e-9 and norm_curr > 1e-9: 
                                cosine_sim_consecutive = np.clip(dot_product / (norm_prev * norm_curr), -1.0, 1.0)
                                logger.debug(f"RUN_EVO_COSINE (L{loop_count}): Cosine sim with PREVIOUS LATENT: {cosine_sim_consecutive:.6f}")
                            else:
                                logger.debug(f"RUN_EVO_COSINE (L{loop_count}): Norm near zero for latent. PrevN:{norm_prev:.2e}, CurrN:{norm_curr:.2e}")
                        elif previous_latent_representation_debug is None and learned_latent_rep_np.size > 0:
                            logger.debug(f"RUN_EVO_COSINE (L{loop_count}): First valid latent representation.")
                        
                        if learned_latent_rep_np.size > 0 : # Sadece boş değilse kopyala
                            previous_latent_representation_debug = learned_latent_rep_np.copy()
                    else: 
                        logger.warning(f"RUN_EVO (L{loop_count}): Latent to NumPy conversion failed.")
                else: 
                    logger.warning(f"RUN_EVO (L{loop_count}): RepresentationLearner.learn returned None.")
            
            # ... (Kalan Memory, Cognition, MotorControl, Interaction kodları aynı)
            relevant_mem = []; core_mem = memories.get('core_memory')
            if core_mem:
                if learned_latent_rep_np is not None: core_mem.store(learned_latent_rep_np, {'timestamp': time.time(), 'loop': loop_count})
                relevant_mem = core_mem.retrieve(learned_latent_rep_np, num_results=num_memories_to_retrieve)

            concepts = []; cog_core = cognition_modules.get('core_cognition')
            if cog_core and hasattr(cog_core, 'learning_module') and cog_core.learning_module:
                try:
                    ret_concepts = cog_core.learning_module.get_concepts()
                    if isinstance(ret_concepts, list): concepts = ret_concepts
                except Exception as e: logger.error(f"RUN_EVO (L{loop_count}): Error getting concepts: {e}", exc_info=False)
            
            decision = None
            if cog_core:
                try:
                    decision = cog_core.decide(processed_inputs, learned_latent_rep_np, relevant_mem, concepts)
                    # Dont open
                    # if decision: logger.info(f"RUN_EVO (L{loop_count}): Decision: {str(decision)}")
                except Exception as e: logger.error(f"RUN_EVO (L{loop_count}): CognitionCore.decide error: {e}", exc_info=False)

            response = None; motor_core = motor_control_modules.get('core_motor_control')
            if motor_core:
                try:
                    response = motor_core.generate_response(decision)
                    if response: logger.debug(f"RUN_EVO (L{loop_count}): Response: {str(response)[:100]}...")
                except Exception as e: logger.error(f"RUN_EVO (L{loop_count}): MotorControlCore.generate_response error: {e}", exc_info=False)

            interaction_api = interaction_modules.get('core_interaction')
            if interaction_api and response is not None:
                try: interaction_api.send_output(response)
                except Exception as e: logger.error(f"RUN_EVO (L{loop_count}): InteractionAPI.send_output error: {e}", exc_info=False)


            elapsed = time.time() - start_time; wait_key_duration = 1
            if show_camera_feed_active:
                remaining_wait = loop_interval - elapsed
                wait_key_duration = int(remaining_wait * 1000) if remaining_wait > 0.001 else 1
                key_pressed = cv2.waitKey(wait_key_duration) & 0xFF
                if key_pressed == ord('q'): logger.info("RUN_EVO: 'q' pressed. Exiting."); break
                elif key_pressed == ord('p'):
                    logger.info("RUN_EVO: 'p' pressed. Pausing... Press 'p' again to resume.")
                    while True:
                        if cv2.waitKey(100) & 0xFF == ord('p'): logger.info("RUN_EVO: Resuming..."); break
            else:
                sleep_duration = loop_interval - elapsed
                if sleep_duration > 0: time.sleep(sleep_duration)
            
            final_elapsed = time.time() - start_time
            if final_elapsed > loop_interval * 1.25: 
                 logger.warning(f"RUN_EVO (L{loop_count}): Loop overrun: {final_elapsed:.4f}s (target: {loop_interval:.2f}s)")

    except KeyboardInterrupt: logger.warning("Ctrl+C detected. Evo shutting down...")
    except Exception as e: logger.critical(f"Evo main loop critical error: {e}", exc_info=True)
    finally:
        cleanup_modules(module_objects)
        if show_camera_feed_active: cv2.destroyAllWindows()
    logger.info("Evo has been shut down.")

if __name__ == '__main__':
    run_evo()