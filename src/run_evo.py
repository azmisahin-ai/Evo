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

# Varsayılan görüntüleme ayarları (config'de belirtilmezse bunlar kullanılır)
DEFAULT_SHOW_CAMERA_FEED = False 
DEFAULT_SHOW_PROCESSED_MAIN = False
DEFAULT_SHOW_PROCESSED_EDGES = False

# Global değişkenler olarak tanımlayalım ki fonksiyon içinde değerleri okunabilsin/değiştirilebilsin
# Ancak daha iyi bir pratik, bunları bir sınıfın üyesi yapmak veya fonksiyona parametre olarak geçmektir.
# Şimdilik basitlik için global tutalım.
show_camera_feed_active = DEFAULT_SHOW_CAMERA_FEED
show_processed_main_active = DEFAULT_SHOW_PROCESSED_MAIN
show_processed_edges_active = DEFAULT_SHOW_PROCESSED_EDGES

def run_evo():
    global logger
    global show_camera_feed_active, show_processed_main_active, show_processed_edges_active # global değişkenleri belirt

    config_path = "config/main_config.yaml"
    config = load_config_from_yaml(config_path)

    setup_logging(config=config) 
    logger = logging.getLogger(__name__) 

    if not config:
        logging.critical(f"Evo: Config file {config_path} failed. Exiting."); return

    initialize_compute_backend(config)
    current_backend = get_backend()
    current_device = get_device()

    # Debug görüntüleme ayarlarını config'den oku
    debug_config = config.get('debug', {}) # 'debug' bölümü yoksa boş dict
    show_camera_feed_active = get_config_value(debug_config, 'show_camera_feed', 
                                               default=DEFAULT_SHOW_CAMERA_FEED, expected_type=bool)
    show_processed_main_active = get_config_value(debug_config, 'show_processed_main_image', 
                                                  default=DEFAULT_SHOW_PROCESSED_MAIN, expected_type=bool)
    show_processed_edges_active = get_config_value(debug_config, 'show_processed_edges_image', 
                                                 default=DEFAULT_SHOW_PROCESSED_EDGES, expected_type=bool)
    
    logger.info(f"Display Settings - Raw Feed: {'ON' if show_camera_feed_active else 'OFF'}, Processed Main: {'ON' if show_processed_main_active else 'OFF'}, Processed Edges: {'ON' if show_processed_edges_active else 'OFF'}")

    logger.info(f"Evo awakening... Backend: {current_backend}, Device: {str(current_device)}")
    logger.info(f"Config loaded from: {config_path}")

    module_objects, can_run_main_loop = initialize_modules(config)

    if not can_run_main_loop:
        logger.critical("Evo: Core modules failed. Shutting down."); cleanup_modules(module_objects)
        if show_camera_feed_active or show_processed_main_active or show_processed_edges_active: cv2.destroyAllWindows()
        return

    # ... (sensör, işlemci vb. kısayolları aynı) ...
    sensors = module_objects.get('sensors', {})
    processors = module_objects.get('processors', {})
    representers = module_objects.get('representers', {})
    memories = module_objects.get('memories', {})
    cognition_modules = module_objects.get('cognition', {})
    motor_control_modules = module_objects.get('motor_control', {})
    interaction_modules = module_objects.get('interaction', {})

    logger.info("Evo cognitive loop ready.")
    loop_interval_cfg = get_config_value(config, 'cognitive_loop_interval', default=0.15, expected_type=(float,int,np.floating))
    loop_interval = float(loop_interval_cfg) if isinstance(loop_interval_cfg, (float,int,np.floating)) and float(loop_interval_cfg) > 0 else 0.15
    num_mem_retrieve_cfg = get_config_value(config, 'memory', 'num_retrieved_memories', default=5, expected_type=(int,np.integer))
    num_memories_to_retrieve = int(num_mem_retrieve_cfg) if isinstance(num_mem_retrieve_cfg, (int,np.integer)) and int(num_mem_retrieve_cfg) >=0 else 5
    
    representation_config = config.get('representation', {})
    expected_input_order_for_learner = representation_config.get('_expected_input_order_', []) 
    if expected_input_order_for_learner: logger.info(f"RUN_EVO: Expected input order for learner: {expected_input_order_for_learner}")
    
    logger.info("RUN_EVO: RepresentationLearner (Autoencoder) will be used and trained.")

    try:
        loop_count = 0
        while True:
            loop_count += 1; start_time = time.time()
            raw_inputs = {}; current_raw_frame = None
            if sensors.get('vision'): 
                raw_inputs['visual'] = sensors['vision'].capture_frame()
                if raw_inputs['visual'] is not None: current_raw_frame = raw_inputs['visual']
            if sensors.get('audio'): raw_inputs['audio'] = sensors['audio'].capture_chunk()

            if show_camera_feed_active and current_raw_frame is not None:
                try: cv2.imshow("Evo Raw Camera Feed", current_raw_frame)
                except Exception as e: logger.error(f"Error displaying raw camera feed: {e}", exc_info=False)

            processed_inputs = {}; vis_proc_output_for_display = None
            if processors.get('vision') and raw_inputs.get('visual') is not None: 
                processed_inputs['visual'] = processors['vision'].process(raw_inputs['visual'])
                if show_processed_main_active or show_processed_edges_active: # Sadece gerekliyse al
                    vis_proc_output_for_display = processed_inputs['visual']
            if processors.get('audio') and raw_inputs.get('audio') is not None: 
                processed_inputs['audio'] = processors['audio'].process(raw_inputs['audio'])

            if vis_proc_output_for_display and isinstance(vis_proc_output_for_display, dict):
                if show_processed_main_active:
                    main_img_tensor = vis_proc_output_for_display.get('main_image')
                    if main_img_tensor is not None:
                        try:
                            img_np = to_numpy(main_img_tensor.cpu())
                            if img_np is not None:
                                if img_np.ndim == 3 and img_np.shape[0] in [1, 3]: img_np = img_np.transpose(1, 2, 0)
                                img_to_show = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.01 and img_np.min() >= -0.01 else img_np.astype(np.uint8) # 0-1 aralığı kontrolü için küçük tolerans
                                cv2.imshow("Evo Processed - Main", img_to_show)
                        except Exception as e: logger.error(f"Error displaying processed main image: {e}", exc_info=False)
                
                if show_processed_edges_active:
                    edges_tensor = vis_proc_output_for_display.get('edges')
                    if edges_tensor is not None:
                        try:
                            edges_np = to_numpy(edges_tensor.cpu())
                            if edges_np is not None:
                                if edges_np.ndim == 3 and edges_np.shape[0] == 1: edges_np = edges_np.squeeze(0)
                                edges_to_show = (edges_np * 255).astype(np.uint8) if edges_np.max() <= 1.01 and edges_np.min() >= -0.01 else edges_np.astype(np.uint8)
                                cv2.imshow("Evo Processed - Edges", edges_to_show)
                        except Exception as e: logger.error(f"Error displaying processed edges image: {e}", exc_info=False)
            
            # ... (Özellik birleştirme, Temsil Öğrenme, Bellek, Bilişse, Motor, Etkileşim kodları aynı kalır) ...
            combined_features_list = []; vis_out = processed_inputs.get('visual', {}); aud_out = processed_inputs.get('audio', {})
            has_valid_feature = False
            for key_full in expected_input_order_for_learner:
                src, name = key_full.split('_', 1); data = None
                if src == "vision" and isinstance(vis_out, dict): data = vis_out.get(name)
                elif src == "audio": 
                    if isinstance(aud_out, dict) and name in aud_out : data = aud_out.get(name)
                    elif name == "audio_features" and not isinstance(aud_out, dict): data = aud_out
                if data is not None:
                    if isinstance(data, (torch.Tensor, np.ndarray)): combined_features_list.append(data.flatten()); has_valid_feature = True
                    else: logger.warning(f"RUN_EVO (L{loop_count}): Feat '{key_full}' type {type(data)} not Tensor/Array.")
                # else: logger.debug(f"RUN_EVO (L{loop_count}): Expected feat '{key_full}' not found.") # Çok sık log
            if not has_valid_feature and combined_features_list: combined_features_list.clear()

            learned_rep_np = None; final_combined_input = None
            if combined_features_list:
                if current_backend == "pytorch":
                    try:
                        target_device = torch.device("cpu") 
                        tensors = [d.to(target_device) if isinstance(d, torch.Tensor) else torch.from_numpy(d).float().to(target_device) for d in combined_features_list]
                        if tensors: final_combined_input = torch.cat(tensors)
                        else: logger.warning(f"RUN_EVO (L{loop_count}): No valid tensors to concat.")
                    except Exception as e: logger.error(f"RUN_EVO (L{loop_count}): Error concat PyTorch tensors: {e}", exc_info=True)
                elif current_backend == "numpy":
                    valid_arrays = [arr for arr in combined_features_list if isinstance(arr, np.ndarray)]
                    if valid_arrays: final_combined_input = np.concatenate(valid_arrays)
                    else: logger.warning(f"RUN_EVO (L{loop_count}): No valid NumPy arrays to concat.")
            
            if final_combined_input is not None and representers.get('main_learner'):
                output_learner = representers['main_learner'].learn(final_combined_input)
                if output_learner is not None: learned_rep_np = to_numpy(output_learner)
                if learned_rep_np is None and output_learner is not None: logger.warning(f"RUN_EVO (L{loop_count}): Conversion of learner output to NumPy failed.")
                elif output_learner is None: logger.warning(f"RUN_EVO (L{loop_count}): RepresentationLearner.learn returned None.")
            
            if learned_rep_np is not None: logger.debug(f"RUN_EVO (L{loop_count}): Latent Rep (NP) Shape: {learned_rep_np.shape}")

            relevant_mem = []; core_mem = memories.get('core_memory')
            if core_mem:
                if learned_rep_np is not None: core_mem.store(learned_rep_np, {'timestamp': time.time(), 'loop': loop_count})
                relevant_mem = core_mem.retrieve(learned_rep_np, num_results=num_memories_to_retrieve)

            concepts = []; cog_core = cognition_modules.get('core_cognition')
            if cog_core and hasattr(cog_core, 'learning_module') and cog_core.learning_module:
                try:
                    ret_concepts = cog_core.learning_module.get_concepts()
                    if isinstance(ret_concepts, list): concepts = ret_concepts
                except Exception as e: logger.error(f"RUN_EVO (L{loop_count}): Error getting concepts: {e}", exc_info=False)
            
            decision = None
            if cog_core:
                try:
                    decision = cog_core.decide(processed_inputs, learned_rep_np, relevant_mem, concepts)
                    #if decision: logger.info(f"RUN_EVO (L{loop_count}): Decision: {str(decision)}")
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
            # Sadece herhangi bir görüntüleme aktifse waitKey çağır
            if show_camera_feed_active or show_processed_main_active or show_processed_edges_active:
                remaining_wait = loop_interval - elapsed
                wait_key_duration = int(remaining_wait * 1000) if remaining_wait > 0.001 else 1
                key_pressed = cv2.waitKey(wait_key_duration) & 0xFF
                if key_pressed == ord('q'): logger.info("RUN_EVO: 'q' pressed. Exiting."); break
                elif key_pressed == ord('p'):
                    logger.info("RUN_EVO: 'p' pressed. Pausing... Press 'p' again to resume.")
                    while True:
                        if cv2.waitKey(100) & 0xFF == ord('p'): logger.info("RUN_EVO: Resuming..."); break
            else: # Görüntüleme kapalıysa
                sleep_duration = loop_interval - elapsed
                if sleep_duration > 0: time.sleep(sleep_duration)
            
            final_elapsed = time.time() - start_time
            if final_elapsed > loop_interval * 1.25: 
                 logger.warning(f"RUN_EVO (L{loop_count}): Loop overrun: {final_elapsed:.4f}s (target: {loop_interval:.2f}s)")

    except KeyboardInterrupt: logger.warning("Ctrl+C detected. Evo shutting down...")
    except Exception as e: logger.critical(f"Evo main loop critical error: {e}", exc_info=True)
    finally:
        cleanup_modules(module_objects)
        # Herhangi bir görüntüleme aktifse tüm pencereleri kapat
        if show_camera_feed_active or show_processed_main_active or show_processed_edges_active: 
            cv2.destroyAllWindows()
    logger.info("Evo has been shut down.")

if __name__ == '__main__':
    run_evo()