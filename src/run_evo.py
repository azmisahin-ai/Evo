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

# Kamera görüntüsünü gösterme bayrağı (config'den de alınabilir)
SHOW_CAMERA_FEED_CONFIG_KEY = ('debug_flags', 'show_camera_feed') # Örnek bir config yolu
# Eğer config'de yoksa varsayılan olarak True/False ayarlanabilir.
# Şimdilik sabit bırakalım:
SHOW_CAMERA_FEED = False # Üretim benzeri için False, debug için True yapabilirsiniz.

def run_evo():
    global logger

    config_path = "config/main_config.yaml"
    config = load_config_from_yaml(config_path)

    setup_logging(config=config) 
    logger = logging.getLogger(__name__) 

    if not config:
        logging.critical(f"Evo cannot start: Configuration file {config_path} failed to load.")
        return

    initialize_compute_backend(config)
    current_backend = get_backend()
    current_device = get_device()

    # Kamera feed gösterimi için config'den değer oku (opsiyonel)
    # show_feed_flag = get_config_value(config, *SHOW_CAMERA_FEED_CONFIG_KEY, default=SHOW_CAMERA_FEED, expected_type=bool)
    # Global SHOW_CAMERA_FEED'i override et
    # global SHOW_CAMERA_FEED
    # SHOW_CAMERA_FEED = show_feed_flag
    # logger.info(f"Camera feed display: {'Enabled' if SHOW_CAMERA_FEED else 'Disabled'}")


    logger.info(f"Evo awakening... Backend: {current_backend}, Device: {str(current_device)}")
    logger.info(f"Configuration loaded from: {config_path}")

    module_objects, can_run_main_loop = initialize_modules(config)

    if not can_run_main_loop:
        logger.critical("Evo's core modules failed to initialize. Shutting down.")
        cleanup_modules(module_objects)
        if SHOW_CAMERA_FEED: cv2.destroyAllWindows()
        return

    sensors = module_objects.get('sensors', {})
    processors = module_objects.get('processors', {})
    representers = module_objects.get('representers', {})
    memories = module_objects.get('memories', {})
    cognition_modules = module_objects.get('cognition', {})
    motor_control_modules = module_objects.get('motor_control', {})
    interaction_modules = module_objects.get('interaction', {})

    logger.info("Evo cognitive loop ready.")
    loop_interval_cfg = get_config_value(config, 'cognitive_loop_interval', default=0.15, expected_type=(float, int, np.floating), logger_instance=logger)
    loop_interval = float(loop_interval_cfg) if isinstance(loop_interval_cfg, (float, int, np.floating)) and float(loop_interval_cfg) > 0 else 0.15

    num_memories_to_retrieve_cfg = get_config_value(config, 'memory', 'num_retrieved_memories', default=5, expected_type=(int, np.integer), logger_instance=logger)
    num_memories_to_retrieve = int(num_memories_to_retrieve_cfg) if isinstance(num_memories_to_retrieve_cfg, (int, np.integer)) and int(num_memories_to_retrieve_cfg) >=0 else 5
    
    representation_config = config.get('representation', {})
    expected_input_order_for_learner = representation_config.get('_expected_input_order_', []) 
    if expected_input_order_for_learner: # Sadece doluysa logla
        logger.info(f"RUN_EVO: Expected input order for learner: {expected_input_order_for_learner}")
    
    logger.info("RUN_EVO: RepresentationLearner (Autoencoder) will be used and trained.")

    try:
        loop_count = 0
        while True:
            loop_count += 1
            start_time = time.time()

            raw_inputs = {}
            current_raw_frame = None # Görüntüleme için
            vision_sensor = sensors.get('vision')
            if vision_sensor: 
                raw_inputs['visual'] = vision_sensor.capture_frame()
                if SHOW_CAMERA_FEED : current_raw_frame = raw_inputs['visual'] # Sadece gösterilecekse kopyala
            
            audio_sensor = sensors.get('audio')
            if audio_sensor: 
                raw_inputs['audio'] = audio_sensor.capture_chunk()

            if SHOW_CAMERA_FEED and current_raw_frame is not None:
                try:
                    cv2.imshow("Evo Camera Feed", current_raw_frame)
                except Exception as e_imshow:
                    logger.error(f"Error displaying camera feed: {e_imshow}", exc_info=False)

            processed_inputs = {}
            vision_processor = processors.get('vision')
            if vision_processor and raw_inputs.get('visual') is not None:
                 processed_inputs['visual'] = vision_processor.process(raw_inputs['visual'])
            
            audio_processor = processors.get('audio')
            if audio_processor and raw_inputs.get('audio') is not None:
                 processed_inputs['audio'] = audio_processor.process(raw_inputs['audio'])
            
            combined_features_list = []
            vis_out = processed_inputs.get('visual', {})
            aud_out = processed_inputs.get('audio')
            has_valid_feature_for_concat = False

            for feature_key_full in expected_input_order_for_learner:
                source_module, feature_name = feature_key_full.split('_', 1)
                feature_data_to_add = None
                if source_module == "vision" and isinstance(vis_out, dict): feature_data_to_add = vis_out.get(feature_name)
                elif source_module == "audio" and isinstance(aud_out, dict): feature_data_to_add = aud_out.get(feature_name)
                elif source_module == "audio" and feature_name == "audio_features": feature_data_to_add = aud_out # Doğrudan tensor/array ise

                if feature_data_to_add is not None:
                    if isinstance(feature_data_to_add, (torch.Tensor, np.ndarray)):
                         combined_features_list.append(feature_data_to_add.flatten())
                         has_valid_feature_for_concat = True
                    else: logger.warning(f"RUN_EVO (L{loop_count}): Feat '{feature_key_full}' type {type(feature_data_to_add)} not Tensor/Array.")
                else: logger.debug(f"RUN_EVO (L{loop_count}): Expected feat '{feature_key_full}' not found for concat.")
            
            if not has_valid_feature_for_concat and combined_features_list: combined_features_list.clear()

            learned_representation_np = None
            representation_learner = representers.get('main_learner')
            final_combined_input = None

            if combined_features_list:
                if current_backend == "pytorch":
                    try:
                        # Birleştirme CPU'da yapılabilir, RepresentationLearner kendi cihazına alır.
                        target_concat_device = torch.device("cpu") 
                        tensors_on_device = []
                        for t_data in combined_features_list: # Flatten zaten yapıldı
                            if isinstance(t_data, torch.Tensor): tensors_on_device.append(t_data.to(target_concat_device))
                            elif isinstance(t_data, np.ndarray): tensors_on_device.append(torch.from_numpy(t_data).float().to(target_concat_device))
                        if tensors_on_device: final_combined_input = torch.cat(tensors_on_device)
                        else: logger.warning(f"RUN_EVO (L{loop_count}): No valid PyTorch tensors to concat.")
                    except Exception as e: logger.error(f"RUN_EVO (L{loop_count}): Error concat PyTorch tensors: {e}", exc_info=True)
                elif current_backend == "numpy":
                    # ... (NumPy birleştirme kodu aynı kalır) ...
                    try:
                        valid_arrays = [arr for arr in combined_features_list if isinstance(arr, np.ndarray)]
                        if valid_arrays: final_combined_input = np.concatenate(valid_arrays)
                        else: logger.warning(f"RUN_EVO (L{loop_count}): No valid NumPy arrays to concat.")
                    except Exception as e: logger.error(f"RUN_EVO (L{loop_count}): Error concat NumPy arrays: {e}", exc_info=True)
            
            if final_combined_input is not None:
                if representation_learner:
                    output_from_learner = representation_learner.learn(final_combined_input) # Artık tensor veya array dönebilir
                    if output_from_learner is not None:
                        learned_representation_np = to_numpy(output_from_learner) # Her zaman NumPy'a çevir
                        if learned_representation_np is not None:
                            logger.debug(f"RUN_EVO (L{loop_count}): Latent Rep (NP) Shape: {learned_representation_np.shape}")
                        else: logger.warning(f"RUN_EVO (L{loop_count}): Conversion of learner output to NumPy failed.")
                    else: logger.warning(f"RUN_EVO (L{loop_count}): RepresentationLearner.learn returned None.")
                else: logger.error(f"RUN_EVO (L{loop_count}): RepresentationLearner is None.")
            # ... (Kalan Memory, Cognition, MotorControl, Interaction aynı) ...
            relevant_memory_entries = []
            core_memory_instance = memories.get('core_memory')
            if core_memory_instance:
                 if learned_representation_np is not None:
                      core_memory_instance.store(learned_representation_np, metadata={'timestamp': time.time(), 'loop': loop_count})
                 relevant_memory_entries = core_memory_instance.retrieve(learned_representation_np, num_results=num_memories_to_retrieve)

            current_concepts = []
            core_cognition_instance = cognition_modules.get('core_cognition')
            if core_cognition_instance and hasattr(core_cognition_instance, 'learning_module') and core_cognition_instance.learning_module:
                try:
                    concepts = core_cognition_instance.learning_module.get_concepts()
                    if isinstance(concepts, list): current_concepts = concepts
                except Exception as e:
                     logger.error(f"RUN_EVO (L{loop_count}): Error getting concepts: {e}", exc_info=False)
            
            decision = None
            if core_cognition_instance:
                 try:
                     decision = core_cognition_instance.decide(
                         processed_inputs, 
                         learned_representation_np, 
                         relevant_memory_entries,   
                         current_concepts           
                     )
                     #if decision: logger.info(f"RUN_EVO (L{loop_count}): Cognition decision: {str(decision)}")
                 except Exception as e:
                      logger.error(f"RUN_EVO (L{loop_count}): CognitionCore.decide error: {e}", exc_info=False)

            response_output = None
            core_motor_control_instance = motor_control_modules.get('core_motor_control')
            if core_motor_control_instance:
                 try:
                    response_output = core_motor_control_instance.generate_response(decision)
                    if response_output: logger.debug(f"RUN_EVO (L{loop_count}): Generated response: {str(response_output)[:100]}...")
                 except Exception as e:
                      logger.error(f"RUN_EVO (L{loop_count}): MotorControlCore.generate_response error: {e}", exc_info=False)

            core_interaction_instance = interaction_modules.get('core_interaction')
            if core_interaction_instance and response_output is not None:
               try:
                  core_interaction_instance.send_output(response_output)
               except Exception as e:
                   logger.error(f"RUN_EVO (L{loop_count}): InteractionAPI.send_output error: {e}", exc_info=False)

            elapsed_time = time.time() - start_time
            wait_key_time = 1 
            if SHOW_CAMERA_FEED:
                remaining_time_for_waitkey = loop_interval - elapsed_time
                wait_key_time = int(remaining_time_for_waitkey * 1000) if remaining_time_for_waitkey > 0.001 else 1
                key_pressed = cv2.waitKey(wait_key_time) & 0xFF
                if key_pressed == ord('q'):
                    logger.info("RUN_EVO: 'q' key pressed. Exiting loop.")
                    break 
                elif key_pressed == ord('p'): # 'p' tuşu ile duraklat/devam et (opsiyonel)
                    logger.info("RUN_EVO: 'p' key pressed. Pausing... Press 'p' again to resume.")
                    while True:
                        if cv2.waitKey(100) & 0xFF == ord('p'):
                            logger.info("RUN_EVO: Resuming...")
                            break
            else:
                sleep_time = loop_interval - elapsed_time
                if sleep_time > 0: time.sleep(sleep_time)

            final_elapsed_time = time.time() - start_time
            if final_elapsed_time > loop_interval * 1.2: 
                 logger.warning(f"RUN_EVO (L{loop_count}): Loop overrun: {final_elapsed_time:.4f}s (target: {loop_interval:.2f}s)")

    except KeyboardInterrupt:
        logger.warning("Ctrl+C detected. Evo shutting down...")
    except Exception as e:
        logger.critical(f"Evo main loop critical error: {e}", exc_info=True)
    finally:
        cleanup_modules(module_objects)
        if SHOW_CAMERA_FEED: cv2.destroyAllWindows()

    logger.info("Evo has been shut down.")

if __name__ == '__main__':
    run_evo()