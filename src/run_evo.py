# src/run_evo.py
import logging
import time
import numpy as np
import torch # PyTorch işlemleri için

from src.core.logging_utils import setup_logging
from src.core.config_utils import load_config_from_yaml, get_config_value
from src.core.module_loader import initialize_modules, cleanup_modules
from src.core.compute_utils import initialize_compute_backend, get_backend, get_device, to_numpy # to_numpy önemli

logger = None 

def run_evo():
    global logger 

    config_path = "config/main_config.yaml"
    config = load_config_from_yaml(config_path)

    # Loglama config yüklendikten sonra, compute_backend'den önce başlatılmalı
    setup_logging(config=config) 
    logger = logging.getLogger(__name__) # Ana logger'ı al

    if not config:
        # setup_logging config yoksa default logger kullanır, bu yüzden burada loglayabiliriz
        logging.critical(f"Evo cannot start: Configuration file {config_path} failed to load.")
        return

    # Hesaplama backend'ini ve cihazını başlat
    initialize_compute_backend(config)
    current_backend = get_backend()
    current_device = get_device() # PyTorch için torch.device objesi, NumPy için "cpu" string'i

    logger.info(f"Evo awakening... Backend: {current_backend}, Device: {current_device}")
    logger.info(f"Configuration loaded from: {config_path}")

    module_objects, can_run_main_loop = initialize_modules(config)

    if not can_run_main_loop:
        logger.critical("Evo's core modules failed to initialize. Shutting down.")
        cleanup_modules(module_objects) # Deneyebileceği kadarını temizle
        return

    # Modül objelerine kolay erişim
    sensors = module_objects.get('sensors', {})
    processors = module_objects.get('processors', {})
    representers = module_objects.get('representers', {})
    memories = module_objects.get('memories', {})
    cognition_modules = module_objects.get('cognition', {})
    motor_control_modules = module_objects.get('motor_control', {})
    interaction_modules = module_objects.get('interaction', {})

    logger.info("Evo cognitive loop ready.")
    loop_interval = get_config_value(config, 'cognitive_loop_interval', default=0.1, expected_type=(float, int), logger_instance=logger)
    loop_interval = float(loop_interval) if isinstance(loop_interval, (int, float)) and loop_interval > 0 else 0.1

    num_memories_to_retrieve = get_config_value(config, 'memory', 'num_retrieved_memories', default=5, expected_type=int, logger_instance=logger)
    num_memories_to_retrieve = num_memories_to_retrieve if num_memories_to_retrieve >=0 else 5
    
    # RepresentationLearner'dan beklenen girdi sırasını al (debug/doğrulama için)
    representation_config = config.get('representation', {})
    expected_input_order_for_learner = representation_config.get('_expected_input_order_', [])


    try:
        while True:
            start_time = time.time()

            raw_inputs = {}
            vision_sensor = sensors.get('vision')
            audio_sensor = sensors.get('audio') # AudioSensor'ı al
            if vision_sensor: raw_inputs['visual'] = vision_sensor.capture_frame()
            if audio_sensor: raw_inputs['audio'] = audio_sensor.capture_chunk() # audio_sensor.capture_chunk()

            processed_inputs = {}
            vision_processor = processors.get('vision')
            audio_processor = processors.get('audio') # audio_processor'ı al
            if vision_processor and raw_inputs.get('visual') is not None:
                 processed_inputs['visual'] = vision_processor.process(raw_inputs['visual'])
            if audio_processor and raw_inputs.get('audio') is not None:
                 processed_inputs['audio'] = audio_processor.process(raw_inputs['audio'])
            
            # İşlemci çıktılarını RepresentationLearner için birleştir
            # Bu birleştirme, module_loader'da hesaplanan input_dim sırasına uygun olmalı
            # VisionProcessor: {'main_image': tensor, 'edges': tensor} (C,H,W)
            # AudioProcessor: tensor (Dim,)
            
            combined_features_list = []
            vis_out = processed_inputs.get('visual', {}) # dict döner
            aud_out = processed_inputs.get('audio')      # tensor/array döner

            # module_loader'dan gelen expected_input_order_for_learner listesine göre birleştir.
            # Bu, sıranın tutarlı olmasını sağlar.
            for feature_key in expected_input_order_for_learner:
                source, name = feature_key.split('_', 1) # "vision_main_image" -> ("vision", "main_image")
                
                feature_data = None
                if source == "vision" and isinstance(vis_out, dict):
                    feature_data = vis_out.get(name)
                elif source == "audio": # Audio tek bir özellik seti döndürür, ismi 'audio_features' idi.
                    if name == "audio_features": # AudioProcessor.get_output_shape_info()'daki anahtar
                        feature_data = aud_out
                
                if feature_data is not None:
                    # PyTorch backend'inde tüm girdiler zaten tensör olmalı.
                    # flatten() işlemi PyTorch tensörleri için de çalışır.
                    if isinstance(feature_data, torch.Tensor) or isinstance(feature_data, np.ndarray):
                         combined_features_list.append(feature_data.flatten())
                    else:
                         logger.warning(f"RUN_EVO: Feature '{feature_key}' is not a Tensor/Array. Type: {type(feature_data)}. Skipping.")
                else:
                    logger.warning(f"RUN_EVO: Expected feature '{feature_key}' not found in processed_inputs. Concatenation might be incorrect.")
                    # Eksik özellik durumunda, input_dim eşleşmeyebilir. Hata vermek veya sıfırlarla doldurmak gerekebilir.
                    # Şimdilik logla ve devam et. RepresentationLearner boyut hatası verecektir.

            learned_representation_np = None # Sonuç NumPy array olacak
            representation_learner = representers.get('main_learner')

            if representation_learner and combined_features_list:
                final_combined_input = None
                if current_backend == "pytorch":
                    try:
                        # Tüm tensörlerin aynı cihazda olduğundan emin ol (RL'nin cihazı)
                        learner_device = representation_learner.device 
                        tensors_on_device = [t.to(learner_device) for t in combined_features_list if isinstance(t, torch.Tensor)]
                        if tensors_on_device:
                             final_combined_input = torch.cat(tensors_on_device)
                        else:
                             logger.warning("RUN_EVO: No valid PyTorch tensors to concatenate for learner.")
                    except Exception as e:
                        logger.error(f"RUN_EVO: Error concatenating PyTorch tensors for learner: {e}", exc_info=True)
                
                elif current_backend == "numpy":
                    try:
                        valid_arrays = [arr for arr in combined_features_list if isinstance(arr, np.ndarray)]
                        if valid_arrays:
                            final_combined_input = np.concatenate(valid_arrays)
                        else:
                            logger.warning("RUN_EVO: No valid NumPy arrays to concatenate for learner.")
                    except Exception as e:
                        logger.error(f"RUN_EVO: Error concatenating NumPy arrays for learner: {e}", exc_info=True)
                
                if final_combined_input is not None:
                    # RepresentationLearner.learn metodu, backend'e göre ya tensor ya da np.array alıp,
                    # backend'e uygun bir çıktı (PyTorch RL için tensor, NumPy RL için array) verir.
                    output_from_learner = representation_learner.learn(final_combined_input)
                    
                    # Çıktıyı her zaman NumPy'a çevir (Memory ve sonrası için)
                    if output_from_learner is not None:
                        learned_representation_np = to_numpy(output_from_learner) 
                        if learned_representation_np is not None:
                            logger.debug(f"RUN_EVO: Learned Representation (NumPy). Shape: {learned_representation_np.shape}")
                        else:
                            logger.warning("RUN_EVO: Conversion of learner output to NumPy failed.")
                    else:
                        logger.warning("RUN_EVO: RepresentationLearner.learn returned None.")
                else:
                    logger.debug("RUN_EVO: No combined input to send to RepresentationLearner.")
            elif representation_learner: # RL var ama combined_features_list boş
                logger.debug("RUN_EVO: No features to combine for RepresentationLearner.")


            relevant_memory_entries = []
            core_memory_instance = memories.get('core_memory')
            if core_memory_instance:
                 if learned_representation_np is not None:
                      core_memory_instance.store(learned_representation_np, metadata={'timestamp': time.time()})
                 relevant_memory_entries = core_memory_instance.retrieve(learned_representation_np, num_results=num_memories_to_retrieve)

            current_concepts = []
            core_cognition_instance = cognition_modules.get('core_cognition')
            if core_cognition_instance and hasattr(core_cognition_instance, 'learning_module') and core_cognition_instance.learning_module:
                try:
                    concepts = core_cognition_instance.learning_module.get_concepts()
                    if isinstance(concepts, list): current_concepts = concepts
                except Exception as e:
                     logger.error(f"RUN_EVO: Error getting concepts: {e}", exc_info=False)
            
            decision = None
            if core_cognition_instance:
                 try:
                     decision = core_cognition_instance.decide(
                         processed_inputs, # Ham işlenmiş girdiler (dict of tensors/arrays)
                         learned_representation_np, # NumPy array
                         relevant_memory_entries,   # List of NumPy arrays
                         current_concepts           # List
                     )
                     if decision: logger.debug(f"RUN_EVO: Cognition decision: {str(decision)[:100]}...")
                 except Exception as e:
                      logger.error(f"RUN_EVO: CognitionCore.decide error: {e}", exc_info=False)

            response_output = None
            core_motor_control_instance = motor_control_modules.get('core_motor_control')
            if core_motor_control_instance:
                 try:
                    response_output = core_motor_control_instance.generate_response(decision)
                    if response_output: logger.debug(f"RUN_EVO: Generated response: {str(response_output)[:100]}...")
                 except Exception as e:
                      logger.error(f"RUN_EVO: MotorControlCore.generate_response error: {e}", exc_info=False)

            core_interaction_instance = interaction_modules.get('core_interaction')
            if core_interaction_instance and response_output is not None:
               try:
                  core_interaction_instance.send_output(response_output)
               except Exception as e:
                   logger.error(f"RUN_EVO: InteractionAPI.send_output error: {e}", exc_info=False)

            elapsed_time = time.time() - start_time
            sleep_time = loop_interval - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif elapsed_time > loop_interval * 1.2: # %20 tolerans
                 logger.warning(f"RUN_EVO: Cognitive loop overrun: {elapsed_time:.4f}s (target: {loop_interval:.2f}s)")

    except KeyboardInterrupt:
        logger.warning("Ctrl+C detected. Evo shutting down...")
    except Exception as e:
        logger.critical(f"Evo main loop critical error: {e}", exc_info=True)
    finally:
        cleanup_modules(module_objects)

    logger.info("Evo has been shut down.")

if __name__ == '__main__':
    run_evo()