# src/run_evo.py
#
# Evo projesinin ana çalıştırma noktası.
# Evo'nın temel yaşam döngüsünü (bilişsel döngü) başlatır ve yönetir.
# Konfigürasyonu yükler, modülleri başlatır, döngüyü çalıştırır ve kaynakları temizler.

import logging
import time
# NumPy, temel veri yapıları için gerekli, ancak doğrudan run_evo'da kullanılmıyor
# Modüllerin girdilerini/çıktılarını göstermek için loglarda kullanışlı olabilir.
import numpy as np
# import sys # Programı sonlandırmak gerekirse import edilebilir

# Loglama ve Konfigürasyon yardımcı fonksiyonlarını import et
from src.core.logging_utils import setup_logging
from src.core.config_utils import load_config_from_yaml, get_config_value
# Modül başlatma yardımcı fonksiyonlarını import et
from src.core.module_loader import initialize_modules, cleanup_modules


# Bu dosyanın kendi logger'ını oluştur (setup_logging'den sonra kullanılacak)
# __name__ 'src.run_evo' olacak
# Başlangıçta logger'ı None yapalım, setup_logging sonrası tekrar alacağız.
logger = None # type: logging.Logger


def run_evo():
    """
    Evo'nın çekirdek bilişsel döngüsünü ve arayüzlerini başlatır ve çalıştırır.
    Bu fonksiyon çağrıldığında Evo "canlanır".

    Adımlar:
    1. Loglama sistemini yapılandır.
    2. Yapılandırma dosyasını yükle.
    3. Ana modülleri (Sense, Process, Represent, Memory, Cognition, MotorControl, Interaction) başlat.
    4. Modül başlatma sırasında kritik hata yoksa ana bilişsel döngüyü çalıştır.
    5. Kullanıcı durdurduğunda (Ctrl+C) veya kritik bir hata oluştuğunda kaynakları temizle.
    """
    global logger # Global logger değişkenini kullanacağımızı belirt

    # --- Başlatma Aşaması ---

    # 1a. Konfigürasyonu Yükle (Loglama setup'ından önce yüklenmeli ki loglama config'i kullanılabilsin)
    config_path = "config/main_config.yaml" # Yapılandırma dosyasının yolu
    config = load_config_from_yaml(config_path)

    # 1b. Loglama sistemini yüklenen config ile yapılandır
    setup_logging(config=config) # <<< setup_logging'e yüklenen config gönderildi
    logger = logging.getLogger(__name__) # Logger objesini tekrar al

    # Config'in başarıyla yüklenip yüklenmediğini kontrol et
    if not config:
        logger.critical(f"Evo, yapılandırma yüklenemediği için başlatılamıyor. Lütfen {config_path} dosyasını kontrol edin.")
        return

    logger.info("Evo canlanıyor...")
    logger.info(f"Konfigürasyon başarıyla yüklendi: {config_path}")
    # logger.debug(f"Yüklenen Konfigürasyon Detayları: {config}") # Gerekirse açılabilir

    # 2. Modülleri Başlat
    module_objects, can_run_main_loop = initialize_modules(config)

    if not can_run_main_loop:
        logger.critical("Evo'nın temel modülleri başlatılamadığı için program sonlandırılıyor.")
        cleanup_modules(module_objects)
        return

    # Modül objelerine kolay erişim
    sensors = module_objects.get('sensors', {})
    processors = module_objects.get('processors', {})
    representers = module_objects.get('representers', {})
    memories = module_objects.get('memories', {})
    cognition_modules = module_objects.get('cognition', {})
    motor_control_modules = module_objects.get('motor_control', {})
    interaction_modules = module_objects.get('interaction', {})

    logger.info("Evo bilişsel döngüye hazır.")

    # --- Ana Bilişsel Döngü (Yaşam Döngüsü) ---
    if can_run_main_loop:
        logger.info("Evo'nın bilişsel döngüsü başlatıldı...")
        loop_interval = get_config_value(config, 'cognitive_loop_interval', default=0.1, expected_type=(float, int), logger_instance=logger)

        if not isinstance(loop_interval, (int, float)) or loop_interval <= 0:
             logger.warning(f"RUN_EVO: Konfigürasyondan alınan geçersiz cognitive_loop_interval değeri ({loop_interval}). Varsayılan 0.1 kullanılıyor.")
             loop_interval = 0.1
        else:
             loop_interval = float(loop_interval)

        num_memories_to_retrieve = get_config_value(config, 'memory', 'num_retrieved_memories', default=5, expected_type=int, logger_instance=logger)
        if num_memories_to_retrieve < 0:
             logger.warning(f"RUN_EVO: Konfigürasyondan alınan negatif num_retrieved_memories değeri ({num_memories_to_retrieve}). Varsayılan 5 kullanılıyor.")
             num_memories_to_retrieve = 5

        try:
            while True:
                start_time = time.time()

                # --- Bilgi Akışı (Sense -> Process -> Represent -> Memory -> Cognition -> Motor -> Interact) ---

                # Duyu Verisini Yakala (Faz 0)
                raw_inputs = {}
                vision_sensor = sensors.get('vision')
                audio_sensor = sensors.get('audio')
                if vision_sensor:
                    raw_inputs['visual'] = vision_sensor.capture_frame()
                if audio_sensor:
                    raw_inputs['audio'] = audio_sensor.capture_chunk()

                # DEBUG Log: Ham Veri
                # (Loglama seviyesi DEBUG olduğunda bu loglar görünür)
                if raw_inputs.get('visual') is not None: logger.debug(f"RUN_EVO: Raw Visual Input shape: {raw_inputs['visual'].shape}")
                if raw_inputs.get('audio') is not None: logger.debug(f"RUN_EVO: Raw Audio Input shape: {raw_inputs['audio'].shape}")

                # Yakalanan Veriyi İşle (Faz 1 Başlangıcı)
                processed_inputs = {}
                vision_processor = processors.get('vision')
                audio_processor = processors.get('audio')
                if vision_processor:
                     processed_inputs['visual'] = vision_processor.process(raw_inputs.get('visual'))
                if audio_processor:
                     processed_inputs['audio'] = audio_processor.process(raw_inputs.get('audio'))

                # DEBUG Log: İşlenmiş Veri
                vis_proc_out = processed_inputs.get('visual')
                aud_proc_out = processed_inputs.get('audio')
                if isinstance(vis_proc_out, dict): logger.debug(f"RUN_EVO: Processed Visual Output keys: {list(vis_proc_out.keys())}")
                if isinstance(aud_proc_out, np.ndarray): logger.debug(f"RUN_EVO: Processed Audio Output shape: {aud_proc_out.shape}")


                # İşlenmiş Veriden Temsil Öğren (Faz 1 Devamı)
                learned_representation = None
                representation_learner = representers.get('main_learner')
                if representation_learner:
                     learned_representation = representation_learner.learn(processed_inputs)

                # DEBUG Log: Öğrenilmiş Temsil
                if isinstance(learned_representation, np.ndarray): logger.debug(f"RUN_EVO: Learned Representation shape: {learned_representation.shape}")

                # Temsili Hafızaya Kaydet ve Hafızadan Bilgi Al (Faz 2)
                relevant_memory_entries = []
                core_memory_instance = memories.get('core_memory')
                if core_memory_instance:
                     if learned_representation is not None: # Sadece geçerli bir temsil varsa sakla
                          core_memory_instance.store(learned_representation, metadata={'timestamp': time.time()}) # Metadata eklenebilir
                     relevant_memory_entries = core_memory_instance.retrieve(learned_representation, num_results=num_memories_to_retrieve)

                # DEBUG Log: Geri Çağrılan Bellek Girdileri
                if relevant_memory_entries: logger.debug(f"RUN_EVO: Retrieved {len(relevant_memory_entries)} memories.")


                # <<< HATA DÜZELTME BAŞLANGICI: current_concepts alınması >>>
                current_concepts = [] # Başlangıçta boş
                core_cognition_instance = cognition_modules.get('core_cognition')
                if core_cognition_instance and hasattr(core_cognition_instance, 'learning_module'):
                    learning_module_instance = core_cognition_instance.learning_module
                    if learning_module_instance:
                        try:
                            concepts = learning_module_instance.get_concepts()
                            if isinstance(concepts, list):
                                 current_concepts = concepts
                                 if concepts: logger.debug(f"RUN_EVO: Using {len(concepts)} current concepts for decision.")
                            else:
                                 logger.warning("RUN_EVO: LearningModule.get_concepts() liste döndürmedi. Boş liste kullanılıyor.")
                                 current_concepts = []
                        except AttributeError:
                             logger.warning("RUN_EVO: LearningModule'de 'get_concepts' metodu bulunamadı.")
                             current_concepts = []
                        except Exception as e:
                             logger.error(f"RUN_EVO: LearningModule.get_concepts çağrılırken hata: {e}", exc_info=False) # exc_info=False for brevity in loop
                             current_concepts = []
                    else:
                        logger.debug("RUN_EVO: CognitionCore içinde LearningModule bulunamadı veya None.")
                else:
                     logger.debug("RUN_EVO: CognitionCore veya LearningModule bulunamadı, kavramlar kullanılamıyor.")
                # <<< HATA DÜZELTME SONU >>>


                # Hafıza ve temsile göre Bilişsel işlem yap (Faz 3)
                decision = None
                # core_cognition_instance yukarıda zaten alındı.
                if core_cognition_instance:
                     try:
                         decision = core_cognition_instance.decide(
                             processed_inputs,
                             learned_representation,
                             relevant_memory_entries,
                             current_concepts
                         )
                         if decision: logger.debug(f"RUN_EVO: Cognition decision: {decision}")
                     except Exception as e:
                          logger.error(f"RUN_EVO: CognitionCore.decide sırasında hata: {e}", exc_info=False) # exc_info=False for brevity in loop


                # Karara göre bir Tepki Üret (Faz 3 Devamı)
                response_output = None
                core_motor_control_instance = motor_control_modules.get('core_motor_control')
                if core_motor_control_instance:
                     try:
                        response_output = core_motor_control_instance.generate_response(decision)
                        if response_output: logger.debug(f"RUN_EVO: Generated response: {response_output[:100]}...") # Log yanıtın başını gösterir
                     except Exception as e:
                          logger.error(f"RUN_EVO: MotorControlCore.generate_response sırasında hata: {e}", exc_info=False)


                # Tepkiyi Dışarı Aktar (Faz 3 Tamamlanması)
                core_interaction_instance = interaction_modules.get('core_interaction')
                if core_interaction_instance and response_output is not None: # Sadece bir yanıt varsa gönder
                   try:
                      core_interaction_instance.send_output(response_output)
                      logger.debug("RUN_EVO: Response sent to interaction channels.")
                   except Exception as e:
                       logger.error(f"RUN_EVO: InteractionAPI.send_output sırasında hata: {e}", exc_info=False)


                # --- Döngü Gecikmesi ---
                elapsed_time = time.time() - start_time
                sleep_time = loop_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    if elapsed_time > loop_interval * 1.5: # Hedefin %150'sinden uzun sürerse uyar
                         logger.warning(f"RUN_EVO: Cognitive loop took longer than expected: {elapsed_time:.4f}s (target: {loop_interval:.2f}s)")

        except KeyboardInterrupt:
            logger.warning("Ctrl+C algılandı. Evo durduruluyor...")
        except Exception as e:
            logger.critical(f"Evo'nın ana döngüsünde beklenmedik kritik hata: {e}", exc_info=True)
        finally:
            # --- Kaynakları Temizleme ---
            cleanup_modules(module_objects)

        logger.info("Evo durduruldu.")


# Ana çalıştırma noktası
if __name__ == '__main__':
    run_evo()