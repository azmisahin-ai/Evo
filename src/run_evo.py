# src/run_evo.py

import logging
import time
import numpy as np

# Loglama ve Konfigürasyon yardımcı fonksiyonlarını import et
from src.core.logging_utils import setup_logging # Mevcut import
from src.core.config_utils import load_config_from_yaml # <<< Yeni import

# Temel modülleri import edeceğiz
from src.senses.vision import VisionSensor
from src.senses.audio import AudioSensor
# Processing modülleri import et
from src.processing.vision import VisionProcessor
from src.processing.audio import AudioProcessor
# Representation modüllerini import et
from src.representation.models import RepresentationLearner
# Memory modülü import et
from src.memory.core import Memory # core.py dosyasındaki Memory sınıfı
# Cognition modülü import et
from src.cognition.core import CognitionCore # core.py dosyasındaki CognitionCore sınıfı
# Motor Control modülü import et
from src.motor_control.core import MotorControlCore # core.py dosyasındaki MotorControlCore sınıfı
# Interaction modülü import et
from src.interaction.api import InteractionAPI # api.py dosyasındaki InteractionAPI sınıfı


# Konfigürasyon yükleme fonksiyonu kaldırıldı, yerine load_config_from_yaml kullanılacak.
# def load_config(): # <<< BU FONKSİYON SİLİNDİ
#     """Basit placeholder konfigürasyon yükleyici."""
#     ... (içeriği silin) ...
#     return { ... }

def run_evo():
    """
    Evo'nın çekirdek bilişsel döngüsünü ve arayüzlerini başlatır.
    Bu fonksiyon çağrıldığında Evo "canlanır".
    """
    # Loglama sistemini merkezi utility ile yapılandır (Her zaman ilk adım olmalı)
    setup_logging(level=logging.DEBUG) # DEBUG loglarını görmek için DEBUG seviyesiyle başlat

    # Bu dosyanın kendi logger'ını oluştur (setup_logging'den sonra)
    # __name__ 'src.run_evo' olacak
    logger = logging.getLogger(__name__)

    logger.info("Evo canlanıyor...")

    # --- Konfigürasyonu Yükle --- <<< Burası güncellendi
    config_path = "config/main_config.yaml" # Yapılandırma dosyasının yolu
    config = load_config_from_yaml(config_path) # <<< Yeni yükleme fonksiyonu çağrıldı

    if not config: # Eğer config yüklenemezse (dosya yok veya hata var)
        logger.critical(f"Evo, yapılandırma yüklenemediği için başlatılamıyor. Lütfen {config_path} dosyasını kontrol edin.")
        # Burada programı sonlandırmak mantıklı olabilir
        # sys.exit(1) # Eğer sys import ediliyorsa
        return # run_evo fonksiyonundan çık


    logger.info(f"Konfigürasyon yüklendi: {config_path}") # <<< Log mesajı güncellendi

    # Modül objelerini depolamak için sözlükler
    sensors = {}
    processors = {}
    representers = {}
    memories = {}
    cognition_modules = {}
    motor_control_modules = {}
    interaction_modules = {}


    # Başlatma başarılı olduysa main loop'u çalıştırmak için flag
    can_run_main_loop = True

    # --- Modülleri Başlatma ---
    logger.info("Modüller başlatılıyor...")

    # Faz 0: Duyusal Sensörleri Başlat - Bireysel Hata Yönetimi ile
    logger.info("Faz 0: Duyusal sensörler başlatılıyor (Bireysel hata yönetimi aktif)...")
    try:
        # Config'ten ilgili bölümleri alırken .get() kullanmak güvenlidir,
        # yaml dosyası eksik bölümler içerse bile hata vermez.
        logger.info("VisionSensor başlatılıyor...")
        sensors['vision'] = VisionSensor(config.get('vision', {}))
        if not (sensors.get('vision') and getattr(sensors['vision'], 'is_camera_available', False)):
             logger.warning("VisionSensor tam başlatılamadı veya kamera açılamadı.")
             # Simüle edilmiş girdi kullanılacaksa objenin is_camera_available False olacak. Objeyi None yapma!
    except Exception as e:
        logger.critical(f"VisionSensor başlatılırken kritik hata oluştu: {e}", exc_info=True)
        sensors['vision'] = None # Kritik hata durumunda objeyi None yap


    try:
        logger.info("AudioSensor başlatılıyor...")
        sensors['audio'] = AudioSensor(config.get('audio', {}))
        if not (sensors.get('audio') and getattr(sensors['audio'], 'is_audio_available', False)):
             logger.warning("AudioSensor tam başlatılamadı veya ses akışı aktif değil.")
             # Simüle edilmiş girdi kullanılacaksa objenin is_audio_available False olacak. Objeyi None yapma!
    except Exception as e:
        logger.critical(f"AudioSensor başlatılırken kritik hata oluştu: {e}", exc_info=True)
        sensors['audio'] = None # Kritik hata durumunda objeyi None yap


    # Başlatma durumunu kontrol et ve logla (Sözlükteki objelerin None olup olmadığına ve internal flag'lere bakarak)
    # Sadece None olmayan ve internal flag'i True olan sensörleri say
    active_sensors = [name for name, sensor in sensors.items()
                      if sensor and (getattr(sensor, 'is_camera_available', False) or getattr(sensor, 'is_audio_available', False))]
    if active_sensors:
        logger.info(f"Duyusal Sensörler başlatıldı ({', '.join(active_sensors)} aktif).")
    else:
        logger.warning("Hiçbir duyusal sensör aktif değil. Evo girdi alamayacak.")


    # Faz 1 Başlangici: Processing Modülleri Başlat
    logger.info("Faz 1: Processing modülleri başlatılıyor...")
    try:
        processors['vision'] = VisionProcessor(config.get('processing_vision', {})) # <<< Config'ten alındı
        processors['audio'] = AudioProcessor(config.get('processing_audio', {})) # <<< Config'ten alındı
        logger.info("Processing modülleri başarıyla başlatıldı.")
    except Exception as e:
         logger.critical(f"Processing modülleri başlatılırken kritik hata oluştu: {e}", exc_info=True)
         # Processing kritik hata verirse sensörler çalışsa bile anlamı yok, main loop'u engelle.
         can_run_main_loop = False


    # Faz 1 Devami: Representation Modülleri Başlat
    logger.info("Faz 1: Representation modülleri başlatılıyor...")
    # Representation modülü processing çıktısına bağımlı, eğer processing yoksa başlatmanın anlamı yok.
    # Veya dummy/placeholder representation modülü başlatılabilir.
    # Şimdilik processing başarılıysa başlatmayı deneyelim.
    if can_run_main_loop: # Eğer processing başlatma hatası olmadıysa
        try:
            representers['main_learner'] = RepresentationLearner(config.get('representation', {})) # <<< Config'ten alındı
            logger.info("Representation modülü başarıyla başlatıldı.")
        except Exception as e:
            logger.critical(f"Representation modülü başlatılırken kritik hata oluştu: {e}", exc_info=True)
            can_run_main_loop = False # Representation kritik hata verirse main loop'u engelle.
    else:
         # Bu durum loglanmalı, çünkü can_run_main_loop zaten False yapıldı.
         logger.warning("Processing modülleri başlatılamadığı için Representation modülleri atlandı.")


    # Faz 2 Başlangici: Memory Modülü Başlat
    logger.info("Faz 2: Memory modülü başlatılıyor...")
    # Memory modülü representation'a bağımlı. Eğer representation yoksa başlatmanın anlamı yok.
    if can_run_main_loop: # Eğer processing ve representation başlatma hatası olmadıysa
        try:
            memories['core_memory'] = Memory(config.get('memory', {})) # <<< Config'ten alındı
            logger.info("Memory modülü başarıyla başlatıldı.")
        except Exception as e:
            logger.critical(f"Memory modülü başlatılırken kritik hata oluştu: {e}", exc_info=True)
            # Memory kritik hata verirse ana döngüyü engellemeyebilir (projeye bağlı),
            # ama başlangıçta engellemek daha güvenli.
            can_run_main_loop = False
    else:
         logger.warning("Önceki modüller başlatılamadığı için Memory modülü atlandı.")

    # Faz 3 Başlangici: Cognition Modülü Başlat
    logger.info("Faz 3: Cognition modülü başlatılıyor...")
    # Cognition modülü representation ve memory'ye bağımlı. Eğer bunlar yoksa başlatmanın anlamı yok.
    if can_run_main_loop: # Eğer processing, representation ve memory başlatma hatası olmadıysa
        try:
            cognition_modules['core_cognition'] = CognitionCore(config.get('cognition', {})) # <<< Config'ten alındı
            logger.info("Cognition modülü başarıyla başlatıldı.")
        except Exception as e:
            logger.critical(f"Cognition modülü başlatılırken kritik hata oluştu: {e}", exc_info=True)
            # Cognition kritik hata verirse main loop'u engelle.
            can_run_main_loop = False
    else:
         logger.warning("Önceki modüller başlatılamadığı için Cognition modülü atlandı.")

    # Faz 3 Devami: Motor Control Modülü Başlat
    logger.info("Faz 3: Motor Control modülü başlatılıyor...")
    # Motor Control modülü Cognition'a bağımlı. Eğer Cognition yoksa başlatmanın anlamı yok.
    if can_run_main_loop: # Eğer processing, representation, memory ve cognition başlatma hatası olmadıysa
        try:
            motor_control_modules['core_motor_control'] = MotorControlCore(config.get('motor_control', {})) # <<< Config'ten alındı
            logger.info("Motor Control modülü başarıyla başlatıldı.")
        except Exception as e:
            logger.critical(f"Motor Control modülü başlatılırken kritik hata oluştu: {e}", exc_info=True)
            # Motor Control kritik hata verirse main loop'u engelle.
            can_run_main_loop = False
    else:
         logger.warning("Önceki modüller başlatılamadığı için Motor Control modülü atlandı.")

    # Faz 3 Tamamlanması: Interaction Modülü Başlat
    logger.info("Faz 3: Interaction modülü başlatılıyor...")
    # Interaction modülü Motor Control'e bağımlı. Eğer Motor Control yoksa başlatmanın anlamı yok.
    # Ancak Interaction API'si dış dünya ile bağlantı kurduğu için, bu modülün başlatılması
    # main loop'un çalışmasını engellememeli (eğer kritik bir hata değilse, sadece API'nin çalışmadığı anlamına gelir).
    # O yüzden bu modülün başlatılması can_run_main_loop'u False yapmamalı.
    # Bireysel try-except kullanıyoruz.
    try:
        interaction_modules['core_interaction'] = InteractionAPI(config.get('interaction', {})) # <<< Config'ten alındı
        # Eğer API bir thread içinde çalışacaksa burada start() çağrılmalı
        # if hasattr(interaction_modules['core_interaction'], 'start'):
        #      interaction_modules['core_interaction'].start()
        logger.info("Interaction modülü başarıyla başlatıldı.")
    except Exception as e:
        logger.critical(f"Interaction modülü başlatılırken kritik hata oluştu: {e}", exc_info=True)
        interaction_modules['core_interaction'] = None # Hata durumında objeyi None yap


    # Tüm temel modül kategorileri başlatıldı mı kontrolü (eğer can_run_main_loop True ise)
    # Interaction modülü kritik başlatma hatası vermediyse buraya gelinir.
    if can_run_main_loop:
         # Sadece ana pipeline modüllerinin (Sense, Process, Represent, Memory, Cognition, MotorControl)
         # kategori sözlüklerinin None olmaması ve içlerindeki objelerin None olmaması kontrol edilebilir.
         # Interaction modülü None olsa bile ana döngü çalışabilir (sadece çıktı veremez).
         pipeline_modules_ok = sensors and processors and representers and memories and cognition_modules and motor_control_modules
         # Check if all required *objects* within these categories are not None (based on our init logic)
         # Sensör objeleri None olmayabilir ama is_available False olabilir, o yüzden all(sensors.values()) tek başına yeterli değil.
         # Ama diğer kritik modüllerin (Process, Represent, Memory, Cognition, MotorControl) None olmaması gerekiyor.
         all_critical_objects_ok = all(processors.values()) and \
                                   all(representers.values()) and \
                                   all(memories.values()) and \
                                   all(cognition_modules.values()) and \
                                   all(motor_control_modules.values())


         if active_sensors and all_critical_objects_ok: # En az bir aktif sensör ve tüm kritik modül objeleri var mı?
            # Basitçe loglayalım:
             logger.info("Tüm ana pipeline modül kategorileri başarıyla başlatıldı. Evo bilişsel döngüye hazır.")
             if not interaction_modules.get('core_interaction'):
                  logger.warning("Interaction modülü başlatılamadı. Evo çıktı veremeyebilir.")

         else:
              # Bu durum should not happen if can_run_main_loop is True with current logic,
              # unless some non-critical module init fails silently (which we handle with logging).
              logger.warning("Bazı temel ana pipeline modül kategorileri başlatılamadı veya eksik. Evo bilişsel döngüsü sınırlı çalışabilir.")
              # Hangi kategorilerin eksik olduğunu loglayabiliriz:
              # missing_categories = [cat_name for cat_name, cat_dict in [('Sensors', sensors), ('Processors', processors), ('Representers', representers), ('Memories', memories), ('Cognition', cognition_modules), ('MotorControl', motor_control_modules)] if not cat_dict or any(v is None for v in cat_dict.values())]
              # if missing_categories:
              #      logger.warning(f"Eksik veya None olan temel modül objeleri: {', '.join(missing_categories)}")


    else:
         logger.critical("Modül başlatma hataları nedeniyle Evo'nın bilişsel döngüsü başlatılamadı.")


    # --- Ana Bilişsel Döngü ---
    # Sadece can_run_main_loop True ise döngüye gir
    if can_run_main_loop:
        logger.info("Evo'nın bilişsel döngüsü başlatıldı...")
        # Konfigürasyondan değerleri alırken .get() kullanmak veya değerin varlığını kontrol etmek önemlidir.
        # config_utils.load_config_from_yaml hata durumunda {} döndürdüğü için .get() kullanımı daha güvenlidir.
        loop_interval = config.get('cognitive_loop_interval', 0.1) # <<< Config'ten alındı
        num_memories_to_retrieve = config.get('memory', {}).get('num_retrieved_memories', 5) # <<< Config'ten alındı

        try:
            while True: # Main loop runs as long as no unhandled error or KeyboardInterrupt
                start_time = time.time()

                # --- Duyu Verisini Yakala (Faz 0) ---
                raw_inputs = {}
                # Sadece sensör objesi None değilse capture metodunu çağır.
                # Capture metodu kendi içinde donanım kontrolü ve simülasyon mantığı içerir.
                if sensors.get('vision'):
                    raw_inputs['visual'] = sensors['vision'].capture_frame()
                # else: raw_inputs['visual'] = None # Sensor objesi hiç oluşturulamadıysa girdi None

                # DEBUG Log: Yakalanan Ham Görsel Veri
                if raw_inputs.get('visual') is not None:
                     logger.debug(f"RUN_EVO: Raw Visual Input yakalandı. Shape: {raw_inputs['visual'].shape}, Dtype: {raw_inputs['visual'].dtype}")
                else:
                     logger.debug("RUN_EVO: Raw Visual Input None.")


                if sensors.get('audio'):
                    raw_inputs['audio'] = sensors['audio'].capture_chunk()
                # else: raw_inputs['audio'] = None # Sensor objesi hiç oluşturulamadıysa girdi None

                # DEBUG Log: Yakalanan Ham Ses Verisi
                if raw_inputs.get('audio') is not None:
                     logger.debug(f"RUN_EVO: Raw Audio Input yakalandı. Shape: {raw_inputs['audio'].shape}, Dtype: {raw_inputs['audio'].dtype}")
                else:
                     logger.debug("RUN_EVO: Raw Audio Input None.")


                # logging.debug("Duyusal veri yakalama tamamlandı.")


                # --- Yakalanan Veriyi İşle (Faz 1 Başlangıcı) ---
                processed_inputs = {}
                # Sadece işlemci objesi None değilse ve girdi None değilse process metodunu çağır
                if processors.get('vision') and raw_inputs.get('visual') is not None:
                     processed_inputs['visual'] = processors['vision'].process(raw_inputs['visual'])
                     # Process metotları kendi içlerinde loglama yapıyor.
                     # if processed_inputs.get('visual') is not None: # Bu kontrol process içinde yapılıyor
                     #      logger.debug(f"Görsel input işlendi ve islendi. Output Shape: {processed_inputs['visual'].shape}, Dtype: {processed_inputs['visual'].dtype}")


                if processors.get('audio') and raw_inputs.get('audio') is not None:
                     processed_inputs['audio'] = processors['audio'].process(raw_inputs['audio'])
                     # Process metotları kendi içlerinde loglama yapıyor.
                     # if processed_inputs.get('audio') is not None: # Bu kontrol process içinde yapılıyor
                     #      logger.debug(f"Sesli input işlendi ve islendi. Output Energy: {processed_inputs['audio']:.4f}") # Log enerji değeri

                # DEBUG Log: İşlenmiş Görsel Veri
                if processed_inputs.get('visual') is not None:
                     logger.debug(f"RUN_EVO: Processed Visual Output. Shape: {processed_inputs['visual'].shape}, Dtype: {processed_inputs['visual'].dtype}")
                else:
                     logger.debug("RUN_EVO: Processed Visual Output None.")

                # DEBUG Log: İşlenmiş Ses Verisi
                if processed_inputs.get('audio') is not None:
                     logger.debug(f"RUN_EVO: Processed Audio Output (Energy). Value: {processed_inputs['audio']:.4f}")
                else:
                     logger.debug("RUN_EVO: Processed Audio Output None.")


                # logger.debug("İşlenmiş veri işleme tamamlandı.")


                # --- İşlenmiş Veriden Temsil Öğren (Faz 1 Devamı) ---
                learned_representation = None
                # Sadece representation learner objesi None değilse ve processed_inputs boş değilse
                if representers.get('main_learner') and processed_inputs:
                     learned_representation = representers['main_learner'].learn(processed_inputs)
                     # learned_representation'ın None olup olmadığı RepresentationLearner içinde loglanıyor.
                     if learned_representation is not None:
                         logger.debug(f"RUN_EVO: Learned Representation. Shape: {learned_representation.shape}, Dtype: {learned_representation.dtype}")


                # else: logger.debug("RUN_EVO: Representation Learner mevcut değil veya processed_inputs boş. Temsil öğrenilemedi.")


                # --- Temsili Hafızaya Kaydet ve/veya Hafızadan Bilgi Al (Faz 2) ---
                relevant_memory_entries = [] # Başlangıçta ilgili bellek girdileri yok
                # Sadece memory objesi None değilse ve öğrenilmiş temsil None değilse
                if memories.get('core_memory') and learned_representation is not None:
                     # Öğrenilen temsili hafızaya kaydet
                     memories['core_memory'].store(learned_representation)
                     # Hafızadan ilgili bilgi çağır (placeholder retrieve fonksiyonu)
                     relevant_memory_entries = memories['core_memory'].retrieve(
                         learned_representation, # Sorgu temsili (şimdilik placeholder retrieve'de kullanılmıyor ama yapı bu)
                         num_results=num_memories_to_retrieve
                     )
                     if relevant_memory_entries:
                          logger.debug(f"RUN_EVO: Hafızadan {len(relevant_memory_entries)} ilgili girdi geri çağrıldı (placeholder).")
                          # Geri çağrılan girdilerin içeriğini loglamak DEBUG seviyesinde çok fazla çıktı üretebilir,
                          # sadece sayısını veya basit özetini loglamak daha iyi.
                          # Örneğin: logger.debug(f"RUN_EVO: Hafızadan ilgili girdi temsil şekli: {relevant_memory_entries[0]['representation'].shape}...")

                     # else: logger.debug("RUN_EVO: Hafızadan ilgili girdi çağrılamadı.")
                # else:
                #     # logger.debug("RUN_EVO: Memory modülü veya öğrenilmiş temsil mevcut değil. Hafıza işlemi atlandı.")
                #     pass # Debug logu çok sık gelebilir, gerek yok.


                # --- Hafıza ve temsile göre Bilişsel işlem yap (Faz 3 Başlangıcı) ---
                decision = None # Başlangıçta bir karar yok
                # Sadece cognition objesi None değilse ve işlenecek girdi varsa (temsil VEYA ilgili bellek)
                if cognition_modules.get('core_cognition') and (learned_representation is not None or relevant_memory_entries):
                     decision = cognition_modules['core_cognition'].decide(
                         learned_representation, # Temsil None olabilir
                         relevant_memory_entries # Liste boş olabilir
                     )
                     if decision is not None:
                          logger.debug(f"RUN_EVO: Bilişsel karar alındı (placeholder): {decision}")
                     # else: logger.debug("RUN_EVO: Bilişsel karar üretilemedi veya None döndü.")

                # else:
                #      # logger.debug("RUN_EVO: Cognition modülü veya işlenecek girdi (temsil/bellek) mevcut değil.")
                #      pass # Debug logu çok sık gelebilir.

                # --- Karara göre bir Tepki Üret (Faz 3 Devamı) ---
                response_output = None # Başlangıçta bir tepki yok
                # Sadece motor control objesi None değilse ve bir karar alındıysa
                if motor_control_modules.get('core_motor_control') and decision is not None:
                     response_output = motor_control_modules['core_motor_control'].generate_response(decision)
                     # if response_output is not None: # Bu kontrol generate_response içinde yapılıyor
                     #      logger.debug(f"RUN_EVO: Motor kontrol tepki üretti (placeholder). Output: '{response_output}'")

                # else:
                #     # logger.debug("RUN_EVO: Motor Control modülü veya karar mevcut değil.")
                #     pass # Debug logu çok sık gelebilir.

                # --- Tepkiyi Dışarı Aktar (Faz 3 Tamamlanması) ---
                # Sadece interaction objesi None değilse ve üretilmiş bir tepki varsa
                if interaction_modules.get('core_interaction') and response_output is not None:
                   # InteractionAPI'nin send_output metodu artık INFO loglamayı kendi içinde yapıyor.
                   # DEBUG loglama send_output içinde OutputChannel'a gönderilmeden yapılıyor.
                   # Burada sadece send_output çağrısı yeterli
                   interaction_modules['core_interaction'].send_output(response_output)
                # else:
                #     # logger.debug("RUN_EVO: Interaction modülü veya tepki mevcut değil. Çıktı gönderilemedi.")
                #     pass # Debug logu çok sık gelebilir.


                # --- Döngü Gecikmesi ---
                # Döngünün belirli bir hızda çalışmasını sağla
                elapsed_time = time.time() - start_time
                sleep_time = loop_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Döngü intervalinden uzun süren durumlar için uyarı (DEBUG seviyesinde)
                    # INFO seviyesine çekilirse her seferinde loglanır ki bu istenmeyebilir
                    logger.debug(f"RUN_EVO: Bilişsel döngü {loop_interval} saniyeden daha uzun sürdü ({elapsed_time:.4f}s). İşlem yükü yüksek olabilir.")


                # Gelecekte döngüyü sonlandıracak bir mekanizma eklenecek (örn. kullanıcı sinyali, içsel durum)
                # if cognition_modules.get('core_cognition') and cognition_modules['core_cognition'].should_stop(): break # Örnek: Evo uykuya dalarsa

        except KeyboardInterrupt:
            logger.warning("Ctrl+C algılandı. Evo durduruluyor...")
        except Exception as e: # Catch any other exception during the main loop
            logger.critical(f"Evo'nın ana döngüsünde beklenmedik kritik hata: {e}", exc_info=True)

        finally:
            # --- Kaynakları Temizleme ---
            logger.info("Evo kaynakları temizleniyor...")
            # Check if objects exist in the dictionaries before stopping or cleanup
            if sensors.get('vision'):
                logger.info("VisionSensor kapatılıyor...")
                sensors['vision'].stop_stream()
            if sensors.get('audio'):
                logger.info("AudioSensor kapatılıyor...")
                sensors['audio'].stop_stream()

        # Processor'lar genellikle kapatma gerektirmez ama emin olmak için kontrol edilebilir veya cleanup metodu eklenebilir
        # if processors.get('vision') and hasattr(processors['vision'], 'cleanup'):
        #      logger.info("VisionProcessor temizleniyor...")
        #      processors['vision'].cleanup()
        # if processors.get('audio') and hasattr(processors['audio'], 'cleanup'):
        #      logger.info("AudioProcessor temizleniyor...")
        #      processors['audio'].cleanup()

        # Representation Learner da genellikle kapatma gerektirmez
        # if representers.get('main_learner') and hasattr(representers['main_learner'], 'cleanup'):
        #      logger.info("RepresentationLearner temizleniyor...")
        #      representers['main_learner'].cleanup()

        # Memory modülü eğer kalıcı depolama kullanıyorsa cleanup gerektirebilir.
        # if memories.get('core_memory') and hasattr(memories['core_memory'], 'cleanup'):
        #     logger.info("Memory modülü temizleniyor...")
        #     memories['core_memory'].cleanup()

        # Cognition modülü genellikle kapatma gerektirmez
        # if cognition_modules.get('core_cognition') and hasattr(cognition_modules['core_cognition'], 'cleanup'):
        #      logger.info("Cognition modülü temizleniyor...")
        #      cognition_modules['core_cognition'].cleanup()

        # Motor Control modülü genellikle kapatma gerektirmez
        # if motor_control_modules.get('core_motor_control') and hasattr(motor_control_modules['core_motor_control'], 'cleanup'):
        #      logger.info("Motor Control modülü temizleniyor...")
        #      motor_control_modules['core_motor_control'].cleanup()

        # Interaction modülü eğer bir thread veya servis başlattıysa stop metodu gerektirir.
        if interaction_modules.get('core_interaction') and hasattr(interaction_modules['core_interaction'], 'stop'):
             logger.info("InteractionAPI kapatılıyor...")
             interaction_modules['core_interaction'].stop()
        # elif interaction_modules.get('core_interaction') and hasattr(interaction_modules['core_interaction'], 'cleanup'):
        #      logger.info("InteractionAPI temizleniyor...")
        #      interaction_modules['core_interaction'].cleanup()


        logger.info("Evo durduruldu.")


# Ana çalıştırma noktası
if __name__ == '__main__':
    run_evo()