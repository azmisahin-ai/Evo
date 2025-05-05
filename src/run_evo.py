# src/run_evo.py

import logging
import time
# Temel modülleri import edeceğiz
from src.senses.vision import VisionSensor
from src.senses.audio import AudioSensor
# Processing modülleri import et
from src.processing.vision import VisionProcessor
from src.processing.audio import AudioProcessor
# Representation modüllerini import et
from src.representation.models import RepresentationLearner
# Memory modülünü import et
from src.memory.core import Memory # core.py dosyasındaki Memory sınıfı
# Cognition modülünü import et
from src.cognition.core import CognitionCore # core.py dosyasındaki CognitionCore sınıfı
# Motor Control modülünü import et
from src.motor_control.core import MotorControlCore # core.py dosyasındaki MotorControlCore sınıfı
# Interaction modülünü import et
from src.interaction.api import InteractionAPI # api.py dosyasındaki InteractionAPI sınıfı


# Konfigürasyon yükleme (şimdilik basit bir placeholder)
# Gelecekte config/main_config.yaml dosyasından okunacak
def load_config():
    """Basit placeholder konfigürasyon yükleyici."""
    # TODO: config/main_config.yaml dosyasından gerçek konfigürasyonu yükle
    # TODO: YAML okuma kütüphanesi (PyYAML) requirements.txt'e ve kuruluma eklenmeli
    return {
        'vision': {
            'camera_index': 0, # Varsayılan kamera indeksi
            'dummy_width': 640, # Simüle kare genişliği
            'dummy_height': 480 # Simüle kare yüksekliği
        },
        'audio': {
            'audio_rate': 44100,
            'audio_chunk_size': 1024,
            # 'audio_input_device_index': None # None varsayılanı kullanır
        },
        'processing_vision': { # Vision Processor için konfigürasyon
             'output_width': 64, # Processing çıktısı boyutu
             'output_height': 64 # Processing çıktısı boyutu
        },
         'processing_audio': { # Audio Processor için konfigürasyon
             # Örneğin, MFCC sayısı gibi ayarlar buraya gelebilir
             # output_features: ['energy', 'mfcc'] # Gelecekte
             'output_dim': 1 # Şu anki enerji çıktısı için boyut
         },
         'representation': { # Representation Learner için konfigürasyon
             # input_dim, processing çıktılarının boyutuna göre dinamik belirlenmeli veya burada hesaplanmalı.
             # Şimdilik config'de belirtelim ve VisionProcessor çıktı boyutu ile AudioProcessor çıktı boyutuna göre manuel ayarlayalım.
             # Eğer Processing Vision 64x64 gri dönerse: 64*64=4096. Audio 1 enerji dönerse: 1. input_dim = 4096 + 1 = 4097
             'input_dim': 4096 + 1, # Vision (64*64) + Audio (1)
             'representation_dim': 128 # Öğrenilecek temsil boyutu
         },
         'memory': { # Memory modülü için konfigürasyon
             'max_memory_size': 1000, # Saklanacak maksimum temsil sayısı (geçici hafıza)
             'num_retrieved_memories': 5 # Her döngüde kaç hafıza girdisi geri çağrılacak (placeholder için)
             # Gelecekte kalıcı depolama ayarları buraya gelebilir
         },
         'cognition': { # Cognition modülü için konfigürasyon
             # Örneğin, karar eşikleri veya model yolları buraya gelebilir
         },
         'motor_control': { # Motor Control modülü için konfigürasyon
             # Örneğin, çıktı tipi (metin, ses vb.) veya sentezleyici ayarları
         },
         'interaction': { # Interaction modülü için konfigürasyon
             'enabled_channels': ['console'], # Aktif çıktı kanalları listesi. Şimdilik sadece 'console'.
             'channel_configs': { # Kanal bazlı özel ayarlar
                 'web_api': {'port': 5000} # 'web_api' kanalı aktif edilirse kullanılacak örnek ayar
             }
         },
        'cognitive_loop_interval': 0.1 # Bilişsel döngünün ne sıklıkla çalışacağı (saniye)
    }

def run_evo():
    """
    Evo'nın çekirdek bilişsel döngüsünü ve arayüzlerini başlatır.
    Bu fonksiyon çağrıldığında Evo "canlanır".
    """
    # Logging ayarları (dosya başında veya ayrı bir utility fonksiyonda olabilir)
    # İlk logging config ayarı, modül başlatma hatalarını görebilmek için try bloğu dışında olmalı
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # src içindeki detaylı logları göstermek için seviyeyi DEBUG yap
    logging.getLogger('src').setLevel(logging.DEBUG)

    logging.info("Evo canlanıyor...")

    # Konfigürasyonu yükle
    config = load_config()
    logging.info("Konfigürasyon yüklendi.")

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
    logging.info("Modüller başlatılıyor...")

    # Faz 0: Duyusal Sensörleri Başlat - Bireysel Hata Yönetimi ile
    logging.info("Faz 0: Duyusal sensörler başlatılıyor (Bireysel hata yönetimi aktif)...")
    try:
        logging.info("VisionSensor başlatılıyor...")
        sensors['vision'] = VisionSensor(config.get('vision', {}))
        if not (sensors.get('vision') and getattr(sensors['vision'], 'is_camera_available', False)):
             logging.warning("VisionSensor tam başlatılamadı veya kamera açılamadı.")
             # Simüle edilmiş girdi kullanılacaksa objenin is_camera_available False olacak. Objeyi None yapma!
    except Exception as e:
        logging.critical(f"VisionSensor başlatılırken kritik hata oluştu: {e}", exc_info=True)
        sensors['vision'] = None # Kritik hata durumunda objeyi None yap


    try:
        logging.info("AudioSensor başlatılıyor...")
        sensors['audio'] = AudioSensor(config.get('audio', {}))
        if not (sensors.get('audio') and getattr(sensors['audio'], 'is_audio_available', False)):
             logging.warning("AudioSensor tam başlatılamadı veya ses akışı aktif değil.")
             # Simüle edilmiş girdi kullanılacaksa objenin is_audio_available False olacak. Objeyi None yapma!
    except Exception as e:
        logging.critical(f"AudioSensor başlatılırken kritik hata oluştu: {e}", exc_info=True)
        sensors['audio'] = None # Kritik hata durumunda objeyi None yap


    # Başlatma durumunu kontrol et ve logla (Sözlükteki objelerin None olup olmadığına ve internal flag'lere bakarak)
    # Sadece None olmayan ve internal flag'i True olan sensörleri say
    active_sensors = [name for name, sensor in sensors.items()
                      if sensor and (getattr(sensor, 'is_camera_available', False) or getattr(sensor, 'is_audio_available', False))]
    if active_sensors:
        logging.info(f"Duyusal Sensörler başlatıldı ({', '.join(active_sensors)} aktif).")
    else:
        logging.warning("Hiçbir duyusal sensör aktif değil. Evo girdi alamayacak.")


    # Faz 1 Başlangici: Processing Modülleri Başlat
    logging.info("Faz 1: Processing modülleri başlatılıyor...")
    try:
        processors['vision'] = VisionProcessor(config.get('processing_vision', {}))
        processors['audio'] = AudioProcessor(config.get('processing_audio', {}))
        logging.info("Processing modülleri başarıyla başlatıldı.")
    except Exception as e:
         logging.critical(f"Processing modülleri başlatılırken kritik hata oluştu: {e}", exc_info=True)
         # Processing kritik hata verirse sensörler çalışsa bile anlamı yok, main loop'u engelle.
         can_run_main_loop = False


    # Faz 1 Devami: Representation Modülleri Başlat
    logging.info("Faz 1: Representation modülleri başlatılıyor...")
    # Representation modülü processing çıktısına bağımlı, eğer processing yoksa başlatmanın anlamı yok.
    # Veya dummy/placeholder representation modülü başlatılabilir.
    # Şimdilik processing başarılıysa başlatmayı deneyelim.
    if can_run_main_loop: # Eğer processing başlatma hatası olmadıysa
        try:
            representers['main_learner'] = RepresentationLearner(config.get('representation', {}))
            logging.info("Representation modülü başarıyla başlatıldı.")
        except Exception as e:
            logging.critical(f"Representation modülü başlatılırken kritik hata oluştu: {e}", exc_info=True)
            can_run_main_loop = False # Representation kritik hata verirse main loop'u engelle.
    else:
         # Bu durum loglanmalı, çünkü can_run_main_loop zaten False yapıldı.
         logging.warning("Processing modülleri başlatılamadığı için Representation modülleri atlandı.")


    # Faz 2 Başlangici: Memory Modülü Başlat
    logging.info("Faz 2: Memory modülü başlatılıyor...")
    # Memory modülü representation'a bağımlı. Eğer representation yoksa başlatmanın anlamı yok.
    if can_run_main_loop: # Eğer processing ve representation başlatma hatası olmadıysa
        try:
            memories['core_memory'] = Memory(config.get('memory', {}))
            logging.info("Memory modülü başarıyla başlatıldı.")
        except Exception as e:
            logging.critical(f"Memory modülü başlatılırken kritik hata oluştu: {e}", exc_info=True)
            # Memory kritik hata verirse ana döngüyü engellemeyebilir (projeye bağlı),
            # ama başlangıçta engellemek daha güvenli.
            can_run_main_loop = False
    else:
         logging.warning("Önceki modüller başlatılamadığı için Memory modülü atlandı.")

    # Faz 3 Başlangici: Cognition Modülü Başlat
    logging.info("Faz 3: Cognition modülü başlatılıyor...")
    # Cognition modülü representation ve memory'ye bağımlı. Eğer bunlar yoksa başlatmanın anlamı yok.
    if can_run_main_loop: # Eğer processing, representation ve memory başlatma hatası olmadıysa
        try:
            cognition_modules['core_cognition'] = CognitionCore(config.get('cognition', {}))
            logging.info("Cognition modülü başarıyla başlatıldı.")
        except Exception as e:
            logging.critical(f"Cognition modülü başlatılırken kritik hata oluştu: {e}", exc_info=True)
            # Cognition kritik hata verirse main loop'u engelle.
            can_run_main_loop = False
    else:
         logging.warning("Önceki modüller başlatılamadığı için Cognition modülü atlandı.")

    # Faz 3 Devami: Motor Control Modülü Başlat
    logging.info("Faz 3: Motor Control modülü başlatılıyor...")
    # Motor Control modülü Cognition'a bağımlı. Eğer Cognition yoksa başlatmanın anlamı yok.
    if can_run_main_loop: # Eğer processing, representation, memory ve cognition başlatma hatası olmadıysa
        try:
            motor_control_modules['core_motor_control'] = MotorControlCore(config.get('motor_control', {}))
            logging.info("Motor Control modülü başarıyla başlatıldı.")
        except Exception as e:
            logging.critical(f"Motor Control modülü başlatılırken kritik hata oluştu: {e}", exc_info=True)
            # Motor Control kritik hata verirse main loop'u engelle.
            can_run_main_loop = False
    else:
         logging.warning("Önceki modüller başlatılamadığı için Motor Control modülü atlandı.")

    # Faz 3 Tamamlanması: Interaction Modülü Başlat
    logging.info("Faz 3: Interaction modülü başlatılıyor...")
    # Interaction modülü Motor Control'e bağımlı. Eğer Motor Control yoksa başlatmanın anlamı yok.
    # Ancak Interaction API'si dış dünya ile bağlantı kurduğu için, bu modülün başlatılması
    # main loop'un çalışmasını engellememeli (eğer kritik bir hata değilse, sadece API'nin çalışmadığı anlamına gelir).
    # O yüzden bu modülün başlatılması can_run_main_loop'u False yapmamalı.
    # Bireysel try-except kullanıyoruz.
    try:
        interaction_modules['core_interaction'] = InteractionAPI(config.get('interaction', {}))
        # Eğer API bir thread içinde çalışacaksa burada start() çağrılmalı
        # if hasattr(interaction_modules['core_interaction'], 'start'):
        #      interaction_modules['core_interaction'].start()
        logging.info("Interaction modülü başarıyla başlatıldı.")
    except Exception as e:
        logging.critical(f"Interaction modülü başlatılırken kritik hata oluştu: {e}", exc_info=True)
        interaction_modules['core_interaction'] = None # Hata durumunda objeyi None yap


    # Tüm temel modül kategorileri başlatıldı mı kontrolü (eğer can_run_main_loop True ise)
    # Interaction modülü kritik başlatma hatası vermediyse buraya gelinir.
    if can_run_main_loop:
         # Sadece ana pipeline modüllerinin (Sense, Process, Represent, Memory, Cognition, MotorControl)
         # kategori sözlüklerinin None olmaması ve içlerindeki objelerin None olmaması kontrol edilebilir.
         # Interaction modülü None olsa bile ana döngü çalışabilir (sadece çıktı veremez).
         # Sensors kategorisindeki objeler None olmaz ama is_available False olabilir.
         # Diğer kritik modüllerin (Process, Represent, Memory, Cognition, MotorControl) None olmaması gerekiyor.
         all_critical_objects_ok = all(processors.values()) and \
                                   all(representers.values()) and \
                                   all(memories.values()) and \
                                   all(cognition_modules.values()) and \
                                   all(motor_control_modules.values())

         # En az bir aktif sensör VE tüm kritik işlem/beyin modülleri başarılı mı?
         if active_sensors and all_critical_objects_ok:
            # Basitçe loglayalım:
             logging.info("Tüm ana pipeline modül kategorileri başarıyla başlatıldı. Evo bilişsel döngüye hazır.")
             if not interaction_modules.get('core_interaction'):
                  logging.warning("Interaction modülü başlatılamadı. Evo çıktı veremeyebilir.")

         else:
              # Bu durum should not happen if can_run_main_loop is True with current logic,
              # unless some non-critical module init fails silently (which we handle with logging).
              # Eğer buraya gelindiyse, ya hiç aktif sensör yok ya da kritik objelerden biri None.
              logging.warning("Bazı temel ana pipeline modül kategorileri başlatılamadı veya eksik. Evo bilişsel döngüsü sınırlı çalışabilir.")
              # Hangi kategorilerin eksik olduğunu loglayabiliriz:
              # missing_categories = [cat_name for cat_name, cat_dict in [('Sensors', sensors), ('Processors', processors), ('Representers', representers), ('Memories', memories), ('Cognition', cognition_modules), ('MotorControl', motor_control_modules)] if not cat_dict or any(v is None for v in cat_dict.values())]
              # if missing_categories:
              #      logging.warning(f"Eksik veya None olan temel modül objeleri: {', '.join(missing_categories)}")


    else:
         logging.critical("Modül başlatma hataları nedeniyle Evo'nın bilişsel döngüsü başlatılamadı.")


    # --- Ana Bilişsel Döngü ---
    # Sadece can_run_main_loop True ise döngüye gir
    if can_run_main_loop:
        logging.info("Evo'nın bilişsel döngüsü başlatıldı...")
        loop_interval = config.get('cognitive_loop_interval', 0.1) # Döngü hızı
        num_memories_to_retrieve = config.get('memory', {}).get('num_retrieved_memories', 5) # Her döngüde kaç hafıza çağırılacak

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


                if sensors.get('audio'):
                    raw_inputs['audio'] = sensors['audio'].capture_chunk()
                # else: raw_inputs['audio'] = None # Sensor objesi hiç oluşturulamadıysa girdi None

                # logging.debug("Duyusal veri yakalama tamamlandı.")


                # --- Yakalanan Veriyi İşle (Faz 1 Başlangıcı) ---
                processed_inputs = {}
                # Sadece işlemci objesi None değilse ve girdi None değilse process metodunu çağır
                if processors.get('vision') and raw_inputs.get('visual') is not None:
                     processed_inputs['visual'] = processors['vision'].process(raw_inputs['visual'])
                     if processed_inputs.get('visual') is not None:
                          logging.debug(f"Görsel input işlendi ve islendi. Output Shape: {processed_inputs['visual'].shape}, Dtype: {processed_inputs['visual'].dtype}")
                     # else: logging.debug("Görsel input işlenemedi veya None döndü.")


                if processors.get('audio') and raw_inputs.get('audio') is not None:
                     processed_inputs['audio'] = processors['audio'].process(raw_inputs['audio'])
                     if processed_inputs.get('audio') is not None:
                          logging.debug(f"Sesli input işlendi ve islendi. Output Energy: {processed_inputs['audio']:.4f}") # Log enerji değeri
                     # else: logging.debug("Sesli input işlenemedi veya None döndü.")


                # logging.debug("İşlenmiş veri işleme tamamlandı.")


                # --- İşlenmiş Veriden Temsil Öğren (Faz 1 Devamı) ---
                learned_representation = None
                # Sadece representation learner objesi None değilse ve processed_inputs boş değilse
                if representers.get('main_learner') and processed_inputs:
                     learned_representation = representers['main_learner'].learn(processed_inputs)
                     # learned_representation'ın None olup olmadığı RepresentationLearner içinde loglanıyor.
                     # learned_representation RepresentationLearner'dan non-None gelirse buradaki DEBUG logu tetiklenir.
                     if learned_representation is not None:
                         logging.debug(f"Ana Döngü: Representation öğrenildi. Shape: {learned_representation.shape}, Dtype: {learned_representation.dtype}")


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
                          logging.debug(f"Ana Döngü: Hafızadan {len(relevant_memory_entries)} ilgili girdi geri çağrıldı (placeholder).")
                          # Geri çağrılan girdilerin içeriğini loglamak DEBUG seviyesinde çok fazla çıktı üretebilir,
                          # sadece sayısını veya basit özetini loglamak daha iyi.
                          # Örneğin: logging.debug(f"Hafızadan ilgili girdi temsil şekli: {relevant_memory_entries[0]['representation'].shape}...")

                     # else: logging.debug("Hafızadan ilgili girdi çağrılamadı.")
                # else:
                #     # logging.debug("Memory modülü veya öğrenilmiş temsil mevcut değil. Hafıza işlemi atlandı.")
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
                          logging.debug(f"Ana Döngü: Bilişsel karar alındı (placeholder): {decision}")
                     # else: logging.debug("Bilişsel karar üretilemedi veya None döndü.")

                # else:
                #      # logging.debug("Cognition modülü veya işlenecek girdi (temsil/bellek) mevcut değil.")
                #      pass # Debug logu çok sık gelebilir.

                # --- Karara göre bir Tepki Üret (Faz 3 Devamı) ---
                response_output = None # Başlangıçta bir tepki yok
                # Sadece motor control objesi None değilse ve bir karar alındıysa
                if motor_control_modules.get('core_motor_control') and decision is not None:
                     response_output = motor_control_modules['core_motor_control'].generate_response(decision)
                     if response_output is not None:
                          logging.debug(f"Ana Döngü: Motor kontrol tepki üretti (placeholder). Output: '{response_output}'")
                     # else: logging.debug("Motor control tepki üretemedi veya None döndü.")

                # else:
                #     # logging.debug("Motor Control modülü veya karar mevcut değil.")
                #     pass # Debug logu çok sık gelebilir.

                # --- Tepkiyi Dışarı Aktar (Faz 3 Tamamlanması) ---
                # Sadece interaction objesi None değilse ve üretilmiş bir tepki varsa
                if interaction_modules.get('core_interaction') and response_output is not None:
                   interaction_modules['core_interaction'].send_output(response_output)
                # else:
                #     # logging.debug("Interaction modülü veya tepki mevcut değil. Çıktı gönderilemedi.")
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
                    logging.debug(f"Bilişsel döngü {loop_interval} saniyeden daha uzun sürdü ({elapsed_time:.4f}s). İşlem yükü yüksek olabilir.")


                # Gelecekte döngüyü sonlandıracak bir mekanizma eklenecek (örn. kullanıcı sinyali, içsel durum)
                # if cognition_modules.get('core_cognition') and cognition_modules['core_cognition'].should_stop(): break # Örnek: Evo uykuya dalarsa

        except KeyboardInterrupt:
            logging.warning("Ctrl+C algılandı. Evo durduruluyor...")
        except Exception as e: # Catch any other exception during the main loop
            logging.critical(f"Evo'nın ana döngüsünde beklenmedik kritik hata: {e}", exc_info=True)

        finally:
            # --- Kaynakları Temizleme ---
            logging.info("Evo kaynakları temizleniyor...")
            # Check if objects exist in the dictionaries before stopping or cleanup
            if sensors.get('vision'):
                logging.info("VisionSensor kapatılıyor...")
                sensors['vision'].stop_stream()
            if sensors.get('audio'):
                logging.info("AudioSensor kapatılıyor...")
                sensors['audio'].stop_stream()

        # Processor'lar genellikle kapatma gerektirmez ama emin olmak için kontrol edilebilir veya cleanup metodu eklenebilir
        # if processors.get('vision') and hasattr(processors['vision'], 'cleanup'):
        #      logging.info("VisionProcessor temizleniyor...")
        #      processors['vision'].cleanup()
        # if processors.get('audio') and hasattr(processors['audio'], 'cleanup'):
        #      logging.info("AudioProcessor temizleniyor...")
        #      processors['audio'].cleanup()

        # Representation Learner da genellikle kapatma gerektirmez
        # if representers.get('main_learner') and hasattr(representers['main_learner'], 'cleanup'):
        #      logging.info("RepresentationLearner temizleniyor...")
        #      representers['main_learner'].cleanup()

        # Memory modülü eğer kalıcı depolama kullanıyorsa cleanup gerektirebilir.
        # if memories.get('core_memory') and hasattr(memories['core_memory'], 'cleanup'):
        #     logging.info("Memory modülü temizleniyor...")
        #     memories['core_memory'].cleanup()

        # Cognition modülü genellikle kapatma gerektirmez
        # if cognition_modules.get('core_cognition') and hasattr(cognition_modules['core_cognition'], 'cleanup'):
        #      logging.info("Cognition modülü temizleniyor...")
        #      cognition_modules['core_cognition'].cleanup()

        # Motor Control modülü genellikle kapatma gerektirmez
        # if motor_control_modules.get('core_motor_control') and hasattr(motor_control_modules['core_motor_control'], 'cleanup'):
        #      logging.info("Motor Control modülü temizleniyor...")
        #      motor_control_modules['core_motor_control'].cleanup()

        # Interaction modülü eğer bir thread veya servis başlattıysa stop metodu gerektirir.
        if interaction_modules.get('core_interaction') and hasattr(interaction_modules['core_interaction'], 'stop'):
             logging.info("InteractionAPI kapatılıyor...")
             interaction_modules['core_interaction'].stop()
        # elif interaction_modules.get('core_interaction') and hasattr(interaction_modules['core_interaction'], 'cleanup'):
        #      logging.info("InteractionAPI temizleniyor...")
        #      interaction_modules['core_interaction'].cleanup()


        logging.info("Evo durduruldu.")


# Ana çalıştırma noktası
if __name__ == '__main__':
    run_evo()