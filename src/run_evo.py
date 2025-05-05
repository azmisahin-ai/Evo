# src/run_evo.py

import logging
import time
# Temel modülleri import edeceğiz
from src.senses.vision import VisionSensor
from src.senses.audio import AudioSensor
# Processing modüllerini import et
from src.processing.vision import VisionProcessor
from src.processing.audio import AudioProcessor
# Representation modüllerini import et
from src.representation.models import RepresentationLearner
# Diğer modüller (memory, cognition, motor_control, interaction) geldikçe buraya eklenecek

# Konfigürasyon yükleme (şimdilik basit bir placeholder)
# Gelecekte config/main_config.yaml dosyasından okunacak
def load_config():
    """Basit placeholder konfigürasyon yükleyici."""
    # TODO: config/main_config.yaml dosyasından gerçek konfigürasyonu yükle
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
         },
         'representation': { # Representation Learner için konfigürasyon
             'representation_dim': 128 # Öğrenilecek temsil boyutu
         },
        'cognitive_loop_interval': 0.1 # Bilişsel döngünün ne sıklıkla çalışacağı (saniye)
    }

def run_evo():
    """
    Evo'nun çekirdek bilişsel döngüsünü ve arayüzlerini başlatır.
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
    representers = {} # Representation modülleri için yeni kategori
    other_modules = {} # Memory, cognition, interaction vb. buraya gelecek

    # Başlatma başarılı olduysa main loop'u çalıştırmak için flag
    can_run_main_loop = True

    # --- Modülleri Başlatma ---
    logging.info("Modüller başlatılıyor...")

    # Faz 0: Duyusal Sensörleri Başlat - Bireysel Hata Yönetimi ile
    logging.info("Faz 0: Duyusal sensörler başlatılıyor (Bireysel hata yönetimi aktif)...")
    try:
        logging.info("VisionSensor başlatılıyor...")
        sensors['vision'] = VisionSensor(config.get('vision', {}))
        if not getattr(sensors['vision'], 'is_camera_available', False):
             logging.warning("VisionSensor tam başlatılamadı veya kamera açılamadı.")
             # Simüle edilmiş girdi kullanılacak, obje None değil.
    except Exception as e:
        logging.critical(f"VisionSensor başlatılırken kritik hata oluştu: {e}", exc_info=True)
        sensors['vision'] = None # Kritik hata durumunda objeyi None yap


    try:
        logging.info("AudioSensor başlatılıyor...")
        sensors['audio'] = AudioSensor(config.get('audio', {}))
        if not getattr(sensors['audio'], 'is_audio_available', False):
             logging.warning("AudioSensor tam başlatılamadı veya ses akışı aktif değil.")
             # Simüle edilmiş girdi kullanılacak, obje None değil.
    except Exception as e:
        logging.critical(f"AudioSensor başlatılırken kritik hata oluştu: {e}", exc_info=True)
        sensors['audio'] = None # Kritik hata durumunda objeyi None yap


    # Faz 1 Başlangıcı: Processing Modülleri Başlat
    logging.info("Faz 1: Processing modülleri başlatılıyor...")
    try:
        processors['vision'] = VisionProcessor(config.get('processing_vision', {}))
        processors['audio'] = AudioProcessor(config.get('processing_audio', {}))
        logging.info("Processing modülleri başarıyla başlatıldı.")
    except Exception as e:
         logging.critical(f"Processing modülleri başlatılırken kritik hata oluştu: {e}", exc_info=True)
         # Processing kritik hata verirse sensörler çalışsa bile anlamı yok, main loop'u engelle.
         can_run_main_loop = False


    # Faz 1 Devamı: Representation Modülleri Başlat
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
         logging.warning("Processing modülleri başlatılamadığı için Representation modülleri atlandı.")


    # TODO: Diğer modülleri burada başlat (Faz 2 ve sonrası).
    # Bunların başlatılması main loop'u engellememeli (kritik değillerse),
    # bu yüzden bireysel try-except kullanın ve kritik hata durumunda objeyi None yapın.
    # try:
    #     logging.info("Memory modülü başlatılıyor...")
    #     other_modules['memory'] = MemoryModule(config.get('memory', {}))
    #     logging.info("Memory modülü başarıyla başlatıldı.")
    # except Exception as e:
    #     logging.critical(f"Memory modülü başlatılırken kritik hata oluştu: {e}", exc_info=True)
    #     other_modules['memory'] = None # Kritik hata durumunda objeyi None yap
    #     can_run_main_loop = False # Kritik modül başlatılamazsa main loop'u çalıştırma


    if can_run_main_loop:
         # Başlatma sırasında en az bir sensör, tüm işlemci ve tüm representation modülleri başarılıysa
         if active_sensors and processors and representers: # Temel modüllerin varlığını kontrol et
             logging.info("Temel modüller başarıyla başlatıldı. Evo bilişsel döngüye hazır.")
         else:
              logging.warning("Temel modüllerin bazıları başlatılamadı. Evo bilişsel döngüsü sınırlı çalışabilir veya durdurulabilir.")
              # Daha katı bir kural eklenebilir: Eğer temel modüller (sensores, processors, representers) tam değilse can_run_main_loop = False yap.
              # Şimdilik loglayıp devam edelim.
    else:
         logging.critical("Modül başlatma hataları nedeniyle Evo'nun bilişsel döngüsü başlatılamadı.")


    # --- Ana Bilişsel Döngü ---
    # Sadece can_run_main_loop True ise döngüye gir
    if can_run_main_loop:
        logging.info("Evo'nun bilişsel döngüsü başlatıldı...")
        loop_interval = config.get('cognitive_loop_interval', 0.1) # Döngü hızı

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
                     if processed_inputs['visual'] is not None:
                          logging.debug(f"Görsel input işlendi. Output Shape: {processed_inputs['visual'].shape}, Dtype: {processed_inputs['visual'].dtype}")
                     # else: logging.debug("Görsel input işlenemedi veya None döndü.")


                if processors.get('audio') and raw_inputs.get('audio') is not None:
                     processed_inputs['audio'] = processors['audio'].process(raw_inputs['audio'])
                     if processed_inputs['audio'] is not None:
                          logging.debug(f"Sesli input işlendi. Output Energy: {processed_inputs['audio']:.4f}") # Log enerji değeri
                     # else: logging.debug("Sesli input işlenemedi veya None döndü.")


                # logging.debug("İşlenmiş veri işleme tamamlandı.")


                # --- İşlenmiş Veriden Temsil Öğren (Faz 1 Devamı) ---
                learned_representation = None
                # Sadece representation learner objesi None değilse ve processed_inputs boş değilse
                if representers.get('main_learner') and processed_inputs:
                     learned_representation = representers['main_learner'].learn(processed_inputs)
                     # learned_representation'ın None olup olmadığı RepresentationLearner içinde loglanıyor.


                # TODO: Temsili hafızaya kaydet ve/veya hafızadan bilgi al (Faz 2)
                # if other_modules.get('memory') and learned_representation is not None: # Representation None değilse
                #      other_modules['memory'].store(learned_representation)
                #      relevant_memory = other_modules['memory'].retrieve(learned_representation)
                #      if relevant_memory:
                #           logging.debug(f"Hafızadan ilgili bilgi alındı. Örnek: {relevant_memory[:min(len(relevant_memory), 10)]}...") # İlk 10 elemanı logla


                # TODO: Hafıza ve temsile göre bilişsel işlem yap (anlama, karar verme) (Faz 3+)
                # if other_modules.get('cognition') and (learned_representation is not None or relevant_memory is not None):
                #      decision = other_modules['cognition'].decide(
                #          learned_representation, # Temsil None olabilir
                #          relevant_memory if 'relevant_memory' in locals() else None # Hafıza None olabilir
                #      )
                #      if decision is not None:
                #           logging.debug(f"Bilişsel karar alındı: {decision}")


                # TODO: Karara göre bir tepki üret (Faz 3+)
                # if other_modules.get('motor_control') and decision is not None:
                #      raw_output = other_modules['motor_control'].generate_response(decision)
                #      if raw_output:
                #           logging.debug(f"Tepki üretildi. İlk kısım: {str(raw_output)[:min(len(str(raw_output)), 50)]}...")


                # TODO: Tepkiyi interaction API üzerinden dışarı aktar (Faz 3+)
                # if other_modules.get('interaction') and raw_output is not None:
                #    other_modules['interaction'].send_output(raw_output)
                #    logging.debug("Tepki dışarı aktarıldı.")


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
                # if other_modules.get('cognition') and other_modules['cognition'].should_stop(): break # Örnek: Evo uykuya dalarsa

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


        # TODO: Diğer başlatılan modüllerin kapatılmasını sağla (API gibi thread'ler)
        # for module_name, module_obj in other_modules.items():
        #      if hasattr(module_obj, 'stop'): # Eğer stop metodu varsa
        #           logging.info(f"{module_name} kapatılıyor...")
        #           module_obj.stop()
        #      elif hasattr(module_obj, 'cleanup'): # Eğer cleanup metodu varsa
        #            logging.info(f"{module_name} temizleniyor...")
        #            module_obj.cleanup()


        logging.info("Evo durduruldu.")


# Ana çalıştırma noktası
if __name__ == '__main__':
    run_evo()