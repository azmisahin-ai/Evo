# src/run_evo.py

import logging
import time
# Temel modülleri import edeceğiz
# Import sırasında hata olursa bu script de başlayamaz, o yüzden try-except burada değil
from src.senses.vision import VisionSensor
from src.senses.audio import AudioSensor
# Processing modüllerini import et
from src.processing.vision import VisionProcessor
from src.processing.audio import AudioProcessor
# Diğer modüller (representation, memory, cognition, motor_control, interaction) geldikçe buraya eklenecek

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
             'output_width': 64,
             'output_height': 64
        },
         'processing_audio': { # Audio Processor için konfigürasyon
             # Örneğin, MFCC sayısı gibi ayarlar buraya gelebilir
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
    other_modules = {} # Representation, memory, cognition, interaction vb. buraya gelecek

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


    # Başlatma durumunu kontrol et ve logla (Sözlükteki objelerin None olup olmadığına ve internal flag'lere bakarak)
    active_sensors = [name for name, sensor in sensors.items()
                      if sensor and (getattr(sensor, 'is_camera_available', False) or getattr(sensor, 'is_audio_available', False))]
    if active_sensors:
        logging.info(f"Duyusal Sensörler başlatıldı ({', '.join(active_sensors)} aktif).")
    else:
        logging.warning("Hiçbir duyusal sensör aktif değil. Evo girdi alamayacak.")


    # Faz 1: Processing Modülleri Başlat
    logging.info("Faz 1: Processing modülleri başlatılıyor...")
    try:
        processors['vision'] = VisionProcessor(config.get('processing_vision', {}))
        processors['audio'] = AudioProcessor(config.get('processing_audio', {}))
        logging.info("Processing modülleri başarıyla başlatıldı.")
    except Exception as e:
         logging.critical(f"Processing modülleri başlatılırken kritik hata oluştu: {e}", exc_info=True)
         can_run_main_loop = False # Processing kritik hata verirse main loop'u çalıştırma


    # TODO: Diğer modülleri burada başlat (Faz 1 ve sonrası).
    # Bunların başlatılması main loop'u engellememeli,
    # bu yüzden bireysel try-except kullanın ve kritik hata durumunda objeyi None yapın.
    # try:
    #     logging.info("Representation modülü başlatılıyor...")
    #     other_modules['representation'] = RepresentationModule(config.get('representation', {}))
    #     logging.info("Representation modülü başarıyla başlatıldı.")
    # except Exception as e:
    #     logging.critical(f"Representation modülü başlatılırken kritik hata oluştu: {e}", exc_info=True)
    #     other_modules['representation'] = None # Kritik hata durumunda objeyi None yap
    #     can_run_main_loop = False # Kritik modül başlatılamazsa main loop'u çalıştırma


    if can_run_main_loop:
         logging.info("Temel modüller başarıyla başlatıldı.")
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
                else:
                     raw_inputs['visual'] = None # Sensor objesi hiç oluşturulamadıysa girdi None


                if sensors.get('audio'):
                    raw_inputs['audio'] = sensors['audio'].capture_chunk()
                else:
                     raw_inputs['audio'] = None # Sensor objesi hiç oluşturulamadıysa girdi None

                # logging.debug("Duyusal veri yakalama tamamlandı.")

                # --- Yakalanan Veriyi İşle (Faz 1) ---
                processed_inputs = {}
                # Sadece işlemci objesi None değilse ve girdi None değilse process metodunu çağır
                if processors.get('vision') and raw_inputs.get('visual') is not None:
                     processed_inputs['visual'] = processors['vision'].process(raw_inputs['visual'])
                     if processed_inputs['visual'] is not None:
                          logging.debug(f"Görsel input işlendi. Output Shape: {processed_inputs['visual'].shape}, Dtype: {processed_inputs['visual'].dtype}")
                     else:
                          logging.debug("Görsel input işlenemedi veya None döndü.")

                if processors.get('audio') and raw_inputs.get('audio') is not None:
                     processed_inputs['audio'] = processors['audio'].process(raw_inputs['audio'])
                     if processed_inputs['audio'] is not None:
                          logging.debug(f"Sesli input işlendi. Output Energy: {processed_inputs['audio']:.4f}") # Log enerji değeri
                     else:
                          logging.debug("Sesli input işlenemedi veya None döndü.")


                # TODO: İşlenmiş veriyi representation modülüne ilet (Faz 1 sonrası)
                # if other_modules.get('representation') and processed_inputs: # processed_inputs sözlüğü boş değilse
                #     learned_representation = other_modules['representation'].learn(processed_inputs)
                #     if learned_representation is not None:
                #          logging.debug(f"Representation öğrenildi. Shape: {learned_representation.shape}")


                # TODO: Temsili hafızaya kaydet ve/veya hafızadan bilgi al (Faz 2)
                # if other_modules.get('memory') and 'learned_representation' in locals() and learned_representation is not None:
                #      other_modules['memory'].store(learned_representation)
                #      relevant_memory = other_modules['memory'].retrieve(learned_representation)
                #      if relevant_memory:
                #           logging.debug(f"Hafızadan ilgili bilgi alındı. Örnek: {relevant_memory[:10]}...") # İlk 10 elemanı logla


                # TODO: Hafıza ve temsile göre bilişsel işlem yap (anlama, karar verme) (Faz 3+)
                # if other_modules.get('cognition') and ('learned_representation' in locals() or 'relevant_memory' in locals()):
                #      decision = other_modules['cognition'].decide(
                #          learned_representation if 'learned_representation' in locals() else None,
                #          relevant_memory if 'relevant_memory' in locals() else None
                #      )
                #      if decision is not None:
                #           logging.debug(f"Bilişsel karar alındı: {decision}")


                # TODO: Karara göre bir tepki üret (Faz 3+)
                # if other_modules.get('motor_control') and decision is not None:
                #      raw_output = other_modules['motor_control'].generate_response(decision)
                #      if raw_output:
                #           logging.debug(f"Tepki üretildi. İlk kısım: {str(raw_output)[:50]}...")


                # TODO: Tepkiyi interaction API üzerinden dışarı aktar (Faz 3+)
                # if other_modules.get('interaction') and 'raw_output' in locals() and raw_output is not None:
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
            logging.critical(f"Evo'nun ana döngüsünde beklenmedik kritik hata: {e}", exc_info=True)

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

            # TODO: Diğer başlatılan modüllerin kapatılmasını sağla (API gibi thread'ler)
            # for module_name, module_obj in other_modules.items():
            #      if hasattr(module_obj, 'stop'):
            #           logging.info(f"{module_name} kapatılıyor...")
            #           module_obj.stop()


            logging.info("Evo durduruldu.")


# Ana çalıştırma noktası
if __name__ == '__main__':
    run_evo()