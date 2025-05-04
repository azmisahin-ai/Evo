# src/run_evo.py

import logging
import time
# Temel modülleri import edeceğiz
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # src içindeki detaylı logları göstermek için seviyeyi DEBUG yap
    logging.getLogger('src').setLevel(logging.DEBUG)

    logging.info("Evo canlanıyor...")

    # Konfigürasyonu yükle
    config = load_config()
    logging.info("Konfigürasyon yüklendi.")

    # --- Modülleri Başlatma ---
    # Modül objelerini depolamak için sözlük veya liste kullanmak daha düzenli olabilir.
    sensors = {}
    processors = {}
    # Diğer modül kategorileri geldikçe eklenecek

    modules_initialized_successfully = False # Başlatma başarılı oldu mu?

    try:
        # Faz 0: Duyusal Sensörleri Başlat
        logging.info("Faz 0: Duyusal sensörler başlatılıyor...")
        sensors['vision'] = VisionSensor(config.get('vision', {}))
        sensors['audio'] = AudioSensor(config.get('audio', {}))

        # Başlatma durumunu kontrol et ve logla
        active_sensors = [name for name, sensor in sensors.items() if getattr(sensor, 'is_camera_available', False) or getattr(sensor, 'is_audio_available', False)]
        if active_sensors:
            logging.info(f"Duyusal Sensörler başarıyla başlatıldı ({', '.join(active_sensors)} aktif).")
        else:
            logging.warning("Hiçbir duyusal sensör aktif değil. Evo girdi alamayacak.")


        # Faz 1: Processing Modüllerini Başlat
        logging.info("Faz 1: Processing modülleri başlatılıyor...")
        processors['vision'] = VisionProcessor(config.get('processing_vision', {}))
        processors['audio'] = AudioProcessor(config.get('processing_audio', {}))
        logging.info("Processing modülleri başarıyla başlatıldı.")


        # TODO: Diğer modülleri burada başlat (Faz 1 ve sonrası)
        # representation_module = RepresentationModule(config.get('representation', {}))
        # memory_module = MemoryModule(config.get('memory', {}))
        # cognition_module = CognitionModule(config.get('cognition', {}))
        # motor_control_module = MotorControlModule(config.get('motor_control', {}))
        # interaction_api = InteractionAPI(config.get('interaction', {}))
        # interaction_api.start() # API bir thread içinde çalışabilir

        logging.info("Tüm başlatılması gereken modüller başlatıldı (aktif olanlar loglarda).")
        modules_initialized_successfully = True # Başlatma bloğu hatasız tamamlandı

    except Exception as e:
        logging.critical(f"Modüller başlatılırken kritik hata oluştu: {e}", exc_info=True)
        # Başlatma hatası durumunda döngüye girmeyecek, cleanup yapılacak.


    # --- Ana Bilişsel Döngü ---
    # Sadece modüller başarılı başlatıldıysa döngüye gir
    if modules_initialized_successfully:
        logging.info("Evo'nun bilişsel döngüsü başlatıldı...")
        loop_interval = config.get('cognitive_loop_interval', 0.1) # Döngü hızı

        try:
            while True:
                start_time = time.time()

                # --- Duyu Verisini Yakala (Faz 0) ---
                # Sensör objeleri varsa capture metodlarını çağır
                raw_inputs = {}
                if 'vision' in sensors and sensors['vision']:
                     raw_inputs['visual'] = sensors['vision'].capture_frame()
                if 'audio' in sensors and sensors['audio']:
                     raw_inputs['audio'] = sensors['audio'].capture_chunk()

                # logging.debug("Duyusal veri yakalama tamamlandı.")

                # --- Yakalanan Veriyi İşle (Faz 1) ---
                processed_inputs = {}
                if 'vision' in processors and processors['vision'] and raw_inputs.get('visual') is not None:
                     processed_inputs['visual'] = processors['vision'].process(raw_inputs['visual'])
                     if processed_inputs['visual'] is not None:
                          # logging.debug(f"Görsel input işlendi. Output Shape: {processed_inputs['visual'].shape}")
                          pass # Zaten VisionProcessor içindeki loglar detay veriyor
                     # else: logging.debug("Görsel input işlenemedi veya None döndü.")


                if 'audio' in processors and processors['audio'] and raw_inputs.get('audio') is not None:
                     processed_inputs['audio'] = processors['audio'].process(raw_inputs['audio'])
                     if processed_inputs['audio'] is not None:
                          # logging.debug(f"Sesli input işlendi. Output: {processed_inputs['audio']:.4f}")
                          pass # Zaten AudioProcessor içindeki loglar detay veriyor
                     # else: logging.debug("Sesli input işlenemedi veya None döndü.")


                # TODO: İşlenmiş veriyi representation modülüne ilet (Faz 1 sonrası)
                # learned_representation = representation_module.learn(processed_inputs) # processed_inputs sözlüğünü ilet

                # TODO: Temsili hafızaya kaydet ve/veya hafızadan bilgi al (Faz 2)
                # memory_module.store(learned_representation)
                # relevant_memory = memory_module.retrieve(learned_representation)

                # TODO: Hafıza ve temsile göre bilişsel işlem yap (anlama, karar verme) (Faz 3+)
                # decision = cognition_module.decide(learned_representation, relevant_memory)

                # TODO: Karara göre bir tepki üret (Faz 3+)
                # raw_output = motor_control_module.generate_response(decision)

                # TODO: Tepkiyi interaction API üzerinden dışarı aktar (Faz 3+)
                # if 'interaction_api' in locals() and interaction_api:
                #    interaction_api.send_output(raw_output)


                # --- Döngü Gecikmesi ---
                # Döngünün belirli bir hızda çalışmasını sağla
                elapsed_time = time.time() - start_time
                sleep_time = loop_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Döngü intervalinden uzun süren durumlar için uyarı
                    # DEBUG seviyesinde loglamak, sürekli uyarı vermesini engeller
                    logging.debug(f"Bilişsel döngü {loop_interval} saniyeden daha uzun sürdü ({elapsed_time:.4f}s). İşlem yükü yüksek olabilir.")


                # Gelecekte döngüyü sonlandıracak bir mekanizma eklenecek (örn. kullanıcı sinyali, içsel durum)
                # if 'cognition_module' in locals() and cognition_module and cognition_module.should_stop(): break # Örnek: Evo uykuya dalarsa

        except KeyboardInterrupt:
            logging.warning("Ctrl+C algılandı. Evo durduruluyor...")
        except Exception as e:
            logging.critical(f"Evo'nun ana döngüsünde beklenmedik kritik hata: {e}", exc_info=True) # Detaylı hata logu için exc_info=True

        finally:
            # --- Kaynakları Temizleme ---
            logging.info("Evo kaynakları temizleniyor...")
            # Tüm başlatılmış objelerin stop veya cleanup metodlarını çağır
            if 'vision' in sensors and sensors['vision']:
                sensors['vision'].stop_stream()
            if 'audio' in sensors and sensors['audio']:
                sensors['audio'].stop_stream()

        # Processor'lar genellikle kapatma gerektirmez ama emin olmak için kontrol edilebilir veya cleanup metodu eklenebilir
        # if 'vision' in processors and processors['vision'] and hasattr(processors['vision'], 'cleanup'): processors['vision'].cleanup()
        # if 'audio' in processors and processors['audio'] and hasattr(processors['audio'], 'cleanup'): processors['audio'].cleanup()

        # TODO: Diğer başlatılan modüllerin kapatılmasını sağla (API gibi thread'ler)
        # if 'interaction_api' in locals() and interaction_api: interaction_api.stop()


        logging.info("Evo durduruldu.")

# Ana çalıştırma noktası
if __name__ == '__main__':
    run_evo()