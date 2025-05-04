# src/run_evo.py

import logging
import time
# Temel modülleri import edeceğiz
from src.senses.vision import VisionSensor
from src.senses.audio import AudioSensor
# Yeni processing modüllerini import et
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
    # Modüller başlatılırken hata olursa yakalamak için try-except blokları kullanılabilir.
    vision_sensor = None
    audio_sensor = None
    vision_processor = None
    audio_processor = None
    # representation_module = None # Gelecekte
    # memory_module = None # Gelecekte
    # cognition_module = None # Gelecekte
    # motor_control_module = None # Gelecekte
    # interaction_api = None # Gelecekte


    try:
        # Faz 0: Duyusal Sensörleri Başlat
        logging.info("Faz 0: Duyusal sensörler başlatılıyor...")
        vision_sensor = VisionSensor(config.get('vision', {}))
        audio_sensor = AudioSensor(config.get('audio', {}))
        # Sensörlerin durumu loglar içinde belirtiliyor.


        # Faz 1: Processing Modüllerini Başlat
        logging.info("Faz 1: Processing modülleri başlatılıyor...")
        vision_processor = VisionProcessor(config.get('processing_vision', {})) # processing_vision config'i yok, varsayılan kullanılacak
        audio_processor = AudioProcessor(config.get('processing_audio', {})) # processing_audio config'i yok, varsayılan kullanılacak
        logging.info("Processing modülleri başarıyla başlatıldı.")

        # TODO: Diğer modülleri burada başlat (Faz 1 ve sonrası)
        # representation_module = RepresentationModule(config.get('representation', {}))
        # memory_module = MemoryModule(config.get('memory', {}))
        # cognition_module = CognitionModule(config.get('cognition', {}))
        # motor_control_module = MotorControlModule(config.get('motor_control', {}))
        # interaction_api = InteractionAPI(config.get('interaction', {}))
        # interaction_api.start() # API bir thread içinde çalışabilir


        logging.info("Tüm başlatılması gereken modüller başlatıldı (aktif olanlar loglarda).")

    except Exception as e:
        logging.critical(f"Modüller başlatılırken kritik hata oluştu: {e}", exc_info=True)
        # Hata durumunda tüm başlatılmış kaynakları temizle
        # Bu kısım cleanup_evo fonksiyonuna taşınabilir
        if vision_sensor: vision_sensor.stop_stream()
        if audio_sensor: audio_sensor.stop_stream()
        # TODO: Diğer modüllerin kapatılmasını sağla
        logging.critical("Evo başlangıç hatası nedeniyle durduruldu.")
        return # Başlangıç hatasında Evo'yu durdur

    # --- Ana Bilişsel Döngü ---
    logging.info("Evo'nun bilişsel döngüsü başlatıldı...")
    loop_interval = config.get('cognitive_loop_interval', 0.1) # Döngü hızı

    try:
        while True:
            start_time = time.time()

            # --- Duyu Verisini Yakala (Faz 0) ---
            visual_input = None
            if vision_sensor: # Sensör objesi oluşturulduysa
                 visual_input = vision_sensor.capture_frame()

            audio_input = None
            if audio_sensor: # Sensör objesi oluşturulduysa
                 audio_input = audio_sensor.capture_chunk()

            # logging.debug("Duyu verisi yakalama tamamlandı.")

            # --- Yakalanan Veriyi İşle (Faz 1) ---
            processed_visual = None
            if vision_processor and visual_input is not None: # İşlemci ve girdi varsa
                 processed_visual = vision_processor.process(visual_input)
                 if processed_visual is not None:
                      logging.debug(f"Görsel input işlendi. Output Shape: {processed_visual.shape}")
                 # else: logging.debug("Görsel input işlenemedi veya None döndü.")


            processed_audio = None
            if audio_processor and audio_input is not None: # İşlemci ve girdi varsa
                 processed_audio = audio_processor.process(audio_input)
                 if processed_audio is not None:
                      logging.debug(f"Sesli input işlendi. Output Shape: {processed_audio.shape}")
                 # else: logging.debug("Sesli input işlenemedi veya None döndü.")


            # TODO: İşlenmiş veriyi representation modülüne ilet (Faz 1 sonrası)
            # learned_representation = representation_module.learn(processed_visual, processed_audio)

            # TODO: Temsili hafızaya kaydet ve/veya hafızadan bilgi al (Faz 2)
            # memory_module.store(learned_representation)
            # relevant_memory = memory_module.retrieve(learned_representation)

            # TODO: Hafıza ve temsile göre bilişsel işlem yap (anlama, karar verme) (Faz 3+)
            # decision = cognition_module.decide(learned_representation, relevant_memory)

            # TODO: Karara göre bir tepki üret (Faz 3+)
            # raw_output = motor_control_module.generate_response(decision)

            # TODO: Tepkiyi interaction API üzerinden dışarı aktar (Faz 3+)
            # interaction_api.send_output(raw_output)


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
            # if cognition_module.should_stop(): break # Örnek: Evo uykuya dalarsa

    except KeyboardInterrupt:
        logging.warning("Ctrl+C algılandı. Evo durduruluyor...")
    except Exception as e:
        logging.critical(f"Evo'nun ana döngüsünde beklenmedik kritik hata: {e}", exc_info=True) # Detaylı hata logu için exc_info=True

    finally:
        # --- Kaynakları Temizleme ---
        logging.info("Evo kaynakları temizleniyor...")
        # Tüm başlatılmış objelerin stop veya cleanup metodlarını çağır
        if vision_sensor:
            vision_sensor.stop_stream()
        if audio_sensor:
            audio_sensor.stop_stream()
        # TODO: Diğer başlatılan modüllerin kapatılmasını sağla
        # if interaction_api: interaction_api.stop()
        # Processor'lar genellikle kapatma gerektirmez ama emin olmak için kontrol edilebilir:
        # if vision_processor: vision_processor.cleanup() # Eğer cleanup metodu varsa
        # if audio_processor: audio_processor.cleanup() # Eğer cleanup metodu varsa


        logging.info("Evo durduruldu.")

# Ana çalıştırma noktası
if __name__ == '__main__':
    run_evo()