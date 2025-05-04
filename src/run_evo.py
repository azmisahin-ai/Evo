# src/run_evo.py

import logging
import time
# Temel modülleri import edeceğiz
from src.senses.vision import VisionSensor
from src.senses.audio import AudioSensor
# Diğer modüller (processing, core, interaction) geldikçe buraya eklenecek

# Konfigürasyon yükleme (şimdilik basit bir placeholder)
# Gelecekte config/main_config.yaml dosyasından okunacak
def load_config():
    """Basit placeholder konfigürasyon yükleyici."""
    # TODO: config/main_config.yaml dosyasından gerçek konfigürasyonu yükle
    return {
        'vision': {
            'camera_index': 0 # Varsayılan kamera indeksi
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
    logging.getLogger('src').setLevel(logging.DEBUG) # src içindeki detaylı logları göster

    logging.info("Evo canlanıyor...")

    # Konfigürasyonu yükle
    config = load_config()
    logging.info("Konfigürasyon yüklendi.")

    # --- Faz 0: Duyuların Başlatılması ---
    logging.info("Faz 0: Duyular başlatılıyor...")
    vision_sensor = None
    audio_sensor = None
    try:
        # VisionSensor'ı başlat
        logging.info("VisionSensor başlatılıyor...")
        vision_sensor = VisionSensor(config.get('vision', {}))
        if vision_sensor.cap is None or not vision_sensor.cap.isOpened():
             logging.error("VisionSensor başlatılamadı veya kamera açılamadı. Görsel girdi devre dışı.")

        # AudioSensor'ı başlat
        logging.info("AudioSensor başlatılıyor...")
        audio_sensor = AudioSensor(config.get('audio', {}))
        if audio_sensor.stream is None or not audio_sensor.stream.is_active():
             logging.error("AudioSensor başlatılamadı veya ses akışı aktif değil. İşitsel girdi devre dışı.")

        if (vision_sensor.cap is not None and vision_sensor.cap.isOpened()) or \
           (audio_sensor.stream is not None and audio_sensor.stream.is_active()):
            logging.info("Duyusal Sensörler başarıyla başlatıldı (en az biri aktif).")
        else:
            logging.warning("Hiçbir duyusal sensör aktif değil. Evo girdi alamayacak.")


    except Exception as e:
        logging.critical(f"Duyusal sensörler başlatılırken kritik hata oluştu: {e}")
        # Hata durumunda Sensörleri kapatmayı dene
        if vision_sensor: vision_sensor.stop_stream()
        if audio_sensor: audio_sensor.stop_stream()
        logging.critical("Evo başlangıçta duyusal hata nedeniyle durduruldu.")
        return # Başlangıç hatasında Evo'yu durdur

    # --- Ana Bilişsel Döngü (Duyuları Yakalama ve Loglama) ---
    logging.info("Evo'nun bilişsel döngüsü başlatıldı...")
    loop_interval = config.get('cognitive_loop_interval', 0.1) # Döngü hızı

    try:
        while True:
            start_time = time.time()

            # --- Duyu Verisini Yakala ---
            # Sensörler başlatıldıysa capture metodlarını çağır
            visual_input = None
            if vision_sensor and vision_sensor.cap and vision_sensor.cap.isOpened():
                 visual_input = vision_sensor.capture_frame()

            audio_input = None
            if audio_sensor and audio_sensor.stream and audio_sensor.stream.is_active():
                 audio_input = audio_sensor.capture_chunk()


            # --- Yakalanan Veriyi Logla ---
            # Gerçek veriyi logluyoruz (veya None olduğunu)
            if visual_input is not None:
                logging.debug(f"Görsel Input Yakalandı. Shape: {visual_input.shape}, Dtype: {visual_input.dtype}")
                # TODO: visual_input'u processing modülüne ilet
            else:
                 logging.debug("Görsel Input Alınamadı.")

            if audio_input is not None:
                logging.debug(f"Sesli Input Yakalandı. Shape: {audio_input.shape}, Dtype: {audio_input.dtype}")
                # TODO: audio_input'u processing modülüne ilet
            else:
                 logging.debug("Sesli Input Alınamadı (Sessizlik veya Akış Sorunu).")


            # TODO: Yakalanan veriyi processing, representation, memory, cognition modüllerine ilet ve işle
            # TODO: İşleme sonucunda bir tepki üret (motor_control)
            # TODO: Tepkiyi interaction API üzerinden dışarı aktar

            # --- Döngü Gecikmesi ---
            # Döngünün belirli bir hızda çalışmasını sağla
            elapsed_time = time.time() - start_time
            sleep_time = loop_interval - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logging.warning(f"Bilişsel döngü {loop_interval} saniyeden daha uzun sürdü ({elapsed_time:.4f}s). İşlem yükü yüksek olabilir.")

            # Gelecekte döngüyü sonlandıracak bir mekanizma eklenecek (örn. kullanıcı sinyali)
            # if should_stop_evo(): break

    except KeyboardInterrupt:
        logging.warning("Ctrl+C algılandı. Evo durduruluyor...")
    except Exception as e:
        logging.critical(f"Evo'nun ana döngüsünde beklenmedik hata: {e}", exc_info=True) # Detaylı hata logu için exc_info=True

    finally:
        # --- Kaynakları Temizleme ---
        logging.info("Evo kaynakları temizleniyor...")
        if vision_sensor:
            vision_sensor.stop_stream()
        if audio_sensor:
            audio_sensor.stop_stream()
        # TODO: Diğer başlatılan modüllerin kapatılmasını sağla
        logging.info("Evo durduruldu.")

if __name__ == '__main__':
    run_evo()