# src/run_evo.py

import logging
import time
# Temel modülleri import edeceğiz
from src.senses.vision import VisionSensor
from src.senses.audio import AudioSensor
# Diğer modüller (processing, core, interaction) geldikçe buraya eklenecek

def run_evo():
    """
    Evo'nun çekirdek bilişsel döngüsünü ve arayüzlerini başlatır.
    Bu fonksiyon çağrıldığında Evo "canlanır".
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Evo canlanıyor...")

    # --- Faz 0: Duyuların Başlatılması ---
    logging.info("Faz 0: Duyular başlatılıyor...")
    vision_sensor = None
    audio_sensor = None
    try:
        vision_sensor = VisionSensor()
        audio_sensor = AudioSensor()
        logging.info("Duyular başarıyla başlatıldı.")
    except Exception as e:
        logging.error(f"Duyular başlatılırken hata oluştu: {e}")
        # Hata durumunda Sensörleri kapatmayı dene
        if vision_sensor: vision_sensor.stop_stream()
        if audio_sensor: audio_sensor.stop_stream()
        logging.error("Evo başlangıçta durduruldu.")
        return # Başlangıç hatasında Evo'yu durdur

    # --- Ana Bilişsel Döngü (Şimdilik Basit Bir Placeholder) ---
    # Bu döngü, Evo'nun uyanık kaldığı sürece çalışacak.
    logging.info("Evo'nun bilişsel döngüsü başlatıldı (Placeholder)...")
    try:
        while True:
            # Duyu verisini yakala (Placeholder)
            visual_input = vision_sensor.capture_frame()
            audio_input = audio_sensor.capture_chunk()

            # TODO: Yakalanan veriyi processing, representation, memory, cognition modüllerine ilet
            logging.debug(f"Görsel Input: {visual_input}, Sesli Input: {audio_input}")

            # TODO: İşleme sonucunda bir tepki üret (motor_control)
            # TODO: Tepkiyi interaction API üzerinden dışarı aktar

            # Şimdilik sadece bir simülasyon gecikmesi ekleyelim
            time.sleep(1) # Her saniye bir döngü (örnek)

            # Gelecekte döngüyü sonlandıracak bir mekanizma eklenecek
            # if should_stop_evo(): break

    except KeyboardInterrupt:
        logging.warning("Ctrl+C algılandı. Evo durduruluyor...")
    except Exception as e:
        logging.critical(f"Evo'nun ana döngüsünde kritik hata: {e}")

    finally:
        # --- Kaynakları Temizleme ---
        logging.info("Evo kaynakları temizleniyor...")
        if vision_sensor: vision_sensor.stop_stream()
        if audio_sensor: audio_sensor.stop_stream()
        # TODO: Diğer modüllerin kapatılmasını sağla
        logging.info("Evo durduruldu.")

if __name__ == '__main__':
    run_evo()