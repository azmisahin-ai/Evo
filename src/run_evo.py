# src/run_evo.py

import logging
import time
import numpy as np
# import sys # Programı sonlandırmak gerekirse import edilebilir

# Loglama ve Konfigürasyon yardımcı fonksiyonlarını import et
from src.core.logging_utils import setup_logging
from src.core.config_utils import load_config_from_yaml
# Modül başlatma yardımcı fonksiyonlarını import et
from src.core.module_loader import initialize_modules, cleanup_modules # <<< Yeni import


# load_config fonksiyonu kaldırıldı - load_config_from_yaml kullanılacak.
# def load_config(): ... # <<< BU FONKSİYON SİLİNDİ

def run_evo():
    """
    Evo'nın çekirdek bilişsel döngüsünü ve arayüzlerini başlatır.
    Bu fonksiyon çağrıldığında Evo "canlanır".
    """
    # 1. Loglama sistemini merkezi utility ile yapılandır (Her zaman ilk adım olmalı)
    setup_logging(level=logging.DEBUG) # DEBUG loglarını görmek için DEBUG seviyesiyle başlat

    # Bu dosyanın kendi logger'ını oluştur (setup_logging'den sonra)
    # __name__ 'src.run_evo' olacak
    logger = logging.getLogger(__name__)

    logger.info("Evo canlanıyor...")

    # 2. Konfigürasyonu Yükle
    config_path = "config/main_config.yaml" # Yapılandırma dosyasının yolu
    config = load_config_from_yaml(config_path)

    if not config: # Eğer config yüklenemezse (dosya yok veya hata var)
        logger.critical(f"Evo, yapılandırma yüklenemediği için başlatılamıyor. Lütfen {config_path} dosyasını kontrol edin.")
        # Burada programı sonlandırmak mantıklı olabilir
        # sys.exit(1) # Eğer sys import ediliyorsa
        return # run_evo fonksiyonundan çık


    logger.info(f"Konfigürasyon yüklendi: {config_path}")

    # 3. Modülleri Başlat
    # initialize_modules fonksiyonu başlatılan objeleri ve ana döngünün çalışıp çalışamayacağını döndürür
    module_objects, can_run_main_loop = initialize_modules(config) # <<< Modül başlatma buraya taşındı

    # Başlatma sırasında kritik bir hata olduysa programı sonlandır (initialize_modules bayrağına göre)
    if not can_run_main_loop:
        logger.critical("Evo'nın temel modülleri başlatılamadığı için program sonlandırılıyor.")
        # Kaynakları temizle (initialize_modules tarafından başlatılanları)
        cleanup_modules(module_objects)
        # sys.exit(1) # Eğer sys import ediliyorsa
        return # run_evo fonksiyonundan çık


    # Modül objelerine kolay erişim için değişkenler atayalım (opsiyonel ama kodu sadeleştirebilir)
    # Dictionary'lerden None kontrolü yaparak güvenli erişim sağlamalıyız.
    sensors = module_objects.get('sensors', {})
    processors = module_objects.get('processors', {})
    representers = module_objects.get('representers', {})
    memories = module_objects.get('memories', {})
    cognition_modules = module_objects.get('cognition', {})
    motor_control_modules = module_objects.get('motor_control', {})
    interaction_modules = module_objects.get('interaction', {})


    # --- Ana Bilişsel Döngü ---
    # Sadece can_run_main_loop True ise döngüye gir
    if can_run_main_loop: # Bu kontrol zaten yukarıda yapılıyor, ama döngüye giriş şartı olarak tekrar kontrol etmek netlik sağlar.
        logger.info("Evo'nın bilişsel döngüsü başlatıldı...")
        # Konfigürasyondan değerleri alırken .get() kullanmak veya değerin varlığını kontrol etmek önemlidir.
        # config_utils.load_config_from_yaml hata durumunda {} döndürdüğü için .get() kullanımı daha güvenlidir.
        loop_interval = config.get('cognitive_loop_interval', 0.1) # Config'ten alındı
        # Memory config'i yoksa veya num_retrieved_memories yoksa varsayılan 5 kullanılır
        num_memories_to_retrieve = config.get('memory', {}).get('num_retrieved_memories', 5) # Config'ten alındı


        try:
            # Döngüden önce gerekli objelerin (Processing, Represent, Memory, Cognition, MotorControl)
            # hala None olmadığından emin olalım. initialize_modules bayrağı bunu zaten sağlıyor olmalı,
            # ama eğer döngü içinde dinamik olarak modüllerin None olma durumu olursa burada da kontrol gerekebilir.
            # Şimdilik initialize_modules'ın garantisine güvenelim.

            while True: # Main loop runs as long as no unhandled error or KeyboardInterrupt
                start_time = time.time()

                # --- Duyu Verisini Yakala (Faz 0) ---
                # Sensör objesi None olsa bile çağırma, o objenin varlığını kontrol ederek yapılıyor.
                raw_inputs = {}
                if sensors.get('vision'):
                    raw_inputs['visual'] = sensors['vision'].capture_frame()
                if sensors.get('audio'):
                    raw_inputs['audio'] = sensors['audio'].capture_chunk()

                # DEBUG Log: Yakalanan Ham Görsel/Ses Verisi
                if raw_inputs.get('visual') is not None:
                     logger.debug(f"RUN_EVO: Raw Visual Input yakalandı. Shape: {raw_inputs['visual'].shape}, Dtype: {raw_inputs['visual'].dtype}")
                else:
                     logger.debug("RUN_EVO: Raw Visual Input None.")

                if raw_inputs.get('audio') is not None:
                     logger.debug(f"RUN_EVO: Raw Audio Input yakalandı. Shape: {raw_inputs['audio'].shape}, Dtype: {raw_inputs['audio'].dtype}")
                else:
                     logger.debug("RUN_EVO: Raw Audio Input None.")


                # --- Yakalanan Veriyi İşle (Faz 1 Başlangıcı) ---
                # İşlemci objesi None olsa bile çağırma, o objenin varlığını kontrol ederek yapılıyor.
                # Process metodları None girdiyi kendi içinde yönetmeli.
                processed_inputs = {}
                if processors.get('vision'): # Process modülü objesi var mı?
                     processed_inputs['visual'] = processors['vision'].process(raw_inputs.get('visual')) # Girdi None olabilir
                     # if processed_inputs.get('visual') is not None: ... (Loglar Process içinde)

                if processors.get('audio'): # Process modülü objesi var mı?
                     processed_inputs['audio'] = processors['audio'].process(raw_inputs.get('audio')) # Girdi None olabilir
                     # if processed_inputs.get('audio') is not None: ... (Loglar Process içinde)

                # DEBUG Log: İşlenmiş Görsel/Ses Verisi
                if processed_inputs.get('visual') is not None:
                     logger.debug(f"RUN_EVO: Processed Visual Output. Shape: {processed_inputs['visual'].shape}, Dtype: {processed_inputs['visual'].dtype}")
                else:
                     logger.debug("RUN_EVO: Processed Visual Output None.")

                if processed_inputs.get('audio') is not None:
                     # Output (Energy) sadece sayısal bir değer olduğu için .get() ile erişelim
                     audio_energy = processed_inputs['audio']
                     logger.debug(f"RUN_EVO: Processed Audio Output (Energy). Value: {audio_energy:.4f}")
                else:
                     logger.debug("RUN_EVO: Processed Audio Output None.")


                # --- İşlenmiş Veriden Temsil Öğren (Faz 1 Devamı) ---
                # Representation learner objesi None olsa bile çağırma, o objenin varlığını kontrol ederek yapılıyor.
                # Learn metodu None girdiyi kendi içinde yönetmeli (processed_inputs boş olabilir).
                learned_representation = None
                if representers.get('main_learner'): # Representation modülü objesi var mı?
                     # processed_inputs boş sözlük olabilir, learn metodu bunu yönetmeli.
                     learned_representation = representers['main_learner'].learn(processed_inputs)
                     # if learned_representation is not None: ... (Loglar Representation içinde)

                # DEBUG Log: Öğrenilmiş Temsil
                if learned_representation is not None:
                     logger.debug(f"RUN_EVO: Learned Representation. Shape: {learned_representation.shape}, Dtype: {learned_representation.dtype}")


                # --- Temsili Hafızaya Kaydet ve/veya Hafızadan Bilgi Al (Faz 2) ---
                # Memory objesi None olsa bile çağırma, o objenin varlığını kontrol ederek yapılıyor.
                relevant_memory_entries = [] # Başlangıçta ilgili bellek girdileri yok
                if memories.get('core_memory'): # Memory modülü objesi var mı?
                     # Öğrenilen temsili hafızaya kaydet (representation None olabilir, store metodu yönetmeli)
                     memories['core_memory'].store(learned_representation)
                     # Hafızadan ilgili bilgi çağır (query_representation None olabilir, retrieve metodu yönetmeli)
                     relevant_memory_entries = memories['core_memory'].retrieve(
                         learned_representation, # Sorgu temsili
                         num_results=num_memories_to_retrieve
                     )
                     # if relevant_memory_entries: ... (Loglar Memory içinde)

                # DEBUG Log: Geri Çağrılan Bellek Girdileri
                if relevant_memory_entries:
                     logger.debug(f"RUN_EVO: Hafızadan {len(relevant_memory_entries)} ilgili girdi geri çağrıldı (placeholder).")


                # --- Hafıza ve temsile göre Bilişsel işlem yap (Faz 3 Başlangıcı) ---
                # Cognition objesi None olsa bile çağırma, o objenin varlığını kontrol ederek yapılıyor.
                # Decide metodu None girdileri (representation None, relevant_memory_entries boş liste) yönetmeli.
                decision = None # Başlangıçta bir karar yok
                if cognition_modules.get('core_cognition'): # Cognition modülü objesi var mı?
                     decision = cognition_modules['core_cognition'].decide(
                         learned_representation, # Temsil None olabilir
                         relevant_memory_entries # Liste boş olabilir
                     )
                     # if decision is not None: ... (Loglar Cognition içinde)

                # DEBUG Log: Bilişsel Karar
                if decision is not None:
                     logger.debug(f"RUN_EVO: Bilişsel karar alındı (placeholder): {decision}")


                # --- Karara göre bir Tepki Üret (Faz 3 Devamı) ---
                # Motor Control objesi None olsa bile çağırma, o objenin varlığını kontrol ederek yapılıyor.
                # Generate_response metodu None kararı yönetmeli.
                response_output = None # Başlangıçta bir tepki yok
                if motor_control_modules.get('core_motor_control'): # Motor Control modülü objesi var mı?
                     response_output = motor_control_modules['core_motor_control'].generate_response(decision) # Karar None olabilir
                     # if response_output is not None: ... (Loglar MotorControl içinde)


                # --- Tepkiyi Dışarı Aktar (Faz 3 Tamamlanması) ---
                # Interaction objesi None olsa bile çağırma, o objenin varlığını kontrol ederek yapılıyor.
                # Send_output metodu None çıktıyı yönetmeli.
                # Interaction modülü başlatılamadıysa interaction_modules['core_interaction'] zaten None olacaktır.
                if interaction_modules.get('core_interaction'): # Interaction modülü objesi var mı?
                   # Send_output metodu None çıktıyı kendi içinde yönetiyor.
                   interaction_modules['core_interaction'].send_output(response_output)


                # --- Döngü Gecikmesi ---
                # Döngünün belirli bir hızda çalışmasını sağla
                elapsed_time = time.time() - start_time
                sleep_time = loop_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Döngü intervalinden uzun süren durumlar için uyarı (DEBUG seviyesinde)
                    logger.debug(f"RUN_EVO: Bilişsel döngü {loop_interval} saniyeden daha uzun sürdü ({elapsed_time:.4f}s). İşlem yükü yüksek olabilir.")


                # Gelecekte döngüyü sonlandıracak bir mekanizma eklenecek (örn. kullanıcı sinyali, içsel durum)
                # if cognition_modules.get('core_cognition') and cognition_modules['core_cognition'].should_stop(): break # Örnek: Evo uykuya dalarsa

        except KeyboardInterrupt:
            logger.warning("Ctrl+C algılandı. Evo durduruluyor...")
        except Exception as e: # Catch any other exception during the main loop
            logger.critical(f"Evo'nın ana döngüsünde beklenmedik kritik hata: {e}", exc_info=True)

        finally:
            # --- Kaynakları Temizleme ---
            # Temizleme işlemini yeni yardımcı fonksiyona devret
            cleanup_modules(module_objects) # <<< Temizleme buraya taşındı


    # Eğer initialize_modules sırasında kritik hata olduysa buraya gelinir,
    # kaynak temizleme yukarıda yapıldı.
    logger.info("Evo durduruldu.")


# Ana çalıştırma noktası
if __name__ == '__main__':
    run_evo()