# src/run_evo.py
#
# Evo projesinin ana çalıştırma noktası.
# Evo'nun temel yaşam döngüsünü (bilişsel döngü) başlatır ve yönetir.
# Konfigürasyonu yükler, modülleri başlatır, döngüyü çalıştırır ve kaynakları temizler.

import logging
import time
# NumPy, temel veri yapıları için gerekli, ancak doğrudan run_evo'da kullanılmıyor
# Modüllerin girdilerini/çıktılarını göstermek için loglarda kullanışlı olabilir.
import numpy as np
# import sys # Programı sonlandırmak gerekirse import edilebilir

# Loglama ve Konfigürasyon yardımcı fonksiyonlarını import et
from src.core.logging_utils import setup_logging
from src.core.config_utils import load_config_from_yaml
# Modül başlatma yardımcı fonksiyonlarını import et
from src.core.module_loader import initialize_modules, cleanup_modules


# Bu dosyanın kendi logger'ını oluştur (setup_logging'den sonra kullanılacak)
# __name__ 'src.run_evo' olacak
logger = logging.getLogger(__name__)


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
    # --- Başlatma Aşaması ---

    # 1. Loglama sistemini merkezi utility ile yapılandır
    # Loglama seviyesi ve çıktı hedefleri config'ten okunacak.
    # Config'in kendisi yüklenmeden önce çağrılır, böylece config yükleme hataları da loglanabilir.
    # setup_logging fonksiyonu, config None olsa bile varsayılan ayarlarla çalışır.
    # Config yüklenemese bile, en azından CRITICAL hatalar loglanabilir.
    # Config yüklendikten sonra tekrar çağrılması gerekebilir, ancak current setup'ta ilk çağrı yeterli.
    # Veya setup_logging config alırsa, config yüklendikten sonra çağrılmalı.
    # Current: setup_logging receives config. So, load config first.

    # 1a. Konfigürasyonu Yükle (Loglama setup'ından önce yüklenmeli ki loglama config'i kullanılabilsin)
    config_path = "config/main_config.yaml" # Yapılandırma dosyasının yolu
    config = load_config_from_yaml(config_path) # config_utils kullanarak dosyadan yükle. Hata durumunda boş dict döner.

    # 1b. Loglama sistemini yüklenen config ile yapılandır
    # Eğer config yüklenemezse, setup_logging None alacak ve varsayılan seviyeyi (INFO) kullanacak.
    setup_logging(config=config) # <<< setup_logging'e yüklenen config gönderildi

    # run_evo fonksiyonunun kendi logger'ını oluştur (Loglama setup'ından sonra)
    # __name__ 'src.run_evo' olacak
    logger = logging.getLogger(__name__)

    # Config'in başarıyla yüklenip yüklenmediğini kontrol et
    if not config: # load_config_from_yaml hata vermişse boş dictionary döndürmüş demektir.
        logger.critical(f"Evo, yapılandırma yüklenemediği için başlatılamıyor. Lütfen {config_path} dosyasını kontrol edin.")
        # Bu noktada modüller başlatılmadı, kaynak temizleme gerekmez.
        # Programı burada sonlandırmak en mantıklısı.
        # sys.exit(1) # Eğer sys import ediliyorsa
        return # run_evo fonksiyonundan çık - program main bloğunda sonlanır.

    logger.info("Evo canlanıyor...")
    logger.info(f"Konfigürasyon başarıyla yüklendi: {config_path}")
    # DEBUG seviyesinde tüm config'i loglamak, hata ayıklama için faydalı olabilir.
    # logger.debug(f"Yüklenen Konfigürasyon Detayları: {config}")


    # 2. Modülleri Başlat
    # initialize_modules fonksiyonu başlatılan objeleri içeren bir dict ve ana döngünün çalışıp çalışamayacağını belirten bir bool döndürür.
    # Kritik başlatma hataları initialize_modules içinde CRITICAL olarak loglanır ve can_run_main_loop False yapılır.
    module_objects, can_run_main_loop = initialize_modules(config)

    # Modül başlatma sırasında kritik bir hata olduysa programı sonlandır
    if not can_run_main_loop:
        # initialize_modules zaten kritik hatayı logladı.
        logger.critical("Evo'nın temel modülleri başlatılamadığı için program sonlandırılıyor.")
        # Kaynakları temizle (initialize_modules tarafından başlatılanları)
        # cleanup_modules, None objeleri güvenli bir şekilde işleyebilir.
        cleanup_modules(module_objects)
        # sys.exit(1) # Eğer sys import ediliyorsa
        return # run_evo fonksiyonundan çık - program main bloğunda sonlanır.


    # Modül objelerine kolay erişim için değişkenler atayalım (opsiyonel ama kodu sadeleştirebilir)
    # Bu değişkenler artık initialize_modules'dan dönen dict'in referanslarıdır.
    # Bu dictionary'lerden .get() ile erişirken None kontrolü yapmak döngü içinde güvenlidir.
    sensors = module_objects.get('sensors', {})
    processors = module_objects.get('processors', {})
    representers = module_objects.get('representers', {})
    memories = module_objects.get('memories', {})
    cognition_modules = module_objects.get('cognition', {})
    motor_control_modules = module_objects.get('motor_control', {})
    interaction_modules = module_objects.get('interaction', {})

    # Tüm temel modül kategorileri başarıyla başlatıldıysa logla (initialize_modules bayrağına göre)
    # can_run_main_loop True ise buraya gelinir.
    # initialize_modules içinde zaten kritik hata durumları loglanmıştı.
    logger.info("Evo bilişsel döngüye hazır.")
    # Interaction modülü başlatılamadıysa zaten module_loader içinde warning loglandı.


    # --- Ana Bilişsel Döngü (Yaşam Döngüsü) ---
    # Sadece can_run_main_loop True ise döngüye girilir.
    if can_run_main_loop: # Bu kontrol zaten yukarıda yapılıyor, ama döngüye giriş şartı olarak tekrar kontrol etmek netlik sağlar.
        logger.info("Evo'nın bilişsel döngüsü başlatıldı...")
        # Bilişsel döngü hızını konfigürasyondan al. Yoksa varsayılan 0.1 saniye kullan.
        loop_interval = config.get('cognitive_loop_interval', 0.1)
        # Bellekten kaç anı çağrılacağını konfigürasyondan al. Memory config yoksa veya değer yoksa varsayılan 5 kullan.
        num_memories_to_retrieve = config.get('memory', {}).get('num_retrieved_memories', 5)


        try:
            # Döngü, kullanıcı Ctrl+C ile durdurana veya işlenmemiş bir istisna oluşana kadar devam eder.
            while True:
                start_time = time.time() # Döngü adımının başlangıç zamanı

                # --- Bilgi Akışı (Sense -> Process -> Represent -> Memory -> Cognition -> Motor -> Interact) ---
                # Her adımda, bir önceki adımdan gelen çıktı (veya None) bir sonraki adıma girdi olur.
                # Modül objelerinin None olma durumları ve metotların None girdi alma durumları
                # ilgili modül içindeki hata yönetimi ile ele alınır.

                # Duyu Verisini Yakala (Faz 0)
                raw_inputs = {}
                # Sensör objelerinin varlığını kontrol ederek güvenli çağrı
                if sensors.get('vision'):
                    raw_inputs['visual'] = sensors['vision'].capture_frame() # Hata durumunda None dönebilir
                if sensors.get('audio'):
                    raw_inputs['audio'] = sensors['audio'].capture_chunk() # Hata durumunda None dönebilir

                # DEBUG Log: Yakalanan Ham Görsel/Ses Verisi
                if raw_inputs.get('visual') is not None:
                     logger.debug(f"RUN_EVO: Raw Visual Input yakalandı. Shape: {raw_inputs['visual'].shape}, Dtype: {raw_inputs['visual'].dtype}")
                else:
                     logger.debug("RUN_EVO: Raw Visual Input None.")

                if raw_inputs.get('audio') is not None:
                     logger.debug(f"RUN_EVO: Raw Audio Input yakalandı. Shape: {raw_inputs['audio'].shape}, Dtype: {raw_inputs['audio'].dtype}")
                else:
                     logger.debug("RUN_EVO: Raw Audio Input None.")


                # Yakalanan Veriyi İşle (Faz 1 Başlangıcı)
                processed_inputs = {}
                # İşlemci objelerinin varlığını kontrol ederek güvenli çağrı
                # Process metotları None girdi alabilmeli ve None döndürebilmeli (hata durumunda)
                if processors.get('vision'):
                     processed_inputs['visual'] = processors['vision'].process(raw_inputs.get('visual'))
                if processors.get('audio'):
                     processed_inputs['audio'] = processors['audio'].process(raw_inputs.get('audio'))

                # DEBUG Log: İşlenmiş Görsel/Ses Verisi
                if processed_inputs.get('visual') is not None:
                     logger.debug(f"RUN_EVO: Processed Visual Output. Shape: {processed_inputs['visual'].shape}, Dtype: {processed_inputs['visual'].dtype}")
                else:
                     logger.debug("RUN_EVO: Processed Visual Output None.")

                if processed_inputs.get('audio') is not None:
                     audio_energy = processed_inputs['audio']
                     if audio_energy is not None: # Processed Audio None dönebilir (hata yönetimi)
                         logger.debug(f"RUN_EVO: Processed Audio Output (Energy). Value: {audio_energy:.4f}")
                     else:
                          logger.debug("RUN_EVO: Processed Audio Output (Energy) is None.")
                else:
                     logger.debug("RUN_EVO: Processed Audio Output None.")


                # İşlenmiş Veriden Temsil Öğren (Faz 1 Devamı)
                # Representation learner objesinin varlığını kontrol ederek güvenli çağrı
                # Learn metodu boş processed_inputs dict alabilmeli ve None döndürebilmeli (hata durumunda)
                learned_representation = None
                if representers.get('main_learner'):
                     learned_representation = representers['main_learner'].learn(processed_inputs) # processed_inputs boş olabilir veya None değerler içerebilir

                # DEBUG Log: Öğrenilmiş Temsil
                if learned_representation is not None:
                     logger.debug(f"RUN_EVO: Learned Representation. Shape: {learned_representation.shape}, Dtype: {learned_representation.dtype}")


                # Temsili Hafızaya Kaydet ve/veya Hafızadan Bilgi Al (Faz 2)
                # Memory objesinin varlığını kontrol ederek güvenli çağrı
                # Store/Retrieve metotları None representation alabilmeli ve retrieve None/boş liste döndürebilmeli (hata durumunda)
                relevant_memory_entries = [] # Başlangıçta ilgili bellek girdileri yok
                if memories.get('core_memory'):
                     # Öğrenilen temsili hafızaya kaydet (learned_representation None olabilir)
                     memories['core_memory'].store(learned_representation)
                     # Hafızadan ilgili bilgi çağır (learned_representation None olabilir)
                     # retrieve metodu hata durumunda boş liste döndürmeli
                     relevant_memory_entries = memories['core_memory'].retrieve(
                         learned_representation, # Sorgu temsili
                         num_results=num_memories_to_retrieve
                     )

                # DEBUG Log: Geri Çağrılan Bellek Girdileri
                # relevant_memory_entries boş liste olabilir, bu normaldir.
                if relevant_memory_entries:
                     # logger.debug(f"RUN_EVO: Hafızadan {len(relevant_memory_entries)} ilgili girdi geri çağrıldı (placeholder).")
                     # Çok sık loglanabilir, sadece sayı logu yeterli.
                     logger.debug(f"RUN_EVO: Hafızadan {len(relevant_memory_entries)} girdi çağrıldı.")


                # Hafıza ve temsile göre Bilişsel işlem yap (Faz 3 Başlangıcı)
                # Cognition objesinin varlığını kontrol ederek güvenli çağrı
                # Decide metodu None representation ve boş/None memory listesi alabilmeli, None karar döndürebilmeli (hata durumunda)
                decision = None # Başlangıçta bir karar yok
                if cognition_modules.get('core_cognition'):
                     decision = cognition_modules['core_cognition'].decide(
                         learned_representation, # Temsil None olabilir
                         relevant_memory_entries # Liste boş olabilir
                     )

                # DEBUG Log: Bilişsel Karar
                if decision is not None:
                     logger.debug(f"RUN_EVO: Bilişsel karar alındı (placeholder): {decision}")


                # Karara göre bir Tepki Üret (Faz 3 Devamı)
                # Motor Control objesinin varlığını kontrol ederek güvenli çağrı
                # Generate_response metodu None karar alabilmeli, None çıktı döndürebilmeli (hata durumunda)
                response_output = None # Başlangıçta bir tepki yok
                if motor_control_modules.get('core_motor_control'):
                     response_output = motor_control_modules['core_motor_control'].generate_response(decision) # Karar None olabilir
                     # if response_output is not None: ... (Loglar MotorControl içinde)


                # Tepkiyi Dışarı Aktar (Faz 3 Tamamlanması)
                # Interaction objesinin varlığını kontrol ederek güvenli çağrı
                # Send_output metodu None çıktıyı alabilmeli.
                if interaction_modules.get('core_interaction'):
                   interaction_modules['core_interaction'].send_output(response_output) # response_output None olabilir


                # --- Döngü Gecikmesi ---
                # Döngünün belirli bir hızda çalışmasını sağla
                # İşlem süresini ölç
                elapsed_time = time.time() - start_time
                # Gerekirse bekleme süresini hesapla
                sleep_time = loop_interval - elapsed_time
                # Bekle (sadece süre pozitifse)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Döngü intervalinden uzun süren durumlar için uyarı (DEBUG seviyesinde)
                    # Bu, performans darboğazlarını tespit etmede yardımcı olur.
                    logger.debug(f"RUN_EVO: Bilişsel döngü {loop_interval} saniyeden daha uzun sürdü ({elapsed_time:.4f}s). İşlem yükü yüksek olabilir.")


                # Gelecekte döngüyü sonlandıracak bir mekanizma eklenecek (örn. kullanıcı sinyali, içsel durum)
                # if cognition_modules.get('core_cognition') and cognition_modules['core_cognition'].should_stop(): break # Örnek: Evo uykuya dalarsa

        except KeyboardInterrupt:
            # Kullanıcı Ctrl+C ile programı durdurduğunda bu istisna yakalanır.
            logger.warning("Ctrl+C algılandı. Evo durduruluyor...")
        except Exception as e: # Ana döngü sırasında işlenmemiş diğer tüm istisnaları yakala
            # Bu, modül içi try-except bloklarını atlatan veya döngü mantığında oluşan beklenmedik hatalar için.
            logger.critical(f"Evo'nın ana döngüsünde beklenmedik kritik hata: {e}", exc_info=True)

        finally:
            # --- Kaynakları Temizleme ---
            # Program normal veya hata ile sonlandığında kaynakları temizle.
            # cleanup_modules fonksiyonu, initialize_modules tarafından döndürülen dict'i alır
            # ve modüllerin cleanup/stop_stream metotlarını çağırır (güvenli bir şekilde).
            cleanup_modules(module_objects)

        # initialize_modules sırasında kritik hata olduysa buraya gelinir ve zaten program sonlandırılır.
        # Döngü normal bittiğinde veya hata yakalandığında finally bloğu çalışır ve buraya gelinir.
        logger.info("Evo durduruldu.")


# Ana çalıştırma noktası
if __name__ == '__main__':
    run_evo()