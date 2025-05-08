# src/run_evo.py
#
# Evo projesinin ana çalıştırma noktası.
# Evo'nın temel yaşam döngüsünü (bilişsel döngü) başlatır ve yönetir.
# Konfigürasyonu yükler, modülleri başlatır, döngüyü çalıştırır ve kaynakları temizler.

import logging
import time
# NumPy, temel veri yapıları için gerekli, ancak doğrudan run_evo'da kullanılmıyor
# Modüllerin girdilerini/çıktılarını göstermek için loglarda kullanışlı olabilir.
import numpy as np
# import sys # Programı sonlandırmak gerekirse import edilebilir

# Loglama ve Konfigürasyon yardımcı fonksiyonlarını import et
from src.core.logging_utils import setup_logging
from src.core.config_utils import load_config_from_yaml, get_config_value
# Modül başlatma yardımcı fonksiyonlarını import et
from src.core.module_loader import initialize_modules, cleanup_modules


# Bu dosyanın kendi logger'ını oluştur (setup_logging'den sonra kullanılacak)
# __name__ 'src.run_evo' olacak
# Başlangıçta logger'ı None yapalım, setup_logging sonrası tekrar alacağız.
logger = None # type: logging.Logger


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
    global logger # Global logger değişkenini kullanacağımızı belirt

    # --- Başlatma Aşaması ---

    # 1a. Konfigürasyonu Yükle (Loglama setup'ından önce yüklenmeli ki loglama config'i kullanılabilsin)
    config_path = "config/main_config.yaml" # Yapılandırma dosyasının yolu
    # config_utils kullanarak dosyadan yükle. Hata durumunda boş dict döner.
    # Bu ilk çağrı, config yüklenemese bile default logging seviyesi ile temel loglamayı kurar.
    # Config yüklendiğinde, eğer farklı log ayarları varsa setup_logging tekrar çağrılır.
    # load_config_from_yaml içinde hata oluşursa ve config boş dönerse, bu durum setup_logging'e None config gitmesine sebep olur.
    # setup_logging(config=None) # Config yüklenmeden temel loglama için (isteğe bağlı). Şu anki setup_logging config alabiliyor.
    config = load_config_from_yaml(config_path)

    # 1b. Loglama sistemini yüklenen config ile yapılandır
    # Eğer config yüklenemezse (load_config_from_yaml boş dict döndürürse), setup_logging None alacak ve varsayılan seviyeyi (INFO) kullanacak.
    # Config başarıyla yüklendiyse, setup_logging config'teki loglama ayarlarını kullanır.
    setup_logging(config=config) # <<< setup_logging'e yüklenen config gönderildi

    # run_evo fonksiyonunun kendi logger'ını oluştur (Loglama setup'ından sonra)
    # __name__ 'src.run_evo' olacak
    logger = logging.getLogger(__name__) # Logger objesini tekrar al


    # Config'in başarıyla yüklenip yüklenmediğini kontrol et
    if not config: # load_config_from_yaml hata vermişse boş dictionary döndürmüş demektir.
        # Hata zaten load_config_from_yaml içinde CRITICAL olarak loglandı.
        logger.critical(f"Evo, yapılandırma yüklenemediği için başlatılamıyor. Lütfen {config_path} dosyasını kontrol edin.")
        # Bu noktada modüller başlatılmadı, kaynak temizleme gerekmez.
        # Programı burada sonlandırmak en mantıklısı.
        # sys.exit(1) # Eğer sys import ediliyorsa
        return # run_evo fonksiyonundan çık - program main bloğında sonlanır.

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
        return # run_evo fonksiyonundan çık - program main bloğında sonlanır.


    # Modül objelerine kolay erişim için değişkenler atayalım (opsiyonel ama kodu sadeleştirebilir)
    # Bu değişkenler artık initialize_modules'dan dönen dict'in referanslarıdır.
    # Bu dictionary'lerden .get() ile erişirken None kontrolü yapmak döngü içinde güvenlidir.
    # .get(key, {}) kullanarak, eğer kategori dict yoksa (init hatası vb), boş dict döner
    # ve alt modül erişimleri (.get('module_name')) hata vermez.
    sensors = module_objects.get('sensors', {})
    processors = module_objects.get('processors', {})
    representers = module_objects.get('representers', {})
    memories = module_objects.get('memories', {})
    cognition_modules = module_objects.get('cognition', {})
    motor_control_modules = module_objects.get('motor_control', {})
    interaction_modules = module_objects.get('interaction', {})

    # Tüm temel modül kategorileri başarıyla başlatıldıysa logla (initialize_modules bayrağına göre)
    # can_run_main_loop True ise buraya gelinir.
    # initialize_modules içinde zaten kritik hata durumları loglandmıştı.
    logger.info("Evo bilişsel döngüye hazır.")
    # Interaction modülü başlatılamadıysa initialize_modules içinde warning loglandı.


    # --- Ana Bilişsel Döngü (Yaşam Döngüsü) ---
    # Sadece can_run_main_loop True ise döngüye girilir.
    if can_run_main_loop: # Bu kontrol zaten yukarıda yapılıyor, ama döngüye giriş şartı olarak tekrar kontrol etmek netlik sağlar.
        logger.info("Evo'nın bilişsel döngüsü başlatıldı...")
        # Bilişsel döngü hızını konfigürasyondan al. Yoksa varsayılan 0.1 saniye kullan.
        # Düzeltme: get_config_value çağrısını default=keyword formatına çevir.
        # Config'e göre bu ayar global olarak config'in kökünde.
        loop_interval = get_config_value(config, 'cognitive_loop_interval', default=0.1, expected_type=(float, int), logger_instance=logger)

        if not isinstance(loop_interval, (int, float)) or loop_interval <= 0:
             logger.warning(f"RUN_EVO: Konfigürasyondan alınan geçersiz cognitive_loop_interval değeri ({loop_interval}). Varsayılan 0.1 kullanılıyor.")
             loop_interval = 0.1
        else:
             loop_interval = float(loop_interval) # float'a çevir


        # Bellekten kaç anı çağrılacağını konfigürasyon
        # dan al. Memory config yoksa veya değer yoksa varsayılan 5 kullan.
        # Config dict'inin 'memory' anahtarı altındaki 'num_retrieved_memories' değerini al.
        # Düzeltme: get_config_value çağrısını default=keyword formatına çevir.
        num_memories_to_retrieve = get_config_value(config, 'memory', 'num_retrieved_memories', default=5, expected_type=int, logger_instance=logger)

        # Eğer num_memories_to_retrieve hala int değilse veya negatifse varsayılan ata (utils içinde kontrol edildi ama burada da sağlamlık için).
        # get_config_value expected_type kontrolü yaptığı için bu kontrol artık gerekmiyordu.
        # Negatif kontrolü get_config_value yapmıyor, bu kontrol kalabilir.
        if num_memories_to_retrieve < 0:
             logger.warning(f"RUN_EVO: Konfigürasyondan alınan negatif num_retrieved_memories değeri ({num_memories_to_retrieve}). Varsayılan 5 kullanılıyor.")
             num_memories_to_retrieve = 5


        try:
            # Döngü, kullanıcı Ctrl+C ile durdurana veya işlenmemiş bir istisna oluşana kadar devam eder.
            while True:
                start_time = time.time() # Döngü adımının başlangıç zamanı

                # --- Bil bilgi Akışı (Sense -> Process -> Represent -> Memory -> Cognition -> Motor -> Interact) ---
                # Her adımda, bir önceki adımdan gelen çıktı (veya None veya boş dict/liste) bir sonraki adıma girdi olur.
                # Modül objelerinin None olma durumları ve metotların None/boş girdi alma durumları
                # ilgili modül içindeki hata yönetimi ile ele alınır.

                # Duyu Verisini Yakala (Faz 0)
                raw_inputs = {}
                # Sensör objelerinin varlığını kontrol ederek güvenli çağrı
                # sensors dict'inin kendisi None değil (get() ile varsayılan {} aldık)
                if sensors.get('vision'):
                    # capture_frame None veya numpy array döndürür.
                    raw_inputs['visual'] = sensors['vision'].capture_frame()
                if sensors.get('audio'):
                    # capture_chunk None veya numpy array döndürür.
                    raw_inputs['audio'] = sensors['audio'].capture_chunk()

                # DEBUG Log: Yakalanan Ham Görsel/Ses Verisi
                if raw_inputs.get('visual') is not None:
                     visual_raw = raw_inputs['visual']
                     # Ham görselin numpy array olduğu varsayılır (VisionSensor'a göre)
                     logger.debug(f"RUN_EVO: Raw Visual Input yakalandı. Shape: {visual_raw.shape}, Dtype: {visual_raw.dtype}")
                else:
                     logger.debug("RUN_EVO: Raw Visual Input None.")

                if raw_inputs.get('audio') is not None:
                     audio_raw = raw_inputs['audio']
                     # Ham sesin numpy array olduğu varsayılır (AudioSensor'a göre)
                     logger.debug(f"RUN_EVO: Raw Audio Input yakalandı. Shape: {audio_raw.shape}, Dtype: {audio_raw.dtype}")
                else:
                     logger.debug("RUN_EVO: Raw Audio Input None.")


                # Yakalanan Veriyi İşle (Faz 1 Başlangıcı)
                processed_inputs = {}
                # İşlemci objelerinin varlığını kontrol ederek güvenli çağrı
                # processors dict'inin kendisi None değil (get() ile varsayılan {} aldık)
                # Process metotları None girdi alabilmeli ve None/boş dict/array döndürebilmeli (hata durumunda)
                if processors.get('vision'):
                     # VisionProcessor.process artık dict veya boş dict {} döndürür.
                     processed_inputs['visual'] = processors['vision'].process(raw_inputs.get('visual'))
                if processors.get('audio'):
                     # AudioProcessor.process artık numpy array (shape (2,)) veya None döndürür.
                     processed_inputs['audio'] = processors['audio'].process(raw_inputs.get('audio'))


                # DEBUG Log: İşlenmiş Görsel/Ses Verisi - Çıktı formatlarına uygun loglama
                visual_processed_output = processed_inputs.get('visual')
                # VisionProcessor.process bir dict veya boş dict {} döndürür.
                if isinstance(visual_processed_output, dict): # İşlenmiş görsel çıktı bir dict ise
                     logger.debug(f"RUN_EVO: Processed Visual Output is a dictionary. Keys: {list(visual_processed_output.keys())}.")
                     # Dict içindeki bir örneğin (örn: 'grayscale' veya 'edges') shape/dtype bilgisini logla.
                     if visual_processed_output: # Eğer dict boş değilse
                         # Dict'in ilk öğesini almak veya belirli bir anahtara ('grayscale') erişmek daha güvenli.
                         first_key = list(visual_processed_output.keys())[0] if visual_processed_output else None
                         if first_key and isinstance(visual_processed_output.get(first_key), np.ndarray): # .get() ile güvenli erişim
                              # Eğer ilk değer (veya 'grayscale') numpy array ise shape/dtype logla
                              example_data = visual_processed_output[first_key]
                              logger.debug(f"RUN_EVO: Processed Visual Output Example Feature ('{first_key}'). Shape: {example_data.shape}, Dtype: {example_data.dtype}")
                         else:
                             logger.debug("RUN_EVO: Processed Visual Output dictionary does not contain a loggable numpy array.")
                     else: # Dictionary boş
                         logger.debug("RUN_EVO: Processed Visual Output is an empty dictionary.")
                elif visual_processed_output is not None: # Dict değil ama None da değilse (Beklenmeyen durum olabilir, logla)
                      logger.warning(f"RUN_EVO: Processed Visual Output beklenmeyen tipte: {type(visual_processed_output)}. Dict bekleniyordu.")
                else: # None ise (VisionProcessor'dan None dönme durumu - current implementasyona göre boş dict dönecek ama sağlamlık için)
                      logger.debug("RUN_EVO: Processed Visual Output None.")


                audio_processed_output = processed_inputs.get('audio')
                # AudioProcessor.process bir numpy array (shape (2,)) veya None döndürür.
                if isinstance(audio_processed_output, np.ndarray): # İşlenmiş ses çıktısı bir numpy array ise
                     logger.debug(f"RUN_EVO: Processed Audio Output is numpy array. Shape: {audio_processed_output.shape}, Dtype: {audio_processed_output.dtype}. Values: {audio_processed_output}")
                     # Örnek olarak enerji ve centroid değerlerini logla
                     if audio_processed_output.shape[0] >= 2:
                         logger.debug(f"RUN_EVO: Processed Audio Output (Energy, Centroid): [{audio_processed_output[0]:.4f}, {audio_processed_output[1]:.4f}]")
                     elif audio_processed_output.shape[0] == 1:
                         logger.debug(f"RUN_EVO: Processed Audio Output (Single Value): [{audio_processed_output[0]:.4f}]")
                elif audio_processed_output is not None: # Array değil ama None da değilse (Beklenmeyen durum olabilir, logla)
                      logger.warning(f"RUN_EVO: Processed Audio Output beklenmeyen tipte: {type(audio_processed_output)}. numpy.ndarray bekleniyordu.")
                else: # None ise
                     logger.debug("RUN_EVO: Processed Audio Output None.")


                # İşlenmiş Veriden Temsil Öğren (Faz 1 Devamı)
                # Representation learner objesinin varlığını kontrol ederek güvenli çağrı
                # representers dict'inin kendisi None değil (get() ile varsayılan {} aldık)
                # Learn metodu boş processed_inputs dict alabilmeli ve None/numpy array döndürebilmeli (hata durumında)
                learned_representation = None # Başlangıçta temsil yok
                if representers.get('main_learner'):
                     # RepresentationLearner.learn metodu processed_inputs sözlüğünü bekler.
                     # Bu sözlük VisionProcessor'dan dict, AudioProcessor'dan array içeriyor.
                     learned_representation = representers['main_learner'].learn(processed_inputs)

                # DEBUG Log: Öğrenilmiş Temsil
                if isinstance(learned_representation, np.ndarray): # RepresentationLearner bir numpy array veya None döndürür
                     logger.debug(f"RUN_EVO: Learned Representation. Shape: {learned_representation.shape}, Dtype: {learned_representation.dtype}")
                elif learned_representation is not None: # numpy array değil ama None da değilse
                     logger.warning(f"RUN_EVO: Learned Representation beklenmeyen tipte: {type(learned_representation)}. numpy.ndarray veya None bekleniyordu.")
                else:
                     logger.debug("RUN_EVO: Learned Representation None.")


                # Temsili Hafızaya Kaydet ve/ PENSAR /ou Hafızadan Bilgi Al (Faz 2)
                # Memory objesinin varlığını kontrol ederek güvenli çağrı
                # memories dict'inin kendisi None değil (get() ile varsayılan {} aldık)
                # Store/Retrieve metotları Representation (None/array) alabilmeli ve retrieve None/boş liste döndürebilmeli (hata durumunda)
                relevant_memory_entries = [] # Başlangıçta ilgili bellek girdileri yok, boş liste

                # Bellek objesi varsa işlemleri yap
                core_memory_instance = memories.get('core_memory')
                if core_memory_instance:
                     # Öğrenilen temsili hafızaya kaydet (learned_representation None veya array olabilir)
                     # Memory.store None/array girdiyi güvenle ele alır.
                     core_memory_instance.store(learned_representation)
                     # logger.debug(f"RUN_EVO: Temsil belleğe saklanmaya çalışıldı (başarılı/başarısız: {learned_representation is not None}).")

                     # Hafızadan ilgili bilgi çağır (learned_representation None veya array olabilir)
                     # Memory.retrieve None/array sorguyu ve geçersiz num_results'ı güvenle ele alır, boş liste döndürür.
                     # num_memories_to_retrieve config'ten alınır ve int olduğu kontrol edilir.
                     relevant_memory_entries = core_memory_instance.retrieve(
                         learned_representation, # Sorgu temsili (None veya array olabilir)
                         num_results=num_memories_to_retrieve # Config'ten int olarak alındı
                     )
                     # logger.debug(f"RUN_EVO: Hafızadan geri çağırma tamamlandı.")


                # DEBUG Log: Geri Çağrılan Bellek Girdileri
                # relevant_memory_entries boş liste veya None olabilir, bu normaldir.
                if isinstance(relevant_memory_entries, list):
                     if relevant_memory_entries:
                          logger.debug(f"RUN_EVO: Hafızadan {len(relevant_memory_entries)} ilgili girdi geri çağrıldı.")
                     else:
                          logger.debug("RUN_EVO: Hafızadan ilgili girdi geri çağrılamadı (boş liste).")
                elif relevant_memory_entries is not None: # Liste değil ama None da değilse
                      logger.warning(f"RUN_EVO: Geri çağrılan bellek girdileri beklenmeyen tipte: {type(relevant_memory_entries)}. Liste bekleniyordu.")
                else: # None ise
                     logger.debug("RUN_EVO: Geri çağrılan bellek girdileri None.")

                # Öğrenilmiş kavram temsilcileri listesini al (Cognition decide için)
                # LearningModule objesi varsa ondan iste. Yoksa boş liste kullan.
                current_concepts = [] # Başlangıçta boş
                learning_module_instance = cognition_modules.get('core_cognition', {}).get('learning_module') # CognitionCore içinde tutuluyor olabilir mi? Hayır, CognitionCore init ediyor alt modülleri.

                # Düzeltme: current_concepts bilgisini LearningModule'den alıp decide'a iletelim.
                learning_module_in_cognition = cognition_modules.get('core_cognition', {}).learning_module # CognitionCore objesinin alt modülü
                # Eğer CognitionCore objesi ve onun LearningModule'ü varsa kavramları al.
                if cognition_modules.get('core_cognition') and hasattr(cognition_modules['core_cognition'], 'learning_module') and cognition_modules['core_cognition'].learning_module:
                    try:
                        concepts = cognition_modules['core_cognition'].learning_module.get_concepts()
                        if isinstance(concepts, list):
                             current_concepts = concepts
                        else:
                             logger.warning("RUN_EVO: CognitionCore'dan alınan kavramlar liste değil. Boş liste kullanılıyor.")
                             current_concepts = []
                    except Exception as e:
                         logger.error(f"RUN_EVO: LearningModule.get_concepts çağrılırken hata: {e}", exc_info=True)
                         current_concepts = []


                # Hafıza ve temsile göre Bilişsel işlem yap (Faz 3)
                # Cognition objesinin varlığını kontrol ederek güvenli çağrı
                # cognition_modules dict'inin kendisi None değil (get() ile varsayılan {} aldık)
                # Decide metodu artık processed_inputs, learned_representation, relevant_memory_entries, current_concepts bekliyor.
                # DecisionModule.decide string veya None karar döndürebilmeli (hata durumunda)
                decision = None # Başlangıçta bir karar yok

                # CognitionCore objesi varsa decide metodunu çağır
                core_cognition_instance = cognition_modules.get('core_cognition')
                if core_cognition_instance:
                     # *** HATA DÜZELTME BURADA YAPILDI ***
                     # CognitionCore.decide artık processed_inputs, learned_representation, relevant_memory_entries ve current_concepts argümanlarını bekliyor.
                     decision = core_cognition_instance.decide(
                         processed_inputs, # İşlenmiş Processor çıktıları (dict/None)
                         learned_representation, # Temsil (None/array)
                         relevant_memory_entries, # Bellek girdileri (list/None)
                         current_concepts # Öğrenilmiş kavramlar (list)
                         # internal_state # Gelecekte eklenecek.
                     )
                     # logger.debug(f"RUN_EVO: Bilişsel karar alma tamamlandı.")


                # DEBUG Log: Bilişsel Karar (Loglama artık DecisionModule içinde yapılıyor)
                # DecisionModule.decide string veya None döndürür.
                # if decision is not None: ... # Loglama DecisionModule'e taşındı


                # Karara göre bir Tepki Ü üret (Faz 3 Devamı)
                # Motor Control objesinin varlığını kontrol ederek güvenli çağrı
                # motor_control_modules dict'inin kendisi None değil (get() ile varsayılan {} aldık)
                # Generate_response metodu None karar alabilmeli, None çıktı döndürebilmeli (hata durumunda)
                response_output = None # Başlangıçta bir tepki yok

                # MotorControlCore objesi varsa generate_response metodunu çağır
                core_motor_control_instance = motor_control_modules.get('core_motor_control')
                if core_motor_control_instance:
                     # MotorControlCore.generate_response None kararı güvenle ele alır.
                     response_output = core_motor_control_instance.generate_response(decision) # Karar None olabilir
                     # logger.debug(f"RUN_EVO: Tepki üretme tamamlandı.")
                     # Eğer response_output üretildiyse (None değilse), logu MotorControlCore.generate_response içinde yapılır.


                # Tepkiyi Dışarı Aktar (Faz 3 Tamamlanması)
                # Interaction objesinin varlığını kontrol ederek güvenli çağrı
                # interaction_modules dict'inin kendisi None değil (get() ile varsayılan {} aldık)
                # Send_output metodu None çıktıyı alabilmeli.
                core_interaction_instance = interaction_modules.get('core_interaction')
                if core_interaction_instance:
                   # InteractionAPI.send_output None çıktıyı güvenle ele alır.
                   core_interaction_instance.send_output(response_output) # response_output None olabilir
                   # logger.debug(f"RUN_EVO: Çıktı gönderme tamamlandı.")


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
                    # INFO seviyesinde çok sık olabilir, DEBUG seviyesine düşürelim.
                    # logger.debug(f"RUN_EVO: Bilişsel döngü {loop_interval} saniyeden daha uzun sürdü ({elapsed_time:.4f}s). İşlem yükü yüksek olabilir.")
                    if elapsed_time > loop_interval + 0.1: # Sadece belirgin gecikmeleri logla
                         logger.debug(f"RUN_EVO: Bilişsel döngü {loop_interval:.2f}s hedefinden daha uzun sürdü ({elapsed_time:.4f}s).")


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
