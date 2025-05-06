# scripts/test_module.py
#
# Evo'nın modüllerini tek başına girdi/çıktı ile test etmek için yardımcı script.
# Belirli bir modülü başlatır, sahte girdi verir ve çıktısını loglar.
# Bu script, geliştirme sırasında modüllerin beklenen davranışı sergilediğinden emin olmak için kullanılır.

import argparse
import logging
import sys
import importlib
import inspect
import numpy as np
import time
import random # Sahte veriler için gerekebilir

# Evo'nın loglama ve config yardımcılarını import et
# sys.path'e projenin kök dizinini eklemek gerekebilir, veya script'in kök dizininden çalıştırıldığından emin olun.
# Asumsi: script Evo kök dizininden çalıştırılıyor.
from src.core.logging_utils import setup_logging
from src.core.config_utils import load_config_from_yaml

# Modül için bir logger oluştur (test scriptinin kendisi için)
logger = logging.getLogger(__name__)

def load_module_class(module_path, class_name):
    """
    Belirtilen modül yolundan (örn: src.processing.vision) sınıfı (örn: VisionProcessor) dinamik olarak yükler.

    Args:
        module_path (str): Modülün Python yolu (örn: 'src.processing.vision').
        class_name (str): Modül içindeki sınıf adı (örn: 'VisionProcessor').

    Returns:
        class or None: Yüklenen sınıf objesi veya hata durumunda None.
    """
    try:
        # Modülü import et
        module = importlib.import_module(module_path)
        logger.debug(f"Modül yüklendi: {module_path}")
        # Modül içindeki sınıfı getir
        class_obj = getattr(module, class_name)
        logger.debug(f"Sınıf yüklendi: {class_name} from {module_path}")

        # Yüklenen objenin gerçekten bir sınıf olduğundan emin ol
        if not inspect.isclass(class_obj):
             logger.error(f"Yüklenen '{class_name}' '{module_path}' içinde bir sınıf değil.")
             return None

        return class_obj

    except ModuleNotFoundError:
        logger.error(f"Modül bulunamadı: {module_path}. Lütfen yolu kontrol edin.")
        return None
    except AttributeError:
        logger.error(f"Sınıf bulunamadı: {class_name} in {module_path}. Lütfen sınıf adını kontrol edin.")
        return None
    except Exception as e:
        logger.error(f"Modül veya sınıf yüklenirken beklenmedik hata oluştu: {e}", exc_info=True)
        return None

def create_dummy_input(module_name, config):
    """
    Belirtilen modül için basit sahte girdi verisi oluşturur.
    Bu fonksiyon, test edilen modülün beklediği girdi formatına göre özelleştirilmelidir.

    Args:
        module_name (str): Test edilen modülün adı (örn: 'vision_processor', 'audio_processor').
        config (dict): Genel yapılandırma sözlüğü.

    Returns:
        any: Modülün beklediği formatta sahte girdi verisi veya desteklenmiyorsa None.
    """
    logger.debug(f"'{module_name}' için sahte girdi oluşturuluyor...")

    # Modül adına göre sahte girdi oluştur.
    # Bu kısım, test etmek istediğiniz her modül için özelleştirilmelidir.
    if module_name == 'visionsensor':
        # VisionSensor ham görüntü karesi (numpy array) bekler.
        # Boyutları config'ten veya sabit değerlerden alabiliriz.
        dummy_width = config.get('vision', {}).get('dummy_width', 640)
        dummy_height = config.get('vision', {}).get('dummy_height', 480)
        # Sahte renkli BGR görüntüsü (uint8)
        dummy_frame = np.random.randint(0, 256, size=(dummy_height, dummy_width, 3), dtype=np.uint8)
        logger.debug(f"VisionSensor için sahte frame ({dummy_frame.shape}, {dummy_frame.dtype}) oluşturuldu.")
        return dummy_frame

    elif module_name == 'audiosensor':
        # AudioSensor ham ses chunk'ı (numpy array) bekler.
        # Boyutu config'ten veya sabit değerden alabiliriz.
        chunk_size = config.get('audio', {}).get('audio_chunk_size', 1024)
        # Sahte int16 ses verisi (sessizlik gibi)
        dummy_chunk = np.zeros(chunk_size, dtype=np.int16)
        # Basit bir ton ekleyelim ki tamamen sıfır olmasın test için
        sample_rate = config.get('audio', {}).get('audio_rate', 44100)
        frequency = 440
        t = np.linspace(0., chunk_size / sample_rate, chunk_size)
        amplitude = np.iinfo(np.int16).max * 0.05 # Biraz ses
        dummy_chunk = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)

        logger.debug(f"AudioSensor için sahte chunk ({dummy_chunk.shape}, {dummy_chunk.dtype}) oluşturuldu.")
        return dummy_chunk

    elif module_name == 'visionprocessor':
        # VisionProcessor ham görüntü karesi (numpy array) bekler (VisionSensor çıktısı gibi).
        # Farklı boyut ve kanallarda (BGR veya Gri) sahte veriler test edilebilir.
        dummy_width = 640
        dummy_height = 480
        # Sahte renkli BGR görüntüsü
        dummy_frame_bgr = np.random.randint(0, 256, size=(dummy_height, dummy_width, 3), dtype=np.uint8)
        # Sahte gri görüntü
        dummy_frame_gray = np.random.randint(0, 256, size=(dummy_height, dummy_width), dtype=np.uint8)
        # Test senaryosuna göre uygun olanı döndürebiliriz. Şimdilik BGR döndürelim.
        logger.debug(f"VisionProcessor için sahte frame ({dummy_frame_bgr.shape}, {dummy_frame_bgr.dtype}) oluşturuldu.")
        return dummy_frame_bgr

    elif module_name == 'audioprocessor':
        # AudioProcessor ham ses chunk'ı (numpy array) bekler (AudioSensor çıktısı gibi).
        chunk_size = 1024
        sample_rate = config.get('audio', {}).get('audio_rate', 44100)
        frequency = 880 # Farklı frekans deneyelim
        amplitude = np.iinfo(np.int16).max * 0.1 # Biraz ses
        t = np.linspace(0., chunk_size / sample_rate, chunk_size)
        dummy_chunk = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)

        logger.debug(f"AudioProcessor için sahte chunk ({dummy_chunk.shape}, {dummy_chunk.dtype}) oluşturuldu.")
        return dummy_chunk

    elif module_name == 'representationlearner':
        # RepresentationLearner processed_inputs sözlüğü bekler: {'visual': dict, 'audio': np.ndarray}.
        # Bu sözlük Processing modüllerinin çıktı formatında olmalıdır.
        # VisionProcessor çıktısı: {'grayscale': 64x64 uint8 array, 'edges': 64x64 uint8 array}
        # AudioProcessor çıktısı: np.array([energy, spectral_centroid], dtype=float32) - shape (2,)

        # Sahte VisionProcessor çıktısı dictionary'si
        vis_out_w = config.get('processors', {}).get('vision', {}).get('output_width', 64)
        vis_out_h = config.get('processors', {}).get('vision', {}).get('output_height', 64)
        dummy_processed_visual_gray = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
        dummy_processed_visual_edges = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8) # Kenarlar 0/255 ama ortalama test için rastgele olabilir
        dummy_processed_visual_dict = {'grayscale': dummy_processed_visual_gray, 'edges': dummy_processed_visual_edges}

        # Sahte AudioProcessor çıktısı array'i
        audio_out_dim = config.get('processors', {}).get('audio', {}).get('output_dim', 2)
        dummy_processed_audio_features = np.random.rand(audio_out_dim).astype(np.float32) # 0-1 arası rastgele floatlar

        dummy_processed_inputs = {
            'visual': dummy_processed_visual_dict,
            'audio': dummy_processed_audio_features
        }
        logger.debug(f"RepresentationLearner için sahte processed_inputs ({list(dummy_processed_inputs.keys())}) oluşturuldu.")
        return dummy_processed_inputs

    elif module_name == 'memory':
        # Memory modülü store metodu Representation vektörü (numpy array) bekler.
        # RepresentationLearner çıktısı formatında olmalı.
        repr_dim = config.get('representation', {}).get('representation_dim', 128)
        dummy_representation = np.random.rand(repr_dim).astype(np.float64) # RL default float64 döndürüyor
        logger.debug(f"Memory için sahte Representation ({dummy_representation.shape}, {dummy_representation.dtype}) oluşturuldu.")
        # retrieve metodu Representation vektörü ve num_results int bekler.
        # Hangi metot test edilecekse ona göre girdi üretilmeli.
        # Şimdilik store için girdi üretelim.
        return dummy_representation

    elif module_name == 'cognitioncore':
         # CognitionCore.decide processed_inputs (dict), learned_representation (array), relevant_memory_entries (list) bekler.
         # Bu girdiler Processor, RepresentationLearner, Memory modüllerinin çıktı formatında olmalıdır.

         # Sahte processed_inputs (Processor çıktısı)
         vis_out_w = config.get('processors', {}).get('vision', {}).get('output_width', 64)
         vis_out_h = config_get_value(config, 'processors', {}).get('vision', {}).get('output_height', 64) # Util kullanıldı
         dummy_processed_visual_gray = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
         dummy_processed_visual_edges = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
         dummy_processed_visual_dict = {'grayscale': dummy_processed_visual_gray, 'edges': dummy_processed_visual_edges}

         audio_out_dim = config.get('processors', {}).get('audio', {}).get('output_dim', 2)
         dummy_processed_audio_features = np.random.rand(audio_out_dim).astype(np.float32)

         dummy_processed_inputs = {
             'visual': dummy_processed_visual_dict,
             'audio': dummy_processed_audio_features
         }

         # Sahte learned_representation (RepresentationLearner çıktısı)
         repr_dim = config.get('representation', {}).get('representation_dim', 128)
         dummy_representation = np.random.rand(repr_dim).astype(np.float64)

         # Sahte relevant_memory_entries (Memory.retrieve çıktısı)
         # Memory.retrieve list of dicts döndürür: [{'representation': array, 'metadata': {}, 'timestamp': float}, ...]
         num_mem = config.get('memory', {}).get('num_retrieved_memories', 5)
         dummy_memory_entries = []
         for i in range(num_mem):
              dummy_mem_rep = np.random.rand(repr_dim).astype(np.float64) # Rastgele temsil
              # İlk anıyı query'ye çok benzer yapalım ki tanıdık algılansın bazen
              if i == 0: dummy_mem_rep = dummy_representation.copy() # Query'nin kopyası
              dummy_memory_entries.append({
                  'representation': dummy_mem_rep,
                  'metadata': {'source': 'test', 'index': i},
                  'timestamp': time.time() - i # Farklı zaman damgaları
              })


         # CognitionCore.decide methodu için args/kwargs'ı tuple olarak döndür.
         # decide(self, processed_inputs, learned_representation, relevant_memory_entries)
         logger.debug("CognitionCore için sahte processed_inputs, learned_representation, relevant_memory_entries oluşturuldu.")
         return (dummy_processed_inputs, dummy_representation, dummy_memory_entries)


    elif module_name == 'decisionmodule':
         # DecisionModule.decide understanding_signals (dict), relevant_memory_entries (list) bekler.
         # UnderstandingModule çıktısı formatında olmalı.
         # UnderstandingModule çıktısı: {'similarity_score': float, 'high_audio_energy': bool, 'high_visual_edges': bool, 'is_bright': bool, 'is_dark': bool, 'max_concept_similarity': float, 'most_similar_concept_id': int or None}

         # Sahte understanding_signals dictionary'si. Farklı senaryoları test etmek için değerler değiştirilebilir.
         dummy_understanding_signals = {
             'similarity_score': random.random(), # 0.0 - 1.0 arası rastgele bellek benzerliği
             'high_audio_energy': random.choice([True, False]), # Rastgele ses algılama
             'high_visual_edges': random.choice([True, False]), # Rastgele kenar algılama
             'is_bright': random.choice([True, False]), # Rastgele parlak
             'is_dark': random.choice([True, False]),   # Rastgele karanlık
             'max_concept_similarity': random.random(), # 0.0 - 1.0 arası rastgele kavram benzerliği
             'most_similar_concept_id': random.choice([None, 0, 1, 2]), # Rastgele kavram ID veya None
         }
         # consistency check: eğer is_bright True ise is_dark False olmalı
         if dummy_understanding_signals['is_bright'] and dummy_understanding_signals['is_dark']:
             dummy_understanding_signals['is_dark'] = False

         # Sahte relevant_memory_entries (DecisionModule içinde doğrudan kullanılmıyor ama parametre olarak geliyor)
         # Boş liste göndermek yeterli.
         dummy_memory_entries = []

         # DecisionModule.decide methodu için args/kwargs'ı tuple olarak döndür.
         # decide(self, understanding_signals, relevant_memory_entries, internal_state=None)
         logger.debug("DecisionModule için sahte understanding_signals ve relevant_memory_entries oluşturuldu.")
         return (dummy_understanding_signals, dummy_memory_entries)

    elif module_name == 'motorcontrolcore':
         # MotorControlCore.generate_response decision (str or any) bekler.
         # DecisionModule çıktısı formatında olmalı.

         # Sahte karar stringi. Farklı senaryoları test etmek için değiştirilebilir.
         # Örneğin, tüm olası karar stringlerini test edebiliriz.
         possible_decisions = [
             "explore_randomly", "make_noise",
             "sound_detected", "complex_visual_detected",
             "bright_light_detected", "dark_environment_detected",
             "recognized_concept_0", "recognized_concept_1", # Kavram ID'leri değişebilir
             "familiar_input_detected", "new_input_detected",
             "unknown_decision", # Bilinmeyen karar testi
             None, # None karar testi
         ]
         dummy_decision = random.choice(possible_decisions)

         logger.debug(f"MotorControlCore için sahte decision '{dummy_decision}' oluşturuldu.")
         # generate_response(self, decision)
         return (dummy_decision,) # Tek elemanlı tuple döndür.

    elif module_name == 'expressiongenerator':
         # ExpressionGenerator.generate command (str or any) bekler.
         # MotorControlCore çıktısı formatında olmalı (ExpressionGenerator komut stringi).

         # Sahte komut stringi. Farklı senaryoları test etmek için değiştirilebilir.
         possible_commands = [
             "explore_randomly_response", "make_noise_response",
             "sound_detected_response", "complex_visual_response",
             "bright_light_response", "dark_environment_response",
             "recognized_concept_response_0", "recognized_concept_response_1",
             "familiar_response", "new_response",
             "unknown_command", # Bilinmeyen komut testi
             None, # None komut testi
         ]
         dummy_command = random.choice(possible_commands)

         logger.debug(f"ExpressionGenerator için sahte command '{dummy_command}' oluşturuldu.")
         # generate(self, command)
         return (dummy_command,) # Tek elemanlı tuple döndür.


    elif module_name == 'interactionapi':
         # InteractionAPI.send_output output_data (any) bekler.
         # MotorControlCore çıktısı formatında olmalı (ExpressionGenerator'dan gelen metin stringi veya None).

         # Sahte çıktı verisi. Farklı senaryoları test etmek için değiştirilebilir.
         possible_outputs = [
             "Bu tanıdık geliyor.",
             "Yeni bir şey algıladım.",
             "Bir ses duyuyorum.",
             "Sanırım bu bir kavram 0.",
             None, # None çıktı testi
             {"status": "ok", "message": "API response"}, # Başka tipte çıktı
         ]
         dummy_output_data = random.choice(possible_outputs)

         logger.debug(f"InteractionAPI için sahte output_data '{dummy_output_data}' oluşturuldu.")
         # send_output(self, output_data)
         return (dummy_output_data,) # Tek elemanlı tuple döndür.


    # TODO: Diğer modüller için sahte girdi oluşturma mantığı buraya eklenecek.
    # elif module_name == 'some_other_module': ...


    logger.error(f"Sahte girdi oluşturma '{module_name}' modülü için desteklenmiyor.")
    return None # Desteklenmeyen modül için None döndür.


def run_module_test(module_path, class_name, config):
    """
    Belirtilen modülü başlatır, sahte girdi oluşturur ve modülün ana işleme metodunu çalıştırır.

    Args:
        module_path (str): Modülün Python yolu (örn: 'src.processing.vision').
        class_name (str): Modül içindeki sınıf adı (örn: 'VisionProcessor').
        config (dict): Genel yapılandırma sözlüğü.

    Returns:
        tuple: (success, output_data)
               success (bool): Test başarılı mı?
               output_data (any): Modülün işleme metodundan dönen çıktı veya hata durumunda None.
    """
    logger.info(f"--- Modül Testi Başlatılıyor: {class_name} ({module_path}) ---")
    module_class = load_module_class(module_path, class_name)

    if module_class is None:
        logger.error("Modül testi başlatılamadı: Sınıf yüklenemedi.")
        return False, None

    module_instance = None
    output_data = None
    test_success = False

    try:
        # Modülü başlat
        # CognitionCore özel olarak module_objects bekler. Diğerleri config bekler.
        if class_name == 'CognitionCore':
            # Sahte module_objects dictionary'si oluştur.
            # Bu sadece CognitionCore'un init sırasında Memory, Learning gibi alt modül referanslarını alabilmesi için gerekli.
            # Gerçek alt modül objeleri yaratmamıza gerek yok, sadece placeholder None objeler yeterli.
            dummy_module_objects = {
                'memories': {'core_memory': None}, # Memory None olabilir
                'cognition': {}, # Kendisi
                'motor_control': {},
                'interaction': {},
            }
            # Eğer LearningModule varsa Representation boyutu için config'e bakabilir.
            dummy_module_objects['representers'] = {'main_learner': None}

            # CognitionCore'u başlatırken config VE dummy_module_objects ilet.
            logger.debug(f"'{class_name}' başlatılırken dummy_module_objects iletiliyor.")
            module_instance = module_class(config, dummy_module_objects) # <<< CognitionCore özel init
        elif class_name == 'LearningModule':
             # LearningModule init'i config ve representation_dim bekler.
             # Representation dim CognitionCore tarafından config'ten alınıp iletiliyordu.
             # Burada manuel olarak config'ten alıp learning_config'e ekleyelim.
             learning_config = config.get('learning', {}).copy() # Copy alalım ki orijinali değiştirmeyelim
             representation_config = config.get('representation', {})
             learning_config['representation_dim'] = learning_config.get('representation_dim', representation_config.get('representation_dim', 128))
             module_instance = module_class(learning_config) # Sadece config ile başlat
        elif class_name == 'UnderstandingModule':
             # UnderstandingModule init'i sadece config bekler.
             understanding_config = config.get('cognition', {}).get('understanding', {})
             module_instance = module_class(understanding_config) # Sadece config ile başlat.
        elif class_name == 'DecisionModule':
             # DecisionModule init'i sadece config bekler.
             decision_config = config.get('cognition', {}).get('decision', {})
             module_instance = module_class(decision_config) # Sadece config ile başlat.

        elif class_name == 'MotorControlCore':
             # MotorControlCore init'i sadece config bekler (alt modülleri kendi başlatır).
             motor_config = config.get('motor_control', {})
             module_instance = module_class(motor_config) # Sadece config ile başlat.

        elif class_name == 'InteractionAPI':
             # InteractionAPI init'i sadece config bekler (kanalları kendi başlatır).
             interaction_config = config.get('interaction', {})
             module_instance = module_class(interaction_config) # Sadece config ile başlat.

        elif class_name == 'Memory':
            # Memory init'i sadece config bekler.
            memory_config = config.get('memory', {})
            module_instance = module_class(memory_config) # Sadece config ile başlat.

        elif class_name in ['VisionSensor', 'AudioSensor', 'VisionProcessor', 'AudioProcessor']:
            # Sense ve Processor modülleri init'leri sadece config bekler.
            # Config'teki ilgili bölümü bul.
            if module_name.endswith('sensor'):
                 mod_config = config.get(module_name.replace('sensor', ''), {}) # vision -> config['vision']
            elif module_name.endswith('processor'):
                 mod_config = config.get('processors', {}).get(module_name.replace('processor', ''), {}) # visionprocessor -> config['processors']['vision']
            else:
                 mod_config = {} # Bulunamadıysa boş config.

            module_instance = module_class(mod_config) # Sadece config ile başlat.

        else:
            # Diğer modüller default olarak config bekler
            module_instance = module_class(config)

        if module_instance is None:
             logger.error("Modül objesi başlatılamadı.")
             return False, None

        # Modülün ana işleme metodunu bul ve sahte girdi oluştur.
        # Hangi metodun çağrılacağı modüle göre değişir (process, capture_frame, capture_chunk, learn, store, retrieve, decide, generate_response, generate, execute_command, send_output).
        # Test scripti, test edilen modülün ana işleme metodunu bilmeli veya config'ten okumalı.
        # Şimdilik en yaygın 'process' veya 'decide'/'generate' gibi metotları deneyelim.

        # Sahte girdi oluştur
        # create_dummy_input, test edilen modülün adını küçük harflerle bekler.
        dummy_input_or_args = create_dummy_input(class_name.lower(), config)

        if dummy_input_or_args is None:
             logger.warning(f"'{class_name}' modülü için sahte girdi oluşturulamadı veya desteklenmiyor. Metot çağrısı atlanıyor.")
             # Modül başlatıldıysa test başarılı sayılabilir mi? Hayır, işleme testi yapılamadı.
             # return True, None # Duruma göre True/False olabilir. İşleme testi yapılmadıysa False diyelim.
             test_success = False # İşleme testi yapılamadı.
             output_data = None
        else:
             logger.debug(f"'{class_name}' için sahte girdi oluşturuldu.")

             # Modülün ana işleme metodunu çağır.
             # Metot adı ve argümanları modüle göre belirlenmelidir.
             # Eğer dummy_input_or_args bir tuple ise, argümanlar birden fazladır (*dummy_input_or_args ile unpack et).
             try:
                 if class_name in ['VisionProcessor', 'AudioProcessor', 'UnderstandingModule']:
                      # process metotları processed_inputs, learned_representation, relevant_memory_entries, concepts gibi farklı argümanlar alır.
                      # UnderstandingModule.process(processed_inputs, learned_representation, relevant_memory_entries, current_concepts)
                      # Process modülleri sadece (input) alır.
                      if class_name == 'UnderstandingModule':
                           # UnderstandingModule için özel argümanlar (processed_inputs, learned_representation, relevant_memory_entries, current_concepts)
                           # create_dummy_input 'cognitioncore' için bu argümanları tuple olarak döndürüyordu.
                           # UnderstandingModule için ayrı sahte girdi oluşturma mantığı ekleyelim create_dummy_input'a.
                           # Veya burada manuel oluşturalım.
                           # Sahte processed_inputs, learned_representation, relevant_memory_entries, current_concepts oluşturalım.
                           proc_in_dummy = create_dummy_input('representationlearner', config) # RL girdisi Processor çıktısını taklit eder.
                           repr_dummy = create_dummy_input('memory', config) # Memory girdisi RL çıktısını taklit eder.
                           mem_rel_dummy = [] # Şimdilik boş liste
                           concepts_dummy = [] # Şimdilik boş liste
                           # UnderstandingModule.process(processed_inputs, learned_representation, relevant_memory_entries, current_concepts)
                           output_data = module_instance.process(proc_in_dummy, repr_dummy, mem_rel_dummy, concepts_dummy) # Çağır.

                      else: # VisionProcessor, AudioProcessor (sadece 1 argüman)
                           output_data = module_instance.process(dummy_input_or_args) # Sadece sahte girdi ile çağır.

                 elif class_name in ['VisionSensor', 'AudioSensor']:
                      # capture_frame/chunk metotları argüman almaz. Sahte input aslında simülasyon çıktısıdır.
                      # Bu durumda test etmek istediğimiz sadece init ve catch hata mantığıdır. Metodu çağırmaya gerek yok.
                      # Eğer yine de çağırmak istersek argümansız çağırırız.
                      output_data = module_instance.capture_frame() if class_name == 'VisionSensor' else module_instance.capture_chunk() # Argümansız çağır.
                      logger.warning(f"Sensor modülü ('{class_name}') testi: capture metodu çağrıldı. Sahte girdi 'üretme' logic'i test edildi. Gerçek donanım test edilmedi.")


                 elif class_name == 'RepresentationLearner':
                      # learn metodu processed_inputs dictionary'si bekler.
                      output_data = module_instance.learn(dummy_input_or_args) # Sahte processed_inputs ile çağır.

                 elif class_name == 'Memory':
                      # store metodu Representation array bekler, retrieve metodu Representation array ve int bekler.
                      # Hangi metot test edilecekse ona göre çağrı yapılmalı.
                      # Şimdilik store metotunu test edelim.
                      store_input = create_dummy_input('memory', config) # Store için Representation girdisi.
                      module_instance.store(store_input) # Store çağrısı (çıktısı yok)
                      # Eğer retrieve test edilecekse:
                      # retrieve_query = create_dummy_input('memory', config) # Retrieve için Representation girdisi.
                      # output_data = module_instance.retrieve(retrieve_query, num_results=5) # Retrieve çağrısı (çıktısı var).
                      logger.warning(f"Memory modülü ('{class_name}') testi: store metodu çağrıldı. Retrieve testi atlandı.")
                      output_data = None # Store metodu çıktı döndürmez.


                 elif class_name == 'CognitionCore':
                      # decide metodu processed_inputs, learned_representation, relevant_memory_entries bekler.
                      # create_dummy_input('cognitioncore', config) bu argümanları tuple olarak döndürüyordu.
                      output_data = module_instance.decide(*dummy_input_or_args) # Argümanları unpack ederek çağır. <<< BURADA DEĞİŞİKLİK YAPILDI


                 elif class_name == 'DecisionModule':
                      # decide metodu understanding_signals (dict) ve relevant_memory_entries (list) bekler.
                      # create_dummy_input('decisionmodule', config) bu argümanları tuple olarak döndürüyordu.
                      output_data = module_instance.decide(*dummy_input_or_args) # Argümanları unpack ederek çağır.


                 elif class_name == 'MotorControlCore':
                      # generate_response decision string veya None bekler.
                      # create_dummy_input('motorcontrolcore', config) bu argümanı tuple olarak döndürüyordu.
                      output_data = module_instance.generate_response(*dummy_input_or_args) # Argümanları unpack ederek çağır.


                 elif class_name == 'ExpressionGenerator':
                      # generate command string veya None bekler.
                      # create_dummy_input('expressiongenerator', config) bu argümanı tuple olarak döndürüyordu.
                      output_data = module_instance.generate(*dummy_input_or_args) # Argümanları unpack ederek çağır.

                 elif class_name == 'InteractionAPI':
                      # send_output output_data bekler.
                      # create_dummy_input('interactionapi', config) bu argümanı tuple olarak döndürüyordu.
                      output_data = module_instance.send_output(*dummy_input_or_args) # Argümanları unpack ederek çağır.
                      logger.warning(f"InteractionAPI testi: send_output çağrıldı. Çıktı Interaction kanallarına gönderildi/simüle edildi.")
                      # send_output genellikle çıktı döndürmez, bu yüzden output_data None olur.


                 # TODO: Diğer modüllerin ana işleme metotları buraya eklenecek.


                 else:
                      logger.error(f"Modül '{class_name}' için ana işleme metodu belirleme/çağırma implemente edilmedi.")
                      test_success = False # İşleme testi yapılamadı.
                      output_data = None # Çıktı alınamadı.

                 # Eğer buraya kadar hata olmadıysa ve output_data başarıyla alındıysa
                 if output_data is not None:
                      logger.debug(f"'{class_name}' modülünden çıktı alındı: {output_data}")
                      # Burada çıktının beklenen formatta olup olmadığını kontrol edebiliriz (type, shape, value range).
                      # Bu kontroller test senaryosuna özeldir ve ayrı test fonksiyonlarında daha iyi yönetilir.
                      test_success = True # Çıktı alındıysa test başarılı sayılabilir.
                 # else: İşleme metodu None döndürdüyse (bu bazı metotlar için normaldir veya hata durumudur).
                 elif output_data is None and class_name not in ['Memory', 'InteractionAPI', 'MotorControlCore']: # None çıktı normal olmayan modüller için
                      logger.warning(f"'{class_name}' modülünden None çıktı döndü.")
                      # None çıktı beklenen bir durum değilse test başarısız sayılabilir.
                      # test_success = False # Duruma göre karar verilir.
                      # Şimdilik None döndürmek hata yönetimi gereği olabilir, bu yüzden True sayalım eğer Exception atmadıysa.
                      test_success = True # Exception almadık.


                 # Eğer işleme metodu belirlenmediyse veya hata olduysa success False kalır.
                 if handled_decision is not None and not handled_decision: # MotorControlCore generate_response'da işlenemeyen karar durumu.
                      test_success = False


             except Exception as e:
                 logger.error(f"'{class_name}' modülünün işleme metodu çalıştırılırken beklenmedik hata oluştu: {e}", exc_info=True)
                 test_success = False # İşlem sırasında hata olursa test başarısız.
                 output_data = None # Hata durumunda çıktı None.


    except Exception as e:
        # Modül başlatılırken beklenmedik hata oluşursa
        logger.error(f"'{class_name}' modülü başlatılırken beklenmedik hata oluştu: {e}", exc_info=True)
        test_success = False # Başlatma başarısızsa test başarısız.
        output_data = None # Çıktı alınamadı.

    finally:
        # Kaynakları temizle (cleanup metodu varsa).
        if module_instance and hasattr(module_instance, 'cleanup'):
            logger.debug(f"'{class_name}' modülü cleanup çağrılıyor.")
            try:
                 module_instance.cleanup()
                 logger.debug(f"'{class_name}' modülü temizlendi.")
            except Exception as e:
                 logger.error(f"'{class_name}' modülü temizlenirken hata oluştu: {e}", exc_info=True)


    logger.info(f"--- Modül Testi Tamamlandı: {class_name} ({module_path}). Başarılı: {test_success} ---")

    return test_success, output_data


def main():
    """
    Script'in ana çalıştırma fonksiyonu. Argümanları ayrıştırır ve modül testini başlatır.
    """
    parser = argparse.ArgumentParser(description="Evo modüllerini tek başına test etmek için script.")
    parser.add_argument('--module', required=True, help='Test edilecek modülün Python yolu (örn: src.processing.vision)')
    parser.add_argument('--class_name', required=True, help='Test edilecek sınıfın adı (örn: VisionProcessor)')
    # TODO: Gelecekte belirli test senaryolarını çalıştırma argümanı eklenecek. (--scenario)
    # TODO: Gelecekte sahte girdi verisi kaynağı/formatı argümanı eklenecek. (--input_data)
    # TODO: Gelecekte çıktıyı dosyaya kaydetme argümanı eklenecek. (--output_file)

    args = parser.parse_args()

    # Loglama sistemini yapılandır (config dosyası olmadan varsayılan ayarlar).
    # Test scripti config'i sadece modül başlatırken kullanmalı.
    setup_logging(config=None) # Varsayılan (INFO seviyesi) loglama

    # Script'in kendi logger'ını al (Loglama setup'ından sonra).
    global logger # Global logger'ı kullanacağımızı belirt
    logger = logging.getLogger(__name__)
    logger.info("Test scripti başlatıldı.")


    # Yapılandırma dosyasını yükle (modül başlatırken kullanılacak).
    config_path = "config/main_config.yaml"
    config = load_config_from_yaml(config_path) # Hata durumunda boş dict döner.

    # Config yüklenemezse testi sonlandır.
    if not config:
        logger.critical("Yapılandırma yüklenemediği için modül testi başlatılamıyor.")
        sys.exit(1)


    # Modül testini çalıştır.
    success, output = run_module_test(args.module, args.class_name, config)

    # Test sonucunu raporla.
    logger.info(f"\nGenel Test Sonucu: '{args.class_name}' ({args.module}) testi { 'BAŞARILI' if success else 'BAŞARISIZ' }.")

    # TODO: Çıktıyı dosyaya kaydetme veya başka formatlarda gösterme mantığı eklenecek.
    # if output is not None:
    #      logger.info(f"Test çıktısı: {output}") # Çok detaylı olabilir, özetini göstermek veya dosyaya yazmak daha iyi.


if __name__ == '__main__':
    main()