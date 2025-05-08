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
import random
import os # For creating dummy data files if needed
import json # For logging dict/json outputs

# Evo'nın loglama ve config yardımcılarını import et
# Assumes: script is run from the Evo root directory.
from src.core.logging_utils import setup_logging
from src.core.config_utils import load_config_from_yaml, get_config_value # Import get_config_value
from src.core.utils import cleanup_safely # Import cleanup_safely for robust cleanup


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

# Refactored name to clarify it creates inputs for the METHOD, not the module init
def create_dummy_method_inputs(class_name, config):
    """
    Belirtilen sınıfın ana işleme metodu için basit sahte girdi verisi oluşturur.
    Bu fonksiyon, test edilen sınıfın ana metodunun beklediği girdi formatına göre özelleştirilmelidir.

    Args:
        class_name (str): Test edilen sınıfın adı (örn: 'VisionProcessor', 'AudioProcessor').
                          Bu ad büyük/küçük harf fark etmeden gelebilir.
        config (dict): Genel yapılandırma sözlüğü.

    Returns:
        tuple: Modülün ana metot çağrısı için argümanları içeren bir tuple (pozisyonel argümanlar)
               veya desteklenmiyorsa None.
               Tek argümanlı metotlar için bile tuple döndürülmelidir (örn: (input_data,)).
               Eğer metodun çağrılması test edilmeyecekse veya argüman gerektirmiyorsa boş tuple () döndürülür.
    """
    # class_name'i küçük harfe çevirerek iç logic'te tutarlılık sağla
    class_name_lower = class_name.lower()

    logger.debug(f"'{class_name}' için sahte metot girdisi oluşturuluyor...")

    # Modül adına göre sahte girdi oluştur.
    # Bu kısım, test etmek istediğiniz her modülün ana metodu için özelleştirilmelidir.
    # Döndürülen değerler, metodun beklediği pozisyonel argümanların tuple'ı olmalıdır.

    if class_name_lower in ['visionsensor', 'audiosensor']:
        # VisionSensor.capture_frame() ve AudioSensor.capture_chunk() argüman almaz.
        # create_dummy_input metodu aslında buraya uygun değil, sadece metot argümanlarını hazırlar.
        # Sensörlerin capture metotları test edilecekse argümansız çağrılacak demektir.
        logger.debug(f"'{class_name}' için capture metodu argüman almaz. Boş girdi tuple'ı döndürülüyor.")
        return () # Boş tuple döndür, çağrı argümansız olacak.

    elif class_name_lower == 'visionprocessor':
        # VisionProcessor.process(visual_input) numpy array bekler.
        # Farklı boyut ve kanallarda (BGR veya Gri) sahte veriler test edilebilir.
        # VisionProcessor init'inde dummy_width/height config'i sensör için kullanılır,
        # processor girdisi VisionSensor capture_frame çıktısıdır (genellikle daha büyük).
        # Config'teki VisionSensor dummy boyutlarını kullanalım test için.
        dummy_width = get_config_value(config, 'vision', 'dummy_width', default=640, expected_type=int, logger_instance=logger)
        dummy_height = get_config_value(config, 'vision', 'dummy_height', default=480, expected_type=int, logger_instance=logger)
        # Sahte renkli BGR görüntüsü (uint8)
        dummy_frame = np.random.randint(0, 256, size=(dummy_height, dummy_width, 3), dtype=np.uint8)
        logger.debug(f"VisionProcessor için sahte process girdi frame ({dummy_frame.shape}, {dummy_frame.dtype}) oluşturuldu.")
        return (dummy_frame,) # Tuple içinde döndür.


    elif class_name_lower == 'audioprocessor':
        # AudioProcessor.process(audio_input) int16 numpy array bekler.
        # Config'teki AudioSensor chunk_size'ı kullanalım test için.
        chunk_size = get_config_value(config, 'audio', 'audio_chunk_size', default=1024, expected_type=int, logger_instance=logger)
        sample_rate = get_config_value(config, 'audio', 'audio_rate', default=44100, expected_type=int, logger_instance=logger) # Config'ten al
        # Sahte int16 ses verisi (ton gibi)
        frequency = 880
        amplitude = np.iinfo(np.int16).max * 0.1
        t = np.linspace(0., chunk_size / sample_rate, chunk_size)
        dummy_chunk = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)

        logger.debug(f"AudioProcessor için sahte process girdi chunk ({dummy_chunk.shape}, {dummy_chunk.dtype}) oluşturuldu.")
        return (dummy_chunk,) # Tuple içinde döndür.


    elif class_name_lower == 'representationlearner':
        # RepresentationLearner.learn(processed_inputs) processed_inputs sözlüğü bekler: {'visual': dict, 'audio': np.ndarray}.
        # Processor modüllerinin çıktı formatında olmalıdır.

        # Sahte VisionProcessor çıktısı dictionary'si
        vis_out_w = get_config_value(config, 'processors', 'vision', 'output_width', default=64, expected_type=int, logger_instance=logger)
        vis_out_h = get_config_value(config, 'processors', 'vision', 'output_height', default=64, expected_type=int, logger_instance=logger)
        # Sahte grayscale ve edges arrayleri
        dummy_processed_visual_gray = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
        dummy_processed_visual_edges = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
        dummy_processed_visual_dict = {'grayscale': dummy_processed_visual_gray, 'edges': dummy_processed_visual_edges}

        # Sahte AudioProcessor çıktısı array'i
        audio_out_dim = get_config_value(config, 'processors', 'audio', 'output_dim', default=2, expected_type=int, logger_instance=logger)
        dummy_processed_audio_features = np.random.rand(audio_out_dim).astype(np.float32) # 0-1 arası rastgele floatlar

        dummy_processed_inputs = {
            'visual': dummy_processed_visual_dict,
            'audio': dummy_processed_audio_features
        }
        logger.debug(f"RepresentationLearner için sahte learn girdi processed_inputs ({list(dummy_processed_inputs.keys())}) oluşturuldu.")
        return (dummy_processed_inputs,) # Tuple içinde döndür.


    elif class_name_lower == 'memory':
        # Memory.store(representation, metadata=None) Representation vektörü bekler.
        # Memory.retrieve(query_representation, num_results=None) Representation vektörü ve int bekler.
        # Varsayılan olarak store metodunu test etmek için girdi üretelim.
        # Store için girdi: Representation vektörü
        repr_dim = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)
        dummy_representation = np.random.rand(repr_dim).astype(np.float64) # RL default float64 döndürüyor varsayımı
        # Metadata da isteğe bağlı.
        dummy_metadata = {"source": "test_script", "timestamp": time.time()}

        logger.debug(f"Memory için sahte store girdi Representation ({dummy_representation.shape}, {dummy_representation.dtype}) ve Metadata oluşturuldu.")
        # Store metodunun argümanları (representation, metadata=None)
        return (dummy_representation, dummy_metadata)


    elif class_name_lower == 'cognitioncore':
         # CognitionCore.decide(processed_inputs, learned_representation, relevant_memory_entries, current_concepts=None) bekler.
         # Bunlar Processor, RepresentationLearner, Memory, LearningModule çıktısı formatında olmalıdır.

         # Sahte processed_inputs (Processor çıktısı) - RepresentationLearner girdisi ile aynı mantık
         vis_out_w = get_config_value(config, 'processors', 'vision', 'output_width', default=64, expected_type=int, logger_instance=logger)
         vis_out_h = get_config_value(config, 'processors', 'vision', 'output_height', default=64, expected_type=int, logger_instance=logger)
         dummy_processed_visual_gray = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
         dummy_processed_visual_edges = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
         dummy_processed_visual_dict = {'grayscale': dummy_processed_visual_gray, 'edges': dummy_processed_visual_edges}

         audio_out_dim = get_config_value(config, 'processors', 'audio', 'output_dim', default=2, expected_type=int, logger_instance=logger)
         dummy_processed_audio_features = np.random.rand(audio_out_dim).astype(np.float32)

         dummy_processed_inputs = {
             'visual': dummy_processed_visual_dict,
             'audio': dummy_processed_audio_features
         }

         # Sahte learned_representation (RepresentationLearner çıktısı)
         repr_dim = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)
         dummy_representation = np.random.rand(repr_dim).astype(np.float64)

         # Sahte relevant_memory_entries (Memory.retrieve çıktısı)
         # Memory.retrieve list of dicts döndürür: [{'representation': array, 'metadata': {}, 'timestamp': float}, ...]
         num_mem = get_config_value(config, 'memory', 'num_retrieved_memories', default=5, expected_type=int, logger_instance=logger)
         dummy_memory_entries = []
         for i in range(num_mem):
              dummy_mem_rep = np.random.rand(repr_dim).astype(np.float64)
              # İlk anıyı query'ye çok benzer yapalım ki tanıdık algılansın bazen
              if i == 0: dummy_mem_rep = dummy_representation.copy() + np.random.randn(repr_dim).astype(np.float64) * 0.01
              dummy_memory_entries.append({
                  'representation': dummy_mem_rep,
                  'metadata': {'source': 'test', 'index': i},
                  'timestamp': time.time() - i
              })

         # Sahte current_concepts (LearningModule.get_concepts() çıktısı)
         # LearningModule list of arrays döndürür.
         num_concepts = 3 # Test için 3 sahte kavram
         dummy_concepts = []
         for i in range(num_concepts):
              dummy_concepts.append(np.random.rand(repr_dim).astype(np.float64))


         # CognitionCore.decide methodu için args/kwargs'ı tuple olarak döndür.
         # decide(self, processed_inputs, learned_representation, relevant_memory_entries, current_concepts=None)
         # Argümanları pozisyonel olarak tuple'da döndürüyoruz.
         logger.debug("CognitionCore için sahte decide girdi tuple'ı oluşturuldu.")
         return (dummy_processed_inputs, dummy_representation, dummy_memory_entries, dummy_concepts)


    elif class_name_lower == 'decisionmodule':
         # DecisionModule.decide(understanding_signals, relevant_memory_entries, current_concepts=None) bekler.
         # UnderstandingModule ve Memory çıktı formatında olmalı.

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
         if dummy_understanding_signals.get('is_bright', False) and dummy_understanding_signals.get('is_dark', False):
             dummy_understanding_signals['is_dark'] = False

         # Sahte relevant_memory_entries (DecisionModule içinde doğrudan anı içeriği kullanılmuyor ama parametre olarak geliyor)
         # Boş liste göndermek yeterli.
         dummy_memory_entries = []

         # Sahte current_concepts (DecisionModule içinde kavram temsilleri değil, sadece ID'ler kullanılıyor olabilir?)
         # DecisionModule artık `current_concepts` parametresini alıyor. Boş liste gönderelim.
         dummy_concepts = []


         # DecisionModule.decide methodu için args/kwargs'ı tuple olarak döndür.
         # decide(self, understanding_signals, relevant_memory_entries, current_concepts)
         logger.debug("DecisionModule için sahte decide girdi tuple'ı oluşturuldu.")
         return (dummy_understanding_signals, dummy_memory_entries, dummy_concepts)


    elif class_name_lower == 'motorcontrolcore':
         # MotorControlCore.generate_response(decision) string veya any bekler.
         # DecisionModule çıktısı formatında olmalı (karar stringi).

         # Sahte karar stringi. Farklı senaryoları test etmek için değiştirilebilir.
         # Örneğin, tüm olası karar stringlerini test edebiliriz.
         possible_decisions = [
             "explore_randomly", "make_noise",
             "sound_detected", "complex_visual_detected",
             "bright_light_detected", "dark_environment_detected",
             "recognized_concept_0", "recognized_concept_1",
             "familiar_input_detected", "new_input_detected",
             "unknown_decision", # Bilinmeyen karar testi
             None, # None karar testi
         ]
         dummy_decision = random.choice(possible_decisions)

         logger.debug(f"MotorControlCore için sahte generate_response girdi tuple'ı '{dummy_decision}' oluşturuldu.")
         return (dummy_decision,) # Tuple içinde döndür.


    elif class_name_lower == 'expressiongenerator':
         # ExpressionGenerator.generate(command) string veya any bekler.
         # MotorControlCore çıktısı formatında olmalı (ExpressionGenerator komut stringi).

         # Sahte komut stringi. Farklı senaryoları test etmek için değiştirilebilir.
         possible_commands = [
             "explore_randomly_response", "make_noise_response",
             "sound_detected_response", "complex_visual_response",
             "bright_light_response", "dark_environment_response",
             "recognized_concept_response_0", "recognized_concept_response_1",
             "familiar_response", "new_response",
             "default_response", # Default komut testi
             "unknown_command", # Bilinmeyen komut testi
             None, # None komut testi
         ]
         dummy_command = random.choice(possible_commands)

         logger.debug(f"ExpressionGenerator için sahte generate girdi tuple'ı '{dummy_command}' oluşturuldu.")
         return (dummy_command,) # Tuple içinde döndür.


    elif class_name_lower == 'interactionapi':
         # InteractionAPI.send_output(output_data) any bekler.
         # MotorControlCore çıktısı formatında olmalı (ExpressionGenerator'dan gelen metin stringi veya None).

         # Sahte çıktı verisi. Farklı senaryoları test etmek için değiştirilebilir.
         possible_outputs = [
             "Bu tanıdık geliyor.",
             "Yeni bir şey algıladım.",
             "Bir ses duyuyorum.",
             "Sanırım bu bir kavram 0.",
             None, # None çıktı testi
             {"status": "ok", "message": "API response"}, # Başka tipte çıktı (WebAPI kanal test eder)
         ]
         dummy_output_data = random.choice(possible_outputs)

         logger.debug(f"InteractionAPI için sahte send_output girdi tuple'ı '{dummy_output_data}' oluşturuldu.")
         return (dummy_output_data,) # Tuple içinde döndür.

    elif class_name_lower == 'learningmodule':
         # LearningModule.learn_concepts(representation_list) Representation vektör listesi bekler.
         # Memory'den Representation örneklemi formatında olmalı.
         # Memory'de depolanan Representationlar RepresentationLearner çıktısı formatındadır (shape (repr_dim,), dtype numerical).

         repr_dim = get_config_value(config, 'representation', 'representation_dim', default=128, expected_type=int, logger_instance=logger)
         num_samples = get_config_value(config, 'cognition', 'learning_memory_sample_size', default=50, expected_type=int, logger_instance=logger) # Learning sample size

         # Sahte Representation listesi oluştur.
         dummy_rep_list = []
         for _ in range(num_samples):
              dummy_rep_list.append(np.random.rand(repr_dim).astype(np.float64))

         # Bazı vektörleri birbirine benzer yapalım ki kavram keşfedilebilsin.
         if num_samples > 5:
              # Çok benzer (sim ~1.0)
              dummy_rep_list[1] = dummy_rep_list[0].copy() + np.random.randn(repr_dim).astype(np.float64) * 0.01
              # Biraz benzer (sim < 1.0, >= threshold)
              # Sim 0.7 olacak şekilde bir vektör
              sim_threshold = get_config_value(config, 'cognition', 'new_concept_threshold', default=0.7, expected_type=(int, float), logger_instance=logger)
              vec_similar_to_0 = dummy_rep_list[0].copy()
              norm_vec_similar = np.linalg.norm(vec_similar_to_0)
              if norm_vec_similar > 1e-8:
                   vec_similar_to_0 /= norm_vec_similar # Normalize
              else:
                   vec_similar_to_0 = np.zeros(repr_dim, dtype=np.float64) # Handle zero norm


              # Hedef benzerlik: sim_threshold. Dot product = norm * norm * sim = 1.0 * 1.0 * sim_threshold = sim_threshold
              # Vektör V = [a, b, ...] ve V_ref = [1, 0, ...]. Dot(V, V_ref) = a. Norm(V)=1. Norm(V_ref)=1. Sim = a.
              # Yani ilk eleman a = sim_threshold olursa, diğer elemanlar sqrt(1-a^2) gibi ayarlanırsa sim threshold olur.
              # Burada V_ref random olduğu için tam sim_threshold ayarlamak zor. Rastgele yeterince benzer yapalım.
              dummy_rep_list[2] = dummy_rep_list[0].copy() + np.random.randn(repr_dim).astype(np.float64) * 0.2 # Biraz daha farklı (sim < threshold, > 0)

              dummy_rep_list[4] = dummy_rep_list[3].copy() + np.random.randn(repr_dim).astype(np.float64) * 0.01 # Başka bir kümeye benzer


         logger.debug(f"LearningModule için sahte learn_concepts girdi listesi ({len(dummy_rep_list)} eleman) oluşturuldu.")
         return (dummy_rep_list,) # Tuple içinde döndür.

    elif class_name_lower in ['episodicmemory', 'semanticmemory', 'manipulator', 'locomotioncontroller']:
         # Placeholder veya unimplemented modüller.
         # Bu modüllerin ana metodları henüz net değil.
         # Şimdilik bu modüller için sahte girdi oluşturmayı desteklemediğimizi belirtelim.
         logger.warning(f"'{class_name}' modülü placeholder veya tam implemente edilmemiş. Sahte girdi oluşturma desteklenmiyor.")
         return None # Desteklenmeyen modül için None döndür.


    # TODO: Diğer implemente edildikçe buraya eklenecek modüller için sahte girdi oluşturma mantığı.


    # Eğer buraya gelinirse, isim eşleşmedi demektir.
    logger.error(f"Sahte girdi oluşturma '{class_name}' modülü için implemente edilmedi veya bilinmiyor.")
    return None # Bilinmeyen modül adı için None döndür.

# Refactored run_module_test function
def run_module_test(module_path, class_name, config):
    """
    Belirtilen modülü başlatır, sahte girdi oluşturur ve modülün ana işleme metodunu çalıştırır.

    Args:
        module_path (str): Modülün Python yolu (örn: src.processing.vision).
        class_name (str): Modül içindeki sınıf adı (örn: VisionProcessor).
        config (dict): Genel yapılandırma sözlüğü.

    Returns:
        tuple: (success, output_data)
               success (bool): Test başarılı mı? (Modül başlatıldı mı ve metot hata fırlatmadan çalıştı mı?)
               output_data (any): Modülün işleme metodından dönen çıktı veya metot çağrılamadıysa/hata durumında None.
    """
    logger.info(f"--- Modül Testi Başlatılıyor: {class_name} ({module_path}) ---")
    module_class = load_module_class(module_path, class_name)

    if module_class is None:
        logger.error("Modül testi başlatılamadı: Sınıf yüklenemedi.")
        return False, None

    module_instance = None
    output_data = None
    test_success = True # Başlatma başarısını takip eder.
    method_call_success = None # Metot çağrıldıysa metodun başarısını takip eder.

    try:
        # --- Modülü Başlat ---
        # Çoğu modül sadece config ile başlar. CognitionCore module_objects bekler.
        # Alt modül init argümanlarını hazırlayalım.
        # Config objesi, bu test scripti içinde get_config_value ile okunduğunda doğru değerleri döndürececektir.
        init_args = [config] # İlk argüman her zaman config

        # Eğer CognitionCore ise, dummy_module_objects dictionary'sini de init argümanı olarak ekle.
        # Bu dummy objeler, CognitionCore'un init sırasında null reference hatası almasını engeller.
        # Bu mock benzeri yapı sadece init için geçerlidir.
        if class_name.lower() == 'cognitioncore':
            # Sahte module_objects dictionary'si oluştur.
            # Memory, Learning gibi alt modül referanslarını içerebilir (şimdilik None).
            dummy_module_objects = {
                'memories': {'core_memory': None}, # Memory None olabilir
                'cognition': {}, # Kendisi (CognitionCore)
                'motor_control': {'core_motor_control': None}, # MotorControl None olabilir (alt modülü ExpressionGenerator init'te başlatılır)
                'interaction': {'core_interaction': None}, # Interaction None olabilir
                'representers': {'main_learner': None}, # RepresentationLearner None olabilir
                # Sensör ve Processor objeleri CognitionCore init'i için gerekli değildir.
            }
            init_args.append(dummy_module_objects) # İkinci argüman olarak module_objects ekle.


        logger.debug(f"'{class_name}' modülü başlatılıyor...")
        # Modül objesini başlat. Hata oluşursa exception fırlatılır.
        module_instance = module_class(*init_args) # Argümanları unpack ederek init'i çağır.

        # Eğer init başarıyla tamamlandıysa module_instance None değildir (Sensor init'leri None döndürme ihtimali hariç).
        # Eğer Sensor init hata yönetimi yaparak None döndürdüyse, başlatma başarısız kabul edilir.
        if module_instance is None:
             logger.error(f"'{class_name}' modülü başlatılırken hata oluştu (init None veya False döndürdü).")
             test_success = False # Başlatma başarısızsa test başarısız.
             return False, None # Başlatma başarısız.


        # --- Modülün Ana İşleme Metodunu Bul ve Çağır ---
        # Hangi metodun çağrılacağı test edilen sınıfa göre belirlenmelidir.
        # create_dummy_method_inputs fonksiyonu, metodun beklediği argümanları tuple olarak döndürür.

        # Sahte girdi argümanlarını al
        # create_dummy_method_inputs None döndürürse, o modülün ana metodunu test etmek desteklenmiyor demektir.
        dummy_method_inputs = create_dummy_method_inputs(class_name, config)

        if dummy_method_inputs is None:
             logger.warning(f"'{class_name}' modülü için sahte girdi oluşturulamadı veya ana metot çağrısı implemente edilmedi. Sadece başlatma testi yapıldı.")
             # İşleme testi yapılamadı, ancak başlatma başarılıydı. Başarı init durumuna bağlı.
             # test_success True kaldıysa init başarılıdır.
             final_success_status = test_success # Init başarısı test_success'te.
             logger.debug(f"'{class_name}': İşleme metodu testi atlandı. Başlatma Başarılı: {final_success_status}")
             # Cleanup finally bloğunda yapılacak.
             return final_success_status, None # Metot çağrılmadığı için çıktı yok.

        else:
             # DEBUG: Argümanların şeklini ve içeriğini kontrol et
             logger.debug(f"'{class_name}' için sahte girdi oluşturuldu. Argümanlar: {dummy_method_inputs} (Length: {len(dummy_method_inputs)})")
             # DEBUG: Metod imzasını kontrol etmeden önce metod objesi elde edildi.
             method_to_test = None # Çağrılacak metot objesi
             method_name = None

             # --- Metot Seçimi (Tekrarlanan kod, refactor edilebilir?) ---
             # Her modül için test edilecek ana metodu burada seçiyoruz.
             if class_name.lower() in ['visionprocessor', 'audioprocessor', 'understandingmodule']:
                  method_to_test = getattr(module_instance, 'process', None)
                  method_name = 'process'
             elif class_name.lower() in ['visionsensor']:
                  method_to_test = getattr(module_instance, 'capture_frame', None)
                  method_name = 'capture_frame'
                  # dummy_method_inputs = () # capture metotları argüman almaz, create_dummy_method_inputs boş tuple döndürmeli
                  if dummy_method_inputs: # Ensure it's empty tuple if create_dummy_method_inputs returned non-empty
                       logger.warning(f"'{class_name}.capture_frame' metodu argüman almamalı, sahte girdi {dummy_method_inputs} oluşturuldu. Boş tuple kullanılacak.")
                       dummy_method_inputs = ()
             elif class_name.lower() in ['audiosensor']:
                  method_to_test = getattr(module_instance, 'capture_chunk', None)
                  method_name = 'capture_chunk'
                  # dummy_method_inputs = () # capture metotları argüman almaz
                  if dummy_method_inputs: # Ensure it's empty tuple
                        logger.warning(f"'{class_name}.capture_chunk' metodu argüman almamalı, sahte girdi {dummy_method_inputs} oluşturuldu. Boş tuple kullanılacak.")
                        dummy_method_inputs = ()
             elif class_name.lower() == 'representationlearner':
                  method_to_test = getattr(module_instance, 'learn', None)
                  method_name = 'learn'
             elif class_name.lower() == 'memory':
                  # Memory için store veya retrieve test edilebilir. create_dummy_method_inputs varsayılan olarak store için girdi üretiyor.
                  # Bu scriptin basitliği için sadece store'u test edelim.
                  method_to_test = getattr(module_instance, 'store', None)
                  method_name = 'store'
                  # retrieve test edilecekse, create_dummy_method_inputs retrieve girdilerini döndürmeli ve burası güncellenmeli.
             elif class_name.lower() == 'cognitioncore':
                  method_to_test = getattr(module_instance, 'decide', None)
                  method_name = 'decide'
             elif class_name.lower() == 'decisionmodule':
                  method_to_test = getattr(module_instance, 'decide', None)
                  method_name = 'decide'
             elif class_name.lower() == 'motorcontrolcore':
                  method_to_test = getattr(module_instance, 'generate_response', None)
                  method_name = 'generate_response'
             elif class_name.lower() == 'expressiongenerator':
                  method_to_test = getattr(module_instance, 'generate', None)
                  method_name = 'generate'
             elif class_name.lower() == 'interactionapi':
                  method_to_test = getattr(module_instance, 'send_output', None)
                  method_name = 'send_output'
             elif class_name.lower() == 'learningmodule':
                  method_to_test = getattr(module_instance, 'learn_concepts', None)
                  method_name = 'learn_concepts'
             # --- Metot Seçimi Sonu ---


             if method_to_test is None:
                  logger.error(f"Modül '{class_name}' için test edilecek ana işleme metodu ('{method_name}') bulunamadı.")
                  method_call_success = False # Metot yoksa çağrı başarısız.
                  output_data = None # Metot çağrılamadı.
             else:
                  # DEBUG: Metod imzasını kontrol etmeden önce metod objesi elde edildi.
                  logger.debug(f"Method signature: {inspect.signature(method_to_test)}") # DEBUG: İmza bilgisi
                  logger.debug(f"'{class_name}.{method_name}' metodu çağrılıyor...")
                  try:
                      # Metodu sahte girdi argümanlarıyla çağır.
                      output_data = method_to_test(*dummy_method_inputs) # Tuple'ı unpack ederek argümanları ilet.

                      # Eğer buraya kadar hata olmadıysa, metod çağrısı başarılı.
                      method_call_success = True
                      logger.debug(f"'{class_name}.{method_name}' metodu başarıyla çalıştı.")

                      # Eğer çıktı None değilse logla.
                      if output_data is not None:
                           logger.debug(f"'{class_name}.{method_name}' çıktısı:")
                           # log_output_data(output_data) # Yeni yardımcı fonksiyonu kullan
                           print(output_data) # log_output_data yerine print kullanıldı, log_output_data fonksiyonu yok.
                      else:
                          logger.debug(f"'{class_name}.{method_name}' çıktısı: None")


                  except Exception as e:
                      logger.error(f"'{class_name}.{method_name}' metodu çalıştırılırken beklenmedik hata oluştu: {e}", exc_info=True)
                      method_call_success = False # İşlem sırasında hata olursa çağrı başarısız.
                      output_data = None # Hata durumunda çıktı None.


    except Exception as e:
        # Modül başlatılırken beklenmedik hata oluşursa (module_instance None kalır).
        logger.error(f"'{class_name}' modülü başlatılırken beklenmedik hata oluştu: {e}", exc_info=True)
        test_success = False # Başlatma başarısızsa test başarısız.
        method_call_success = False # Metot çağrılamadı bile.
        output_data = None # Çıktı alınamadı.

    finally:
        # Kaynakları temizle (cleanup metodu varsa).
        # Modül instance'ı başlatılırken hata olduysa None olabilir.
        if module_instance and hasattr(module_instance, 'cleanup'):
            logger.debug(f"'{class_name}' modülü cleanup çağrılıyor.")
            # cleanup_safely kullanarak temizleme sırasındaki hataları yakala
            cleanup_safely(module_instance.cleanup, logger_instance=logger, error_message=f"'{class_name}' modülü temizlenirken hata")
            logger.debug(f"'{class_name}' modülü temizlendi.")

    # Testin genel başarı durumu: Başlatma başarılıysa VE işleme metodu test edildiyse (dummy_method_inputs is not None) ve o başarılıysa True.
    # Eğer işleme metodu test edilmediyse (dummy_method_inputs is None), sadece başlatma başarılıysa True.
    # test_success init başarısını takip ediyor. method_call_success metodun başarısını takip ediyor.

    final_success_status = test_success # Başlatma başarısı

    if dummy_method_inputs is not None: # İşleme metodu test edildiyse...
         final_success_status = final_success_status and method_call_success # Başlatma ve metod başarısı birlikte.

    logger.info(f"'{class_name}': Nihai başarı durumu hesaplandı: {final_success_status}")

    return final_success_status, output_data