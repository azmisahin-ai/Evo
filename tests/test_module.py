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

# Evo'nın loglama ve config yardımcılarını import et
# Assumes: script is run from the Evo root directory.
from src.core.logging_utils import setup_logging
from src.core.config_utils import load_config_from_yaml
from src.core.utils import get_config_value # Import get_config_value

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
                          Bu ad küçük harfli de gelebilir, içinde işlenmelidir.
        config (dict): Genel yapılandırma sözlüğü.

    Returns:
        tuple: Modülün ana metot çağrısı için argümanları içeren bir tuple (pozisyonel argümanlar)
               veya desteklenmiyorsa None.
               Tek argümanlı metotlar için bile tuple döndürülmelidir (örn: (input_data,)).
    """
    # class_name'i küçük harfe çevirerek tutarlılık sağla
    module_base_name = class_name.lower()

    logger.debug(f"'{class_name}' için sahte metot girdisi oluşturuluyor...")

    # Modül adına göre sahte girdi oluştur.
    # Bu kısım, test etmek istediğiniz her modülün ana metodu için özelleştirilmelidir.
    # Döndürülen değerler, metodun beklediği pozisyonel argümanların tuple'ı olmalıdır.

    if module_base_name == 'visionsensor':
        # VisionSensor.capture_frame() veya .capture_chunk() argüman almaz.
        # create_dummy_input metodu aslında buraya uygun değil.
        # Sensorleri test etmek sadece init ve cleanup/stop_stream mantığını kontrol eder.
        # capture metotları argümansız çağrılır.
        logger.debug("VisionSensor/AudioSensor için sahte girdi 'üretme' metodu uygun değil.")
        return () # Boş tuple döndür, çağrı argümansız olacak.

    elif module_base_name == 'audiosensor':
         logger.debug("VisionSensor/AudioSensor için sahte girdi 'üretme' metodu uygun değil.")
         return () # Boş tuple döndür.


    elif module_base_name == 'visionprocessor':
        # VisionProcessor.process(visual_input) numpy array bekler.
        # Farklı boyut ve kanallarda (BGR veya Gri) sahte veriler test edilebilir.
        dummy_width = get_config_value(config, 'vision', {}).get('dummy_width', 640) # Config'ten al
        dummy_height = get_config_value(config, 'vision', {}).get('dummy_height', 480) # Config'ten al
        # Sahte renkli BGR görüntüsü (uint8)
        dummy_frame = np.random.randint(0, 256, size=(dummy_height, dummy_width, 3), dtype=np.uint8)
        logger.debug(f"VisionProcessor için sahte process girdi frame ({dummy_frame.shape}, {dummy_frame.dtype}) oluşturuldu.")
        return (dummy_frame,) # Tuple içinde döndür.


    elif module_base_name == 'audioprocessor':
        # AudioProcessor.process(audio_input) int16 numpy array bekler.
        chunk_size = get_config_value(config, 'audio', {}).get('audio_chunk_size', 1024) # Config'ten al
        sample_rate = get_config_value(config, 'audio', {}).get('audio_rate', 44100) # Config'ten al
        # Sahte int16 ses verisi (ton gibi)
        frequency = 880
        amplitude = np.iinfo(np.int16).max * 0.1
        t = np.linspace(0., chunk_size / sample_rate, chunk_size)
        dummy_chunk = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)

        logger.debug(f"AudioProcessor için sahte process girdi chunk ({dummy_chunk.shape}, {dummy_chunk.dtype}) oluşturuldu.")
        return (dummy_chunk,) # Tuple içinde döndür.


    elif module_base_name == 'representationlearner':
        # RepresentationLearner.learn(processed_inputs) processed_inputs sözlüğü bekler: {'visual': dict, 'audio': np.ndarray}.
        # Processor modüllerinin çıktı formatında olmalıdır.

        # Sahte VisionProcessor çıktısı dictionary'si
        vis_out_w = get_config_value(config, 'processors', {}).get('vision', {}).get('output_width', 64)
        vis_out_h = get_config_value(config, 'processors', {}).get('vision', {}).get('output_height', 64)
        # Sahte grayscale ve edges arrayleri
        dummy_processed_visual_gray = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
        dummy_processed_visual_edges = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
        dummy_processed_visual_dict = {'grayscale': dummy_processed_visual_gray, 'edges': dummy_processed_visual_edges}

        # Sahte AudioProcessor çıktısı array'i
        audio_out_dim = get_config_value(config, 'processors', {}).get('audio', {}).get('output_dim', 2)
        dummy_processed_audio_features = np.random.rand(audio_out_dim).astype(np.float32) # 0-1 arası rastgele floatlar

        dummy_processed_inputs = {
            'visual': dummy_processed_visual_dict,
            'audio': dummy_processed_audio_features
        }
        logger.debug(f"RepresentationLearner için sahte learn girdi processed_inputs ({list(dummy_processed_inputs.keys())}) oluşturuldu.")
        return (dummy_processed_inputs,) # Tuple içinde döndür.


    elif module_base_name == 'memory':
        # Memory.store(representation, metadata=None) Representation vektörü bekler.
        # Memory.retrieve(query_representation, num_results=None) Representation vektörü ve int bekler.
        # Hangi metot test edilecekse ona göre girdi üretilmeli.
        # Test scripti şimdilik sadece store'u çağırıyor gibi davransa da, retrieve girdisi de üretebiliriz.
        # Store için girdi: Representation vektörü
        repr_dim = get_config_value(config, 'representation', {}).get('representation_dim', 128)
        dummy_representation = np.random.rand(repr_dim).astype(np.float64) # RL default float64 döndürüyor
        logger.debug(f"Memory için sahte store girdi Representation ({dummy_representation.shape}, {dummy_representation.dtype}) oluşturuldu.")
        return (dummy_representation,) # Tuple içinde döndür (store metodu için).


    elif module_base_name == 'cognitioncore':
         # CognitionCore.decide(processed_inputs, learned_representation, relevant_memory_entries) bekler.
         # Bunlar Processor, RepresentationLearner, Memory çıktısı formatında olmalıdır.

         # Sahte processed_inputs (Processor çıktısı)
         vis_out_w = get_config_value(config, 'processors', {}).get('vision', {}).get('output_width', 64)
         vis_out_h = get_config_value(config, 'processors', {}).get('vision', {}).get('output_height', 64)
         dummy_processed_visual_gray = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
         dummy_processed_visual_edges = np.random.randint(0, 256, size=(vis_out_h, vis_out_w), dtype=np.uint8)
         dummy_processed_visual_dict = {'grayscale': dummy_processed_visual_gray, 'edges': dummy_processed_visual_edges}

         audio_out_dim = get_config_value(config, 'processors', {}).get('audio', {}).get('output_dim', 2)
         dummy_processed_audio_features = np.random.rand(audio_out_dim).astype(np.float32)

         dummy_processed_inputs = {
             'visual': dummy_processed_visual_dict,
             'audio': dummy_processed_audio_features
         }

         # Sahte learned_representation (RepresentationLearner çıktısı)
         repr_dim = get_config_value(config, 'representation', {}).get('representation_dim', 128)
         dummy_representation = np.random.rand(repr_dim).astype(np.float64)

         # Sahte relevant_memory_entries (Memory.retrieve çıktısı)
         # Memory.retrieve list of dicts döndürür: [{'representation': array, 'metadata': {}, 'timestamp': float}, ...]
         num_mem = get_config_value(config, 'memory', {}).get('num_retrieved_memories', 5)
         dummy_memory_entries = []
         for i in range(num_mem):
              dummy_mem_rep = np.random.rand(repr_dim).astype(np.float64)
              # İlk anıyı query'ye çok benzer yapalım ki tanıdık algılansın bazen
              if i == 0: dummy_mem_rep = dummy_representation.copy()
              dummy_memory_entries.append({
                  'representation': dummy_mem_rep,
                  'metadata': {'source': 'test', 'index': i},
                  'timestamp': time.time() - i
              })

         # CognitionCore.decide methodu için args/kwargs'ı tuple olarak döndür.
         # decide(self, processed_inputs, learned_representation, relevant_memory_entries)
         logger.debug("CognitionCore için sahte decide girdi tuple'ı oluşturuldu.")
         return (dummy_processed_inputs, dummy_representation, dummy_memory_entries)


    elif module_base_name == 'decisionmodule':
         # DecisionModule.decide(understanding_signals, relevant_memory_entries) bekler.
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
         logger.debug("DecisionModule için sahte decide girdi tuple'ı oluşturuldu.")
         return (dummy_understanding_signals, dummy_memory_entries)


    elif module_base_name == 'motorcontrolcore':
         # MotorControlCore.generate_response(decision) string veya any bekler.
         # DecisionModule çıktısı formatında olmalı.

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


    elif module_base_name == 'expressiongenerator':
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


    elif module_base_name == 'interactionapi':
         # InteractionAPI.send_output(output_data) any bekler.
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

         logger.debug(f"InteractionAPI için sahte send_output girdi tuple'ı '{dummy_output_data}' oluşturuldu.")
         return (dummy_output_data,) # Tuple içinde döndür.

    elif module_base_name == 'learningmodule':
         # LearningModule.learn_concepts(representation_list) Representation vektör listesi bekler.
         # Memory'den Representation örneklemi formatında olmalı.
         # Memory'de depolanan Representationlar RepresentationLearner çıktısı formatındadır (shape (repr_dim,), dtype numerical).

         repr_dim = get_config_value(config, 'representation', {}).get('representation_dim', 128)
         num_samples = get_config_value(config, 'cognition', {}).get('learning_memory_sample_size', 50) # Learning sample size

         # Sahte Representation listesi oluştur.
         dummy_rep_list = []
         for _ in range(num_samples):
              dummy_rep_list.append(np.random.rand(repr_dim).astype(np.float64))

         # Bazı vektörleri birbirine benzer yapalım ki kavram keşfedilebilsin.
         if num_samples > 5:
              dummy_rep_list[1] = dummy_rep_list[0].copy() + np.random.randn(repr_dim) * 0.01 # Çok benzer
              dummy_rep_list[2] = dummy_rep_list[0].copy() + np.random.randn(repr_dim) * 0.02 # Biraz benzer
              dummy_rep_list[4] = dummy_rep_list[3].copy() + np.random.randn(repr_dim) * 0.01

         logger.debug(f"LearningModule için sahte learn_concepts girdi listesi ({len(dummy_rep_list)} eleman) oluşturuldu.")
         return (dummy_rep_list,) # Tuple içinde döndür.


    # TODO: Diğer modüller için sahte girdi oluşturma mantığı buraya eklenecek.


    logger.error(f"Sahte girdi oluşturma '{class_name}' modülü için implemente edilmedi veya desteklenmiyor.")
    return None # Desteklenmeyen modül için None döndür.


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
    test_success = True # Başlangıçta başarılı varsayalım, hata olursa False yapacağız.

    try:
        # --- Modülü Başlat ---
        # Çoğu modül sadece config ile başlar. CognitionCore module_objects bekler.
        # LearningModule representation_dim'i config'ten alabilir.
        # Alt modül init argümanlarını hazırlayalım.
        init_args = [config] # İlk argüman her zaman config

        # Eğer CognitionCore ise, dummy_module_objects dictionary'sini de init argümanı olarak ekle.
        if class_name == 'CognitionCore':
            # Sahte module_objects dictionary'si oluştur.
            # Memory, Learning gibi alt modül referanslarını içerebilir (şimdilik None).
            # Bu sadece CognitionCore'un init sırasında hata almaması için.
            dummy_module_objects = {
                'memories': {'core_memory': None}, # Memory None olabilir
                'cognition': {}, # Kendisi
                'motor_control': {'core_motor_control': None}, # MotorControl None olabilir (alt modülü ExpressionGenerator init'te başlatılır)
                'interaction': {'core_interaction': None}, # Interaction None olabilir
                'representers': {'main_learner': None}, # RepresentationLearner None olabilir
            }
            init_args.append(dummy_module_objects) # İkinci argüman olarak module_objects ekle.


        # Modül objesini başlat.
        module_instance = module_class(*init_args) # Argümanları unpack ederek init'i çağır.

        if module_instance is None:
             logger.error("Modül objesi başlatılamadı.")
             return False, None


        # --- Modülün Ana İşleme Metodunu Bul ve Çağır ---
        # Hangi metodun çağrılacağı test edilen sınıfa göre belirlenmelidir.
        # create_dummy_method_inputs fonksiyonu, metodun beklediği argümanları tuple olarak döndürmelidir.
        # Bu argümanlar, modül başlatıldıktan sonra metoda iletilecektir.

        # Sahte girdi oluşturma ve metod çağırma için ayrı bir fonksiyon yapısı daha temiz olabilir.
        # Şimdilik burada devam edelim ama gelecekte refactoring hedefi olsun.

        # Sahte girdi argümanlarını al
        dummy_method_inputs = create_dummy_method_inputs(class_name, config)

        if dummy_method_inputs is None:
             logger.warning(f"'{class_name}' modülü için sahte girdi oluşturulamadı veya metot çağrısı implemente edilmedi.")
             # İşleme testi yapılamadı, ancak başlatma başarılıydı. Testi kısmen başarılı sayabiliriz.
             test_success = True # Başlatma başarılıydı
             output_data = None
        else:
             logger.debug(f"'{class_name}' için sahte girdi oluşturuldu.")

             # Test edilecek metodu belirle ve çağır.
             # Hangi metodun çağrılacağı test edilen sınıfa bağlıdır.
             method_to_test = None # Çağrılacak metot objesi

             if class_name in ['VisionProcessor', 'AudioProcessor', 'UnderstandingModule']:
                  method_to_test = module_instance.process
             elif class_name in ['VisionSensor', 'AudioSensor']:
                  # Sensörlerin capture metotları test ediliyor. Argüman almazlar.
                  method_to_test = module_instance.capture_frame if class_name == 'VisionSensor' else module_instance.capture_chunk
                  dummy_method_inputs = () # Argüman almadıkları için dummy input tuple'ı boş olmalı.
                  logger.debug(f"Sensor modülü ('{class_name}') testi: capture metodu çağrılıyor.")

             elif class_name == 'RepresentationLearner':
                  method_to_test = module_instance.learn
             elif class_name == 'Memory':
                  # Memory'nin store veya retrieve metodu test edilebilir. Şimdilik sadece store'u test edelim.
                  method_to_test = module_instance.store
                  # create_dummy_method_inputs Memory için store girdisini döndürüyor.
                  logger.debug(f"Memory modülü ('{class_name}') testi: store metodu çağrılıyor.")
                  # Retrieve test edilecekse, create_dummy_method_inputs retrieve girdilerini döndürmeli.
                  # method_to_test = module_instance.retrieve
                  # num_results config'ten alınmalı retrieve için.


             elif class_name == 'CognitionCore':
                  method_to_test = module_instance.decide
             elif class_name == 'DecisionModule':
                  method_to_test = module_instance.decide
             elif class_name == 'MotorControlCore':
                  method_to_test = module_instance.generate_response
             elif class_name == 'ExpressionGenerator':
                  method_to_test = module_instance.generate
             elif class_name == 'InteractionAPI':
                  method_to_test = module_instance.send_output


             # TODO: Diğer modüllerin ana işleme metotları buraya eklenecek.


             if method_to_test is None:
                  logger.error(f"Modül '{class_name}' için test edilecek ana işleme metodu belirlenemedi.")
                  test_success = False
                  output_data = None
             else:
                  try:
                      # Metodu sahte girdi argümanlarıyla çağır.
                      output_data = method_to_test(*dummy_method_inputs) # Tuple'ı unpack ederek argümanları ilet.

                      # Eğer buraya kadar hata olmadıysa, temel işlem başarılı.
                      test_success = True
                      logger.debug(f"'{class_name}' modülünün işleme metodu başarıyla çalıştı.")
                      # Eğer çıktı None değilse logla.
                      if output_data is not None:
                           logger.debug(f"'{class_name}' modülünden çıktı alındı: {output_data}")
                      # else: Çıktı None ise (bazı metotlar için normaldir) loga gerek yok.


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
    # setup_logging None alabilir ve varsayılan (INFO) ile başlar.
    setup_logging(config=None)

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