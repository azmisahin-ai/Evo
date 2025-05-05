# src/core/module_loader.py
import logging

# Başlatılacak tüm modül sınıflarını import et
# TODO: Bu importların Config'ten veya daha dinamik bir şekilde yönetilmesi düşünülebilir (gelecekte)
from src.senses.vision import VisionSensor
from src.senses.audio import AudioSensor
from src.processing.vision import VisionProcessor
from src.processing.audio import AudioProcessor
from src.representation.models import RepresentationLearner # models.py dosyasındaki RepresentationLearner sınıfı
from src.memory.core import Memory # core.py dosyasındaki Memory sınıfı
from src.cognition.core import CognitionCore # core.py dosyasındaki CognitionCore sınıfı
from src.motor_control.core import MotorControlCore # core.py dosyasındaki MotorControlCore sınıfı
from src.interaction.api import InteractionAPI # api.py dosyasındaki InteractionAPI sınıfı


# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)


def initialize_modules(config):
    """
    Verilen yapılandırmaya göre Evo'nın tüm ana modüllerini başlatır.
    Başlatma sırasında oluşabilecek hataları yönetir ve loglar.

    Args:
        config (dict): Modül başlatma için yapılandırma ayarları.

    Returns:
        tuple: (module_objects_dict, can_run_main_loop_flag)
               module_objects_dict (dict): Başlatılan modül objelerini içeren sözlük.
                                           Kategori isimleri anahtar, o kategorinin sözlüğü değerdir.
                                           (örn: {'sensors': {'vision': VisionSensor_obj, 'audio': AudioSensor_obj}, ...})
                                           Başlatılamayan modüllerin değeri None olur.
               can_run_main_loop_flag (bool): Ana döngünün çalıştırılıp çalıştırılamayacağını belirten bayrak.
                                            Kritik bir hata (Process, Represent, Memory, Cognition, MotorControl başlatma hatası)
                                            olduğunda False döner.
    """
    logger.info("Modüller başlatılıyor...")

    # Modül objelerini depolamak için sözlükler
    module_objects = {
        'sensors': {},
        'processors': {},
        'representers': {},
        'memories': {},
        'cognition': {},
        'motor_control': {},
        'interaction': {},
    }

    can_run_main_loop = True # Başlatma başarılı olduysa main loop'u çalıştırmak için flag


    # --- Faz 0: Duyusal Sensörleri Başlat ---
    logger.info("Faz 0: Duyusal sensörler başlatılıyor...")
    try:
        logger.info("VisionSensor başlatılıyor...")
        module_objects['sensors']['vision'] = VisionSensor(config.get('vision', {}))
        if not (module_objects['sensors'].get('vision') and getattr(module_objects['sensors']['vision'], 'is_camera_available', False)):
             logger.warning("VisionSensor tam başlatılamadı veya kamera açılamadı. Simüle edilmiş girdi kullanılacak.")
    except Exception as e:
        logger.critical(f"VisionSensor başlatılırken kritik hata oluştu: {e}", exc_info=True)
        module_objects['sensors']['vision'] = None


    try:
        logger.info("AudioSensor başlatılıyor...")
        module_objects['sensors']['audio'] = AudioSensor(config.get('audio', {}))
        if not (module_objects['sensors'].get('audio') and getattr(module_objects['sensors']['audio'], 'is_audio_available', False)):
             logger.warning("AudioSensor tam başlatılamadı veya ses akışı aktif değil.")
    except Exception as e:
        logger.critical(f"AudioSensor başlatılırken kritik hata oluştu: {e}", exc_info=True)
        module_objects['sensors']['audio'] = None


    # Aktif sensör kontrolü
    active_sensors = [name for name, sensor in module_objects['sensors'].items()
                      if sensor and (getattr(sensor, 'is_camera_available', False) or getattr(sensor, 'is_audio_available', False))]
    if active_sensors:
        logger.info(f"Duyusal Sensörler başlatıldı ({', '.join(active_sensors)} aktif).")
    else:
        logger.warning("Hiçbir duyusal sensör aktif değil. Evo girdi alamayacak.")
        # Sensörlerin hiçbiri olmaması ana döngüyü engellemeyebilir ama anlamını azaltır.
        # Şimdilik kritik hata değil, ama ileride policy olarak eklenebilir.


    # --- Faz 1 Başlangici: Processing Modülleri Başlat ---
    logger.info("Faz 1: Processing modülleri başlatılıyor...")
    try:
        # Eğer ilgili sensör başlatılamadıysa işlemcisini de başlatmanın anlamı yok, None olarak bırakabiliriz.
        # Veya işlemci kendi içinde None girdiyi yönetmeli. Mevcut işlemciler None girdiyi yönetiyor.
        # O yüzden sensör None olsa bile işlemciyi başlatmayı deneyebiliriz.
        module_objects['processors']['vision'] = VisionProcessor(config.get('processing_vision', {}))
        module_objects['processors']['audio'] = AudioProcessor(config.get('processing_audio', {}))
        logger.info("Processing modülleri başarıyla başlatıldı.")
        # Kritik hata kontrolü: En az bir işlemci objesi None mu?
        if any(v is None for v in module_objects['processors'].values()):
             can_run_main_loop = False
             logger.critical("Bazı Processing modülleri başlatılamadı. Ana döngü engelleniyor.")

    except Exception as e:
         logger.critical(f"Processing modülleri başlatılırken kritik hata oluştu: {e}", exc_info=True)
         can_run_main_loop = False


    # --- Faz 1 Devami: Representation Modülleri Başlat ---
    logger.info("Faz 1: Representation modülleri başlatılıyor...")
    if can_run_main_loop: # Önceki fazlarda kritik hata yoksa dene
        try:
            module_objects['representers']['main_learner'] = RepresentationLearner(config.get('representation', {}))
            logger.info("Representation modülü başarıyla başlatıldı.")
            # Kritik hata kontrolü
            if module_objects['representers']['main_learner'] is None:
                 can_run_main_loop = False
                 logger.critical("Representation modülü başlatılamadı. Ana döngü engelleniyor.")

        except Exception as e:
            logger.critical(f"Representation modülü başlatılırken kritik hata oluştu: {e}", exc_info=True)
            can_run_main_loop = False
    else:
         logger.warning("Önceki modüller başlatılamadığı için Representation modülleri atlandı.")


    # --- Faz 2 Başlangici: Memory Modülü Başlat ---
    logger.info("Faz 2: Memory modülü başlatılıyor...")
    if can_run_main_loop: # Önceki fazlarda kritik hata yoksa dene
        try:
            module_objects['memories']['core_memory'] = Memory(config.get('memory', {}))
            logger.info("Memory modülü başarıyla başlatıldı.")
             # Kritik hata kontrolü
            if module_objects['memories']['core_memory'] is None:
                 can_run_main_loop = False
                 logger.critical("Memory modülü başlatılamadı. Ana döngü engelleniyor.")
        except Exception as e:
            logger.critical(f"Memory modülü başlatılırken kritik hata oluştu: {e}", exc_info=True)
            can_run_main_loop = False
    else:
         logger.warning("Önceki modüller başlatılamadığı için Memory modülü atlandı.")

    # --- Faz 3 Başlangici: Cognition Modülü Başlat ---
    logger.info("Faz 3: Cognition modülü başlatılıyor...")
    if can_run_main_loop: # Önceki fazlarda kritik hata yoksa dene
        try:
            module_objects['cognition']['core_cognition'] = CognitionCore(config.get('cognition', {}))
            logger.info("Cognition modülü başarıyla başlatıldı.")
             # Kritik hata kontrolü
            if module_objects['cognition']['core_cognition'] is None:
                 can_run_main_loop = False
                 logger.critical("Cognition modülü başlatılamadı. Ana döngü engelleniyor.")
        except Exception as e:
            logger.critical(f"Cognition modülü başlatılırken kritik hata oluştu: {e}", exc_info=True)
            can_run_main_loop = False
    else:
         logger.warning("Önceki modüller başlatılamadığı için Cognition modülü atlandı.")

    # --- Faz 3 Devami: Motor Control Modülü Başlat ---
    logger.info("Faz 3: Motor Control modülü başlatılıyor...")
    if can_run_main_loop: # Önceki fazlarda kritik hata yoksa dene
        try:
            module_objects['motor_control']['core_motor_control'] = MotorControlCore(config.get('motor_control', {}))
            logger.info("Motor Control modülü başarıyla başlatıldı.")
             # Kritik hata kontrolü
            if module_objects['motor_control']['core_motor_control'] is None:
                 can_run_main_loop = False
                 logger.critical("Motor Control modülü başlatılamadı. Ana döngü engelleniyor.")
        except Exception as e:
            logger.critical(f"Motor Control modülü başlatılırken kritik hata oluştu: {e}", exc_info=True)
            can_run_main_loop = False
    else:
         logger.warning("Önceki modüller başlatılamadığı için Motor Control modülü atlandı.")

    # --- Faz 3 Tamamlanması: Interaction Modülü Başlat ---
    # Interaction modülü Motor Control'e bağımlı ama başlatılamaması ana döngüyü engellemeyebilir.
    # O yüzden bu modülün başlatılması can_run_main_loop'u False yapmamalı (Non-kritik gibi ele alıyoruz başlatma hatasını).
    logger.info("Faz 3: Interaction modülü başlatılıyor...")
    try:
        module_objects['interaction']['core_interaction'] = InteractionAPI(config.get('interaction', {}))
        logger.info("Interaction modülü başarıyla başlatıldı.")
         # Kritik hata kontrolü (Interaction'ı başlatılamaması ana döngüyü durdurmamalı policy'mize göre)
        if module_objects['interaction']['core_interaction'] is None:
             logger.warning("Interaction modülü başlatılamadı. Evo çıktı veremeyebilir.")

    except Exception as e:
        logger.critical(f"Interaction modülü başlatılırken hata oluştu: {e}", exc_info=True)
        module_objects['interaction']['core_interaction'] = None # Hata durumında objeyi None yap


    # Tüm ana pipeline modül kategorileri başarıyla başlatıldı mı kontrolü (eğer can_run_main_loop True ise)
    if can_run_main_loop:
         # Sadece ana pipeline modüllerinin (Process, Represent, Memory, Cognition, MotorControl)
         # kategori sözlüklerinin içlerindeki objelerin None olmaması gerekiyor.
         # Sensörlerin ve Interaction'ın None olması ana döngüyü durdurmuyor policy'mize göre.
         all_critical_objects_ok = all(module_objects['processors'].values()) and \
                                   all(module_objects['representers'].values()) and \
                                   all(module_objects['memories'].values()) and \
                                   all(module_objects['cognition'].values()) and \
                                   all(module_objects['motor_control'].values())

         # Ayrıca en az bir sensörün aktif olması da mantıksal bir gereklilik olabilir,
         # ama bu şu an can_run_main_loop'u False yapmıyor.
         # if not active_sensors: logger.warning("Hiçbir sensör aktif değil, Evo girdi alamayacak.")


         if all_critical_objects_ok: # Tüm kritik modül objeleri var mı?
             logger.info("Tüm ana pipeline modül kategorileri başarıyla başlatıldı. Evo bilişsel döngüye hazır.")
             # Interaction modülü başlatılamadıysa zaten yukarıda warning loglandı.
         else:
              # Bu durum should not happen if can_run_main_loop is True with current logic,
              # unless some non-critical module init fails silently (which we handle with logging).
              logger.warning("Bazı temel ana pipeline modül kategorileri başlatılamadı veya eksik. Evo bilişsel döngüsü sınırlı çalışabilir.")


    else:
         logger.critical("Modül başlatma hataları nedeniyle Evo'nın bilişsel döngüsü başlatılamadı.")


    return module_objects, can_run_main_loop # Başlatılan objeleri ve bayrağı döndür


def cleanup_modules(module_objects):
    """
    Başlatılan modül kaynaklarını temizler.

    Args:
        module_objects (dict): initialize_modules fonksiyonundan dönen modül objeleri sözlüğü.
    """
    logger.info("Evo kaynakları temizleniyor...")
    # Ters sırada temizlemek genellikle iyi bir fikirdir (bağımlılıklar nedeniyle)
    module_categories_in_cleanup_order = ['interaction', 'motor_control', 'cognition', 'memory', 'representers', 'processors', 'sensors']

    for category_name in module_categories_in_cleanup_order:
        if category_name in module_objects:
            category_dict = module_objects[category_name]
            for module_name, module_instance in list(category_dict.items()): # Liste kopyası üzerinde dönmek güvenlidir
                if module_instance and hasattr(module_instance, 'cleanup'):
                    logger.info(f"Temizleniyor: {category_name.capitalize()} - {module_name}...")
                    try:
                        module_instance.cleanup()
                        logger.info(f"Temizlendi: {category_name.capitalize()} - {module_name}.")
                    except Exception as e:
                         logger.error(f"Temizleme sırasında hata: {category_name.capitalize()} - {module_name}: {e}", exc_info=True)
                    # Temizlenen veya hata veren objeyi sözlükten çıkarabiliriz (isteğe bağlı)
                    # del category_dict[module_name] # Dikkat: Döngü sırasında dict'i değiştirmek sorun yaratabilir


        # Sensörler için stop_stream metotları var, cleanup metotları değil (mevcut kodda)
        # Stop stream çağrıları manuel yapılmalı veya cleanup metoduna taşınmalı.
        # Mevcut run_evo.py'deki finally bloğunda yapılıyordu, şimdi buraya taşıyalım.
        if category_name == 'sensors':
             if module_objects['sensors'].get('vision') and hasattr(module_objects['sensors']['vision'], 'stop_stream'):
                 logger.info("VisionSensor akışı durduruluyor...")
                 try:
                     module_objects['sensors']['vision'].stop_stream()
                     logger.info("VisionSensor akışı durduruldu.")
                 except Exception as e:
                      logger.error(f"VisionSensor akış durdurma hatası: {e}", exc_info=True)

             if module_objects['sensors'].get('audio') and hasattr(module_objects['sensors']['audio'], 'stop_stream'):
                  logger.info("AudioSensor akışı durduruluyor...")
                  try:
                      module_objects['sensors']['audio'].stop_stream()
                      logger.info("AudioSensor akışı durduruldu.")
                  except Exception as e:
                       logger.error(f"AudioSensor akış durdurma hatası: {e}", exc_info=True)


    logger.info("Tüm Evo kaynakları temizlendi.")