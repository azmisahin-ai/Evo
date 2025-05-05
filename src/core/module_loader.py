# src/core/module_loader.py
#
# Evo'nın ana modüllerini (Sense, Process, Represent, Memory, Cognition, MotorControl, Interaction)
# başlatmak ve sonlandırmak için yardımcı fonksiyonları içerir.
# Modül başlatma sırasında oluşabilecek hataları yönetir.

import logging

# Başlatılacak tüm ana modül sınıflarını import et
# TODO: Bu importların dinamik hale getirilmesi veya bir modül listesi üzerinden yönetilmesi düşünülebilir (Gelecekte refactoring).
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
    Verilen yapılandırmaya (config) göre Evo'nın tüm ana modüllerini başlatır.

    Her modül kategorisi için belirlenen sınıfları config'in ilgili bölümünü
    kullanarak başlatmayı dener. Başlatma sırasında oluşabilecek hataları yakalar,
    loglar ve kritik hatalar durumunda ana döngünün çalışmasını engelleyecek
    bir bayrak döndürür.

    Args:
        config (dict): Modül başlatma için yapılandırma ayarları.
                       Genellikle load_config_from_yaml fonksiyonundan döner.

    Returns:
        tuple: (module_objects_dict, can_run_main_loop_flag)
               module_objects_dict (dict): Başlatılan modül objelerini içeren iç içe sözlük.
                                           Ana anahtarlar modül kategorileridir (örn: 'sensors', 'processors').
                                           İç sözlüklerde modül adı (örn: 'vision', 'audio') anahtar,
                                           başlatılan objesi değerdir. Başlatılamayan modüllerin değeri None olur.
               can_run_main_loop_flag (bool): Ana bilişsel döngünün çalıştırılıp çalıştırılamayacağını belirten bayrak.
                                            Eğer temel pipeline için kritik modüllerden (Processing, Represent, Memory, Cognition, MotorControl)
                                            herhangi biri başlatılamazsa False döner. Sensörler ve Interaction'ın başlatılamaması
                                            şimdilik ana döngüyü durdurmaz (Non-kritik başlatma hatası).
    """
    logger.info("Modüller başlatılıyor...")

    # Modül objelerini kategori bazında depolamak için sözlükler.
    # Her kategori için bir iç sözlük kullanılır.
    module_objects = {
        'sensors': {}, # Duyusal girdiyi yakalayan modüller
        'processors': {}, # Ham duyudan özellik çıkaran modüller
        'representers': {}, # İşlenmiş duyudan içsel temsil öğrenen modüller
        'memories': {}, # Bilgiyi depolayan ve geri çağıran modüller
        'cognition': {}, # Anlama ve karar alma modülleri
        'motor_control': {}, # Kararlardan dışsal tepki üreten modüller
        'interaction': {}, # Dış dünya ile girdi/çıktı arayüzleri
    }

    # Ana bilişsel döngünün çalıştırılıp çalıştırılamayacağını belirten bayrak.
    # Başlangıçta True, kritik bir hata oluştuğunda False yapılır.
    can_run_main_loop = True

    # Modül kategorileri ve içindeki beklenen modül/sınıf eşleştirmeleri.
    # Kritik modüllerin listesi (başlatılamaması can_run_main_loop'u False yapar).
    critical_module_categories = ['processors', 'representers', 'memories', 'cognition', 'motor_control']

    # Modül başlatma sırası (bağımlılıklar nedeniyle belirli bir sıra izlenebilir).
    # Şimdilik Faz sıralamasına göre başlatalım.
    module_init_order = [
        ('sensors', {'vision': VisionSensor, 'audio': AudioSensor}),
        ('processors', {'vision': VisionProcessor, 'audio': AudioProcessor}),
        ('representers', {'main_learner': RepresentationLearner}),
        ('memories', {'core_memory': Memory}),
        ('cognition', {'core_cognition': CognitionCore}),
        ('motor_control', {'core_motor_control': MotorControlCore}),
        ('interaction', {'core_interaction': InteractionAPI}),
    ]

    # Her modül kategorisi ve içindeki modülleri başlatma döngüsü
    for category_name, module_classes in module_init_order:
        logger.info(f"--- Faz {category_name.capitalize()} Modülleri Başlatılıyor ---")

        # Bu kategoride kritik bir hata oldu mu takip etmek için bayrak
        critical_error_in_category = False

        # Kategori içindeki her modülü başlatmayı dene
        for module_name, module_class in module_classes.items():
            logger.info(f"{module_class.__name__} ({module_name}) başlatılıyor...")
            module_config = config.get(module_name, {}) # Konfigürasyondan ilgili modülün ayarlarını al

            try:
                # Modül sınıfının instance'ını oluştur (başlat)
                instance = module_class(module_config)
                module_objects[category_name][module_name] = instance
                logger.info(f"{module_class.__name__} ({module_name}) başarıyla başlatıldı.")

                # Kritik hata kontrolü: Eğer modül kritik bir kategoride ve başlatılan instance None ise (başlatıcı içinde hata oluşmuşsa)
                # Bu kontrol, modül başlatıcılarının hata durumunda None döndürmesi prensibine dayanır.
                if category_name in critical_module_categories and instance is None:
                    logger.critical(f"Kritik modül '{module_name}' ({module_class.__name__}) başlatma sırasında None döndürdü.")
                    critical_error_in_category = True # Bu kategoride kritik hata oldu

            except Exception as e:
                # Başlatma sırasında beklenmedik bir istisna oluşursa
                logger.critical(f"Modül '{module_name}' ({module_class.__name__}) başlatılırken kritik hata oluştu: {e}", exc_info=True)
                module_objects[category_name][module_name] = None # Hata durumunda objeyi None yap
                critical_error_in_category = True # Bu kategoride kritik hata oldu

        # Eğer kritik bir modül kategorisinde hata olduysa, ana döngüyü engelle
        if category_name in critical_module_categories and critical_error_in_category:
             can_run_main_loop = False
             logger.critical(f"Kritik modül kategorisi '{category_name}' başlatılamadığı için ana döngü engelleniyor.")
             # Diğer modül başlatmalarını atlayabiliriz (optimization) veya devam edebiliriz.
             # Devam etmek, hangi diğer modüllerin de hata verdiğini görmemizi sağlar. Devam edelim.


    # Sensör başlatma durumu hakkında özel log (hepsi None olsa bile ana döngü engellenmiyor)
    active_sensors = [name for name, sensor in module_objects['sensors'].items()
                      if sensor and (getattr(sensor, 'is_camera_available', False) or getattr(sensor, 'is_audio_available', False))]
    if not active_sensors and can_run_main_loop: # Ana döngü çalışacak ama sensör yok
         logger.warning("Hiçbir duyusal sensör aktif değil. Evo girdi alamayacak, ancak temel iskelet başlatıldı.")


    # Başlatma durumunu genel olarak logla
    if can_run_main_loop:
         logger.info("Tüm kritik modüller başarıyla başlatıldı. Evo bilişsel döngüye hazır.")
         # Eğer Interaction başlatılamadıysa burada warning loglanabilir veya initialize_modules içinde yapıldıysa gerek yok.
         if module_objects['interaction'].get('core_interaction') is None:
              logger.warning("Interaction modülü başlatılamadı. Evo çıktı veremeyebilir.")

    else:
         logger.critical("Modül başlatma hataları nedeniyle Evo'nın bilişsel döngüsü başlatılamadı.")


    return module_objects, can_run_main_loop


def cleanup_modules(module_objects):
    """
    initialize_modules fonksiyonu tarafından başlatılan modül kaynaklarını temizler.

    Modül kategorilerini ters başlatma sırasına göre ele alır ve her modülün
    cleanup veya stop_stream metodunu (varsa) güvenli bir şekilde çağırır.

    Args:
        module_objects (dict): initialize_modules fonksiyonundan dönen modül objeleri sözlüğü.
    """
    logger.info("Evo kaynakları temizleniyor...")

    # Temizleme sırası (başlatma sırasının tersi veya bağımlılıklara göre)
    # interaction -> motor_control -> cognition -> memory -> representers -> processors -> sensors
    module_categories_in_cleanup_order = [
        'interaction',
        'motor_control',
        'cognition',
        'memories', # Sözlük anahtarı 'memories'
        'representers', # Sözlük anahtarı 'representers'
        'processors',
        'sensors',
    ]

    # Her modül kategorisini temizleme sırasında ele al
    for category_name in module_categories_in_cleanup_order:
        if category_name in module_objects: # Eğer bu kategori initialize_modules'da oluşturulduysa
            category_dict = module_objects[category_name]
            # Kategori içindeki her modül objesini temizle
            # Sözlük üzerinde dönerken değişiklik yapmamak için .items() yerine listesi üzerinde dönülür.
            for module_name, module_instance in list(category_dict.items()):
                # Eğer obje None değilse ve cleanup metodu varsa
                if module_instance and hasattr(module_instance, 'cleanup'):
                    logger.info(f"Temizleniyor: {category_name.capitalize()} - {module_name} ({type(module_instance).__name__})...")
                    try:
                        module_instance.cleanup()
                        logger.info(f"Temizlendi: {category_name.capitalize()} - {module_name}.")
                    except Exception as e:
                         logger.error(f"Temizleme sırasında hata: {category_name.capitalize()} - {module_name}: {e}", exc_info=True)
                    # Hata veren veya temizlenen objeyi sözlükten çıkarmak, tekrar temizlenmesini engeller (isteğe bağlı).
                    # del category_dict[module_name] # Dikkat: Döngü sırasında dict'i değiştirmek sorun yaratabilir


        # Sensörler için stop_stream metotları var, cleanup metotları değil (mevcut kodda).
        # Bunları manuel olarak veya cleanup metodlarına taşımalıyız.
        # Mevcut run_evo.py'deki finally bloğunda yapılıyordu, şimdi buraya taşıdık.
        # İlgili sensör objelerinin varlığını kontrol et.
        if category_name == 'sensors':
             # VisionSensor akışını durdur
             if module_objects['sensors'].get('vision') and hasattr(module_objects['sensors']['vision'], 'stop_stream'):
                 logger.info("VisionSensor akışı durduruluyor...")
                 try:
                     module_objects['sensors']['vision'].stop_stream()
                     logger.info("VisionSensor akışı durduruldu.")
                 except Exception as e:
                      logger.error(f"VisionSensor akış durdurma hatası: {e}", exc_info=True)

             # AudioSensor akışını durdur
             if module_objects['sensors'].get('audio') and hasattr(module_objects['sensors']['audio'], 'stop_stream'):
                  logger.info("AudioSensor akışı durduruluyor...")
                  try:
                      module_objects['sensors']['audio'].stop_stream()
                      logger.info("AudioSensor akışı durduruldu.")
                  except Exception as e:
                       logger.error(f"AudioSensor akış durdurma hatası: {e}", exc_info=True)


    logger.info("Tüm Evo kaynakları temizlendi.")