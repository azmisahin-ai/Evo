# src/core/module_loader.py
#
# Evo'nın ana modüllerini (Sense, Process, Represent, Memory, Cognition, MotorControl, Interaction)
# başlatmak ve sonlandırmak için yardımcı fonksiyonları içerir.
# Modül başlatma ve temizleme süreçlerini merkezi olarak yönetir ve hataları loglar.

import logging # Loglama için

# Başlatılacak tüm ana modül sınıflarını import et
# TODO: Modül sınıflarının import edilmesi veya referanslarının yönetilmesi
#       daha dinamik hale getirilebilir (örn: config'ten okuma, registry desenini kullanma).
#       Bu, yeni bir modül eklendiğinde module_loader.py'nin güncellenme ihtiyacını azaltır. (Gelecekte refactoring).
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
# 'src.core.module_loader' adında bir logger döndürür.
logger = logging.getLogger(__name__)


def _initialize_single_module(module_name, module_class, config, category_name, is_critical_category):
    """
    Tek bir modülü başlatmayı dener ve sonucu döndürür.

    Başlatma sırasında oluşabilecek hataları yönetir ve loglar.
    Kritik kategorideki modüllerin başlatma hatası durumunu döndürür.

    Args:
        module_name (str): Modülün adı (örn: 'vision', 'core_memory').
        module_class (class): Başlatılacak modül sınıfı.
        config (dict): Genel yapılandırma sözlüğü.
        category_name (str): Modülün ait olduğu kategori adı (örn: 'sensors', 'processors').
        is_critical_category (bool): Bu modülün ait olduğu kategorinin kritik olup olmadığı.

    Returns:
        tuple: (instance, critical_error_occurred)
               instance (object or None): Başlatılan modül objesi veya hata durumunda None.
               critical_error_occurred (bool): Eğer modül kritik bir kategoride ve başlatma
                                               başarısız olduysa True, aksi halde False.
    """
    instance = None
    critical_error_occurred = False
    module_config = config.get(module_name, {}) # Konfigürasyondan ilgili modülün ayarlarını al

    logger.info(f"Başlatılıyor: {module_class.__name__} ({module_name})...")

    try:
        # Modül sınıfının instance'ını oluştur (başlat)
        # Modül başlatıcılarının (__init__ metotlarının) hata durumunda None döndürmesi veya exception atması beklenir.
        instance = module_class(module_config)

        # Başlatma sırasında None döndürülmesi de bir hata göstergesidir.
        if instance is None:
            logger.error(f"Modül '{module_name}' ({module_class.__name__}) başlatma sırasında None döndürdü.")
            # None döndürülmesi durumunda exception atılmadığı için hatayı manuel logladık.
            # Eğer kritik kategoride ise bu bir kritik başlatma hatasıdır.
            if is_critical_category:
                critical_error_occurred = True

        # Eğer instance başarıyla oluşturulduysa (None değilse)
        elif is_critical_category:
            # Kritik bir modül başarıyla başlatıldı.
            logger.info(f"Başarıyla başlatıldı: {module_class.__name__} ({module_name}).")
            # Başarı durumunda critical_error_occurred False kalır.
        else: # Non-kritik kategori
            logger.info(f"Başarıyla başlatıldı: {module_class.__name__} ({module_name}).")
            # Non-kritik modül başarıyla başlatıldı.

    except Exception as e:
        # Başlatma sırasında beklenmedik bir istisna oluşursa
        logger.critical(f"Modül '{module_name}' ({module_class.__name__}) başlatılırken kritik hata oluştu: {e}", exc_info=True)
        instance = None # Hata durumunda objeyi None yap
        # Eğer kritik kategoride ise bu bir kritik başlatma hatasıdır.
        if is_critical_category:
            critical_error_occurred = True
        # Non-kritik kategoride exception olsa bile critical_error_occurred False kalır (initialize_modules logic).

    return instance, critical_error_occurred


def _cleanup_single_module(module_name, module_instance, category_name):
    """
    Tek bir modül objesinin kaynaklarını temizlemeyi dener.

    cleanup veya stop_stream metodunu (varsa) güvenli bir şekilde çağırır.
    Temizleme sırasında oluşabilecek hataları loglar.

    Args:
        module_name (str): Modülün adı.
        module_instance (object or None): Temizlenecek modül objesi. None olabilir.
        category_name (str): Modülün ait olduğu kategori adı.
    """
    # Eğer obje None değilse
    if module_instance:
        # Öncelikle genel cleanup metodunu dene
        if hasattr(module_instance, 'cleanup'):
            logger.info(f"Temizleniyor: {category_name.capitalize()} - {module_name} ({type(module_instance).__name__})...")
            try:
                module_instance.cleanup()
                logger.info(f"Temizlendi: {category_name.capitalize()} - {module_name}.")
            except Exception as e:
                 # Temizleme sırasında hata oluşursa logla.
                 logger.error(f"Temizleme sırasında hata: {category_name.capitalize()} - {module_name}: {e}", exc_info=True)

        # Sensörler için özel olarak stop_stream metodunu çağır (cleanup'ları olmayabilir).
        # TODO: Sensörlerin stop_stream mantığını kendi cleanup metotlarına taşımak daha temiz olur. (Gelecekte refactoring).
        elif category_name == 'sensors' and hasattr(module_instance, 'stop_stream'):
             logger.info(f"Temizleniyor (stream): {category_name.capitalize()} - {module_name} ({type(module_instance).__name__})...")
             try:
                 module_instance.stop_stream()
                 logger.info(f"Temizlendi (stream): {category_name.capitalize()} - {module_name}.")
             except Exception as e:
                  logger.error(f"Temizleme (stream) sırasında hata: {category_name.capitalize()} - {module_name}: {e}", exc_info=True)

        # Eğer obje None değil ama cleanup/stop_stream metotları yoksa bilgilendirici log (DEBUG seviyesinde)
        # elif not hasattr(module_instance, 'cleanup') and not hasattr(module_instance, 'stop_stream'):
        #      logger.debug(f"Temizleme metodu yok: {category_name.capitalize()} - {module_name} ({type(module_instance).__name__}). Temizleme atlandı.")
    # else:
         # Obje zaten None ise temizlemeye gerek yok, loglamaya da gerek yok.
         # logger.debug(f"Temizleniyor: {category_name.capitalize()} - {module_name}: Obje None, temizleme atlandı.")


def initialize_modules(config):
    """
    Verilen yapılandırmaya (config) göre Evo'nın tüm ana modüllerini başlatır.

    Her modül kategorisi için belirlenen sınıfları config'in ilgili bölümünü
    kullanarak _initialize_single_module yardımcı fonksiyonu aracılığıyla başlatır.
    Başlatma sırasında oluşabilecek hataları yönetir ve loglar.
    Kritik kategorideki modüllerin başlatma hatası durumunda ana döngünün çalışmasını engelleyecek
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
                                           Config'te olmayan veya bilinmeyen modül isimleri yoksayılır.
               can_run_main_loop_flag (bool): Ana bilişsel döngünün çalıştırılıp çalıştırılamayacağını belirten bayrak.
                                            Eğer temel pipeline için kritik modüllerden (Processing, Represent, Memory, Cognition, MotorControl)
                                            herhangi biri başlatılamazsa (_initialize_single_module False döndürürse), False döner.
                                            Sensörler ve Interaction'ın başlatılamaması (None instance veya exception)
                                            şimdilik ana döngüyü durdurmaz (Non-kritik başlatma hatası olarak ele alınır).
    """
    logger.info("Modüller başlatılıyor...")

    module_objects = {
        'sensors': {}, 'processors': {}, 'representers': {},
        'memories': {}, 'cognition': {}, 'motor_control': {}, 'interaction': {},
    }

    can_run_main_loop = True # Başlangıçta True, kritik bir hata oluştuğunda False yapılır.

    # Modül kategorileri, içindeki beklenen modül/sınıf eşleştirmeleri ve kritiklik durumları.
    # Bu liste başlatma sırasını da belirler.
    # Sözlük anahtarları: module_objects dict'indeki kategori isimleri.
    # Değerler: (bu kategori kritik mi?, {bu kategorideki modül_adı: ModülSınıfı})
    module_definitions = [
        ('sensors', False, {'vision': VisionSensor, 'audio': AudioSensor}),
        ('processors', True, {'vision': VisionProcessor, 'audio': AudioProcessor}), # Kritik kategori
        ('representers', True, {'main_learner': RepresentationLearner}), # Kritik kategori
        ('memories', True, {'core_memory': Memory}), # Kritik kategori
        ('cognition', True, {'core_cognition': CognitionCore}), # Kritik kategori
        ('motor_control', True, {'core_motor_control': MotorControlCore}), # Kritik kategori
        ('interaction', False, {'core_interaction': InteractionAPI}), # Non-kritik başlatma hatası
    ]

    # Her modül kategorisi ve içindeki modülleri başlatma döngüsü
    for category_name, is_critical_category, module_classes_dict in module_definitions:
        logger.info(f"--- {category_name.capitalize()} Modülleri Başlatılıyor ---")

        # Bu kategori içindeki başlatma hatalarını takip etmek için bayrak
        critical_error_in_category = False

        # Kategori içindeki her modülü _initialize_single_module yardımcı fonksiyonu ile başlatmayı dene
        for module_name, module_class in module_classes_dict.items():
            instance, error_occurred_for_single_module = _initialize_single_module(
                module_name, module_class, config, category_name, is_critical_category
            )
            module_objects[category_name][module_name] = instance # Başlatılan instance'ı veya None'ı kaydet

            if error_occurred_for_single_module:
                critical_error_in_category = True # Eğer tek modül başlatmada hata olduysa kategoride kritik hata oldu olarak işaretle

        # Eğer kritik bir modül kategorisinde başlatma hatası olduysa, ana döngüyü engelle
        if is_critical_category and critical_error_in_category:
             can_run_main_loop = False
             # Kritik hata _initialize_single_module içinde loglandı.
             # Burada sadece ana döngü bayrağının değiştiğini loglayabiliriz.
             # logger.critical(f"Kritik modül kategorisi '{category_name}' başlatılamadığı için ana döngü engelleniyor.")
             # Diğer modül başlatmalarına devam ediyoruz (hata ayıklama için).


    # Sensör başlatma durumu hakkında özel log (hepsi None olsa bile ana döngü engellenmiyor policy'mize göre)
    active_sensors = [name for name, sensor in module_objects.get('sensors', {}).items() # sensors dict'i init hatasıyla None olabilir
                      if sensor and (getattr(sensor, 'is_camera_available', False) or getattr(sensor, 'is_audio_available', False))]
    if not active_sensors and can_run_main_loop: # Ana döngü çalışacak ama sensör yok
         logger.warning("Hiçbir duyusal sensör aktif değil. Evo girdi alamayacak, ancak temel iskelet başlatıldı.")


    # Başlatma durumunu genel olarak logla
    if can_run_main_loop:
         # Sadece kritik modüllerin başlatılması ana döngü için yeterlidir.
         logger.info("Tüm kritik modüller başarıyla başlatıldı. Evo bilişsel döngüye hazır.")
         # Interaction modülü başlatılamadıysa initialize_modules içinde warning loglandı.
         if module_objects.get('interaction', {}).get('core_interaction') is None:
              logger.warning("Interaction modülü başlatılamadı. Evo çıktı veremeyebilir.")

    else:
         # can_run_main_loop False ise, kritik başlatma hatası _initialize_single_module veya initialize_modules içinde loglandı.
         logger.critical("Modül başlatma hataları nedeniyle Evo'nın bilişsel döngüsü başlatılamadı.")


    return module_objects, can_run_main_loop


def cleanup_modules(module_objects):
    """
    initialize_modules fonksiyonu tarafından başlatılan modül kaynaklarını temizler.

    Modül kategorilerini ters başlatma sırasına göre ele alır ve her modülün
    cleanup veya stop_stream metodunu (varsa) _cleanup_single_module yardımcı
    fonksiyonu aracılığıyla güvenli bir şekilde çağırır.
    Temizleme sırasında oluşabilecek hataları loglar.

    Args:
        module_objects (dict): initialize_modules fonksiyonundan dönen modül objeleri sözlüğü.
                               Bu sözlük None objeler içerebilir (başlatma hatası nedeniyle).
                               Fonksiyon bunları güvenle işler.
    """
    logger.info("Evo kaynakları temizleniyor...")

    # Temizleme sırası (başlatma sırasının tersi veya bağımlılıklara göre)
    # interaction -> motor_control -> cognition -> memory -> representers -> processors -> sensors
    # dictionary anahtarları kullanılır.
    module_categories_in_cleanup_order = [
        'interaction', 'motor_control', 'cognition', 'memories', 'representers', 'processors', 'sensors',
    ]

    # Her modül kategorisini temizleme sırasında ele al
    for category_name in module_categories_in_cleanup_order:
        # Eğer initialize_modules bu kategori için bir sözlük oluşturabildiyse (None değilse)
        category_dict = module_objects.get(category_name, {})
        if category_dict: # Sözlük boş değilse veya en azından dict objesi varsa
            # Kategori içindeki her modül objesini _cleanup_single_module yardımcı fonksiyonu ile temizle
            # Sözlük üzerinde dönerken değişiklik yapmamak veya None objeleri atlamak için .items() yerine listesi kullanılır.
            for module_name, module_instance in list(category_dict.items()):
                 # _cleanup_single_module obje None olsa bile güvenli çalışır.
                 _cleanup_single_module(module_name, module_instance, category_name)


        # Sensörler için stop_stream metotları var, cleanup metotları değil (mevcut kodda).
        # Bunları manuel olarak veya cleanup metodlarına taşımalıyız.
        # Mevcut run_evo.py'deki finally bloğunda yapılıyordu, şimdi _cleanup_single_module içine taşıdık.
        # _cleanup_single_module already handles calling stop_stream for 'sensors' category
        # based on hasattr check. So no extra logic needed here anymore.


    logger.info("Tüm Evo kaynakları temizlendi.")