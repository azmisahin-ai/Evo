# src/core/module_loader.py
#
# Evo'nın ana modüllerini (Sense, Process, Represent, Memory, Cognition, MotorControl, Interaction)
# başlatmak ve sonlandırmak için yardımcı fonksiyonları içerir.
# Modül başlatma ve temizleme süreçlerini merkezi olarak yönetir ve hataları loglar.

import logging # Loglama için

# Yardımcı fonksiyonları import et
from src.core.utils import run_safely, cleanup_safely # <<< run_safely ve cleanup_safely import edildi

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


def _initialize_single_module(module_name, module_class, config, category_name, is_critical_category, **extra_args):
    """
    Tek bir modülü initialize etmeyi dener ve sonucu döndürür.
    run_safely yardımcı fonksiyonunu kullanarak başlatma sırasında oluşabilecek hataları yönetir.
    **extra_args parametresi ile alt modül initlerine ekstra argümanlar iletilmesini sağlar (örn: module_objects).

    Args:
        module_name (str): Modülün adı (örn: 'vision', 'core_memory').
        module_class (class): Başlatılacak modül sınıfı.
        config (dict): Genel yapılandırma sözlüğü. <<< TÜM CONFIG BURAYA GELİYOR
        category_name (str): Modülün ait olduğu kategori adı (örn: 'sensors', 'processors').
        is_critical_category (bool): Bu modülün ait olduğu kategorinin kritik olup olmadığı.
        **extra_args: Alt modül init metoduna iletilecek ekstra keyword argümanları.

    Returns:
        tuple: (instance, critical_error_occurred)
               instance (object or None): Başlatılan modül objesi veya hata durumunda None.
               critical_error_occurred (bool): Eğer modül kritik bir kategoride ve başlatma
                                               başarısız olduysa True, aksi halde False.
    """
    instance = None
    critical_error_occurred = False
    # Artık modül config'ini burada ayıklamaya gerek yok, tüm config'i init'e vereceğiz.
    # module_config = config.get(category_name, {}).get(module_name, {}) # Bu satır kaldırıldı/değiştirildi

    logger.info(f"Başlatılıyor: {module_class.__name__} ({module_name})...")

    # _initialize_single_module'deki try-except bloğunu run_safely ile değiştir.
    # run_safely, exception yakalarsa None döner ve loglar.
    # Başlatma sırasında hata oluştuğunda log seviyesi CRITICAL olmalı.
    # Alt modül init metotlarına config ve **extra_args dictionary'sindeki anahtarları ilet.
    # <<< DEĞİŞİKLİK: İlk argüman olarak alt config yerine TÜM config'i geçiyoruz >>>
    instance = run_safely(
        module_class, # Çalıştırılacak fonksiyon (sınıf __init__ metodu çağrılır)
        config,       # Fonksiyonun (init) ilk argümanı (TÜM config) <<< DEĞİŞTİ
        **extra_args, # Ekstra keyword argümanları (**extra_args dictionary'sini unpack et)
        logger_instance=logger, # Loglama için bu modülün logger'ını gönder
        error_message=f"Modül '{module_name}' ({module_class.__name__}) başlatılırken kritik hata oluştu", # Log mesajı
        error_level=logging.CRITICAL # Log seviyesi
    )

    # run_safely None döndürdüyse (hata olduysa veya init None döndürdüyse) ve kategori kritikse
    # Eğer instance None ise (hata oldu veya init None döndürdü) ve kategori kritikse, kritik hata oldu.
    if instance is None and is_critical_category:
         critical_error_occurred = True
         # Hata run_safely içinde CRITICAL olarak loglandı.

    # Eğer instance başarıyla oluşturulduysa (None değilse)
    elif instance is not None:
        logger.info(f"Başarıyla başlatıldı: {module_class.__name__} ({module_name}).")
        # Başarı durumunda critical_error_occurred False kalır.


    return instance, critical_error_occurred


def _cleanup_single_module(module_name, module_instance, category_name):
    """
    Tek bir modül objesinin kaynaklarını cleanup_safely yardımcı fonksiyonu ile temizlemeyi dener.

    cleanup veya stop_stream metodunu (varsa) güvenli bir şekilde çağırır.
    Temizleme sırasında oluşabilecek hataları loglar.

    Args:
        module_name (str): Modülün adı.
        module_instance (object or None): Temizlenecek modül objesi. None olabilir.
        category_name (str): Modülün ait olduğu kategori adı.
    """
    # Eğer obje None değilse
    if module_instance:
        # Temizleme metotlarını çağırırken cleanup_safely kullan.
        # Temizleme sırasında hata oluştuğunda log seviyesi ERROR olmalı.
        method_to_call = None
        method_name = None

        # Öncelikle genel cleanup metodunu dene
        if hasattr(module_instance, 'cleanup'):
            method_to_call = module_instance.cleanup
            method_name = 'cleanup'
        # Sensörler için özel olarak stop_stream metodunu çağır (cleanup'ları olmayabilir).
        # TODO: Sensörlerin stop_stream mantığını kendi cleanup metotlarına taşımak daha temiz olur. (Gelecekte refactoring).
        elif category_name == 'sensors' and hasattr(module_instance, 'stop_stream'):
             method_to_call = module_instance.stop_stream
             method_name = 'stop_stream'

        if method_to_call:
            logger.info(f"Temizleniyor ({method_name}): {category_name.capitalize()} - {module_name} ({type(module_instance).__name__})...")
            cleanup_safely(
                method_to_call, # Çalıştırılacak temizleme fonksiyonu
                logger_instance=logger, # Loglama için bu modülün logger'ını gönder
                error_message=f"Temizleme ({method_name}) sırasında hata: {category_name.capitalize()} - {module_name} ({type(module_instance).__name__})", # Log mesajı
                error_level=logging.ERROR # Log seviyesi
            )
        # Eğer obje None değil ama cleanup/stop_stream metotları yoksa bilgilendirici log (DEBUG seviyesinde)
        elif not hasattr(module_instance, 'cleanup') and not hasattr(module_instance, 'stop_stream'):
             logger.debug(f"Temizleme metodu yok: {category_name.capitalize()} - {module_name} ({type(module_instance).__name__}). Temizleme atlandı.")

    # else:
         # Obje zaten None ise temizlemeye gerek yok, loglamaya da gerek yok.
         # logger.debug(f"Temizleniyor: {category_name.capitalize()} - {module_name}: Obje None, temizleme atlandı.")


def initialize_modules(config):
    """
    Verilen yapılandırmaya (config) göre Evo'nın tüm ana modüllerini başlatır.

    Her modül kategorisi için belirlenen sınıfları config'in ilgili bölümünü
    kullanarak _initialize_single_module yardımcı fonksiyonu aracılığıyla başlatır.
    Bazı alt modüllere (örn: CognitionCore içindeki LearningModule) diğer modül objelerinin
    referansları init sırasında iletilmelidir. Bu, burada yönetilir.
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
                                            şimdilik ana döngüyi durdurmaz (Non-kritik başlatma hatası olarak ele alınır).
    """
    logger.info("Modüller başlatılıyor...")

    # Başlatılan modül objelerini tutacak dictionary.
    # Bu dictionary, alt modüllerin init metotlarına 'module_objects' keyword argümanı olarak iletilecek.
    module_objects = {
        'sensors': {}, 'processors': {}, 'representers': {},
        'memories': {}, 'cognition': {}, 'motor_control': {}, 'interaction': {},
    }

    can_run_main_loop = True # Başlangıçta True, kritik bir hata oluştuğunda False yapılır.

    # Modül kategorileri, içindeki beklenen modül/sınıf eşleştirmeleri ve kritiklik durumları.
    # Bu liste başlatma sırasını da belirler.
    # Sözlük anahtarları: module_objects dict'indeki kategori isimleri.
    # Değerler: (bu kategori kritik mi?, {bu kategorideki modül_adı: ModülSınıfı})
    # initialize_modules artık module_objects'ı alt modül initlerine iletebilir.
    module_definitions = [
        ('sensors', False, {'vision': VisionSensor, 'audio': AudioSensor}),
        ('processors', True, {'vision': VisionProcessor, 'audio': AudioProcessor}), # Kritik kategori
        ('representers', True, {'main_learner': RepresentationLearner}), # Kritik kategori
        ('memories', True, {'core_memory': Memory}), # Kritik kategori
        # Cognition kritik kategoridir ve başlatılırken tüm module_objects dictionary'sine ihtiyaç duyar.
        ('cognition', True, {'core_cognition': CognitionCore}),
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
            # CognitionCore başlatılırken module_objects dictionary'sini argüman olarak iletmemiz gerekiyor.
            # Diğer modüller şimdilik sadece config argümanı bekliyor.
            extra_init_args = {}
            if category_name == 'cognition': # Eğer bu kategori Cognition ise
                 # CognitionCore'a diğer modül objelerini içeren dict'i ilet.
                 extra_init_args['module_objects'] = module_objects

            instance, error_occurred_for_single_module = _initialize_single_module(
                module_name, module_class, config, category_name, is_critical_category, **extra_init_args # Extra argümanları ilet.
            )
            module_objects[category_name][module_name] = instance # Başlatılan instance'ı veya None'ı kaydet

            if error_occurred_for_single_module:
                critical_error_in_category = True # Eğer tek modül başlatmada hata olduysa kategoride kritik hata oldu olarak işaretle

        # Eğer kritik bir modül kategorisinde başlatma hatası olduysa, ana döngüyü engelle
        if is_critical_category and critical_error_in_category:
             can_run_main_loop = False
             # Kritik hata _initialize_single_module içinde loglandı.


    # Sensör başlatma durumu hakkında özel log (hepsi None olsa bile ana döngü engellenmiyor policy'mize göre)
    # Aktif sensörleri kontrol et (init başarılı ve stream/kamera aktif ise)
    active_sensors = []
    sensors_dict = module_objects.get('sensors', {})
    if sensors_dict:
        for name, sensor in sensors_dict.items():
            if sensor: # Eğer sensör None değilse
                is_active = False
                if name == 'vision' and hasattr(sensor, 'is_camera_available') and sensor.is_camera_available:
                    is_active = True
                elif name == 'audio' and hasattr(sensor, 'is_audio_available') and sensor.is_audio_available:
                    is_active = True
                # Başka sensör tipleri eklenirse buraya kontrol eklenebilir
                if is_active:
                    active_sensors.append(name)

    if not active_sensors and can_run_main_loop: # Ana döngü çalışacak ama sensör yok
         logger.warning("Hiçbir duyusal sensör aktif değil. Evo girdi alamayacak, ancak temel iskelet başlatıldı.")


    # Başlatma durumunu genel olarak logla
    if can_run_main_loop:
         # Sadece kritik modüllerin başlatılması ana döngü için yeterlidir.
         logger.info("Tüm kritik modüller başarıyla başlatıldı. Evo bilişsel döngüye hazır.")
         # Interaction modülü başlatılamadıysa initialize_modules içinde warning loglandı.
         interaction_instance = module_objects.get('interaction', {}).get('core_interaction')
         if interaction_instance is None:
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
        category_dict = module_objects.get(category_name) # .get() ile al, None olabilir
        if category_dict: # Sözlük None değilse (boş olsa bile)
            # Kategori içindeki her modül objesini _cleanup_single_module yardımcı fonksiyonu ile temizle
            # Sözlük üzerinde dönerken değişiklik yapmamak veya None objeleri atlamak için .items() yerine listesi kullanılır.
            # None objeler _cleanup_single_module içinde zaten atlanıyor.
            for module_name, module_instance in list(category_dict.items()):
                 _cleanup_single_module(module_name, module_instance, category_name)
        else:
            logger.debug(f"Temizleme: '{category_name}' kategorisi bulunamadı veya None, atlanıyor.")


    logger.info("Tüm Evo kaynakları temizlendi.")