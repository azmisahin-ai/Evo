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


def initialize_modules(config):
    """
    Verilen yapılandırmaya (config) göre Evo'nın tüm ana modüllerini başlatır.

    Her modül kategorisi ve içindeki modüller için belirlenen sınıfları config'in
    ilgili bölümünü kullanarak başlatmayı dener. Başlatma sırasında oluşabilecek
    hataları yakalar, loglar ve kritik hatalar durumunda ana döngünün çalışmasını
    engelleyecek bir bayrak döndürür.

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
                                            herhangi biri başlatılamazsa (exception atma veya None döndürme), False döner.
                                            Sensörler ve Interaction'ın başlatılamaması (exception atma veya None döndürme)
                                            şimdilik ana döngüyü durdurmaz (Non-kritik başlatma hatası olarak ele alınır).
    """
    logger.info("Modüller başlatılıyor...")

    # Modül objelerini kategori bazında depolamak için sözlükler.
    # Her kategori için bir iç sözlük kullanılır.
    module_objects = {
        'sensors': {}, # Duyusal girdiyi yakalayan modüller (Vision, Audio)
        'processors': {}, # Ham duyudan özellik çıkaran modüller (VisionProcessor, AudioProcessor)
        'representers': {}, # İşlenmiş duyudan içsel temsil öğrenen modüller (RepresentationLearner)
        'memories': {}, # Bilgiyi depolayan ve geri çağıran modüller (Memory)
        'cognition': {}, # Anlama ve karar alma modülleri (CognitionCore)
        'motor_control': {}, # Kararlardan dışsal tepki üreten modüller (MotorControlCore)
        'interaction': {}, # Dış dünya ile girdi/çıktı arayüzleri (InteractionAPI)
    }

    # Ana bilişsel döngünün çalıştırılıp çalıştırılamayacağını belirten bayrak.
    # Başlangıçta True, kritik bir hata oluştuğunda False yapılır.
    can_run_main_loop = True

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
        error_during_category_init = False

        # Kategori içindeki her modülü başlatmayı dene
        for module_name, module_class in module_classes_dict.items():
            logger.info(f"Başlatılıyor: {module_class.__name__} ({module_name})...")
            # Konfigürasyondan ilgili modülün ayarlarını al. Yoksa boş dict döndür (.get() ile güvenli erişim).
            module_config = config.get(module_name, {})

            try:
                # Modül sınıfının instance'ını oluştur (başlat)
                # Modül başlatıcılarının ( __init__ metotlarının) hata durumunda None döndürmesi veya exception atması beklenir.
                instance = module_class(module_config)
                module_objects[category_name][module_name] = instance
                logger.info(f"Başarıyla başlatıldı: {module_class.__name__} ({module_name}).")

                # Kritik hata kontrolü: Eğer modül kritik bir kategoride ve başlatılan instance None ise
                # (Modül başlatıcısı exception atmak yerine None döndürmüşse)
                if is_critical_category and instance is None:
                    logger.critical(f"Kritik modül '{module_name}' ({module_class.__name__}) başlatma sırasında None döndürdü.")
                    error_during_category_init = True # Bu kategoride kritik hata oldu

            except Exception as e:
                # Başlatma sırasında beklenmedik bir istisna oluşursa
                logger.critical(f"Modül '{module_name}' ({module_class.__name__}) başlatılırken kritik hata oluştu: {e}", exc_info=True)
                module_objects[category_name][module_name] = None # Hata durumunda objeyi None yap
                error_during_category_init = True # Bu kategoride kritik hata oldu

        # Eğer kritik bir modül kategorisinde başlatma hatası olduysa, ana döngüyü engelle
        if is_critical_category and error_during_category_init:
             can_run_main_loop = False
             # Kritik hata initialize_modules fonksiyonunun dışında run_evo.py'de loglanacak,
             # ama burada da bilgilendirici bir log faydalı olabilir.
             logger.critical(f"Kritik modül kategorisi '{category_name}' başlatılamadığı için ana döngü engelleniyor.")
             # Diğer modül başlatmalarını atlayabiliriz (optimization) veya devam edebiliriz.
             # Devam etmek, hangi diğer modüllerin de hata verdiğini görmemizi sağlar, bu hata ayıklama için iyidir. Devam edelim.


    # Sensör başlatma durumu hakkında özel log (hepsi None olsa bile ana döngü engellenmiyor policy'mize göre)
    active_sensors = [name for name, sensor in module_objects['sensors'].items()
                      if sensor and (getattr(sensor, 'is_camera_available', False) or getattr(sensor, 'is_audio_available', False))]
    if not active_sensors and can_run_main_loop: # Ana döngü çalışacak ama sensör yok
         logger.warning("Hiçbir duyusal sensör aktif değil. Evo girdi alamayacak, ancak temel iskelet başlatıldı.")


    # Başlatma durumunu genel olarak logla
    if can_run_main_loop:
         # Sadece kritik modüllerin başlatılması ana döngü için yeterlidir.
         logger.info("Tüm kritik modüller başarıyla başlatıldı. Evo bilişsel döngüye hazır.")
         # Interaction modülü başlatılamadıysa initialize_modules içinde warning loglandı.

    else:
         # can_run_main_loop False ise, initialize_modules dışında run_evo.py'de kritik hata loglanacak.
         logger.critical("Modül başlatma hataları nedeniyle Evo'nın bilişsel döngüsü başlatılamadı.")


    return module_objects, can_run_main_loop


def cleanup_modules(module_objects):
    """
    initialize_modules fonksiyonu tarafından başlatılan modül kaynaklarını temizler.

    Modül kategorilerini ters başlatma sırasına göre ele alır ve her modülün
    cleanup veya stop_stream metodunu (varsa) güvenli bir şekilde çağırır.
    Temizleme sırasında oluşabilecek hataları loglar.

    Args:
        module_objects (dict): initialize_modules fonksiyonundan dönen modül objeleri sözlüğü.
                               Bu sözlük None objeler içerebilir, fonksiyon bunları güvenle işler.
    """
    logger.info("Evo kaynakları temizleniyor...")

    # Temizleme sırası (başlatma sırasının tersi veya bağımlılıklara göre)
    # interaction -> motor_control -> cognition -> memory -> representers -> processors -> sensors
    # dictionary anahtarları kullanılır.
    module_categories_in_cleanup_order = [
        'interaction',
        'motor_control',
        'cognition',
        'memories', # Sözlük anahtarı initialize_modules'da 'memories' idi.
        'representers', # Sözlük anahtarı initialize_modules'da 'representers' idi.
        'processors',
        'sensors',
    ]

    # Her modül kategorisini temizleme sırasında ele al
    for category_name in module_categories_in_cleanup_order:
        # Eğer initialize_modules bu kategori için bir sözlük oluşturabildiyse
        if category_name in module_objects:
            category_dict = module_objects[category_name]
            # Kategori içindeki her modül objesini temizle
            # Sözlük üzerinde dönerken değişiklik yapmamak veya None objeleri atlamak için .items() yerine listesi kullanılır.
            for module_name, module_instance in list(category_dict.items()):
                # Eğer obje None değilse ve bir temizleme metodu (cleanup veya stop_stream) varsa
                # Modüllerin cleanup metotları varsa onu, sensörler için stop_stream metodunu çağırıyoruz.
                # TODO: Tutarlılık için tüm modüllerde cleanup metodu olması daha iyi olabilir. (Gelecekte refactoring).

                if module_instance: # Obje None değilse
                    # Öncelikle genel cleanup metodunu dene
                    if hasattr(module_instance, 'cleanup'):
                        logger.info(f"Temizleniyor: {category_name.capitalize()} - {module_name} ({type(module_instance).__name__})...")
                        try:
                            module_instance.cleanup()
                            logger.info(f"Temizlendi: {category_name.capitalize()} - {module_name}.")
                        except Exception as e:
                             logger.error(f"Temizleme sırasında hata: {category_name.capitalize()} - {module_name}: {e}", exc_info=True)

                    # Sensörler için özel olarak stop_stream metodunu çağır (cleanup'ları olmayabilir)
                    # TODO: Sensörlerin stop_stream mantığını kendi cleanup metotlarına taşımak daha temiz olur.
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

        # Eğer initialize_modules bu kategori için bir sözlük oluşturamadıysa (örneğin initialize_modules'ın kendi try/except'i içinde hata olduysa)
        # bu kategori atlanacaktır, bu da beklenen davranıştır.


    logger.info("Tüm Evo kaynakları temizlendi.")