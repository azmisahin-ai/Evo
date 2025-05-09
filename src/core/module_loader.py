# src/core/module_loader.py
import logging
import numpy as np # Bu import doğru ve gerekli
from src.core.utils import run_safely, cleanup_safely
from src.core.config_utils import get_config_value

# Modül sınıfları
from src.senses.vision import VisionSensor
from src.senses.audio import AudioSensor
from src.processing.vision import VisionProcessor
from src.processing.audio import AudioProcessor
from src.representation.models import RepresentationLearner
from src.memory.core import Memory 
from src.cognition.core import CognitionCore 
from src.motor_control.core import MotorControlCore 
from src.interaction.api import InteractionAPI 

logger = logging.getLogger(__name__)

# _initialize_single_module ve _cleanup_single_module fonksiyonları aynı kalabilir (bir önceki mesajdaki gibi)
def _initialize_single_module(module_name, module_class, config, category_name, is_critical_category, **extra_args):
    instance = None
    critical_error_occurred = False
    logger.info(f"Initializing: {module_class.__name__} ({module_name})...")
    
    # Tüm modüllerin __init__(self, full_config) aldığını varsayıyoruz
    # ve kendi config'lerini oradan çektiklerini.
    init_args_for_class = [config] 

    instance = run_safely(
        module_class, 
        *init_args_for_class, 
        **extra_args, 
        logger_instance=logger,
        error_message=f"Critical error initializing module '{module_name}' ({module_class.__name__})",
        error_level=logging.CRITICAL
    )

    if instance is None and is_critical_category:
         critical_error_occurred = True
    elif instance is not None:
        logger.info(f"Successfully initialized: {module_class.__name__} ({module_name}).")
    
    return instance, critical_error_occurred


def _cleanup_single_module(module_name, module_instance, category_name):
    if module_instance:
        method_to_call = None
        method_name = None
        if hasattr(module_instance, 'cleanup'):
            method_to_call = module_instance.cleanup
            method_name = 'cleanup'
        elif category_name == 'sensors' and hasattr(module_instance, 'stop_stream'):
             method_to_call = module_instance.stop_stream
             method_name = 'stop_stream'

        if method_to_call:
            logger.info(f"Cleaning up ({method_name}): {category_name.capitalize()} - {module_name}...")
            cleanup_safely(
                method_to_call, 
                logger_instance=logger,
                error_message=f"Error during cleanup ({method_name}): {module_name}",
                error_level=logging.ERROR
            )
        elif not hasattr(module_instance, 'cleanup') and not hasattr(module_instance, 'stop_stream'):
             logger.debug(f"No cleanup/stop_stream method for: {module_name}. Skipping.")


def initialize_modules(config):
    logger.info("Initializing modules...")
    module_objects = {
        'sensors': {}, 'processors': {}, 'representers': {},
        'memories': {}, 'cognition': {}, 'motor_control': {}, 'interaction': {},
    }
    can_run_main_loop = True

    module_definitions = [
        ('sensors', False, {'vision': VisionSensor, 'audio': AudioSensor}),
        ('processors', True, {'vision': VisionProcessor, 'audio': AudioProcessor}),
        ('memories', True, {'core_memory': Memory}),
        ('cognition', True, {'core_cognition': CognitionCore}),
        ('motor_control', True, {'core_motor_control': MotorControlCore}),
        ('interaction', False, {'core_interaction': InteractionAPI}),
    ]

    # 1. Sensörler ve İşlemciler
    for category_name, is_critical, modules_in_cat in module_definitions:
        if category_name not in ['sensors', 'processors']:
            continue
        logger.info(f"--- Initializing {category_name.capitalize()} Modules ---")
        cat_error = False
        for name, m_class in modules_in_cat.items():
            instance, err = _initialize_single_module(name, m_class, config, category_name, is_critical)
            module_objects[category_name][name] = instance
            if err: cat_error = True
        if is_critical and cat_error: can_run_main_loop = False
    
    if not can_run_main_loop:
        logger.critical("Critical error in Sensor/Processor initialization. Aborting further module loading.")
        return module_objects, False

    # 2. RepresentationLearner için input_dim hesapla
    calculated_input_dim = 0
    expected_input_order = []
    
    vp = module_objects['processors'].get('vision')
    if vp and hasattr(vp, 'get_output_shape_info'):
        shapes = vp.get_output_shape_info()
        for name_key in ['main_image', 'edges']: # Sıra önemli
            if name_key in shapes:
                shape = shapes[name_key] 
                dim_prod = np.prod(shape) 
                calculated_input_dim += dim_prod
                expected_input_order.append(f"vision_{name_key}")
                logger.info(f"Input dim from VisionProcessor '{name_key}': {dim_prod} (shape: {shape})")
    
    ap = module_objects['processors'].get('audio')
    if ap and hasattr(ap, 'get_output_shape_info'):
        shapes = ap.get_output_shape_info() 
        for name_key, shape in shapes.items():
            dim_prod = np.prod(shape)
            calculated_input_dim += dim_prod
            # AudioProcessor.get_output_shape_info() key'i 'audio_features' olmalı.
            # Eğer farklıysa, buradaki 'name_key' yerine sabit bir string kullanılabilir
            # veya AudioProcessor'ın döndürdüğü key'e göre ayarlanabilir.
            # Şimdilik name_key'i kullanalım.
            expected_input_order.append(f"audio_{name_key}") 
            logger.info(f"Input dim from AudioProcessor '{name_key}': {dim_prod} (shape: {shape})")

    if calculated_input_dim == 0:
        logger.error("Calculated input_dim for RepresentationLearner is zero. Cannot proceed.")
        # can_run_main_loop zaten False ise, tekrar False yapmaya gerek yok.
        # Eğer buraya gelindiyse ve calculated_input_dim sıfırsa, bu yeni bir kritik hatadır.
        can_run_main_loop = False 
        return module_objects, can_run_main_loop # Erken çıkış
        
    logger.info(f"Total calculated input_dim for RepresentationLearner: {calculated_input_dim}")
    logger.info(f"Expected feature concatenation order for Learner: {expected_input_order}")

    # Config'i güncelle (RepresentationLearner __init__ buradan okuyacak)
    if 'representation' not in config: config['representation'] = {}
    original_config_input_dim = str(config['representation'].get('input_dim', 'N/A'))
    
    # >>>>>>>>>> DÜZELTME BURADA <<<<<<<<<<
    config['representation']['input_dim'] = int(calculated_input_dim) # Python int'e çevir
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    config['representation']['_expected_input_order_'] = expected_input_order 

    if original_config_input_dim.lower() != "auto" and original_config_input_dim != str(int(calculated_input_dim)): # Karşılaştırmayı da int ile yap
        logger.warning(f"Overriding 'representation.input_dim' from config ({original_config_input_dim}) with calculated: {int(calculated_input_dim)}")
    elif original_config_input_dim.lower() == "auto":
        logger.info(f"Config 'representation.input_dim' ('auto') set to calculated: {int(calculated_input_dim)}")

    # 3. RepresentationLearner'ı Başlat
    logger.info(f"--- Initializing Representers Modules ---")
    category_name, is_critical, modules_in_cat = ('representers', True, {'main_learner': RepresentationLearner})
    cat_error = False
    for name, m_class in modules_in_cat.items():
        instance, err = _initialize_single_module(name, m_class, config, category_name, is_critical)
        module_objects[category_name][name] = instance
        if err: cat_error = True
    if is_critical and cat_error: can_run_main_loop = False

    if not can_run_main_loop:
        logger.critical("Critical error in RepresentationLearner initialization. Aborting further module loading.")
        return module_objects, False

    # 4. Kalan Modülleri Başlat
    for category_name, is_critical, modules_in_cat in module_definitions:
        if category_name in ['sensors', 'processors', 'representers']: 
            continue
        
        # Eğer önceki adımlarda can_run_main_loop False olduysa ve bu kategori kritikse, atla.
        if not can_run_main_loop and is_critical:
            logger.warning(f"Skipping critical category '{category_name}' due to previous critical errors.")
            continue

        logger.info(f"--- Initializing {category_name.capitalize()} Modules ---")
        cat_error = False
        for name, m_class in modules_in_cat.items():
            extra_args = {}
            if m_class == CognitionCore: 
                extra_args['module_objects'] = module_objects
            
            instance, err = _initialize_single_module(name, m_class, config, category_name, is_critical, **extra_args)
            module_objects[category_name][name] = instance
            if err: cat_error = True # Sadece bu modül kritikse can_run_main_loop'u etkiler
        
        if is_critical and cat_error: # Eğer bu kategori kritikse ve içinde hata olduysa
             can_run_main_loop = False 

    # Genel durum logları
    active_sensors_list = []
    # AudioSensor'ın 'is_audio_available' özelliğine sahip olduğunu varsayıyoruz.
    # Eğer yoksa, AudioSensor.py'ye eklenmeli veya buradaki kontrol basitleştirilmeli (sadece instance var mı gibi).
    for sensor_name, sensor_instance in module_objects.get('sensors', {}).items():
        if sensor_instance:
            is_active_flag = False
            if sensor_name == 'vision' and hasattr(sensor_instance, 'is_camera_available') and sensor_instance.is_camera_available:
                is_active_flag = True
            elif sensor_name == 'audio': # AudioSensor için özel durum
                if hasattr(sensor_instance, 'is_audio_available') and sensor_instance.is_audio_available: # Bu özellik AudioSensor'da olmalı
                    is_active_flag = True
                elif not hasattr(sensor_instance, 'is_audio_available'): # Eğer özellik yoksa, instance varlığını kontrol et
                    logger.debug("AudioSensor does not have 'is_audio_available' attribute. Assuming active if instance exists.")
                    is_active_flag = True # Veya daha katı bir kontrol gerekebilir.
            
            if is_active_flag:
                active_sensors_list.append(sensor_name)
    
    if not active_sensors_list and can_run_main_loop: # Eğer ana döngü çalışacak ama aktif sensör yoksa
        logger.warning("No sensory inputs are active. Evo will not receive external data.")

    if can_run_main_loop:
        logger.info("All critical modules initialized successfully. Evo is ready for cognitive loop.")
        if not module_objects.get('interaction', {}).get('core_interaction'):
            logger.warning("Interaction module failed to initialize. Evo might not produce output.")
    else:
        # can_run_main_loop False ise, kritik hata zaten loglanmış olmalı.
        logger.critical("Evo could not start due to critical module initialization errors.")
        
    return module_objects, can_run_main_loop

def cleanup_modules(module_objects):
    logger.info("Cleaning up Evo resources...")
    cleanup_order = ['interaction', 'motor_control', 'cognition', 'memories', 'representers', 'processors', 'sensors']
    for category_name in cleanup_order:
        category_dict = module_objects.get(category_name)
        if category_dict:
            # Temizleme sırasında modül listesini ters çevirerek başlatma sırasının tersini takip et
            for module_name, instance in reversed(list(category_dict.items())): 
                 _cleanup_single_module(module_name, instance, category_name)
    logger.info("Evo resources cleaned up.")