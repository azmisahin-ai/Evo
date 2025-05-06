# tests/unit/senses/test_sensors.py

import pytest
import numpy as np
import os
import sys
import logging
# Mocking için pytest-mock kullanıyoruz. 'mocker' fixture'ı otomatik sağlanır.
import cv2 # VisionSensor cv2 kullanıyor, mocklamadan önce import edilmeli.
import pyaudio # AudioSensor pyaudio kullanıyor, mocklamadan önce import edilmeli.
from unittest.mock import MagicMock, call # MagicMock ve call import edildi

# conftest.py sys.path'i ayarlayacak, bu importlar artık doğrudan çalışmalı.
from src.senses.vision import VisionSensor
from src.senses.audio import AudioSensor
from src.core.config_utils import get_config_value
# setup_logging burada çağrılmayacak, conftest tarafından ayarlanacak.
# from src.core.logging_utils import setup_logging # Eğer conftest kullanıyorsa bu import kalmalı.
from src.core.utils import cleanup_safely # Fixture cleanup için


# Bu test dosyası için bir logger oluştur. Seviye conftest tarafından ayarlanacak.
test_logger = logging.getLogger(__name__)
test_logger.info("src.senses modülleri ve gerekli yardımcılar başarıyla içe aktarıldı.")


# VisionSensor Testleri
@pytest.fixture(scope="function")
def dummy_vision_sensor_config():
    """VisionSensor testi için sahte yapılandırma sözlüğü sağlar."""
    config = {
        'vision': {
            'camera_index': 0,       # Test için varsayılan kamera indeksi
            'dummy_width': 640,      # Simüle kare genişliği
            'dummy_height': 480,     # Simüle kare yüksekliği
            'is_dummy': False,       # Gerçek donanım modu simüle ediliyor (mock kullanılacak)
        },
        # ... diğer genel configler ...
    }
    test_logger.debug("Sahte vision sensor config fixture oluşturuldu.")
    return config

@pytest.fixture(scope="function")
def vision_sensor_instance(dummy_vision_sensor_config, mocker):
    """Sahte yapılandırma ve mock'larla VisionSensor örneği sağlar."""
    test_logger.debug("VisionSensor instance oluşturuluyor (mocking cv2.VideoCapture)...")

    # cv2.VideoCapture sınıfını mock'la.
    # VisionSensor.__init__ içindeki cv2.VideoCapture(self.camera_index) çağrısı artık bu mock sınıfı kullanacak.
    mock_cv2_videocapture_class = mocker.patch('cv2.VideoCapture')

    # Mock VideoCapture sınıfının instance'ı (cv2.VideoCapture() çağrısının döndürdüğü obje)
    # VisionSensor'ın self.cap özniteliği bu mock instance olacak.
    mock_cv2_videocapture_instance = mock_cv2_videocapture_class.return_value

    # Mock VideoCapture instance'ının VisionSensor init'inde kullanılan metotlarını yapılandır.
    mock_cv2_videocapture_instance.isOpened.return_value = True # Simüle: Kamera başarılı açıldı

    # get() metodu VisionSensor'ın init'inde kare boyutlarını almak için kullanılıyor.
    # side_effect ile farklı CAP_PROP değerleri için mock dönüşleri ayarla.
    dummy_height = get_config_value(dummy_vision_sensor_config, 'vision', 'dummy_height', default=480, expected_type=int)
    dummy_width = get_config_value(dummy_vision_sensor_config, 'vision', 'dummy_width', default=640, expected_type=int)
    mock_cv2_videocapture_instance.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: dummy_width, # dummy config'den alınan değerleri kullan
        cv2.CAP_PROP_FRAME_HEIGHT: dummy_height, # dummy config'den alınan değerleri kullan
        # VisionSensor init'te başka CAP_PROP değerleri alıyorsa buraya ekle (örn: fps)
        cv2.CAP_PROP_FPS: 30.0, # Varsayılan FPS değeri ekleyelim
    }.get(prop, 0) # Bilinmeyen prop istenirse varsayılan 0 döndür


    # release() metodu VisionSensor'ın cleanup'ında çağrılıyor. Mock'la.
    # assert_called_once() ile cleanup test edilebilir.
    mock_cv2_videocapture_instance.release.return_value = None

    # read() metodu VisionSensor.capture_frame içinde çağrılıyor.
    # Test fonksiyonları bu mock'un dönüş değerini kendi senaryolarına göre ayarlayacak.
    # Varsayılan olarak başarısız okuma döndürsün (test senaryosunda ezilecek).
    mock_cv2_videocapture_instance.read.return_value = (False, None)


    try:
        # VisionSensor'ı başlat. Bu, mock cv2.VideoCapture'ı kullanacak.
        sensor = VisionSensor(dummy_vision_sensor_config)

        # VisionSensor init'i çalışırken mock sınıfların ve metotların doğru çağrıldığını doğrula.
        mock_cv2_videocapture_class.assert_called_once_with(sensor.camera_index)
        mock_cv2_videocapture_instance.isOpened.assert_called_once() # init'te bir kez çağrıldı
        mock_cv2_videocapture_instance.get.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH)
        mock_cv2_videocapture_instance.get.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT)
        # Init sırasında release çağrılmamalı (fixture başarılı başlatmayı simüle ediyor).
        mock_cv2_videocapture_instance.release.assert_not_called()


        test_logger.debug("VisionSensor instance oluşturuldu (mock cv2.VideoCapture ile).")
        yield sensor # Test fonksiyonuna instance'ı ver (self.cap artık mock instance)

        # --- Fixture Teardown (Test Fonksiyonu Bittikten Sonra Çalışır) ---
        # Sensor instance'ının cleanup metotunu çağır (varsa ve çalıştırılıyorsa).
        # VisionSensor'ın cleanup metodu stop_stream'i çağırmalı. stop_stream de self.cap.release()'i çağırmalı.
        if hasattr(sensor, 'cleanup'):
             test_logger.debug("VisionSensor fixture teardown: cleanup çağrılıyor.")
             # Hata oluşursa logla ama testi kırma (fixture teardown hataları test sonucu dışında raporlanır).
             cleanup_safely(sensor.cleanup, logger_instance=test_logger, error_message="VisionSensor instance cleanup sırasında hata (teardown)")
             test_logger.debug("VisionSensor cleanup çağrıldı.")
             # Cleanup'ın mock release metodunu çağırdığını doğrula.
             # Init sırasında çağrılmadıysa, cleanup sırasında BİR kez çağrılmalı.
             mock_cv2_videocapture_instance.release.assert_called_once()
             test_logger.debug("VisionSensor fixture teardown: Mock release çağrısı doğrulandı.")


    except Exception as e:
        # pytest.fail artık exc_info argümanı almayabilir.
        test_logger.error(f"VisionSensor fixture başlatılırken veya cleanup sırasında hata: {e}", exc_info=True)
        pytest.fail(f"VisionSensor fixture hatası: {e}")


def test_vision_sensor_capture_frame_success(vision_sensor_instance, mocker, dummy_vision_sensor_config):
    """
    VisionSensor'ın capture_frame metodunun başarılı bir okuma durumunda
    mocklanmış donanımdan beklenen formatta bir kare döndürdüğünü test eder.
    """
    test_logger.info("test_vision_sensor_capture_frame_success testi başlatıldı.")

    # fixture tarafından oluşturulan VisionSensor instance'ının içindeki mock cap objesini al.
    mock_cap_instance = vision_sensor_instance.cap
    # Mock cap instance'ının read metodunun dönüş değerini ayarla: (başarı bayrağı, kare verisi)
    dummy_height = get_config_value(dummy_vision_sensor_config, 'vision', 'dummy_height', default=480, expected_type=int)
    dummy_width = get_config_value(dummy_vision_sensor_config, 'vision', 'dummy_width', default=640, expected_type=int)
    # Gerçekçi dummy frame verisi
    mock_frame_data = np.random.randint(0, 256, size=(dummy_height, dummy_width, 3), dtype=np.uint8)
    mock_cap_instance.read.return_value = (True, mock_frame_data) # Başarılı okuma simülasyonu


    # capture_frame metodunu çağır
    try:
        captured_frame = vision_sensor_instance.capture_frame()
        test_logger.debug(f"VisionSensor.capture_frame çağrıldı. Çıktı tipi: {type(captured_frame)}")

    except Exception as e:
        # pytest.fail artık exc_info argümanı almayabilir.
        test_logger.error(f"VisionSensor.capture_frame çalıştırılırken hata: {e}", exc_info=True)
        pytest.fail(f"VisionSensor.capture_frame çalıştırılırken beklenmedik hata: {e}")


    # --- Çıktıyı Kontrol Et (Assert) ---
    # capture_frame metodu bir numpy array döndürmeli.
    assert isinstance(captured_frame, np.ndarray), f"Capture çıktısı numpy array olmalı, alınan tip: {type(captured_frame)}"
    test_logger.debug("Assert geçti: Çıktı tipi numpy array.")

    # numpy array beklenen şekil ve dtype'a sahip olmalı.
    expected_shape = (dummy_height, dummy_width, 3) # BGR
    expected_dtype = np.uint8
    assert captured_frame.shape == expected_shape, f"Capture çıktısı beklenen şekle sahip olmalı. Beklenen: {expected_shape}, Alınan: {captured_frame.shape}"
    assert captured_frame.dtype == expected_dtype, f"Capture çıktısı beklenen dtype'a sahip olmalı. Beklenen: {expected_dtype}, Alınan: {captured_frame.dtype}"
    test_logger.debug("Assert geçti: Çıktı beklenen şekil ve dtype'a sahip.")

    # capture_frame içinde mock read metodun bir kez çağrıldığını doğrula.
    mock_cap_instance.read.assert_called_once()
    test_logger.debug("Assert geçti: Mock read metodu bir kez çağrıldı.")

    # Returned array'in mock data ile aynı olduğunu doğrula (VisionSensor'ın capture_frame içinde kareyi işlemediğini varsayarak).
    assert np.array_equal(captured_frame, mock_frame_data)
    test_logger.debug("Assert geçti: Yakalanan kare mock veri ile aynı.")

    # is_camera_available True kalmalı (başarılı okuma oldu).
    assert vision_sensor_instance.is_camera_available, "is_camera_available True kalmalıydı."
    test_logger.debug("Assert geçti: is_camera_available True.")


    test_logger.info("test_vision_sensor_capture_frame_success testi başarıyla tamamlandı.")


def test_vision_sensor_capture_frame_failure(vision_sensor_instance, mocker):
    """
    VisionSensor'ın capture_frame metodunun kare okuma başarısız olduğunda
    beklenen çıktıyı (dummy frame) döndürdüğünü ve simüle moda geçtiğini test eder.
    """
    test_logger.info("test_vision_sensor_capture_frame_failure testi başlatıldı.")

    # fixture tarafından oluşturulan VisionSensor instance'ının içindeki mock cap objesini al.
    mock_cap_instance = vision_sensor_instance.cap
    # Mock cap instance'ının read metodunun dönüş değerini ayarla: (başarısızlık bayrağı, None)
    mock_cap_instance.read.return_value = (False, None) # Okuma başarısız simülasyonu


    # capture_frame metodunu çağır
    try:
        captured_frame = vision_sensor_instance.capture_frame()
        test_logger.debug(f"VisionSensor.capture_frame çağrıldı. Çıktı tipi: {type(captured_frame)}")

    except Exception as e:
        # pytest.fail artık exc_info argümanı almayabilir.
        test_logger.error(f"VisionSensor.capture_frame çalıştırılırken hata: {e}", exc_info=True)
        pytest.fail(f"VisionSensor.capture_frame çalıştırılırken beklenmedik hata: {e}")


    # --- Çıktıyı Kontrol Et (Assert) ---
    # Okuma başarısız olduğunda capture_frame dummy kare döndürmeli.
    assert isinstance(captured_frame, np.ndarray), f"Capture çıktısı numpy array (dummy frame) olmalı, alınan tip: {type(captured_frame)}"
    test_logger.debug("Assert geçti: Çıktı tipi numpy array (dummy frame).")

    # Dummy kare beklenen şekil ve dtype'a sahip olmalı (config'den gelen dummy boyutları).
    dummy_height = get_config_value(vision_sensor_instance.config, 'vision', 'dummy_height', default=480, expected_type=int)
    dummy_width = get_config_value(vision_sensor_instance.config, 'vision', 'dummy_width', default=640, expected_type=int)
    expected_shape = (dummy_height, dummy_width, 3) # BGR
    expected_dtype = np.uint8
    assert captured_frame.shape == expected_shape, f"Capture çıktısı beklenen (dummy) şekle sahip olmalı. Beklenen: {expected_shape}, Alınan: {captured_frame.shape}"
    assert captured_frame.dtype == expected_dtype, f"Capture çıktısı beklenen (dummy) dtype'a sahip olmalı. Beklenen: {expected_dtype}, Alınan: {captured_frame.dtype}"
    test_logger.debug("Assert geçti: Çıktı beklenen (dummy) şekil ve dtype'a sahip.")

    # capture_frame içinde mock read metodun bir kez çağrıldığını doğrula.
    mock_cap_instance.read.assert_called_once()
    test_logger.debug("Assert geçti: Mock read metodu bir kez çağrıldı.")

    # is_camera_available bayrağının False olarak güncellendiğini kontrol et.
    assert not vision_sensor_instance.is_camera_available, "is_camera_available False olarak güncellenmeliydi."
    test_logger.debug("Assert geçti: is_camera_available False olarak güncellendi.")

    test_logger.info("test_vision_sensor_capture_frame_failure testi başarıyla tamamlandı.")

# TODO: VisionSensor init/capture sırasında Exception fırlatıldığında ne olduğu testleri (mock ile simüle ederek).
# TODO: VisionSensor is_dummy=True config'i ile başlatıldığında gerçek donanımı mocklamadığını (veya hiç çağırmadığını), sahte veri ürettiğini test etme.
# TODO: capture_chunk (varsa) metodu testi.
# TODO: cleanup metodu testi (mock release çağrılıyor mu?) -> Fixture teardown'da zaten test ediliyor.


# AudioSensor Testleri
@pytest.fixture(scope="function")
def dummy_audio_sensor_config():
    """AudioSensor testi için sahte yapılandırma sözlüğü sağlar."""
    config = {
        'audio': {
            'audio_rate': 44100,        # Ses örnekleme oranı
            'audio_chunk_size': 1024,   # Her seferinde okunacak ses örneği sayısı
            'audio_input_device_index': None, # Varsayılan cihaz (mocklanacak)
            'is_dummy': False,         # Gerçek donanım modu simüle ediliyor (mock kullanılacak)
        },
        # ... diğer genel configler ...
    }
    test_logger.debug("Sahte audio sensor config fixture oluşturuldu.")
    return config

@pytest.fixture(scope="function")
def audio_sensor_instance(dummy_audio_sensor_config, mocker):
    """Sahte yapılandırma ve mock'larla AudioSensor örneği sağlar."""
    test_logger.debug("AudioSensor instance oluşturuluyor (mocking pyaudio.PyAudio)...")

    # pyaudio.PyAudio sınıfını mock'la.
    # AudioSensor.__init__ içindeki pyaudio.PyAudio() çağrısı artık bu mock sınıfı kullanacak.
    mock_pyaudio_class = mocker.patch('pyaudio.PyAudio')

    # Mock PyAudio sınıfının instance'ı (PyAudio() çağrısının döndürdüğü obje)
    # AudioSensor'ın self.p özniteliği bu mock instance olacak.
    mock_pyaudio_instance = mock_pyaudio_class.return_value

    # Mock PyAudio instance'ının getDefaultInputDeviceInfo() metodunu mock'la (varsayılan cihaz bulunuyor simülasyonu).
    # AudioSensor.__init__ config'de audio_input_device_index=None ise bu metodu çağırır.
    # Mock dönüş değeri, open() metodunda kullanılacak cihaz indeksini simüle eder.
    mock_default_device_info = {'index': 0, 'name': 'Mock Default Audio Device'} # Varsayılan cihaz indeksi 0 olarak simüle edildi.
    mock_pyaudio_instance.getDefaultInputDeviceInfo.return_value = mock_default_device_info

    # Mock PyAudio instance'ının open() metodunu mock'la.
    # Bu open() metodu da bir Stream objesi döndürmeli, o objeyi de mock'la.
    mock_stream_instance = mocker.Mock() # Ayrı bir mock Stream objesi oluştur.
    mock_pyaudio_instance.open.return_value = mock_stream_instance # open() mock'u bu mock stream objesini döndürsün.

    # PyAudio instance'ının terminate() metotunu mock'la (cleanup'ta çağrılıyor).
    mock_pyaudio_instance.terminate.return_value = None


    # Stream objesinin AudioSensor tarafından kullanılan metotlarını mock'la.
    mock_stream_instance.stop_stream.return_value = None
    mock_stream_instance.close.return_value = None
    # read() metodu AudioSensor.capture_chunk içinde çağrılıyor.
    # Test fonksiyonları bu mock'un dönüş değerini kendi senaryolarına göre ayarlayacak.
    # Varsayılan boş baytlar döndürsün (test senaryosunda ezilecek).
    chunk_size = get_config_value(dummy_audio_sensor_config, 'audio', 'audio_chunk_size', default=1024, expected_type=int)
    bytes_per_sample = 2 # int16 için
    mock_stream_instance.read.return_value = b'\x00' * chunk_size * bytes_per_sample


    try:
        # AudioSensor'ı başlat. Bu, mock pyaudio.PyAudio'yu kullanacak.
        sensor = AudioSensor(dummy_audio_sensor_config)

        # AudioSensor init'i çalışırken mock sınıfların ve metotların doğru çağrıldığını doğrula.
        mock_pyaudio_class.assert_called_once() # pyaudio.PyAudio() bir kez çağrılır.

        # Eğer config'de audio_input_device_index None ise getDefaultInputDeviceInfo çağrılır ve open o index ile çağrılır.
        # Eğer config'de int ise getDefaultInputDeviceInfo çağrılmaz ve open o int index ile çağrılır.
        device_index_used_by_sensor = sensor.audio_input_device_index # Config'den None veya int
        if device_index_used_by_sensor is None:
            mock_pyaudio_instance.getDefaultInputDeviceInfo.assert_called_once() # Varsayılan cihaz bilgisi alınıyor mu?
            # AudioSensor code sonra bu default info'dan index'i alıp open'da kullanır.
            # Yani open() metodu getDefaultInputDeviceInfo'dan dönen index ile çağrılacaktır.
            device_index_for_open_call = mock_default_device_info['index'] # Mock getDefaultInputDeviceInfo'nun dönüşündeki index
        else:
            mock_pyaudio_instance.getDefaultInputDeviceInfo.assert_not_called() # Config'de belirtildiyse default info alınmaz.
            device_index_for_open_call = device_index_used_by_sensor # Config'deki int index kullanılır.


        # AudioSensor init'i PyAudio instance'ının open metotunu doğru argümanlarla çağırdığını doğrula.
        expected_open_args = {
            'rate': sensor.audio_rate,
            'channels': 1, # AudioSensor tek kanal ses varsayıyor
            'format': pyaudio.paInt16, # AudioSensor int16 formatı varsayıyor
            'input': True,
            'frames_per_buffer': sensor.audio_chunk_size,
            'input_device_index': device_index_for_open_call # <<< Open çağrısında kullanılan actual index
        }
        mock_pyaudio_instance.open.assert_called_once_with(**expected_open_args)

        # Init sırasında terminate, stop_stream, close çağrılmamalı.
        mock_pyaudio_instance.terminate.assert_not_called()
        mock_stream_instance.stop_stream.assert_not_called()
        mock_stream_instance.close.assert_not_called()


        test_logger.debug("AudioSensor instance oluşturuldu (mock pyaudio ile).")
        yield sensor # Test fonksiyonuna instance'ı ver (self.p, self.stream artık mock instance'lar)

        # --- Fixture Teardown (Test Fonksiyonu Bittikten Sonra Çalışır) ---
        # Sensor instance'ının cleanup metotunu çağır (varsa ve çalıştırılıyorsa).
        # AudioSensor'ın cleanup metodu stop_stream() ve terminate_pyaudio() çağırmalı.
        if hasattr(sensor, 'cleanup'):
             test_logger.debug("AudioSensor fixture teardown: cleanup çağrılıyor.")
             # Hata oluşursa logla ama testi kırma.
             cleanup_safely(sensor.cleanup, logger_instance=test_logger, error_message="AudioSensor instance cleanup sırasında hata (teardown)")
             test_logger.debug("AudioSensor cleanup çağrıldı.")
             # Cleanup'ın mock metotları çağırdığını doğrula.
             # Init sırasında çağrılmadıysa, cleanup sırasında bir kez çağrılmalı.
             mock_pyaudio_instance.terminate.assert_called_once()
             mock_stream_instance.stop_stream.assert_called_once() # stop_stream AudioSensor code'da bir kez çağrılıyor.
             mock_stream_instance.close.assert_called_once() # close AudioSensor code'da bir kez çağrılıyor.
             test_logger.debug("AudioSensor fixture teardown: Mock calls doğrulandı.")


    except Exception as e:
        # pytest.fail artık exc_info argümanı almayabilir.
        test_logger.error(f"AudioSensor fixture başlatılırken veya cleanup sırasında hata: {e}", exc_info=True)
        pytest.fail(f"AudioSensor fixture hatası: {e}")


def test_audio_sensor_capture_chunk_success(audio_sensor_instance, mocker, dummy_audio_sensor_config):
    """
    AudioSensor'ın capture_chunk metodunun başarılı bir okuma durumunda
    mocklanmış donanımdan beklenen formatta bir ses bloğu (chunk) döndürdüğünü test eder.
    """
    test_logger.info("test_audio_sensor_capture_chunk_success testi başlatıldı.")

    # fixture tarafından oluşturulan AudioSensor instance'ının içindeki mock stream objesini al.
    mock_stream_instance = audio_sensor_instance.stream # AudioSensor'da stream objesi self.stream olarak saklanıyor.
    # Mock stream instance'ının read metodunun dönüş değerini ayarla (bytes verisi).
    chunk_size = get_config_value(dummy_audio_sensor_config, 'audio', 'audio_chunk_size', default=1024, expected_type=int)
    bytes_per_sample = 2 # int16 için
    # Sahte raw ses verisi (bytes) - np array oluşturup bytes'a çevirerek daha gerçekçi test verisi oluştur.
    dummy_audio_np = (np.random.rand(chunk_size) * 32767).astype(np.int16) # Simüle int16 ses verisi
    mock_audio_bytes = dummy_audio_np.tobytes()

    mock_stream_instance.read.return_value = mock_audio_bytes # Mock read'den dönecek bytes veri


    # capture_chunk metodunu çağır
    try:
        captured_chunk = audio_sensor_instance.capture_chunk()
        test_logger.debug(f"AudioSensor.capture_chunk çağrıldı. Çıktı tipi: {type(captured_chunk)}")

    except Exception as e:
        # pytest.fail artık exc_info argümanı almayabilir.
        test_logger.error(f"AudioSensor.capture_chunk çalıştırılırken hata: {e}", exc_info=True)
        pytest.fail(f"AudioSensor.capture_chunk çalıştırılırken beklenmedik hata: {e}")


    # --- Çıktıyı Kontrol Et (Assert) ---
    # capture_chunk metodu, okuduğu bayt veriyi numpy array'e çevirmeli (int16).
    assert isinstance(captured_chunk, np.ndarray), f"Capture chunk çıktısı numpy array olmalı, alınan tip: {type(captured_chunk)}"
    test_logger.debug("Assert geçti: Çıktı tipi numpy array.")

    # numpy array beklenen şekil ve dtype'a sahip olmalı (işlenmiş ham ses, int16).
    expected_shape = (chunk_size,)
    expected_dtype = np.int16
    assert captured_chunk.shape == expected_shape, f"Capture chunk çıktısı beklenen şekle sahip olmalı. Beklenen: {expected_shape}, Alınan: {captured_chunk.shape}"
    assert captured_chunk.dtype == expected_dtype, f"Capture chunk çıktısı beklenen dtype'a sahip olmalı. Beklenen: {expected_dtype}, Alınan: {captured_chunk.dtype}"
    test_logger.debug("Assert geçti: Çıktı beklenen şekil ve dtype'a sahip.")

    # capture_chunk içinde mock stream read metodun bir kez doğru argümanla (chunk_size ve exception_on_overflow=False) çağrıldığını doğrula.
    # pyaudio'nun read metodu genelde frames ve exception_on_overflow=False bekler.
    mock_stream_instance.read.assert_called_once_with(chunk_size, exception_on_overflow=False)
    test_logger.debug("Assert geçti: Mock stream read metodu bir kez çağrıldı (beklenen argümanlarla).")

    # Returned array'in mock data'dan doğru şekilde çevrildiğini doğrula.
    # np.frombuffer, bytes'ı numpy array'e çevirir.
    assert np.array_equal(captured_chunk, dummy_audio_np) # bytes verisi int16 numpy array'e doğru çevrildi mi?
    test_logger.debug("Assert geçti: Yakalanan chunk mock veri ile aynı (numpy).")

    # is_audio_available True kalmalı (başarılı okuma oldu).
    assert audio_sensor_instance.is_audio_available, "is_audio_available True kalmalıydı."
    test_logger.debug("Assert geçti: is_audio_available True.")


    test_logger.info("test_audio_sensor_capture_chunk_success testi başarıyla tamamlandı.")

# TODO: AudioSensor capture_chunk sırasında Exception (IOError veya diğeri) fırlatıldığında ne olduğu testleri (mock read'in side_effect'ini ayarlayarak).
# TODO: AudioSensor init sırasında gerçek pyaudio.PyAudio() veya open() hata fırlattığında ne olduğu testleri (mock'ların side_effect'ini ayarlayarak).
# TODO: Farklı audio_input_device_index değerleri ile init testi (mock open çağrısının argümanını assert ederek).
# TODO: is_dummy config'i True ise simüle veri ürettiğini test etme.
# TODO: cleanup metodu testi (mock stop_stream, close, terminate çağrılıyor mu?) -> Fixture teardown'da zaten test ediliyor.