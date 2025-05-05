# src/motor_control/core.py
#
# Evo'nın motor control çekirdeğini temsil eder.
# Bilişsel çekirdekten gelen kararları dış dünyaya yönelik tepkilere (output) dönüştürür.
# ExpressionGenerator, Manipulator, LocomotionController gibi alt modülleri koordine eder.

import logging # Loglama için.

# Yardımcı fonksiyonları import et
from src.core.utils import check_input_not_none, check_input_type, cleanup_safely # utils fonksiyonları kullanılmış

# Alt modül sınıflarını import et (Placeholder sınıflar)
from .expression import ExpressionGenerator # <<< Yeni import
from .manipulation import Manipulator # <<< Yeni import
from .locomotion import LocomotionController # <<< Yeni import


# Bu modül için bir logger oluştur
# 'src.motor_control.core' adında bir logger döndürür.
logger = logging.getLogger(__name__)


class MotorControlCore:
    """
    Evo'nın motor control çekirdek sınıfı (Koordinatör/Yönetici).

    CognitionCore'dan gelen bir eylem kararını girdi olarak alır.
    Bu karara dayanarak, ExpressionGenerator, Manipulator, LocomotionController gibi
    alt modüller aracılığıyla dışarıya gönderilecek bir tepki (response) üretir veya
    fiziksel bir eylem (manipülasyon, lokomosyon) gerçekleştirir.
    Şimdilik çok basit placeholder tepki üretme mantığı içerir.
    Hata durumlarında işlemleri loglar ve programın çökmesini engeller.
    """
    def __init__(self, config):
        """
        MotorControlCore modülünü başlatır.

        Alt modülleri (ExpressionGenerator, Manipulator, LocomotionController) başlatmayı dener.
        Başlatma sırasında hata oluşursa alt modüllerin objeleri None kalabilir.

        Args:
            config (dict): Motor control çekirdek yapılandırma ayarları.
                           Alt modüllere ait ayarlar kendi adları altında beklenir
                           (örn: {'expression': {...}, 'manipulation': {...}, 'locomotion': {...}}).
                           Gelecekte genel çıktı tipi ('text', 'audio', 'visual', 'physical')
                           veya varsayılan sentezleyici/aktüatör ayarları buraya gelebilir.
        """
        self.config = config
        logger.info("MotorControl modülü başlatılıyor...")

        self.expression_generator = None # İfade üretme (metin/ses/görsel çıktı) modülü objesi.
        self.manipulator = None # Manipülasyon (robot kolu vb.) modülü objesi.
        self.locomotion_controller = None # Lokomosyon (hareket) modülü objesi.


        # Alt modülleri başlatmayı dene (Gelecek TODO).
        # Başlatma hataları kendi içlerinde veya _initialize_single_module gibi bir utility ile yönetilmeli.
        # Şu anki module_loader initiate_modules fonksiyonu bu sınıfın init'ini çağırıyor.
        # Alt modüllerin başlatılması initialize_modules içinde değil, burada (ana modül init içinde) olmalıdır.
        # Ancak alt modüllerin init hataları MotorControl modülünün kendisinin başlatılmasını (initialize_modules'da)
        # KRİTİK olarak işaretlememelidir (policy'mize göre MotorControl kritik). Alt modül hatası,
        # MotorControlCore'un kendisi için non-kritik bir hata olabilir.

        # TODO: Alt modüller MotorControl'ün init'i içinde başlatılacak.
        # try:
        #     expression_config = config.get('expression', {})
        #     self.expression_generator = ExpressionGenerator(expression_config)
        #     if self.expression_generator is None: logger.error("MotorControlCore: ExpressionGenerator başlatılamadı.")
        # except Exception as e: logger.error(f"MotorControlCore: ExpressionGenerator başlatılırken hata: {e}", exc_info=True); self.expression_generator = None

        # try:
        #     manipulation_config = config.get('manipulation', {})
        #     self.manipulator = Manipulator(manipulation_config)
        #     if self.manipulator is None: logger.error("MotorControlCore: Manipulator başlatılamadı.")
        # except Exception as e: logger.error(f"MotorControlCore: Manipulator başlatılırken hata: {e}", exc_info=True); self.manipulator = None

        # try:
        #     locomotion_config = config.get('locomotion', {})
        #     self.locomotion_controller = LocomotionController(locomotion_config)
        #     if self.locomotion_controller is None: logger.error("MotorControlCore: LocomotionController başlatılamadı.")
        # except Exception as e: logger.error(f"MotorControlCore: LocomotionController başlatılırken hata: {e}", exc_info=True); self.locomotion_controller = None

        pass # Şimdilik alt modül başlatma yok


        logger.info("MotorControl modülü başlatıldı.")


    def generate_response(self, decision):
        """
        Bilişsel karara dayanarak dış dünyaya bir tepki (response) üretir veya bir eylem gerçekleştirir.

        CognitionCore'dan gelen 'decision' girdisini alır.
        Bu karara göre, ExpressionGenerator, Manipulator veya LocomotionController gibi alt modülleri
        kullanarak dışarıya gönderilecek bir tepki (output_data) üretir veya fiziksel bir eylem başlatır.
        Karar None ise veya işlenemeyen bir karar ise None döndürür.
        Tepki üretme veya eylem başlatma sırasında hata oluşursa None döndürür.

        Args:
            decision (str or any): Cognition modülünden gelen karar.
                                    Şimdilik 'processing_and_remembering' gibi bir string beklenir,
                                    ancak None da olabilir.
                                    Gelecekte daha yapısal bir format (örn: dict {'action': '...', 'params': {...}}) beklenir.

        Returns:
            str or any or None: Üretilen tepki (çıktı olarak Interaction modülüne iletilecek string,
                                 ses verisi, görsel data vb.) veya eylem başlatıldıysa (çıktı yoksa) None,
                                 ya da karar None ise, işlenemeyen bir karar ise ya da
                                 işlem sırasında hata durumunda None.
        """
        # Hata yönetimi: Karar None ise işlem yapma. check_input_not_none kullan.
        if not check_input_not_none(decision, input_name="decision", logger_instance=logger):
             return None # Karar None ise None döndürerek tepki üretmeyi atla.

        # Hata yönetimi: Kararın beklenen tipte (şimdilik str) olup olmadığını kontrol et.
        # Gelecekte karar formatı değişirse burası da değişmeli (örn: dict bekleniyorsa check_input_type(decision, dict, ...)).
        # Eğer karar string değilse uyarı verip işlenemeyen karar gibi ele alalım.
        if not check_input_type(decision, str, input_name="decision", logger_instance=logger):
             logger.warning(f"MotorControlCore.generate_response: Karar beklenmeyen tipte: {type(decision)}. String bekleniyordu.")
             # Bu durumda try bloğundaki else (bilinmeyen karar) kısmına düşecektir veya exception atarsa catch edilir.


        output_data = None # Üretilen çıktıyı tutacak değişken. Başlangıçta None.

        try:
            # TODO: Gelecekte: Gelen kararın tipine ve içeriğine bakarak hangi alt modülün (expression, manipulation, locomotion) kullanılacağına karar ver.
            # Örneğin, eğer karar {'action': 'generate_text', 'content': '...'} gibi bir şeyse ExpressionGenerator'ı kullan.
            # Eğer karar {'action': 'move', 'direction': 'forward'} gibi bir şeyse LocomotionController'ı kullan.
            # Eğer karar {'action': 'grasp', 'target': '...'} gibi bir şeyse Manipulator'ı kullan.

            # Basit Placeholder Karar Yönlendirme ve Tepki Üretme Mantığı:
            # Alt modüller (expression_generator vb.) başlatıldıysa onları kullanmayı dene.
            # Başlatılmadıysa veya karar alt modüllere uygun değilse placeholder mantığı kullan.

            # Gelecekte Kullanım Örneği (Alt Modüller):
            # if self.expression_generator and decision.startswith("generate_"): # Örnek basit yönlendirme
            #    # Kararın geri kalanını alt modüle ilet
            #    expression_command = decision.replace("generate_", "") # Örn: "generate_response_hi" -> "response_hi"
            #    output_data = self.expression_generator.generate(expression_command)
            #    # Eğer ifade üretildiyse (output_data None değilse), bu output_data Interaction'a gönderilecek.
            # elif self.locomotion_controller and decision.startswith("move_"): # Örnek başka yönlendirme
            #    # Hareket komutunu alt modüle ilet. Lokomosyon genellikle bir çıktı döndürmez, sadece eylem yapar.
            #    locomotion_command = decision.replace("move_", "") # Örn: "move_forward" -> "forward"
            #    self.locomotion_controller.execute_command(locomotion_command)
            #    output_data = None # Fiziksel eylem yapıldığında Interaction'a bir çıktı göndermeyebiliriz.

            # else: # Alt modüllere uygun bir karar değilse veya alt modüller başlatılamadıysa:
            # Geçici placeholder mantığı (alt modüller yokken veya başlatılamadıysa kullanılır):
            if decision == "processing_and_remembering":
                # Eğer karar 'processing_and_remembering' ise bu sabit metin yanıtını üret.
                output_data = "Çevreyi algılıyorum ve hatırlıyorum." # Basit metin yanıtı.
            # Gelecekte eklenebilecek diğer karar türleri için örnekler:
            # elif decision == "greet": # Eğer karar "selamla" ise
            #      output_data = "Merhaba! Nasılsın?"
            # elif decision == "detect_object": # Eğer karar "nesne tespit edildi" ise
            #      output_data = "Bir şey görüyorum."
            # elif decision.startswith("response_"): # Eğer karar belirli bir yanıt şablonu ise
            #      response_key = decision.split("_")[1] # Örneğin "response_affirmative" -> "affirmative"
            #      output_data = self._get_templated_response(response_key) # Bir şablon fonksiyonu kullan.
            else:
                 # Gelen karar beklenmeyen veya mevcut mantıkla işlenemeyen bir karar ise.
                 # check_input_type ile string değilse zaten loglandı. Burada geçerli ama bilinmeyen string kararlar ele alınır.
                 logger.warning(f"MotorControlCore.generate_response: Bilinmeyen veya işlenemeyen karar: '{decision}'. Varsayilan tepki üretiliyor.")
                 output_data = "Ne yapacağımı bilemedim." # Bilinmeyen kararlar için varsayılan tepki.

            # TODO: Gelecekte farklı çıktı formatları için (ses, görsel) burada kontrol ve yönlendirme yapılacak.
            # Eğer üretilen output_data beklenmeyen bir formatta ise (örn: metin beklerken sayı gelmişse) hata logla.
            # if output_data is not None and not check_input_type(output_data, str, ...):
            #      logger.error(...) # Üretilen çıktı tipi yanlışsa hata.


            # DEBUG logu: Üretilen tepki (None değilse).
            # if output_data is not None: # Zaten None değilse buraya gelinir.
            #      logger.debug(f"MotorControlCore.generate_response: Tepki üretildi. Output: '{output_data}'")


        except Exception as e:
            # Tepki üretme veya eylem başlatma işlemi sırasında beklenmedik bir hata olursa logla.
            # Örneğin, string birleştirme hataları veya gelecekteki sentezleyici/aktüatör çağrı hataları.
            logger.error(f"MotorControlCore.generate_response: Tepki üretme/Eylem başlatma sırasında beklenmedik hata: {e}", exc_info=True)
            return None # Hata durumunda None döndürerek main loop'un çökmesini engelle.

        # Başarılı durumda üretilen tepkiyi veya eylem başlatıldıysa None'ı döndür.
        return output_data

    def cleanup(self):
        """
        MotorControlCore modülü kaynaklarını temizler.

        Alt modüllerin (ExpressionGenerator, Manipulator, LocomotionController)
        cleanup metotlarını (varsa) çağırır.
        module_loader.py bu metodu program sonlanırken çağırır (varsa).
        """
        logger.info("MotorControl modülü objesi siliniyor...")
        # Alt modüllerin cleanup metotlarını çağır (varsa).
        # cleanup_safely yardımcı fonksiyonunu kullanabiliriz.
        if self.expression_generator:
             cleanup_safely(self.expression_generator.cleanup, logger_instance=logger, error_message="MotorControl: ExpressionGenerator temizlenirken hata")
        if self.manipulator:
             cleanup_safely(self.manipulator.cleanup, logger_instance=logger, error_message="MotorControl: Manipulator temizlenirken hata")
        if self.locomotion_controller:
             cleanup_safely(self.locomotion_controller.cleanup, logger_instance=logger, error_message="MotorControl: LocomotionController temizlenirken hata")


        logger.info("MotorControl modülü objesi silindi.")