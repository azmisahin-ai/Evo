# src/motor_control/core.py
#
# Evo'nın motor control çekirdeğini temsil eder.
# Bilişsel çekirdekten gelen kararları dış dünyaya yönelik tepkilere (output) dönüştürür.
# ExpressionGenerator, Manipulator, LocomotionController gibi alt modülleri koordine eder.

import logging # Loglama için.

# Yardımcı fonksiyonları import et
from src.core.utils import check_input_not_none, check_input_type, cleanup_safely # utils fonksiyonları kullanılmış

# Alt modül sınıflarını import et
from .expression import ExpressionGenerator # ExpressionGenerator MotorControl'den gelen komutu alıp çıktı üretiyor.
# from .manipulation import Manipulator # Gelecekte kullanılacak
# from .locomotion import LocomotionController # Gelecekte kullanılacak


# Bu modül için bir logger oluştur
logger = logging.getLogger(__name__)


class MotorControlCore:
    """
    Evo'nın motor control çekirdek sınıfı (Koordinatör/Yönetici).

    CognitionCore'dan gelen bir eylem kararını girdi olarak alır.
    Bu karara dayanarak, ExpressionGenerator (metin/ses/görsel çıktı) gibi alt modülleri
    kullanarak dışarıya gönderilecek bir tepki (response) üretir veya
    fiziksel bir eylem (manipülasyon, lokomosyon) gerçekleştirir.
    Mevcut implementasyon: Basit karar stringlerine göre farklı metin tepkileri üretir.
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

        # ExpressionGenerator objesini başlatmayı dene (Alt modül başlatma örneği).
        # Başlatma hatası durumunda objesi None kalır.
        self.expression_generator = None
        try:
             expression_config = config.get('expression', {}) # config'ten alt modül ayarlarını al.
             # ExpressionGenerator init metodu hata durumunda None döndürmeli veya exception atmalı.
             self.expression_generator = ExpressionGenerator(expression_config)
             if self.expression_generator is None:
                  logger.error("MotorControlCore: ExpressionGenerator başlatılamadı (init None döndürdü).")
        except Exception as e:
             logger.error(f"MotorControlCore: ExpressionGenerator başlatılırken beklenmedik hata: {e}", exc_info=True)
             self.expression_generator = None # Hata durumunda None olduğundan emin ol.


        self.manipulator = None # Manipülasyon (robot kolu vb.) modülü objesi (Gelecek TODO).
        self.locomotion_controller = None # Lokomosyon (hareket) modülü objesi (Gelecek TODO).


        # TODO: Gelecekte: Manipulator ve LocomotionController alt modüllerini burada başlatma mantığı eklenecek.
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


        logger.info("MotorControl modülü başlatıldı.")


    def generate_response(self, decision):
        """
        Bilişsel karara dayanarak dış dünyaya bir tepki (response) üretir veya bir eylem gerçekleştirir.

        CognitionCore'dan gelen 'decision' girdisini alır.
        Bu karara göre, ExpressionGenerator gibi alt modülleri kullanarak
        dışarıya gönderilecek bir tepki (output_data) üretir.
        Mevcut implementasyon: Karar stringlerine göre farklı metin tepkileri üretir.
        Karar None ise veya işlenemeyen bir karar ise None döndürür.
        Tepki üretme veya eylem başlatma sırasında hata oluşursa None döndürür.

        Args:
            decision (str or any): Cognition modülünden gelen karar.
                                    Beklenen format: "sound_detected", "complex_visual_detected", "familiar_input_detected", "new_input_detected", "explore_randomly", "make_noise" stringleri veya None.
                                    Gelecekte daha yapısal bir format (örn: dict {'action': '...', 'params': {...}}) beklenir.

        Returns:
            str or any or None: Üretilen tepki (Interaction modülüne iletilecek string, ses verisi, görsel data vb.)
                                 veya hata durumunda ya da tepki üretilemezse None.
        """
        # Hata yönetimi: Karar None ise veya beklenmeyen tipte ise. check_input_not_none kullan.
        # decision string veya None bekleniyor şimdilik.
        if not check_input_not_none(decision, input_name="decision for MotorControl", logger_instance=logger) and decision is not None:
             # Eğer karar None değil ama geçerli tipi değilse (string değilse) uyarı ver.
             logger.warning(f"MotorControlCore.generate_response: Karar beklenmeyen tipte: {type(decision)}. String veya None bekleniyordu.")
             # Bu durumda altındaki else if/else bloklarına düşecektir.
             # Eğer ExpressionGenerator yoksa veya bilinen komutlardan değilse yine varsayılan tepkiye düşer.


        output_data = None # Üretilen çıktıyı tutacak değişken. Başlangıçta None.
        handled_decision = False # Kararın bilinen bir mantıkla ele alınıp alınmadığını tutar.
        expression_command = None # ExpressionGenerator'a gönderilecek komut stringi.

        try:
            # Karar Yönlendirme ve Tepki Üretme Mantığı (Faz 3):
            # Gelen karar stringine göre ExpressionGenerator'a gönderilecek komutu belirle.
            # Öncelik sırası burada MotorControl'de yönetiliyor:
            # explore_randomly/make_noise > sound_detected > complex_visual_detected > familiar_input_detected > new_input_detected > diğer/None
            # Ancak karar önceliği DecisionModule'de belirlendiği için burada sadece kararı ExpressionGenerator komutuna EŞLEŞTİRİYORUZ.

            if decision == "explore_randomly":
                expression_command = "explore_randomly_response"
                handled_decision = True
            elif decision == "make_noise":
                expression_command = "make_noise_response"
                handled_decision = True
            elif decision == "sound_detected":
                expression_command = "sound_detected_response"
                handled_decision = True
            elif decision == "complex_visual_detected":
                expression_command = "complex_visual_response"
                handled_decision = True
            elif decision == "familiar_input_detected":
                expression_command = "familiar_response"
                handled_decision = True
            elif decision == "new_input_detected":
                expression_command = "new_response"
                handled_decision = True
            # TODO: Gelecekte fiziksel eylem kararları (move_forward, grasp_object vb.) buraya eklenecek.
            # elif decision == "move_forward":
            #      if self.locomotion_controller:
            #           self.locomotion_controller.execute_command("forward")
            #           handled_decision = True
            #           output_data = None # Fiziksel eylemlerde çıktı üretmeyebiliriz.
            #      else: logger.warning("MotorControlCore: LocomotionController yok, hareket kararı işlenemedi."); handled_decision = False # İşlenemedi olarak işaretle.


            # Belirlenen ExpressionGenerator komutunu çalıştır (Eğer bir komut belirlendiyse)
            if handled_decision and expression_command is not None:
                 # ExpressionGenerator varsa onu kullan, yoksa varsayılan metni üret (ExpressionGenerator'da tanımlı fallback).
                 if self.expression_generator:
                      output_data = self.expression_generator.generate(expression_command)
                      # generate None döndürürse output_data None kalır, bu kabul edilebilir (örn: ExpressionGenerator o komutu bilmiyorsa).
                 # else: ExpressionGenerator yoksa veya generate None döndürdüyse output_data None kalır.

            # Eğer gelen karar None ise VEYA bilinen bir komuta eşleşmediyse/işlenirken hata olduysa
            if not handled_decision:
                 # Gelen karar None ise (DecisionModule'den None geldiyse) veya burada eşleşmediyse
                 if decision is not None: # Karar None değildi ama eşleşmedi
                      logger.warning(f"MotorControlCore.generate_response: Karar '{decision}' bilinen bir eyleme dönüştürülemedi veya işlenirken hata oluştu. Varsayilan tepki üretiliyor.")

                 # Varsayılan fallback metin yanıtını üret.
                 # ExpressionGenerator varsa varsayılan yanıt için onu kullanmayı dene.
                 if self.expression_generator:
                      output_data = self.expression_generator.generate("default_response") # Varsayılan yanıt için komut
                      # generate None döndürürse output_data None kalır, bu kabul edilebilir.
                 else:
                      output_data = "Ne yapacağımı bilemedim." # ExpressionGenerator yoksa sabit fallback metin.
                 # output_data hala None ise (ExpressionGenerator default_response için None döndürürse), bu döndürülür.


            # TODO: Gelecekte farklı çıktı formatları için (ses, görsel) burada kontrol ve yönlendirme yapılacak.
            # Örneğin, eğer üretilen output_data bir ses array'i ise, bunu Interaction'a 'audio_output' gibi bir anahtarla gönderecek şekilde ayarlama.
            # Şu an ExpressionGenerator sadece string döndürüyor varsayıyoruz.

        except Exception as e:
            # Tepki üretme veya eylem başlatma işlemi sırasında beklenmedik bir hata olursa logla.
            logger.error(f"MotorControlCore.generate_response: Tepki üretme/Eylem başlatma sırasında beklenmedik hata: {e}", exc_info=True)
            return None # Hata durumunda None döndürerek main loop'un çökmesini engelle.

        # Başarılı durumda üretilen tepkiyi veya eylem başlatıldıysa (çıktı None ise) None'ı döndür.
        # output_data, üretilen string yanıt veya None olabilir.
        if output_data is not None:
             # Üretilen çıktı None değilse (örn: metin yanıtıysa) debug logla.
             # Loglama artık ExpressionGenerator'da yapılıyor. Buradaki debug log çok detaylı olabilir.
             # logger.debug(f"MotorControlCore.generate_response: Tepki üretildi. Output: '{output_data}'")
             pass # Loglama ExpressionGenerator'a taşındı.
        # else: Üretilen çıktı None ise (örn: fiziksel eylem veya generate None döndürdü) loga gerek yok.

        return output_data # String metin yanıtı veya None döndürülür.

    def cleanup(self):
        """
        MotorControlCore modülü kaynaklarını temizler.

        Alt modüllerin (ExpressionGenerator, Manipulator, LocomotionController)
        cleanup metotlarını (varsa) çağırır.
        module_loader.py bu metotu program sonlanırken çağrır (varsa).
        """
        logger.info("MotorControl modülü objesi siliniyor...")
        # Alt modüllerin cleanup metotlarını çağır (varsa).
        # cleanup_safely yardımcı fonksiyonunu kullanabiliriz.
        if self.expression_generator and hasattr(self.expression_generator, 'cleanup'):
             cleanup_safely(self.expression_generator.cleanup, logger_instance=logger, error_message="MotorControl: ExpressionGenerator temizlenirken hata")
        if self.manipulator and hasattr(self.manipulator, 'cleanup'):
             cleanup_safely(self.manipulator.cleanup, logger_instance=logger, error_message="MotorControl: Manipulator temizlenirken hata")
        if self.locomotion_controller and hasattr(self.locomotion_controller, 'cleanup'):
             cleanup_safely(self.locomotion_controller.cleanup, logger_instance=logger, error_message="MotorControl: LocomotionController temizlenirken hata")


        logger.info("MotorControl modülü objesi silindi.")