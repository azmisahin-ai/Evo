# src/motor_control/core.py
#
# Evo'nın motor kontrol çekirdeğini temsil eder.
# Bilişsel çekirdekten gelen kararları dış dünyaya yönelik tepkilere (output) dönüştürür.

import logging # Loglama için.

# Yardımcı fonksiyonları import et
from src.core.utils import check_input_not_none, check_input_type # <<< Yeni importlar


# Bu modül için bir logger oluştur
# 'src.motor_control.core' adında bir logger döndürür.
logger = logging.getLogger(__name__)

# motor_control/expression.py dosyası şu an sadece placeholder olabilir.
# Gelecekte bu sınıflar veya fonksiyonlar burada import edilecek ve kullanılacak.
# from .expression import ExpressionGenerator # Gelecek TODO


class MotorControlCore:
    """
    Evo'nın motor kontrol çekirdek sınıfı.

    CognitionCore'dan gelen bir eylem kararını girdi olarak alır.
    Bu karara dayanarak, Evo'nın dış dünyaya (Interaction modülü aracılığıyla)
    ileteceği bir tepki (response) üretir.
    Şimdilik çok basit placeholder tepki üretme mantığı içerir.
    Gelecekte metin sentezi, ses sentezi, görsel çıktı üretimi veya
    fiziksel eylem komutları gibi yetenekler eklenecek.
    Hata durumlarında işlemleri loglar ve programın çökmesini engeller.
    """
    def __init__(self, config):
        """
        MotorControlCore modülünü başlatır.

        Args:
            config (dict): Motor kontrol çekirdek yapılandırma ayarları.
                           Gelecekte çıktı tipi ('text', 'audio', 'visual') veya
                           sentezleyici/aktüatör ayarları buraya gelebilir.
        """
        self.config = config
        logger.info("MotorControl modülü başlatılıyor...")
        # Alt modüllerin başlatılması buraya gelebilir (Gelecek TODO).
        # self.expression_generator = ExpressionGenerator(config.get('expression', {})) # Gelecek

        logger.info("MotorControl modülü başlatıldı.")

    def generate_response(self, decision):
        """
        Bilişsel karara dayanarak dış dünyaya bir tepki (response) üretir.

        CognitionCore'dan gelen 'decision' string'ini girdi olarak alır.
        Bu karara göre basit bir çıktı string'i (tepki) üretir.
        Karar None veya işlenemeyen bir karar ise None döndürür.
        Tepki üretme sırasında hata oluşursa None döndürür.

        Args:
            decision (str or None): Cognition modülünden gelen karar.
                                    Şimdilik 'processing_and_remembering' gibi bir string beklenir,
                                    ancak None da olabilir.
                                    Gelecekte daha yapısal bir format (örn: dict) beklenir.

        Returns:
            str or None: Üretilen tepki (çıktı olarak Interaction modülüne iletilecek bir string)
                         veya karar None ise, işlenemeyen bir karar ise ya da
                         tepki üretme sırasında hata durumunda None.
                         Gelecekte daha yapısal bir çıktı formatı beklenir (örn: {'type': 'text', 'content': '...'}).
        """
        # Hata yönetimi: Karar None ise işlem yapma. check_input_not_none kullan.
        if not check_input_not_none(decision, input_name="decision", logger_instance=logger):
             return None # Karar None ise None döndürerek tepki üretmeyi atla.

        # Hata yönetimi: Kararın beklenen tipte (şimdilik str) olup olmadığını kontrol et.
        # Gelecekte karar formatı değişirse burası da değişmeli.
        # Eğer karar string değilse uyarı verip işlenemeyen karar gibi ele alalım.
        if not check_input_type(decision, str, input_name="decision", logger_instance=logger):
             logger.warning(f"MotorControlCore.generate_response: Karar beklenmeyen tipte: {type(decision)}. String bekleniyordu.")
             # Bu durumda işlenemeyen karar mantığına düşecektir try bloğu içinde.
             # Ya da burada None döndürebiliriz: return None
             # Şimdilik loglayıp devam etsin ve try bloğundaki else'e düşsün.


        response_output = None # Üretilen çıktıyı tutacak değişken. Başlangıçta None.

        try:
            # Basit Placeholder Tepki Üretme Mantığı:
            # Gelen karar string'ine göre önceden tanımlanmış basit bir metin tepkisi üret.
            # Gelecekte: Karara göre daha dinamik ve karmaşık metinler, sesler, görseller veya fiziksel eylemler üretilecek.
            if decision == "processing_and_remembering":
                # Eğer karar 'processing_and_remembering' ise bu sabit metin yanıtını üret.
                response_output = "Çevreyi algılıyorum ve hatırlıyorum." # Basit metin yanıtı.
            # Gelecekte eklenebilecek diğer karar türleri için örnekler:
            # elif decision == "greet": # Eğer karar "selamla" ise
            #      response_output = "Merhaba! Nasılsın?"
            # elif decision == "detect_object": # Eğer karar "nesne tespit edildi" ise
            #      response_output = "Bir şey görüyorum."
            # elif decision.startswith("response_"): # Eğer karar belirli bir yanıt şablonu ise
            #      response_key = decision.split("_")[1] # Örneğin "response_affirmative" -> "affirmative"
            #      response_output = self._get_templated_response(response_key) # Bir şablon fonksiyonu kullan.
            else:
                 # Gelen karar beklenmeyen veya mevcut mantıkla işlenemeyen bir karar ise.
                 # check_input_type ile string değilse zaten loglandı. Burada geçerli ama bilinmeyen string kararlar ele alınır.
                 logger.warning(f"MotorControlCore.generate_response: Bilinmeyen veya işlenemeyen karar: '{decision}'. Varsayilan tepki üretiliyor.")
                 response_output = "Ne yapacağımı bilemedim." # Bilinmeyen kararlar için varsayılan tepki.


            # Gelecekte Kullanım Örneği (Alt Modüller):
            # if decision['type'] == 'text_response':
            #    response_output = self.expression_generator.generate_text(decision['content']) # Metin üretme alt modülü
            # elif decision['type'] == 'play_sound':
            #    response_output = self.expression_generator.generate_sound(decision['sound_id']) # Ses üretme alt modülü
            # elif decision['type'] == 'move':
            #    self.locomotion_module.execute_movement(decision['params']) # Hareket alt modülü (bu durumda None dönebilir)


            # DEBUG logu: Üretilen tepki (None değilse).
            # if response_output is not None: # Zaten None değilse buraya gelinir.
            #      logger.debug(f"MotorControlCore.generate_response: Motor kontrol tepki üretti (placeholder). Output: '{response_output}'")


        except Exception as e:
            # Tepki üretme işlemi sırasında beklenmedik bir hata oluşursa logla.
            # Örneğin, string birleştirme hataları veya gelecekteki sentezleyici/aktüatör çağrı hataları.
            logger.error(f"MotorControlCore.generate_response: Tepki üretme sırasında beklenmedik hata: {e}", exc_info=True)
            return None # Hata durumunda None döndürerek main loop'un çökmesini engelle.

        # Başarılı durumda üretilen tepkiyi döndür.
        return response_output

    def cleanup(self):
        """
        MotorControlCore modülü kaynaklarını temizler.

        Şimdilik özel bir kaynak kullanmadığı için temizleme adımı içermez,
        sadece bilgilendirme logu içerir.
        Gelecekte alt modüllerin (ExpressionGenerator vb.) cleanup metotları varsa
        burada çağrılmaları gerekir.
        module_loader.py bu metodu program sonlanırken çağırır (varsa).
        """
        logger.info("MotorControl modülü objesi siliniyor...")
        # Alt modüllerin cleanup metotlarını çağır (varsa) (Gelecek TODO).
        # if hasattr(self.expression_generator, 'cleanup'): self.expression_generator.cleanup()

        logger.info("MotorControl modülü objesi silindi.")