# src/motor_control/expression.py
import logging
import random # Rastgele seçimler için
from src.core.utils import check_input_not_none

logger = logging.getLogger(__name__)

class ExpressionGenerator:
    def __init__(self, config):
        self.config = config # Config ileride kullanılabilir (örn: dil, ses tonu)
        logger.info("ExpressionGenerator initializing...")
        logger.info("ExpressionGenerator initialized.")

    def generate(self, decision_or_command):
        if not check_input_not_none(decision_or_command, "decision_or_command for ExpressionGenerator", logger) and decision_or_command is not None:
             logger.warning(f"ExpressionGenerator: Received non-string command: {type(decision_or_command)}. Defaulting.")
             decision_or_command = "default_response" # Varsayılana düş
        elif decision_or_command is None:
            logger.debug("ExpressionGenerator: Command is None. Defaulting.")
            decision_or_command = "default_response"

        logger.debug(f"ExpressionGenerator: Generating expression for '{decision_or_command}'")

        output_text = None

        # DecisionModule'den gelen kararlara göre ifadeler
        # Bu anahtarlar DecisionModule.decide() içindeki karar stringleriyle eşleşmeli
        responses = {
            "explore_surroundings": [
                "Etrafıma bir göz atayım.",
               #  "Acaba etrafta ne var?",
               #  "Biraz keşif yapma zamanı."
            ],
            "make_a_random_sound": [
                "Bip bop!",
               #  "Vızzz!",
               #  "Bir ses çıkarıyorum: hummm."
            ],
            "focus_on_new_detail": [
                "Bu ilginç görünüyor, daha yakından bakmalıyım.",
               #  "Şuna odaklanayım.",
               #  "Bu detayı incelemek istiyorum."
            ],
            "react_to_loud_sound": [
                "Bu ses de neydi?!",
               #  "Bir gürültü duydum!",
               #  "Sesli bir şey oldu."
            ],
            "examine_complex_visual": [
                "Bu karmaşık bir görüntü.",
               #  "Burada çok fazla detay var.",
               #  "Bu gördüğüm şeyi anlamaya çalışıyorum."
            ],
            "acknowledge_bright_light": [
                "Ne kadar parlak!",
               #  "Işık gözümü alıyor.",
               #  "Çok aydınlık burası."
            ],
            "acknowledge_darkness": [
                "Burası oldukça karanlık.",
               #  "Işıkları açsak mı?",
               #  "Görüş mesafem azaldı."
            ],
            "observe_familiar_input": [
                "Bunu daha önce görmüştüm.",
               #  "Bu bana tanıdık geliyor.",
               #  "Evet, bunu hatırlıyorum."
            ],
            "perceive_new_stimulus": [
                "Bu da ne? Yeni bir şey.",
               #  "Daha önce böyle bir şeyle karşılaşmamıştım.",
               #  "Yeni bir uyaran algıladım."
            ],
            "default_response": [ # MotorControlCore'dan "default_response" komutu gelirse
                "Ne yapacağımdan emin değilim.",
               #  "Biraz kafam karıştı.",
               #  "Hmm, ilginç."
            ]
            # "interact_with_concept_X" için özel işlem aşağıda
        }

        if decision_or_command in responses:
            output_text = random.choice(responses[decision_or_command])
        elif isinstance(decision_or_command, str) and decision_or_command.startswith("interact_with_concept_"):
            try:
                concept_id_str = decision_or_command.split("_")[-1]
                # concept_id_str'ın sayı olup olmadığını kontrol etmeye gerek yok, string olarak kullanabiliriz.
                output_text = random.choice([
                    f"Ah, bu {concept_id_str} numaralı kavrama benziyor.",
                    # f"Bu {concept_id_str} kavramıyla ilgili bir şeyler düşünüyorum.",
                    # f"Kavram {concept_id_str} ile ne yapabilirim acaba?"
                ])
            except IndexError:
                logger.warning(f"ExpressionGenerator: Malformed 'interact_with_concept_' command: {decision_or_command}")
                output_text = random.choice(responses["default_response"])
        else: # Bilinmeyen bir komut/karar ise
            logger.warning(f"ExpressionGenerator: Unknown command/decision '{decision_or_command}'. Using default response.")
            output_text = random.choice(responses["default_response"])
        
        if output_text:
            logger.debug(f"ExpressionGenerator: Produced output: '{output_text}' for command: '{decision_or_command}'")
        
        return output_text

    def cleanup(self):
        logger.info("ExpressionGenerator cleaning up.")
        pass