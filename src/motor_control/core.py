# src/motor_control/core.py
import logging
from src.core.utils import check_input_not_none 
# check_input_type ve cleanup_safely (bu dosyada cleanup_safely zaten doğru kullanılıyor)
from .expression import ExpressionGenerator

logger = logging.getLogger(__name__)

class MotorControlCore:
    def __init__(self, config):
        self.config = config
        logger.info("MotorControlCore initializing...")
        self.expression_generator = None
        try:
             # config.get('motor_control', {}).get('expression', {}) daha doğru olabilir,
             # ama ExpressionGenerator için özel bir config bölümü yoksa genel config'den alabilir.
             # Şimdilik ExpressionGenerator'ın kendi config'ini config.get('expression', {}) ile aldığını varsayalım.
             expression_module_config = config.get('motor_control', {}).get('expression', {}) # MotorControl altındaki expression config'i
             if not expression_module_config and 'expression' in config : # Eğer motor_control altında yoksa, ana config'de mi diye bak
                 expression_module_config = config.get('expression',{})

             self.expression_generator = ExpressionGenerator(expression_module_config)
             if self.expression_generator is None: # ExpressionGenerator __init__ None dönerse (pek olası değil)
                  logger.error("MotorControlCore: ExpressionGenerator initialization returned None.")
        except Exception as e:
             logger.error(f"MotorControlCore: Error initializing ExpressionGenerator: {e}", exc_info=True)
             self.expression_generator = None
        
        logger.info("MotorControlCore initialized.")

    def generate_response(self, decision_str): # Argüman adını decision_str yapalım
        if not check_input_not_none(decision_str, "decision_str for MotorControlCore", logger) and decision_str is not None:
             logger.warning(f"MotorControlCore: Received non-string decision: {type(decision_str)}. Expected string or None.")
             # Karar None değilse ve string değilse, ExpressionGenerator'a "default_response" gönder
             expression_command = "default_response"
        elif decision_str is None:
            logger.debug("MotorControlCore: Decision is None. Generating default response.")
            expression_command = "default_response"
        else: # decision_str geçerli bir string
            # DecisionModule'den gelen kararları ExpressionGenerator komutlarına çevir
            # Veya doğrudan ExpressionGenerator'ın bu kararları işlemesini sağla
            # Şimdilik ExpressionGenerator'ın bu karar string'lerini doğrudan anladığını varsayalım
            # ve ExpressionGenerator'ı buna göre güncelleyelim.
            # Bu, MotorControlCore'daki if/elif karmaşasını azaltır.
            expression_command = decision_str # Kararı doğrudan komut olarak kullan

        output_data = None
        if self.expression_generator:
            try:
                output_data = self.expression_generator.generate(expression_command)
                if output_data is None and expression_command != "default_response": # Eğer özel bir komut için None döndüyse
                    logger.warning(f"MotorControlCore: ExpressionGenerator returned None for command '{expression_command}'. Falling back to default.")
                    output_data = self.expression_generator.generate("default_response") # Tekrar default'u dene
            except Exception as e:
                logger.error(f"MotorControlCore: Error calling ExpressionGenerator.generate for command '{expression_command}': {e}", exc_info=True)
                # Hata durumunda da default response'u dene (eğer expression_generator varsa)
                if self.expression_generator:
                    try:
                        output_data = self.expression_generator.generate("default_response")
                    except Exception as e_default:
                        logger.error(f"MotorControlCore: Error generating default response after an initial error: {e_default}", exc_info=True)
                        output_data = "An internal error occurred in motor control." # Son çare
                else:
                    output_data = "Motor expression system is unavailable."
        else: # ExpressionGenerator yoksa
            logger.error("MotorControlCore: ExpressionGenerator is not available. Cannot generate response.")
            output_data = "I am unable to express myself right now." # Sabit fallback

        if output_data is not None:
            logger.debug(f"MotorControlCore: Generated response for decision '{decision_str}': '{str(output_data)[:100]}...'")
        else: # Eğer output_data hala None ise (örn: default_response bile üretilemediyse)
            logger.error(f"MotorControlCore: Failed to generate any response for decision '{decision_str}'. Output is None.")
            output_data = "No response could be generated." # Nihai fallback

        return output_data

    def cleanup(self):
        logger.info("MotorControlCore cleaning up.")
        if self.expression_generator and hasattr(self.expression_generator, 'cleanup'):
             try:
                 self.expression_generator.cleanup()
             except Exception as e:
                 logger.error(f"MotorControlCore: Error during ExpressionGenerator cleanup: {e}", exc_info=True)
        logger.info("MotorControlCore cleaned up.")