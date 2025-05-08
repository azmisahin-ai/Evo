# src/interaction/api.py
#
# Evo'nın dış dünya ile iletişim arayüzünü temsil eder.
# Motor Control'den gelen tepkileri alır ve aktif çıktı kanallarına gönderir.
# Farklı çıktı kanallarını yönetir.
# Kanal başlatma, gönderme ve temizleme sırasında oluşabilecek hataları yönetir.
# Gelecekte dış dünyadan girdi de alacak (Input kanalları).

import logging # For logging.
# import threading # Might be needed if Web API runs as a thread (Future).
# import requests # Might be needed for sending output to API endpoints (Future).
# import json # Might be needed for JSON format (Future).

# Import utility functions
from src.core.config_utils import get_config_value
from src.core.utils import check_input_not_none, check_input_type # <<< check_input_not_none, check_input_type imported
from .output_channels import ConsoleOutputChannel, WebAPIOutputChannel, OutputChannel # Import channel classes


# Create a logger for this module
# Returns a logger named 'src.interaction.api'.
logger = logging.getLogger(__name__)


class InteractionAPI:
    """
    Evo's external world communication interface class.

    Receives responses (output_data) from the MotorControl module and sends them to enabled output channels.
    Manages different output channels.
    Handles potential errors during channel initialization, sending, and cleanup.
    Will also handle external input in the future (Input channels).
    """
    def __init__(self, config):
        """
        Initializes the InteractionAPI module.

        Reads active output channels from config and initializes them.
        Manages potential errors during channel initialization.

        Args:
            config (dict): Full configuration settings for the system.
                           InteractionAPI will read its relevant sections from this dict,
                           specifically settings under 'interaction'.
        """
        self.config = config # InteractionAPI receives the full config
        logger.info("InteractionAPI module initializing...")

        # Get the list of enabled channels from configuration using get_config_value.
        # Check for list type for enabled_channels. Default is ['console'].
        # Corrected: Use default= keyword format.
        # Based on config, these settings are under the 'interaction' key.
        self.enabled_channels = get_config_value(config, 'interaction', 'enabled_channels', default=['console'], expected_type=list, logger_instance=logger)

        # Get channel-specific settings from configuration using get_config_value.
        # Check for dictionary type for channel_configs. Default is {}.
        # Corrected: Use default= keyword format.
        self.channel_configs = get_config_value(config, 'interaction', 'channel_configs', default={}, expected_type=dict, logger_instance=logger)

        # Ensure enabled_channels is a list even if get_config_value returned None or wrong type (which it shouldn't with expected_type).
        # This is a safety check, but expected_type should prevent None.
        if not isinstance(self.enabled_channels, list):
             logger.error("InteractionAPI: enabled_channels config value is not a valid list. Using empty channel list.")
             self.enabled_channels = []


        self.output_channels = {} # Dictionary to hold initialized active output channel objects.

        logger.info(f"InteractionAPI: Enabled channels from config: {self.enabled_channels}")

        # Mapping of supported output channel class names to their classes.
        # This dictionary must be updated as new channel types are added.
        channel_classes = {
            'console': ConsoleOutputChannel, # Defined in output_channels.py
            'web_api': WebAPIOutputChannel, # Defined in output_channels.py (Placeholder implementation exists)
            # Future channels added here (e.g., 'file': FileOutputChannel, 'robot': RobotOutputChannel)
        }

        # Try to initialize each active channel specified in the configuration.
        # The type of self.enabled_channels list is already checked by get_config_value.
        # Now check if each item in the list is a string.
        for channel_name in self.enabled_channels:
            if not isinstance(channel_name, str):
                 logger.warning(f"InteractionAPI: Unexpected item type in 'enabled_channels' list: {type(channel_name)}. Expected string. Skipping this item.")
                 continue # Skip this item if it's not a string.

            # Check if the corresponding channel class is defined.
            channel_class = channel_classes.get(channel_name)
            if channel_class:
                # Channel class found, now try to initialize it.
                # Get the specific config for this channel from channel_configs, use an empty dict if none exists.
                # self.channel_configs is ensured to be a dict (by get_config_value). get() is safe.
                channel_config = self.channel_configs.get(channel_name, {})
                try:
                    # Create and initialize the channel object.
                    # Channel init methods are expected to return None on failure or raise exceptions.
                    # Pass the channel-specific config dict to the channel's __init__.
                    channel_instance = channel_class(channel_config)
                    # If initialized successfully (is not None)
                    if channel_instance is not None:
                         self.output_channels[channel_name] = channel_instance
                         logger.info(f"InteractionAPI: OutputChannel '{channel_name}' initialized successfully.")
                    else:
                         # If channel init returned None (meaning it handled its own error internally)
                         logger.error(f"InteractionAPI: OutputChannel '{channel_name}' returned None during initialization.")

                except Exception as e:
                    # If an unexpected exception occurred while initializing the channel.
                    # Policy: This type of error during initialization is considered non-critical,
                    # the channel will simply be unavailable.
                    logger.error(f"InteractionAPI: Error during OutputChannel '{channel_name}' initialization: {e}", exc_info=True)
                    # It is important NOT to add the failed channel to the active channels dictionary.


            else:
                # Warning for channel names in config that do not have a corresponding class in channel_classes.
                logger.warning(f"InteractionAPI: Unknown OutputChannel name in config: '{channel_name}'. Skipping this channel.")

        # Log the overall status of the initialized InteractionAPI module.
        # Show the list of active output channels.
        logger.info(f"InteractionAPI module initialized. Active Output Channels: {list(self.output_channels.keys())}")

        # If the Web API channel is active, the logic to start the API server could go here (in a separate thread/process?).
        # This logic has been moved to the InteractionAPI.start() method.
        # if 'web_api' in self.output_channels and hasattr(self.output_channels['web_api'], 'start_server'):
        #      logger.info("InteractionAPI: Starting Web API server...")
        #      self.output_channels['web_api'].start_server() # Method to start the Web API

        # TODO: Logic to initialize input channels will go here (Future TODO).
        # self._initialize_input_channels() # Future TODO


    # ... (send_output, start, stop methods - same as before) ...


    def send_output(self, output_data):
        """
        MotorControl'den gelen çıktıyı (tepki) tüm aktif çıktı kanallarına gönderir.

        Her aktif OutputChannel objesinin `send` metodunu çağırır.
        Bir kanala gönderme sırasında hata oluşsa bile diğer kanallara gönderme devam eder.
        Gönderilecek veri None ise işlem yapmaz.

        Args:
            output_data (any): Motor Control modülünden gelen gönderilecek çıktı verisi (tepki).
                               Formatı kanaldan kanala değişebilir (str, dict, numpy array vb.).
                               None olabilir, bu durumda gönderme atlanır.
        """
        # Hata yönetimi: Gönderilecek veri None ise işlem yapma. check_input_not_none kullan.
        if not check_input_not_none(output_data, input_name="output_data", logger_instance=logger):
            # Çıktı verisi None ise, gönderme işlemi atlanır. Bu bir hata değil.
            return

        # DEBUG logu: Çıktının hangi kanallara gönderileceği.
        logger.debug(f"InteractionAPI: Çikti {list(self.output_channels.keys())} kanallarına gönderiliyor.")

        # Her aktif kanala çıktıyı gönderme döngüsü.
        # Hata yönetimi: Bir kanala gönderme hatası diğerlerini etkilememeli.
        # output_channels sözlüğü üzerinde dönerken, send metodu içinde bu sözlükte değişiklik
        # yapılmadığı varsayılır. Eğer send metodu kanalı pasifize edip sözlükten silerse,
        # döngünün bir kopyası üzerinde dönmek (list(self.output_channels.items())) daha güvenli olabilir.
        for channel_name, channel_instance in list(self.output_channels.items()):
            # channel_instance'ın None olup olmadığını kontrol et (başlatma hatası nedeniyle None olabilir).
            if channel_instance:
                try:
                    # Kanalın send metodunu çağır.
                    # send metotlarının kendi içlerinde de hata yakalama olmalı.
                    # Burada yakaladığımız hata, send metodunun kendisinin çağrılması sırasında
                    # veya send metodunun içindeki *işlenmemiş* bir hatadan kaynaklanır.
                    channel_instance.send(output_data)
                    # DEBUG logu: Hangi kanala gönderim yapıldığı.
                    # logger.debug(f"InteractionAPI: Çikti OutputChannel '{channel_name}' kanalına gönderildi.")

                except Exception as e:
                    # Kanalın send metodunu çağırırken veya çalıştırırken beklenmedik bir hata oluşursa logla.
                    logger.error(f"InteractionAPI: OutputChannel '{channel_name}' send metodu çalıştırılırken beklenmedik hata: {e}", exc_info=True)
                    # Hata veren bu kanalın bir daha kullanılmaması için aktif listesinden çıkarılması düşünülebilir
                    # (Gelecekteki bir iyileştirme/policy). Şu an sadece loglayıp devam ediyoruz.
                    # del self.output_channels[channel_name] # Döngü sırasında dict'i değiştirmek sorun yaratabilir!
            # else:
                 # Eğer channel_instance None ise, bu zaten başlatma sırasında loglanmıştır.
                 # Burada tekrar loglamaya gerek yok.
                 # logger.debug(f"InteractionAPI: OutputChannel '{channel_name}' objesi None, gönderme atlandı.")


    def start(self):
        """
        Interaction arayüzlerini başlatır (örn: Web API sunucusu).

        Bu metot, run_evo.py tarafından program başlatıldığında çağrılır.
        Şimdilik placeholder. Eğer bir API servisi thread veya process olarak
        çalışacaksa burası kullanılacak.
        """
        logger.info("InteractionAPI başlatılıyor (Placeholder)...")
        # Eğer Web API kanalı aktifse ve sunucuyu başlatma yeteneği varsa, burası çağrılabilir.
        # if 'web_api' in self.output_channels and hasattr(self.output_channels['web_api'], 'start_server'):
        #      logger.info("InteractionAPI: Web API sunucusu başlatılıyor...")
        #      self.output_channels['web_api'].start_server() # Web API başlatma metodu


    def stop(self):
        """
        Interaction arayüzlerini durdurur ve tüm aktif çıktı kanallarının kaynaklarını temizler.

        Bu metot, run_evo.py tarafından program sonlanırken çağrılır.
        """
        logger.info("InteractionAPI durduruluyor...")
        # Tüm aktif çıktı kanallarının cleanup metotlarını çağır.
        # Sözlük üzerinde dönerken değişiklik yapmamak için list() kullanarak bir kopya üzerinde dönülür.
        for channel_name, channel_instance in list(self.output_channels.items()):
            # Eğer obje None değilse (başlatma hatası olmadıysa) ve cleanup metodu varsa
            if channel_instance and hasattr(channel_instance, 'cleanup'):
                logger.info(f"InteractionAPI: OutputChannel '{channel_name}' temizleniyor...")
                try:
                    channel_instance.cleanup()
                    logger.info(f"InteractionAPI: OutputChannel '{channel_name}' temizlendi.")
                except Exception as e:
                     # Temizleme sırasında beklenmedik bir hata oluşursa logla.
                     logger.error(f"InteractionAPI: OutputChannel '{channel_name}' temizlenirken hata oluştu: {e}", exc_info=True)
                # Hata veren veya temizlenen kanalı listeden çıkarmak (isteğe bağlı).
                # if channel_name in self.output_channels: # Zaten kopya üzerinde dönüyoruz, bu kontrol gerekli olmayabilir.
                #      del self.output_channels[channel_name]

        # Aktif kanallar sözlüğünü tamamen boşalt (temizlendiğini belirtmek için).
        self.output_channels = {} # Veya self.output_channels = None

        # Eğer API sunucusu bir thread/process olarak başlatıldıysa, durdurma mantığı buraya gelecek (Gelecek TODO).
        # if hasattr(self, 'api_thread') and self.api_thread and self.api_thread.is_alive():
        #      logger.info("InteractionAPI: API sunucusu durduruluyor...")
        #      self.api_thread.stop() # API thread'inin stop metodu olmalı
        #      self.api_thread.join() # Thread'in bitmesini bekle


        logger.info("InteractionAPI objesi silindi.")


    # Gelecekte:
    # def receive_input(self):
    #     """Dış dünyadan girdi alır (örn: API endpoint'inden gelen istek)."""
    #     pass # Implement edilecek input alma mekanizmaları