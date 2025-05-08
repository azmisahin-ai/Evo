# src/interaction/output_channels.py
#
# Defines Evo's output channels to the external world.
# Different types of outputs (text, audio, visual) are directed to specific channels.

import logging # For logging.
# import time # For timing if needed.
# import requests # Might be needed for sending requests to WebAPI (Future).
# import json # Might be needed for JSON format (Future).
# import flask # Might be needed for WebAPI server (Future).

# Import utility functions
from src.core.config_utils import get_config_value
from src.core.utils import check_input_type # <<< check_input_type imported


# Create a logger for this module
# Returns a logger named 'src.interaction.output_channels'.
logger = logging.getLogger(__name__)

# --- OutputChannel Base Class ---
class OutputChannel:
    """
    Base class for different output channels.

    All specific output channel classes (ConsoleOutputChannel, WebAPIOutputChannel, etc.)
    should inherit from this class. It defines base initialization (__init__),
    output sending (send), and resource cleanup (cleanup) methods.
    """
    def __init__(self, name, config):
        """
        Initializes the base of an OutputChannel.

        Each channel has a name and its specific configuration. It creates its own logger.
        Checks the type of the config input.

        Args:
            name (str): The name of the channel (e.g., 'console', 'web_api'). Expected type: string.
            config (dict): Specific configuration settings for this channel. Expected type: dictionary.
                           This is typically the sub-dictionary from main_config.yaml for this channel.
        """
        # Error handling: Is name a string? It's usually expected to be a string when initialized.
        # check_input_type(name, str, input_name="channel name", logger_instance=logger) # Raising TypeError might be more appropriate

        # Error handling: Is config a dict? Use check_input_type.
        # This check is done in the base class __init__ so it's consistent.
        if not check_input_type(config, dict, input_name=f"{name} channel config", logger_instance=logger):
             # Log a warning if config is not a dict and use an empty dict instead.
             logger.warning(f"OutputChannel '{name}': Configuration has unexpected type: {type(config)}. Dictionary expected. Using empty dictionary {{}}.")
             config = {} # Use an empty dictionary if the type is invalid.


        self.name = name
        self.config = config # Store the channel-specific config
        # Each channel creates its own named logger.
        # The logger name will be 'src.interaction.output_channels.ChannelName'.
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.logger.info(f"OutputChannel '{self.name}' initializing.")
        # Base class initialization is complete.

    def send(self, output_data):
        """
        Method to send processed output to the relevant channel.

        This method is not implemented in the base class and must be overridden
        by subclasses with their specific logic.
        The format of output_data varies by channel.

        Args:
            output_data (any): The output data to be sent, typically from Motor Control.
                               Its format depends on the channel (string, number, dict, numpy array, etc.).

        Raises:
            NotImplementedError: If subclasses do not implement this method.
        """
        # General input checks here are not meaningful as the format varies.
        # Subclasses' send methods should validate their own inputs (output_data).
        raise NotImplementedError("Subclasses must implement the 'send' method.")

    def cleanup(self):
        """
        Cleans up resources used by the channel.

        This method is defined as a placeholder in the base class. Subclasses that use
        specific resources (files, network connections, threads, etc.) must override this method
        to implement their cleanup logic.
        Called by module_loader.py and InteractionAPI.stop() when the program terminates (if it exists).
        """
        self.logger.info(f"OutputChannel '{self.name}' cleaning up.")
        pass # By default, there's nothing to clean up.


# --- Console Output Channel ---
class ConsoleOutputChannel(OutputChannel):
    """
    An output channel that writes output to the system console (terminal).

    Typically used for text-based outputs.
    """
    def __init__(self, config):
        """
        Initializes the ConsoleOutputChannel.

        Args:
            config (dict): Channel configuration settings for this channel.
                           Currently, no specific settings are expected for the console channel.
                           The base class checks the config type.
        """
        # Call the base class's __init__ method. Set the channel name to "console".
        # Pass the specific config dict received from InteractionAPI.
        super().__init__("console", config)
        self.logger.info("ConsoleOutputChannel initialized.")

    def send(self, output_data):
        """
        Writes the output to the console.

        Attempts to convert the incoming output_data to a string and prints it to the console
        with a standard format. Catches and logs errors during conversion or printing.

        Args:
            output_data (any): The data to be printed to the console. A string is typically expected.
                               If not a string, it attempts to convert using str().
        """
        # Error handling: Check if the incoming data is a string. Use check_input_type.
        # If not a string, log a warning and try to convert to string.
        if not check_input_type(output_data, str, input_name="output_data for Console", logger_instance=self.logger):
             self.logger.warning(f"OutputChannel '{self.name}': Unexpected output type: {type(output_data)}. String expected. Attempting to convert to string.")
             try:
                 # str() function can convert most Python objects to a string.
                 output_to_print = str(output_data)
             except Exception as e:
                 # If an error occurs during string conversion, log an error and stop the send operation.
                 self.logger.error(f"OutputChannel '{self.name}': Could not convert output to string: {e}", exc_info=True)
                 return # Skip sending if conversion fails.
        else:
             # If the incoming data is already a string, use it directly.
             output_to_print = output_data


        # DEBUG log: Info about receiving and preparing the raw output.
        self.logger.debug(f"OutputChannel '{self.name}': Raw output received, processing/preparing (Console).")

        try:
            # Print the processed output (string) to the console.
            self.logger.info(f"Evo Output: '{output_to_print}'")

            # DEBUG log: Info that the output was successfully printed to the console.
            self.logger.debug(f"OutputChannel '{self.name}': Output printed to console.")

        except Exception as e:
             # Catch any unexpected error during printing to the console (rare).
             self.logger.error(f"OutputChannel '{self.name}': Error printing to console: {e}", exc_info=True)
             # Not much more to do in case of a printing error, just log the error.


    def cleanup(self):
        """
        Cleans up ConsoleOutputChannel resources.

        Console output does not require specific resources, so it includes no cleanup steps
        beyond calling the base class's cleanup method (which only logs).
        """
        # Informational log.
        self.logger.info(f"ConsoleOutputChannel '{self.name}' cleaning up.")
        # Call the base class's cleanup method (only logs).
        super().cleanup()


# --- Web API Output Channel (Placeholder) ---
class WebAPIOutputChannel(OutputChannel):
    """
    An output channel that sends output to a Web API endpoint (Placeholder).

    This channel allows the InteractionAPI to communicate with the external world
    via HTTP or similar protocols.
    """
    def __init__(self, config):
        """
        Initializes the WebAPIOutputChannel.

        Args:
            config (dict): Channel configuration settings for this channel.
                           This is typically the sub-dictionary from main_config.yaml for this channel,
                           containing settings like 'port' and 'host'.
                           'port': The port the API is running on (int, default 5000).
                           'host': The host the API is running on (str, default '127.0.0.1').
                           The base class checks the config type.
        """
        # Call the base class's __init__ method. Set the channel name to "web_api".
        # Pass the specific config dict received from InteractionAPI.
        super().__init__("web_api", config) # self.config is now the channel-specific config

        # Get settings from the channel-specific config using get_config_value.
        # Corrected: Use default= keyword format.
        # The keys 'port' and 'host' are expected directly within the self.config dict.
        self.port = get_config_value(self.config, 'port', default=5000, expected_type=int, logger_instance=self.logger)
        self.host = get_config_value(self.config, 'host', default='127.0.0.1', expected_type=str, logger_instance=self.logger)


        self.logger.info(f"WebAPIOutputChannel initialized. Port: {self.port}, Host: {self.host}")
        # Logic to start the API server could go here (in a separate thread/process?) (Future TODO).
        # self._start_api_server() # Future TODO


    def send(self, output_data):
        """
        Sends the output to the Web API endpoint (Placeholder implementation).

        Receives the incoming output_data (e.g., dict, string) and simulates sending it
        to a specified API endpoint, like an HTTP POST request. Currently, this operation
        is just logged.
        Catches and logs errors if they occur during the simulated sending process.

        Args:
            output_data (any): The output data to be sent. A dict or string that can be
                               converted to JSON is typically expected.
        """
        # Error handling: Optional validation of the incoming data's validity.
        # For example, can check if output_data is not None using check_input_not_none
        # or check if the format to be sent (e.g., dict, str) is of an expected type using check_input_type.
        # InteractionAPI.send_output method already doesn't call this if output_data is None.
        # Here, we might want to check the format of output_data (e.g., is it a dict? a string?).
        # if not check_input_type(output_data, (dict, str), input_name="output_data for WebAPI", logger_instance=self.logger):
        #      self.logger.warning(f"OutputChannel '{self.name}': Unexpected output type: {type(output_data)}. dict or str expected.")
        #      # If type is invalid, skip sending.
        #      return


        self.logger.debug(f"OutputChannel '{self.name}': Raw output received, processing/preparing (Web API).")

        try:
            # Logic for sending to the Web API endpoint will go here (Future TODO).
            # For example, sending a POST request using the 'requests' library.
            # api_url = f"http://{self.host}:{self.port}/output" # Endpoint URL obtained from config.
            # headers = {'Content-Type': 'application/json'}
            # try:
            #     # Convert output data to JSON format (if necessary)
            #     # if isinstance(output_data, dict):
            #     #      json_data = json.dumps(output_data)
            #     # elif isinstance(output_data, str):
            #     #      json_data = output_data # If it's already a string
            #     # else:
            #     #      # Unsupported output_data type
            #     #      self.logger.warning(f"WebAPIOutputChannel: Unsupported output type: {type(output_data)}. Skipping send.")
            #     #      return # Skip sending

            #     # Send the POST request
            #     # response = requests.post(api_url, headers=headers, data=json_data, timeout=5) # Adding a timeout is good practice
            #     # response.raise_for_status() # Raises an Exception for bad HTTP status codes (4xx or 5xx)

            #     # Log successful send
            #     # self.logger.debug(f"WebAPIOutputChannel: Output successfully sent to API. Status Code: {response.status_code}")

            # except requests.exceptions.RequestException as e:
            #     # Specific errors from the 'requests' library (e.g., connection error, timeout, bad HTTP status code)
            #     self.logger.error(f"WebAPIOutputChannel: Error sending to API: {e}", exc_info=True)
            # except Exception as e:
            #      # Other unexpected errors (e.g., JSON conversion or others)
            #      self.logger.error(f"WebAPIOutputChannel: Unexpected error during API send: {e}", exc_info=True)

            # For now, just log and simulate the sending process.
            self.logger.info(f"WebAPIOutputChannel: Simulated sending output to API endpoint. Data sent: {output_data}")


        except Exception as e:
             # General error catching the main try block within the send method.
             self.logger.error(f"OutputChannel '{self.name}': Unexpected error during send operation: {e}", exc_info=True)
             # Not much more to do in case of error, just log it.


    def cleanup(self):
        """
        Cleans up WebAPIOutputChannel resources.

        Logic to stop the API server (if started here) or close open connections
        could go here.
        Called by module_loader.py and InteractionAPI.stop() when the program terminates (if it exists).
        """
        # Informational log.
        self.logger.info(f"WebAPIOutputChannel '{self.name}' cleaning up.")
        # Logic to shut down the API server could go here (if it was started here and has a stop method).
        # if hasattr(self, 'api_server') and self.api_server:
        #      self.logger.info(f"WebAPIOutputChannel: Shutting down API server (Port: {self.port})...")
        #      self.api_server.shutdown() # Method of a Flask development server or another server object


        # Call the base class's cleanup method (only logs).
        super().cleanup()

# TODO: Other output channel classes to be added in the future (e.g., FileOutputChannel, RoboticArmChannel) defined here.
# class FileOutputChannel(OutputChannel): ...
# class RoboticArmChannel(OutputChannel): ...