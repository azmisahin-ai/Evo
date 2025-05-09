# src/interaction/output_channels.py

import logging
import numpy as np
import requests
import json
import threading # ThreadPoolExecutor için gerekli olmasa da, bazen lock vb. için gerekebilir diye import kalabilir.
from concurrent.futures import ThreadPoolExecutor # <<< Arka plan işlemleri için eklendi

# Import utility functions
from src.core.config_utils import get_config_value
from src.core.utils import check_input_type


# Create a logger for this module
logger = logging.getLogger(__name__)

# --- OutputChannel Base Class ---
class OutputChannel:
    # ... (Base sınıf aynı kalabilir) ...
    def __init__(self, name: str, config: dict):
        if not isinstance(config, dict):
             logger.warning(f"OutputChannel '{name}': Configuration has unexpected type: {type(config)}. Dictionary expected. Using empty dictionary {{}}.")
             config = {}
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.logger.info(f"OutputChannel '{self.name}' initializing.")

    def send(self, output_data):
        raise NotImplementedError("Subclasses must implement the 'send' method.")

    def cleanup(self):
        self.logger.info(f"OutputChannel '{self.name}' cleaning up.")

# --- Console Output Channel ---
class ConsoleOutputChannel(OutputChannel):
    # ... (Console kanalı aynı kalabilir) ...
    def __init__(self, config: dict):
        super().__init__("console", config)
        self.logger.info("ConsoleOutputChannel initialized.")

    def send(self, output_data):
        output_to_print = None
        if isinstance(output_data, str):
            output_to_print = output_data
        else:
            self.logger.warning(f"OutputChannel '{self.name}': Unexpected output type: {type(output_data)}. String expected. Attempting conversion.")
            try:
                output_to_print = str(output_data)
            except Exception as e:
                self.logger.error(f"OutputChannel '{self.name}': Could not convert output to string: {e}", exc_info=True)
                return

        self.logger.debug(f"OutputChannel '{self.name}': Preparing to print to console.")
        try:
            # print(f"Evo Output: '{output_to_print}'")
            self.logger.info(f"OutputChannel '{self.name}': Successfully printed to console: '{output_to_print[:80]}...'")
        except Exception as e:
            self.logger.error(f"OutputChannel '{self.name}': Error printing to console: {e}", exc_info=True)

    def cleanup(self):
        self.logger.info(f"ConsoleOutputChannel '{self.name}' cleaning up.")
        super().cleanup()


# --- Web API Output Channel (Non-Blocking) ---
class WebAPIOutputChannel(OutputChannel):
    """
    An output channel that sends output to a Web API endpoint asynchronously
    using a background thread pool. It does not block the main cognitive loop.
    """
    def __init__(self, config: dict):
        """
        Initializes the WebAPIOutputChannel and the background executor.
        """
        super().__init__("web_api", config)

        self.port = get_config_value(self.config, 'port', default=5000, expected_type=int, logger_instance=self.logger)
        self.host = get_config_value(self.config, 'host', default='127.0.0.1', expected_type=str, logger_instance=self.logger)
        self.endpoint = get_config_value(self.config, 'endpoint', default='/evo_output', expected_type=str, logger_instance=self.logger)
        self.api_url = f"http://{self.host}:{self.port}{self.endpoint}"
        # Arka plan istekleri için bir thread havuzu oluştur
        # max_workers: Aynı anda en fazla kaç isteğin gönderileceğini belirler.
        # Bu değeri config'den almak da mümkün olabilir. Küçük bir değerle başlamak iyi olur.
        self.max_workers = get_config_value(self.config, 'max_workers', default=3, expected_type=int, logger_instance=self.logger)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        self.logger.info(f"WebAPIOutputChannel initialized. Target URL: {self.api_url}. Max concurrent requests: {self.max_workers}")

    def _send_request_task(self, url: str, headers: dict, payload: str):
        """
        The actual task run by the background thread to send the request.
        Handles errors and logging within the thread.
        """
        thread_id = threading.get_ident() # Hangi thread'in çalıştığını loglamak için (opsiyonel)
        self.logger.debug(f"[Thread-{thread_id}] Sending POST request to {url}")
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=10) # Timeout burada hala önemli
            response.raise_for_status() # Hatalı durum kodları için exception fırlat
            self.logger.info(f"[Thread-{thread_id}] API request successful to {url}. Status: {response.status_code}")
            # self.logger.debug(f"[Thread-{thread_id}] API Response: {response.text[:100]}...")

        except requests.exceptions.ConnectionError:
            # Bu log artık arka plan thread'inden gelecek
            self.logger.error(f"[Thread-{thread_id}] Connection error sending to {url}. Is the API server running?")
        except requests.exceptions.Timeout:
            # Bu log artık arka plan thread'inden gelecek
            self.logger.error(f"[Thread-{thread_id}] Timeout error sending to {url}.")
        except requests.exceptions.RequestException as e:
            # Bu log artık arka plan thread'inden gelecek
            self.logger.error(f"[Thread-{thread_id}] Error sending to API ({url}): {e}")
        except Exception as e:
            # Bu log artık arka plan thread'inden gelecek
            self.logger.error(f"[Thread-{thread_id}] Unexpected error during background API send to {url}: {e}", exc_info=True)


    def send(self, output_data):
        """
        Serializes data and submits the send request task to the background thread pool.
        Returns immediately without waiting for the request to complete.
        """
        self.logger.debug(f"OutputChannel '{self.name}': Preparing data for background API send to: {self.api_url}")

        headers = {'Content-Type': 'application/json'}
        json_payload = None

        try:
            # Veriyi JSON'a çevirme mantığı aynı kalıyor
            if isinstance(output_data, dict):
                json_payload = json.dumps(output_data)
            elif isinstance(output_data, str):
                json_payload = json.dumps({"message": output_data})
            elif isinstance(output_data, (int, float, bool)):
                 json_payload = json.dumps({"value": output_data})
            elif isinstance(output_data, np.ndarray):
                 self.logger.debug("Attempting to convert numpy array to list for WebAPI.")
                 json_payload = json.dumps({"numpy_array": output_data.tolist()})
            else:
                self.logger.warning(f"WebAPIOutputChannel: Unsupported output type {type(output_data)}. Attempting str conversion.")
                try:
                    json_payload = json.dumps({"raw_message": str(output_data)})
                except Exception as str_err:
                     self.logger.error(f"WebAPIOutputChannel: Could not convert output of type {type(output_data)} to string for JSON: {str_err}")
                     return # JSON oluşturulamadıysa gönderme

            # <<< DEĞİŞİKLİK: İsteği arka plana gönder >>>
            if json_payload:
                # _send_request_task fonksiyonunu thread havuzuna gönder.
                # submit çağrısı hemen döner, görevin tamamlanmasını beklemez.
                self.executor.submit(self._send_request_task, self.api_url, headers, json_payload)
                # Sadece görevin gönderildiğini logla. Sonuç logları _send_request_task içinden gelecek.
                self.logger.debug(f"OutputChannel '{self.name}': Task submitted to send data to {self.api_url}.")

        except TypeError as json_err:
             # JSON serileştirme hatası hala ana thread'de olabilir
             self.logger.error(f"WebAPIOutputChannel: Could not serialize output data to JSON before submitting: {json_err}. Data: {output_data}")
        except Exception as e:
            # Beklenmedik diğer hatalar (örn. executor çalışmıyorsa?)
            self.logger.error(f"WebAPIOutputChannel: Unexpected error during task submission: {e}", exc_info=True)


    def cleanup(self):
        """
        Cleans up WebAPIOutputChannel resources, including shutting down the thread pool.
        """
        self.logger.info(f"WebAPIOutputChannel '{self.name}' cleaning up.")
        # Thread havuzunu kapat.
        # wait=False: Bekleyen görevlerin bitmesini bekleme, hemen çık. Bu, programın hızlı kapanmasını sağlar
        #             ancak son gönderilen birkaç isteğin tamamlanmama riski vardır.
        # wait=True: Bekleyen tüm görevlerin bitmesini bekle. Kapanışı geciktirebilir ama daha güvenlidir.
        #           "Teslimatı düşünmeyelim" dediğiniz için False daha uygun.
        if self.executor:
             self.logger.info(f"Shutting down WebAPI background executor (wait=False)...")
             self.executor.shutdown(wait=False)
             self.logger.info(f"WebAPI background executor shut down.")
        super().cleanup()

# --- OpenTelemetry Output Channel (Future Placeholder) ---
# class OpenTelemetryOutputChannel(OutputChannel):
#     """
#     An output channel that sends Evo's internal state or events
#     as OpenTelemetry signals (traces, metrics, logs).
#     """
#     def __init__(self, config: dict):
#         super().__init__("opentelemetry", config)
#         # Initialize OpenTelemetry exporter (e.g., OTLP exporter) based on config
#         # self.exporter = ...
#         # self.tracer = trace.get_tracer("evo.cognition")
#         # self.meter = metrics.get_meter("evo.performance")
#         # self.event_logger = ... # OTel logging API
#         self.logger.info("OpenTelemetryOutputChannel initialized.")
#
#     def send(self, output_data):
#         """ Sends data as OTel signals. """
#         # output_data should ideally be structured (e.g., dict) to indicate
#         # what kind of signal to create (trace span, metric update, log event).
#         self.logger.debug(f"OutputChannel '{self.name}': Preparing to send OTel data.")
#         # Example:
#         # if isinstance(output_data, dict) and output_data.get("type") == "metric":
#         #     metric_name = output_data.get("name")
#         #     value = output_data.get("value")
#         #     attributes = output_data.get("attributes", {})
#         #     # self.meter.create_gauge(metric_name).set(value, attributes=attributes)
#         #     self.logger.info(f"Sent OTel metric: {metric_name}={value}")
#         # elif isinstance(output_data, dict) and output_data.get("type") == "trace_event":
#         #     # with self.tracer.start_as_current_span(output_data.get("name", "evo_event")) as span:
#         #     #     span.set_attributes(output_data.get("attributes", {}))
#         #     self.logger.info(f"Sent OTel trace event: {output_data.get('name')}")
#         # else: # Default to sending as a log
#         #     # self.event_logger.emit(...)
#         #     self.logger.info(f"Sent OTel log: {output_data}")
#         pass # Placeholder
#
#     def cleanup(self):
#         """ Shuts down the OTel exporter. """
#         self.logger.info(f"OpenTelemetryOutputChannel '{self.name}' cleaning up.")
#         # if hasattr(self, 'exporter') and self.exporter:
#         #     self.exporter.shutdown()
#         super().cleanup()