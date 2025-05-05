# src/interaction/api.py

import logging
# Gelecekte ağ iletişimi veya cihaz kontrolü için kütüphaneler buraya eklenecek
# import flask # Örnek: Web API için
# import robotics_interface_lib # Örnek: Robot kontrolü için

class InteractionAPI:
    """
    Evo'nun dış dünya ile iletişim kurduğu arayüzü temsil eder.
    MotorControl modülünden gelen tepkileri dış kanallara iletir.
    """
    def __init__(self, config=None):
        logging.info("InteractionAPI modülü başlatılıyor...")
        self.config = config if config is not None else {}

        # İletişim kanalı ayarları (örneğin, ağ portu, cihaz adresi) burada tanımlanabilir
        # self.port = self.config.get('api_port', 5000) # Örnek API portu

        # Gelecekte burada iletişim kanalı (örneğin bir web sunucusu thread'i) başlatılacak
        logging.info("InteractionAPI modülü başlatıldı.")
        # if self.config.get('start_server', False):
        #      self._start_server() # Örnek: Sunucuyu başlat

    def send_output(self, output_data):
        """
        MotorControl modülünden gelen çıktıyı alır ve dış dünyaya (şimdilik konsola) iletir.
        output_data: MotorControl modülünden gelen tepki (string veya başka format)
        """
        if output_data is None:
            # logging.debug("InteractionAPI: Gönderilecek çıktı verisi yok.")
            return # None çıktı gelirse gönderme

        # logging.debug(f"InteractionAPI: Çikti alindi. İçerik ilk 50 karakter: '{str(output_data)[:50]}'")

        # --- Gerçek Dışarı Aktarma Mantığı Buraya Gelecek (Faz 3 ve sonrası) ---
        # Örnek: Bir ağ soketine gönderme, bir robota komut gönderme, GUI'de gösterme vb.

        # Şimdilik sadece konsola logla
        logging.info(f"Evo Çıktısı: {output_data}")

        # logging.debug("InteractionAPI: Çikti başarıyla iletildi (placeholder: konsol).")


    # Gelecekte kullanılacak, ağ sunucusu gibi başlatılması gereken metot
    # def _start_server(self):
    #      logging.info(f"InteractionAPI sunucusu başlatılıyor (Port: {self.port})...")
    #      # Sunucu başlatma kodu buraya gelecek (örneğin, bir thread içinde Flask çalıştırmak)
    #      pass

    # Gelecekte kullanılacak, kapatılması gereken metot
    # def stop(self):
    #     logging.info("InteractionAPI durduruluyor...")
    #     # Sunucuyu durdurma kodu buraya gelecek
    #     pass


    def __del__(self):
        """
        Nesne silindiğinde kaynakları temizler.
        """
        # Eğer _start_server gibi bir metot kullanıldıysa, stop metodu çağrılmalıdır.
        # if hasattr(self, 'stop'):
        #      self.stop()
        logging.info("InteractionAPI objesi silindi.")

# Modülü bağımsız test etmek için örnek kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print("InteractionAPI modülü test ediliyor...")

    api = InteractionAPI() # Varsayılan config ile başlat

    # Sahte çıktıları ile test et
    dummy_output_1 = "Merhaba dünya!"
    dummy_output_2 = "Bir kuş uçuyor."
    dummy_output_3 = "" # Boş string testi
    dummy_output_4 = None # None testi

    print("\nÇikti 'Merhaba dünya!' ile test et:")
    api.send_output(dummy_output_1)

    print("\nÇikti 'Bir kuş uçuyor.' ile test et:")
    api.send_output(dummy_output_2)

    print("\nBoş string çikti ile test et:")
    api.send_output(dummy_output_3) # Boş string loglanır

    print("\nNone çikti ile test et:")
    api.send_output(dummy_output_4) # None ise işlem yapmaz


    print("\nInteractionAPI modülü testi bitti.")