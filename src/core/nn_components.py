# src/core/nn_components.py

import logging
import numpy as np
# NumPy'ı temel matematik işlemleri için kullanıyoruz.
# Gelecekte GPU hızlandırma için farklı backend'ler düşünülebilir.

class Dense:
    """
    Temel bir Dense (Fully Connected) Sinir Ağı Katmanı implementasyonu (from scratch).
    Girdi vektörünü ağırlık matrisiyle çarpar ve sapma (bias) ekler.
    """
    def __init__(self, input_size, output_size, activation=None):
        logging.info(f"Dense katmanı başlatılıyor: Input={input_size}, Output={output_size}, Activation={activation}")
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation # Aktivasyon fonksiyonu (şimdilik sadece adı)

        # Ağırlık matrisini (weights) ve sapma vektörünü (bias) başlatma
        # Genellikle küçük rastgele değerlerle başlatılır (He, Xavier init vb.)
        # Şimdilik basit rastgele başlatma kullanalım.
        # Weights shape: (input_size, output_size)
        # Bias shape: (output_size,)
        # Başlatmada ölçeklendirme önemlidir (örn: sqrt(2. / input_size))
        limit = np.sqrt(1. / input_size) # Basit başlatma ölçeği (fan-in)
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias = np.zeros(output_size) # Bias'ı genellikle sıfır başlatmak yaygındır.

        logging.info("Dense katmanı başlatıldı.")

    def forward(self, input_data):
        """
        Girdi verisini (input_data) katmandan geçirir (ileri yayılım).
        input_data: Shape (batch_size, input_size) veya (input_size,) olabilir.
        Tek bir örnek için (input_size,) shape bekleniyor şimdilik.
        """
        if input_data is None:
             # logging.debug("Dense katmanı: İşlenecek girdi yok.")
             return None

        # Girdinin NumPy array olduğundan emin olalım ve boyutunu kontrol edelim.
        if not isinstance(input_data, np.ndarray):
             logging.error(f"Dense katmanı: Beklenmeyen girdi formatı (NumPy array bekleniyordu, geldi: {type(input_data)}).")
             return None
        
        # Tek bir örnek (vector) geldiğini varsayalım: Shape (input_size,)
        # Eğer batch geldiyse (batch_size, input_size), matris çarpımı da batch boyutunda çalışır.
        # Şimdilik tek örneği işleyelim. Boyut kontrolü yapalım.
        if input_data.shape[-1] != self.input_size:
             logging.error(f"Dense katmanı: Girdi boyutu yanlış. Beklenen: {self.input_size}, Gelen: {input_data.shape[-1]}.")
             return None
             
        # logging.debug(f"Dense katmanı: Girdi alindi. Shape: {input_data.shape}")

        try:
            # Doğrusal dönüşüm: output = input_data @ weights + bias
            # input_data (input_size,) * weights (input_size, output_size) -> (output_size,)
            linear_output = np.dot(input_data, self.weights) + self.bias

            # Aktivasyon fonksiyonunu uygula (varsa)
            if self.activation == 'relu':
                output_data = np.maximum(0, linear_output) # ReLU aktivasyonu
            # TODO: Diğer aktivasyon fonksiyonları eklenecek (sigmoid, tanh vb.)
            # elif self.activation == 'sigmoid':
            #      output_data = 1 / (1 + np.exp(-linear_output))
            # ...
            else: # activation is None veya tanımlı değilse
                output_data = linear_output # Lineer aktivasyon

            # logging.debug(f"Dense katmanı: Çikti üretildi. Shape: {output_data.shape}")

            return output_data # Çıktı NumPy array (output_size,)

        except Exception as e:
            logging.error(f"Dense katmanı forward sırasında hata oluştu: {e}", exc_info=True)
            return None

    # TODO: Backprop (geri yayılım) metodları gelecekte eklenecek (.backward(), .update_weights())

    def __del__(self):
        """
        Nesne silindiğinde kaynakları temizler.
        NumPy array'ler için özel temizlik gerekmez.
        """
        logging.info(f"Dense katmanı objesi silindi: Input={self.input_size}, Output={self.output_size}")


# Modülü bağımsız test etmek için örnek kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print("Dense katmanı test ediliyor...")

    # Dense katmanı oluştur
    input_dim = 10
    output_dim_linear = 5
    output_dim_relu = 8

    dense_linear = Dense(input_dim, output_dim_linear)
    dense_relu = Dense(input_dim, output_dim_relu, activation='relu')

    # Sahte girdi verisi oluştur
    dummy_input = np.random.rand(input_dim).astype(np.float32)
    print(f"\nSahte girdi verisi oluşturuldu. Shape: {dummy_input.shape}")

    # Lineer katmanı test et
    print("\nLineer Dense katmanı testi:")
    output_linear = dense_linear.forward(dummy_input)
    if output_linear is not None:
        print(f"Çikti alindi. Shape: {output_linear.shape}, Dtype: {output_linear.dtype}")
        if output_linear.shape == (output_dim_linear,):
             print("Çikti boyutu doğru.")
    else:
        print("Lineer katman çikti üretmedi.")

    # ReLU katmanı test et
    print("\nReLU Dense katmanı testi:")
    output_relu = dense_relu.forward(dummy_input)
    if output_relu is not None:
        print(f"Çikti alindi. Shape: {output_relu.shape}, Dtype: {output_relu.dtype}")
        if output_relu.shape == (output_dim_relu,):
             print("Çikti boyutu doğru.")
        # ReLU çıktısında negatif değer olmamalı
        if np.all(output_relu >= 0):
             print("Çikti değerleri >= 0 (ReLU doğru çalışıyor görünüyor).")
        else:
             print("Hata: ReLU çıktısında negatif değer var.")
    else:
        print("ReLU katman çikti üretmedi.")

    # None girdi ile test et
    print("\nNone girdi ile Dense katmanı testi:")
    output_none = dense_linear.forward(None)
    if output_none is None:
        print("None girdi ile çikti doğru şekilde None döndü.")
    else:
         print("None girdi ile çikti None dönmedi (beklenmeyen durum).")


    print("\nDense katmanı testi bitti.")