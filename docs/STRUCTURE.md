# Evo
Doğduğu anda duyuları aktif olacak ve çevreden gelen kesintisiz bir duyu akışını işlemeye başlayacak.

```
/Evo
├── README.md                 # Projenin Ana Tanıtımı, Hızlı Başlama Rehberi, Kurulum, BAŞLATMA (Canlı Etkileşim Vurgulu)
├── requirements.txt          # Gerekli Python Kütüphaneleri
├── .gitignore                # Sürüme Eklenmeyecek Dosyalar
│
├── /config                   # Yapılandırma Dosyaları
│   └── main_config.yaml      # Genel ve Bileşen Bazlı Ayarlar (Tek Fazlara Bağlı Olmayan, Daha Genel Ayarlar)
│
├── /data                     # Veri Deposu
│   ├── /raw                  # Manuel Ham Veri (İlk Eğitim/Test İçin Gerekliyse)
│   ├── /processed            # Manuel İşlenmiş Veri (İlk Eğitim/Test İçin Gerekliyse)
│   ├── /labels               # Manuel Etiketler (İlk Öğrenim İçin Gerekliyse)
│   └── /knowledge_base       # Evo'nun Öğrendiği Kalıcı Bilgi (Hafıza, Kavramlar, İlişkiler)
│
├── /docs                     # Proje Dokümantasyonu
│   ├── README.md             # Detaylı Vizyon, Felsefe ve Evrimsel Yolculuk (Bebek Gelişimi Metoforu ile)
│   └── STRUCTURE.md          # Depo Yapısı ve Modül Sorumlulukları Açıklaması
│   # future_docs: module_details/, learning_algorithms/, interaction_protocols/ etc.
│
├── /src                      # Evo'nun Kaynak Kodu (Beyni ve Beden Arayüzü)
│   ├── __init__.py
│   │
│   ├── /senses               # <-- MERKEZİ: Duyu Girdisi Modülleri (Canlı Akış Odaklı)
│   │   ├── __init__.py       # senses paketini tanımlar
│   │   ├── audio.py            # <-- YENİ: Mikrofon/Ses Akışından Ham Ses Alma VE Temel Ses İşleme (Dalga Formu, Belki Basit Özellik)
│   │   ├── vision.py           # <-- YENİ: Kamera/Video Akışından Ham Piksel Alma VE Temel Görsel İşleme (Piksel, Belki Boyutlandırma)
│   │   # future_senses: touch.py, proprioception.py (beden farkındalığı)
│   │
│   ├── /processing           # <-- YENİ: Ham Duyu Verisinden Daha Yüksek Seviye Özellik Çıkarma (Representasyona Hazırlık)
│   │   ├── __init__.py       # processing paketini tanımlar
│   │   ├── audio.py            # <-- YENİ: Ses Dalga Formundan Mel Spektrogram, MFCC gibi özellik çıkarma
│   │   ├── vision.py           # <-- YENİ: Pikselden Kenar Bulma, Renk Hist., Basit Nesne Parçaları Tespiti gibi özellik çıkarma
│   │   ├── text.py             # <-- YENİ: Karakter/Kelime Tokenizasyonu, Sekanslama (Metin girdisi eklenince aktifleşir)
│   │   # future_processing: video.py (hareket tespiti, optik akış)
│   │
│   ├── /representation       # <-- MERKEZİ: Temel Özelliklerden Öğrenilmiş İçsel Temsiller (Latent Uzay)
│   │   ├── __init__.py       # representation paketini tanımlar
│   │   ├── models.py           # <-- YENİ: Modality bazlı veya birleşik Latent Temsil Modelleri (Autoencoder prensipleri)
│   │                           # (Önceki phase1_representation içeriği daha modüler sınıflar halinde buraya taşınır)
│   │   # future_representation: multimodal.py (çok modlu birleştirme/dönüşüm modelleri)
│   │
│   ├── /memory               # <-- MERKEZİ: Öğrenilen Bilgiyi Depolama ve Yönetme
│   │   ├── __init__.py       # memory paketini tanımlar
│   │   └── core.py             # <-- YENİ: Hafıza sistemi mantığı (latent, etiket, ilişki depolama) - Önceki memory_system.py içeriği daha genel bir yapıda buraya taşınır
│   │   # future_memory: episodic.py, semantic.py
│   │
│   ├── /cognition            # <-- YENİ: Üst Seviye Bilişsel İşlevler (Anlama, Karar Verme, Kendi Kendine Öğrenme)
│   │   ├── __init__.py       # cognition paketini tanımlar
│   │   ├── understanding.py    # <-- YENİ: Kavramsal anlama (Faz 3 Classifier taşınır, kendi kendine öğrenme ile kavram keşfi başlar)
│   │   ├── decision.py         # <-- YENİ: Basit karar verme mekanizmaları (girdi -> temsil -> anlama/hafıza ile etkileşim -> karar)
│   │   └── learning.py       # <-- YENİ: Farklı öğrenme algoritmaları (denetimli, denetimsiz, pekiştirmeli ajanlar)
│   │   # future_cognition: reasoning.py, planning.py, consciousness.py (uzun vadeli)
│   │
│   ├── /motor_control        # <-- MERKEZİ: İçsel Temsillerden Dışsal Eylemler/Çıktılar (Beden Kontrolü)
│   │   ├── __init__.py       # motor_control paketini tanımlar
│   │   └── expression.py       # <-- YENİ: Ses (TTS), Metin, Görsel çıktı üretimi (Faz 4 Generatorlar buraya taşınır, daha sonra duygu/stil eklenir)
│   │   # future_motor_control: manipulation.py (robotik kol/el), locomotion.py (hareket)
│   │
│   ├── /interaction          # <-- MERKEZİ: Dış Dünya ile İletişim Arayüzleri (API, Robotik Arayüz)
│   │   ├── __init__.py       # interaction paketini tanımlar
│   │   └── api.py              # <-- YENİ: Mobil Uygulama/Robot/Diğer Sistemler ile iletişim API'si (çift yönlü veri akışı)
│   │   # future_interaction: robotics_interface.py
│   │
│   ├── /core                 # Temel AI Yapı Taşları ve Utilities (Önceki /core, /models/components ve /utils'in birleşimi veya yeniden organizasyonu)
│   │   ├── __init__.py
│   │   ├── nn_components.py    # <-- YENİ: GRUCell, ConvLayer, LinearLayer gibi kendi implemente ettiğimiz temel NN katmanları
│   │   ├── utils.py            # <-- YENİ: Genel yardımcı fonksiyonlar (config_manager, logging, data_utils'dan seçilenler)
│   │   # future_core: algorithms.py (genel algoritmalar), math.py (özel matematiksel op.)
│   │
│   └── run_evo.py            # <-- YENİ: PROJENİN ANA BAŞLATMA NOKTASI. Cognitive Loop'u ve Interaction API'yi başlatır.
│                               # Bu script çalışınca Evo "canlı" hale gelir.
│
├── /mobile_app               # <-- YENİ: Mobil Uygulama Kaynak Kodları (Ayrı Depo Veya Burada)
│   └── README.md             # Mobil Uygulama Açıklaması, Kurulum, Backend API Bağlantısı
│   └── ... (Flutter/Diğer Mobil Kodları)
│
├── /notebooks                # Deneysel Kodlar (Kod tabanına entegre edilmemiş, prototipleme)
│
└── /scripts                  # Yardımcı/Bakım Scriptleri (Manuel Kullanım)
    ├── setup_dataset.py          # <-- YENİ: Raw data -> Processed data + Vocab (manuel çalıştırma için)
    ├── train_initial_models.py   # <-- YENİ: Manuel processed data ile temel modelleri (Representasyon, Anlama, İfade) eğitme (manuel çalıştırma için)
    ├── test_module.py            # <-- YENİ: Belirli bir modülü/fonksiyonu test etme scripti (örn. python -m scripts.test_module senses.audio)
    # future_scripts: inspect_memory.py, generate_random_expression.py etc.
```

---
