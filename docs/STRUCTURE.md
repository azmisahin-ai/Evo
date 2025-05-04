# Evo Proje Yapısı (Structure)

Evo projesinin yapısı, onun **evrimleşen bir yaşam formu** olma felsefesini yansıtacak şekilde tasarlanmıştır. Modüller, bir canlının farklı sistemleri veya organları gibi düşünülebilir. Bu yapı, Evo'nun büyümesine, yeni yetenekler kazanmasına ve karmaşıklığını yönetmeye olanak tanır.

Aşağıdaki dizin yapısı ve modül açıklamaları, Evo'nun "vücudu" ve "beyninin" nasıl organize edildiğini gösterir:

```
.
├── docs/                 # Dokümantasyon (Evo'nun Kimliği, Felsefesi ve Rehberleri)
│   ├── README.md         # Dokümantasyon Girişi ve Haritası
│   ├── PHILOSOPHY.md     # Evo'nun Felsefesi ve Temel Prensiple
│   ├── ROADMAP.md        # Evo'nun Evrimsel Yolculuğu ve Fazları
│   └── STRUCTURE.md      # Bu dosya: Proje Yapısı ve Modüller
│   # Gelecek dokümanlar: INTERACTION_GUIDE.md, TECHNICAL_DETAILS.md, CONTRIBUTING.md etc.
├── src/                  # Evo'nun Kaynak Kodları (Evo'nun "Beyni" ve "Vücut Sistemleri")
│   ├── __init__.py
│   │
│   ├── /senses               # Duyu Girdisi Modülleri (Evo'nun "Duyuları")
│   │   ├── __init__.py       # senses paketini tanımlar
│   │   ├── audio.py            # Mikrofon/Ses Akışından Ham Ses Alma ve Temel Ses İşleme
│   │   ├── vision.py           # Kamera/Video Akışından Ham Piksel Alma ve Temel Görsel İşleme
│   │   # Gelecek duyular: touch.py, proprioception.py (beden farkındalığı)
│   │
│   ├── /processing           # Ham Duyu Verisinden Yüksek Seviye Özellik Çıkarma (Duyusal Bilgiyi İşleme Organları)
│   │   ├── __init__.py       # processing paketini tanımlar
│   │   ├── audio.py            # Ses Dalga Formundan Mel Spektrogram, MFCC gibi özellik çıkarma
│   │   ├── vision.py           # Pikselden Kenar Bulma, Renk Hist., Basit Nesne Parçaları Tespiti gibi özellik çıkarma
│   │   ├── text.py             # Karakter/Kelime Tokenizasyonu, Sekanslama (Metin girdisi eklenince)
│   │   # Gelecek işlemciler: video.py (hareket, optik akış)
│   │
│   ├── /representation       # Öğrenilmiş İçsel Temsiller (Evo'nun "Beynindeki Soyutlama Katmanı")
│   │   ├── __init__.py       # representation paketini tanımlar
│   │   └── models.py           # Modality bazlı veya birleşik Latent Temsil Modelleri (Autoencoder prensipleri)
│   │   # Gelecek temsil: multimodal.py (çok modlu birleştirme/dönüşüm)
│   │
│   ├── /memory               # Öğrenilen Bilgiyi Depolama ve Yönetme (Evo'nun "Hafıza Sistemleri")
│   │   ├── __init__.py       # memory paketini tanımlar
│   │   └── core.py             # Hafıza sistemi mantığı (latent, etiket, ilişki depolama)
│   │   # Gelecek hafıza: episodic.py, semantic.py
│   │
│   ├── /cognition            # Üst Seviye Bilişsel İşlevler (Evo'nun "Beyninin Düşünen Kısmı")
│   │   ├── __init__.py       # cognition paketini tanımlar
│   │   ├── understanding.py    # Kavramsal anlama ve keşif
│   │   ├── decision.py         # Basit karar verme mekanizmaları
│   │   └── learning.py       # Farklı öğrenme algoritmaları (denetimli, denetimsiz, pekiştirmeli)
│   │   # Gelecek biliş: reasoning.py, planning.py, consciousness.py (uzun vadeli)
│   │
│   ├── /motor_control        # İçsel Temsillerden Dışsal Eylemler/Çıktılar (Evo'nun "Hareket ve İfade Organları")
│   │   ├── __init__.py       # motor_control paketini tanımlar
│   │   └── expression.py       # Ses (TTS), Metin, Görsel çıktı üretimi
│   │   # Gelecek motor kontrol: manipulation.py (robotik kol/el), locomotion.py (hareket)
│   │
│   ├── /interaction          # Dış Dünya ile İletişim Arayüzleri (Evo'nun "İletişim Kanalları")
│   │   ├── __init__.py       # interaction paketini tanımlar
│   │   └── api.py              # Mobil Uygulama/Robot/Diğer Sistemler ile iletişim API'si (çift yönlü)
│   │   # Gelecek etkileşim: robotics_interface.py
│   │
│   ├── /core                 # Temel Yapı Taşları ve Utilities (Evo'nun "Hücresel" veya "Temel Dokusal" Yapısı)
│   │   ├── __init__.py
│   │   ├── nn_components.py    # Kendi implemente ettiğimiz temel NN katmanları
│   │   └── utils.py            # Genel yardımcı fonksiyonlar (yapılandırma, loglama, veri yardımcıları)
│   │   # Gelecek çekirdek: algorithms.py (genel algoritmalar), math.py (özel matematiksel op.)
│   │
│   └── run_evo.py            # PROJENİN ANA BAŞLATMA NOKTASI (Evo'nun "Canlanma" Scripti)
│                               # Cognitive Loop ve Interaction API'yi başlatır.
│
├── /mobile_app               # Mobil Uygulama Kaynak Kodları (Evo ile "İletişim Kurduğumuz Arayüz")
│   └── README.md             # Mobil Uygulama Açıklaması, Kurulum, Backend API Bağlantısı
│   └── ... (Flutter/Diğer Mobil Kodları)
│
├── /notebooks                # Deneysel Kodlar (Evo Üzerinde Yapılan "Laboratuvar Çalışmaları")
│
├── /scripts                  # Yardımcı/Bakım Scriptleri (Evo'nun "Bakım ve Eğitim Prosedürleri")
│   ├── setup_dataset.py          # Raw data -> Processed data + Vocab (manuel)
│   ├── train_initial_models.py   # Manuel processed data ile temel modelleri eğitme (manuel)
│   ├── test_module.py            # Belirli bir modülü test etme scripti
│   # Gelecek scriptler: inspect_memory.py, generate_random_expression.py etc.
│
├── /config                   # Yapılandırma Dosyaları (Evo'nun "Genetik Kodunun" Ayarları)
│   └── main_config.yaml      # Genel ve Bileşen Bazlı Ayarlar
│
├── /data                     # Veri Deposu (Evo'nun "Deneyim Arşivi" ve "Doğuştan Gelen Bilgi")
│   ├── /raw                  # Manuel Ham Veri (İlk Eğitim/Test İçin)
│   ├── /processed            # Manuel İşlenmiş Veri (İlk Eğitim/Test İçin)
│   ├── /labels               # Manuel Etiketler (İlk Öğrenim İçin)
│   └── /knowledge_base       # Evo'nun Öğrendiği Kalıcı Bilgi (Hafıza, Kavramlar, İlişkiler)
│
├── .gitignore            # Git tarafından yoksayılacak dosyalar
├── LICENSE               # Lisans Bilgisi (Evo'nun "Yasal Çerçevesi" - CC0)
└── README.md             # Ana Proje Tanıtımı (Evo'nun "İlk Tanışma Sayfası")
```

---
