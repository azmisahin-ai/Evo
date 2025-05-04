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

### Ana Bileşenler ve Anlamları

*   **`docs/`**: Projenin felsefesini, yapısını, kullanımını ve gelişimini anlatan belgeler. Evo'nun kimliği, geçmişi ve kullanım kılavuzu gibi düşünülebilir.
*   **`src/`**: Projenin ana kod tabanı. Evo'nun "vücudu" ve "beyni".
    *   **`src/senses/`**: Evo'nun **dış dünyadan ham bilgi aldığı** modüller. Farklı "duyular" gibi çalışır. İlk aşamada `camera/` ve `audio/` modülleri temel alınacaktır. Bu modüller kesintisiz akış sağlamayı hedefler.
    *   **`src/processing/`**: Ham duyu verisini **daha işlenmiş, üst düzey özelliklere dönüştüren** modüller. Duyusal korteksler gibi düşünülebilir.
    *   **`src/representation/`**: İşlenmiş duyu verisinden **öğrenilmiş, sıkıştırılmış ve anlamlı içsel temsiller (latent)** oluşturan modüller. Beynin soyutlama ve temsil katmanları gibidir.
    *   **`src/memory/`**: Evo'nun **öğrendiği bilgiyi (temsiller, kavramlar, ilişkiler) depolayan** ve yöneten merkezi birim. Farklı bellek türlerini içerecektir (kısa süreli, uzun süreli, episodik, semantik).
    *   **`src/cognition/`**: Evo'nun **üst seviye bilişsel işlevlerini** barındıran modüller. Anlama, karar verme, kendi kendine öğrenme, akıl yürütme ve planlama gibi süreçler burada gerçekleşir. Beynin prefrontal korteks gibi düşünen kısımlarına benzer.
    *   **`src/motor_control/`**: İçsel temsilleri ve kararları **dışsal eylemlere veya çıktılara (ses, metin, görsel, fiziksel hareket)** dönüştüren modüller. Bir canlının motor korteksi ve kasları gibidir.
    *   **`src/interaction/`**: Evo'nun **dış dünya ile doğrudan iletişim kurduğu arayüzleri** içeren modüller. API'ler veya robotik sürücüler gibi. Konuşma organları veya uzuvlar gibi düşünülebilir.
    *   **`src/core/`**: Temel AI yapı taşlarını (sinir ağı katmanları) ve genel yardımcı fonksiyonları barındıran düşük seviye modüller. Bir canlının hücresel yapısı veya temel dokuları gibidir.
    *   **`run_evo.py`**: Evo'nun **"canlanma" scripti**. Bilişsel döngüyü (`cognitive_loop`) ve dış arayüzleri (API) başlatarak Evo'yu "hayata getirir".
*   **`tests/`**: Evo'nun farklı "organlarının" ve "sistemlerinin" beklendiği gibi çalışıp çalışmadığını kontrol eden testler. Canlının periyodik sağlık kontrolleri gibi düşünülebilir.
*   **`scripts/`**: Projeyi kurmak, veri hazırlamak, temel modelleri eğitmek veya bakımını yapmak için kullanılan **manuel yardımcı scriptler**. Evo'nun eğitim veya bakım prosedürleri gibi düşünülebilir.
*   **Diğer Dizinler (`config`, `data`, `mobile_app`, `notebooks`)**: Projenin genel konfigürasyonu, veri depolama alanları, Evo ile etkileşim kurmak için arayüz projeleri ve deneysel çalışmalar için alanlardır.

Bu yapı, Evo'nun farklı yeteneklerini (duyular, içsel işlem, hafıza, düşünme, eylemler) modüler hale getirerek, her bir parçanın nispeten bağımsız olarak geliştirilmesine ve Evo'nun zamanla "organlarını" ve "sistemlerini" geliştirmesine olanak tanır. Bu, Evo'nun evrimsel büyümesini teknik olarak destekler.

---

