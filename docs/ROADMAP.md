# Evo'nın Evrimsel Yolculuğu: Fazlar ve Kazanımlar (Güncel Durum ve Görev Listesi)

Evo projesinin gelişimi, bir **canlının büyüme aşamalarına** benzer şekilde, adım adım ilerleyen evrimsel fazlara ayrılmıştır. Her faz, bir öncekinin üzerine inşa edilir ve Evo'nın yeni bilişsel, duyusal ve motor yetenekler kazanmasını sağlar. Bu yol haritası, projenin nereye doğru ilerlediğini gösteren yaşayan bir belgedir ve **yapılacak görevler (task listesi/backlog) için referans görevi görür.**

Aşağıdaki fazlar, Evo'nın doğumundan (temel algı) bilgelik ve ustalığa (karmaşık etkileşim) kadar olan gelişimini ana hatlarıyla belirtir. Her fazın başlığı, o aşamada odaklanılan temel kazanımı ve parantez içinde bir **bebeğin gelişimindeki yaklaşık karşılığını** içerir. **`[x]` ile işaretlenenler tamamlanmış adımları (iskelet veya placeholder düzeyinde dahil), `[ ]` ile işaretlenenler ise henüz yapılmamış veya detaylı implementasyon gerektiren görevleri belirtir.**

---

*   **Faz 0: Doğum ve Duyuların Açılması (Temel Algı Akışı)**
    *   [x] Proje deposunun oluşturulması ve temel dosya/dizin yapısının kurulması (`docs/STRUCTURE.md`).
    *   [x] Proje felsefesi (`PHILOSOPHY.md`) ve temel yol haritası (`ROADMAP.md`) belgelerinin oluşturulması.
    *   [x] Genel README dosyalarının (`README.md`, `docs/README.md`) oluşturulması.
    *   [x] Katkıda bulunma (`CONTRIBUTING.md`) ve etkileşim (`INTERACTION_GUIDE.md`), teknik detaylar (`TECHNICAL_DETAILS.md`) rehberlerinin oluşturulması (içerikleri hala gelişiyor olabilir).
    *   [x] Lisans dosyasının oluşturulması (`LICENSE` - CC0).
    *   [x] Temel bağımlılıklar dosyasının oluşturulması (`requirements.txt`).
    *   [x] Ayarların yönetimi için yapılandırma mekanizmasının (basit dictionary) oluşturulması (`config.py` içinde yükleme placeholder'ı).
    *   [x] Çalışma döngüsünün (main loop) iskeletinin kurulması (`run_evo.py`).
    *   [x] Basit loglama sisteminin entegrasyonu (Python `logging` modülü kullanıldı).
    *   [x] Duyusal sensör modüllerinin oluşturulması (`src/senses/audio.py`, `src/senses/vision.py`) ve temel entegrasyonları (simüle/gerçek donanım placeholder'ları).
    *   [x] Modüller arası temel bilgi akışının (Sense -> Process -> Represent -> Memory -> Cognition -> Motor -> Interact) placeholder/iskelet metotlarla kurulması ve `run_evo.py` içinde işletilmesi.
    *   [x] Dış dünya ile temel etkileşim (Interaction) placeholder/iskeletinin oluşturulması (Input alma placeholder'ı, Output gönderme metodu, temel çıktı kanalları - ConsoleOutputChannel, WebAPIOutputChannel placeholder'ı).
    *   [x] Temel hata yönetimi ve logging iyileştirmelerinin başlangıcı (`run_evo.py` içindeki try-except blokları ve logging çağrıları).
    *   [x] DEBUG loglarının neden görünmediği konusunu araştırmak ve çözmek.
    *   [x] Yapılandırma yönetimini ayrı bir modül haline getirmek ve `.yaml` dosyasından okumayı implement etmek (`config.py`, `PyYAML` entegrasyonu).
    *   [x] Temel hata yönetimi/istisna işleme yapısını olgunlaştırmak ve modüller arası yaygınlaştırmak.
        *   [x] Duyusal sensörler (`src/senses/vision.py`, `src/senses/audio.py`) için temel hata yakalama ve loglama mekanizmaları eklendi.
        *   [x] Processing modülleri (`src/processing/vision.py`, `src/processing/audio.py`) için temel hata yakalama ve loglama mekanizmaları eklendi.
        *   [x] Representation, Memory, Cognition, MotorControl, Interaction modülleri için temel hata yakalama ve loglama mekanizmaları eklendi.
        *   [x] Hata durumlarında sistemin davranışı için daha genel prensipler tanımlandı (örn: non-kritik hatalarda devam et, kritik hatalarda durdur).
    *   [ ] **TODO:** Faz 0 ile ilgili genel kod kalitesi ve refactoring.
        *   [x] Modül başlatma ve temizleme mantığı `run_evo.py` dosyasından `src/core/module_loader.py` modülüne taşınarak kod tekrarı azaltıldı.
        *   [x] Loglama altyapısını yapılandırma dosyası (`main_config.yaml`) üzerinden control edilebilir hale getirme (log seviyesi, çıktı hedefleri vb.).
        *   [x] Docstrings ve Yorumlar eklendi/güncellendi.
            *   [x] run_evo.py dosyasına docstring ve yorumlar eklendi/güncellendi.
            *   [x] src/core yardımcı modüllerine (logging_utils, config_utils, module_loader) docstring ve yorumlar eklendi.
            *   [x] src/senses modüllerine (vision, audio) docstring ve yorumlar eklendi.
            *   [x] src/processing modüllerine (vision, audio) docstring ve yorumlar eklendi.
            *   [x] Diğer Faz 0 kapsamındaki modüllere (memory/core, cognition/core, motor_control/core, interaction/api, interaction/output_channels) docstring ve yorumlar eklendi.
        *   [x] Genel kod tekrarı azaltma ve isimlendirme/tutarlılık iyileştirmeleri.
            [ ] Girdi kontrolü yardımcı fonksiyonlarının (`src/core/utils.py`) uygun metotlara uygulanması.
            *   [x] Processing modüllerinde (vision, audio) girdi kontrolleri için src/core/utils.py yardımcı fonksiyonları kullanıldı.
            *   [x] Representation modülünde (models.py - learn metodu) girdi kontrolleri için src/core/utils.py yardımcı fonksiyonları kullanılması.
            *   [x] Memory modülünde (core.py - store, retrieve metotları) girdi kontrolleri için src/core/utils.py yardımcı fonksiyonları kullanılması.
            *   [x] Cognition modülünde (core.py - decide metodu) girdi kontrolleri için src/core/utils.py yardımcı fonksiyonları kullanılması.
            *   [x] MotorControl modülünde (core.py - generate_response metodu) girdi kontrolleri için src/core/utils.py yardımcı fonksiyonları kullanılması.
            *   [x] Interaction modülünde (api.py - send_output metodu, output_channels.py - send metotları) girdi kontrolleri için src/core/utils.py yardımcı fonksiyonları kullanılması.
            *   [x] src/core/utils.py içindeki check_numpy_input fonksiyonunda tuple ndim ve dtype kontrolü düzeltildi.
            *   [x] Diğer modüllerde (Sense, Representation, Memory, Cognition, MotorControl, Interaction) girdi kontrolleri için src/core/utils.py yardımcı fonksiyonları kullanılması.
            *   [x] Module_loader.py içinde tekrar eden modül başlatma ve temizleme kalıpları yardımcı fonksiyonlara taşıldı.
            *   [x] İsimlendirme ve dosya/sınıf sorumluluklarında genel tutarlılık iyileştirmeleri.
                *   [x] Cognition modülü (src/cognition) gözden geçirildi (yapısal niyet belirlendi).
                *   [x] Bellek modülü (src/memory) gözden geçirildi (yapısal niyet belirlendi).
                *   [x] Sense (src/senses) ve Processing (src/processing) modülleri gözden geçirildi.
                *   [x] Representation (src/representation) modülü gözden geçirilecek.
                *   [x] MotorControl (src/motor_control) modülü gözden geçirilecek.
                *   [x] Interaction (src/interaction) modülü gözden geçirilecek.
        *   [ ] Unit testler için temel iskeletin kurulması.


*   **Faz 1: Temel İşleme ve Temsil (Duyusal Veriden Özellik Çıkarma)**
    *   [x] Ham duyu akışını işleyecek Processing modüllerinin oluşturulması (`src/processing/audio.py`, `src/processing/vision.py` - şimdilik passthrough/Placeholder).
    *   [x] Öğrenilmiş içsel temsiller oluşturacak RepresentationLearner modülünün oluşturulması (`src/representation/models.py` - Placeholder/Temel Sınıf).
    *   [x] RepresentationLearner içinde Dense katmanları gibi temel NN yapı taşlarının eklenmesi (Model iskeleti başlangıcı).
    *   [x] Bu modüllerin temel döngüye entegrasyonu (`run_evo.py` içinde çağrılıyorlar).

    *   **Faz 1 Gerçek Implementasyon Görevleri (Mevcut Odak Noktamız - Görsel & İşitsel):**
        *   [x] Ham **görsel** veriden temel, düşük seviyeli özellikler (renk, basit kenarlar, hareket - gelecekte) çıkarma algoritmalarının src/processing/vision.py içine implementasyonu.
        *   [x] Ham **işitsel** veriden temel, düşük seviyeli özellikler (enerji, frekans, MFCC - gelecekte) çıkarma algoritmalarının src/processing/audio.py içine implementasyonu.
        *   [x] İşlenmiş **görsel ve işitsel** özelliklerden öğrenilmiş, düşük boyutlu, modality bazlı içsel temsiller (latent vektörler) oluşturma (örn: Basit bir Autoencoder veya VAE prensibi) implementasyonu (src/representation/models.py).
        *   [x] RepresentationLearner.learn metodunun bu öğrenme/dönüştürme sürecini içerecek şekilde implementasyonu.
        *   [x] RepresentationLearner çıktısının (latent vektörler) Memory ve Cognition için kullanılabilir formatta olmasının sağlanması. (Çıktı 128 boyutlu numpy array, bu modüllerin placeholderları bunu bekliyor)
        *   [x] Faz 1 modüllerinin (Sense, Process, Represent) entegrasyonunun ve çıktı formatlarının detaylı test edilmesi. (Son log çıktısı tüm bu akışın başarılı olduğunu gösterdi).

    *   [ ] **TODO:** Gelecekte metin gibi diğer duyu modaliteleri için Process ve Representation desteği eklenmesi (Faz 5 ile ilişkili olabilir).
    *   [ ] **TODO:** Farklı temsil katmanları veya yöntemleri (örn: anlamsal temsil, sembolik temsil) için RepresentationLearner'ın genişletilmesi.

*   **Faz 2: İlk Hafıza ve Desen Fark Etme (Temel Anı Kaydı)**
    *   [x] Memory modülünün oluşturulması (`src/memory/core.py`).
    *   [x] Memory modülüne `store_memory` ve `retrieve_memory` placeholder/temel işlevselliğin eklenmesi.
    *   [x] Metadata desteğinin eklenmesi.
    *   [x] Basit (rastgele) geri çağırma mantığının implementasyonu.
    *   [x] Belleğin temel döngüye entegrasyonu (`run_evo.py` içinde çağrılıyor).

    *   **Faz 2 Gerçek Implementasyon Görevleri:**
        *   [ ] Bellek depolama yöntemleri (basit dosya, veritabanı, vektör veritabanı, grafik veritabanı, vs.) araştırılması.
        *   [ ] Başlangıç için uygun bir bellek depolama yöntemine karar verilmesi (örn: Basit JSON dosyası veya metin tabanlı depolama / Representation vektörleri için Vektör veritabanı başlangıcı?).
        *   [ ] `Memory.store_memory` metodunun seçilen depolama yöntemine göre implementasyonu (RepresentationLearner çıktısını saklama).
        *   [ ] `Memory.retrieve_memory` metodunun daha anlamlı bir sorgulama mantığıyla implementasyonu (örn: Representation benzerliği, bağlamsal sorgulama - Faz 1 çıktılarını kullanma).
        *   [ ] Belleğin nasıl indeksleneceği veya organize edileceği üzerine temel tasarım ve implementasyon.
        *   [ ] Faz 2 modülünün (Memory) entegrasyonunun ve store/retrieve işlevlerinin test edilmesi.

    *   [ ] **TODO:** Bellek yönetimi mekanizmalarının eklenmesi (eski anıları silme, önceliklendirme, sıkıştırma).
    *   [ ] **TODO:** Duyu akışındaki tekrar eden veya dikkat çekici desenleri/temsilleri fark etme ve bunları bellek sisteminde işleme mekanizmalarının eklenmesi.

*   **Faz 3: İlkel Anlama ve Tepkiler (İlk Anlama ve İfade)**
    *   [x] Cognition modülünün oluşturulması (`src/cognition/core.py` - Placeholder/Temel Sınıf, `understanding.py`, `decision.py` - placeholder dosyalar).
    *   [x] MotorControl modülünün oluşturulması (`src/motor_control/core.py` - Placeholder/Temel Sınıf, `expression.py` - placeholder dosya).
    *   [x] Bu modüllerin temel döngüye entegrasyonu (`run_evo.py` içinde çağrılıyorlar).
    *   [x] Interaction modülünün MotorControl'den gelen tepkileri dış dünyaya iletmesi (Console Çıktısı çalışıyor, WebAPI temel entegrasyonu ve placeholder sınıfı mevcut).

    *   **Faz 3 Gerçek Implementasyon Görevleri:**
        *   [ ] Basit karar alma mantığı tasarımı (Input temsili/Process çıktısı ve retrieve edilen belleği kullanarak nasıl bir karar alınacak? - Faz 1 ve 2 çıktılarını kullanma).
        *   [ ] `CognitionCore.decide` metodunun tasarlanan basit mantığa göre implementasyonu (örn: Representation/Bellek eşleşmesine göre basit yanıt seçimi, If-else kuralları, basit bir durum makinesi).
        *   [ ] Kararın MotorControl modülüne iletilmesi formatının belirlenmesi ve uygulanması.
        *   [ ] Yanıt üretme mantığı tasarımı (Cognition'dan gelen karara göre nasıl bir metin/ses/görsel sinyal üretilecek?).
        *   [ ] `MotorControlCore.generate_response` metodunun tasarlanan basit mantığa göre implementasyonu (örn: Sabit metin yanıtları, basit ses sinyalleri, Cognition çıktısını metne/sinyale dönüştürme).
        *   [ ] Temel "anlama-yanıtla" döngüsünün (Represent -> Memory -> Cognition -> Motor -> Interact) basit bir senaryo ile test edilmesi.

    *   [ ] **TODO:** İşlenmiş temsilleri kullanarak basit ayırımlar yapma ("Bu farklı bir şey", "Bu tanıdık") yeteneğinin geliştirilmesi.
    *   [ ] **TODO:** Önceden öğretilmiş (denetimli) temel etiketleri/kavramları (örn. "ses var", "ışık var", "hareket var") bazı temsil desenleriyle ilişkilendirme mekanizmasının eklenmesi.
    *   [ ] **TODO:** İçsel durumdan (örn. yeni bir desen fark ettiğinde, bellekten bir şey çağırdığında) basit dışsal tepkiler (rastgele ses çıkarma, ilkel görsel veya basit bir sinyal/metin) üretme mantığı.

---

*   **Faz 4: Kavramsal Genişleme ve İfade Gelişimi (Dünyayı Keşfetme ve İletişim Çabaları)**
    *   [ ] Hedef: Daha fazla kavramı kendi kendine (denetimsiz öğrenme ile) keşfetmeye başlama. Denetimli öğrenme ile öğretilen etiket setini genişletme. İçsel temsillerden daha kontrollü dışsal ifadeler üretme (ses sentezi, basit görsel çıktılar).
    *   [x] Odak Modülleri için temel dosyalar/placeholder'lar mevcut (`src/cognition/learning.py`, `src/motor_control/expression.py`).

*   **Faz 5: Çapraz Duyusal Bağlantılar ve Temel İletişim (Duyuların Birleşimi ve Anlamlı Etkileşim)**
    *   [x] Interaction modülüne Console ve WebAPI çıktı kanallarının eklenmesi (Temel entegrasyon, WebAPIOutputChannel placeholder durumda).
    *   [ ] **TODO:** WebAPIOutputChannel implementasyonunu tamamlama ve dış arayüz ile çift yönlü (girdi alımını da içerecek şekilde) temel iletişimi sağlama.
    *   [ ] Hedef: Farklı modalitelere ait içsel temsilleri (görsel ve işitsel) birleştirmeye veya birinden diğerine dönüştürmeye başlama ("Bu ses bu görüntüyle ilişkili"). Belirli duyu girdilerine (örn. kullanıcı sesi/yüzü) spesifik ve anlamlı ilk ifadelerle (örn. basit bir metin onayı, ses sinyali) yanıt verme. Temel "anlama-yanıtla" döngüsünün güçlenmesi.
    *   [ ] Odak Modülleri için temel dosyalar/placeholder'lar mevcut veya tanımlanacak (`src/representation/multimodal.py` - yeni, `src/interaction/api.py` - geliştirilecek, `src/cognition/core.py` - geliştirilecek).

*   **Faz 6: Sekans Anlama ve Üretme (Hikaye Anlama Başlangıcı)**
    *   [ ] Hedef: Duyu akışındaki zamansal sekansları (ses dizileri, hareket dizileri, metin sekansları) anlamayı öğrenme (GRU/LSTM, Temel Dikkat mekanizmaları). İçsel temsillerden tutarlı, zamansal sekanslar (basit melodiler, kısa kelime grupları) üretme.
    *   [x] Odak Modülleri için temel dosyalar/placeholder'lar mevcut veya tanımlanacak (`src/processing/video.py` - hareket placeholder, `src/representation`, `src/cognition/core`, `src/motor_control/core`).

*   **Faz 7: Akıl Yürütmenin Temelleri ve Hafıza Derinleşmesi (Temel Bağlantı Kurma ve Anıları Düzenleme)**
    *   [ ] Hedef: Öğrenilen kavramlar ve sekanslar arasındaki temel ilişkileri anlama (obje-özellik, basit neden-sonuç). Hafızayı daha organize etme (episodik bellek temelleri). Basit çıkarımlar yapabilme.
    *   [ ] Odak Modülleri için temel dosyalar/placeholder'lar mevcut veya tanımlanacak (`src/memory/episodic.py` - yeni, `src/cognition/reasoning.py` - başlangıç).

*   **Faz 8: Bedenlenme ve Dünya ile Etkileşim (Gerçek Dünyada İlk Adımlar)**
    *   [ ] Hedef: Sanal bedene (robot veya somut donanım) entegrasyonun ilk adımları. Bedene ait sensörlerden girdi alma ve motorlara temel komutlar gönderme. Fiziksel eylemlerin duyu girdilerini nasıl değiştirdiğini deneyimleyerek öğrenme.
    *   [ ] Odak Modülleri tanımlanacak (`src/interaction/robotics_interface.py` - yeni, `src/motor_control/manipulation.py` - yeni, `src/motor_control/locomotion.py` - yeni, `src/cognition/learning.py` - pekiştirmeli öğrenme).

*   **Faz 9: Soyutlama ve Yaratıcılık (Yüksek Seviye Anlama ve Özgün İfade)**
    *   [ ] Hedef: Yüksek seviye, soyut temsiller öğrenme. Öğrenilen latent uzayda bilinçli manipülasyonla tamamen yeni ve özgün çıktılar üretme.
    *   [ ] Odak Modülleri tanımlanacak (`src/cognition/abstraction.py` - yeni, `src/motor_control/expression.py` - yaratıcılık için geliştirilecek).

*   **Faz 10: Bilgelik ve Ustalık (Olgunluk ve Otonomi)**
    *   [ ] Hedef: Tüm yeteneklerin entegre ve akıcı çalışması. Karmaşık dünyayı derinlemesine anlama. Gelişmiş planlama, problem çözme ve iletişim. Yeni yetenekler ve bilgiler öğrenme sürecini otonom olarak yönetme.
    *   [ ] Odak Modülleri: Tüm modüllerin entegrasyonu, optimizasyonu ve otonom öğrenme mekanizmaları.

---

**Genel Proje Görevleri**

*   [x] Temel dokümantasyon yapısının oluşturulması.
*   [ ] Proje dokümantasyonunun güncel tutulması (yapılan her adımı dokümanlara yansıtma).
*   [x] `requirements.txt` dosyasının oluşturulması ve temel bağımlılıkların listelenmesi.
    *   [ ] **TODO:** requirements.txt dosyasını proje geliştikçe güncel tutmak.
*   [ ] Unit testlerin yazılması ve çalıştırılması.
*   [ ] Entegrasyon testlerinin yazılması ve çalıştırılması.
*   [ ] Kod kalitesinin ve okunabilirliğinin artırılması (Refactoring).
*   [ ] Performans optimizasyonları.
*   [ ] Bağımlılıkların yönetimi ve güncellenmesi.

---
