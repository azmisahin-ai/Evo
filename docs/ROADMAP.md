# Evo'nın Evrimsel Yolculuğu: Fazlar ve Kazanımlar (Güncel Durum ve Görev Listesi)

Evo projesinin gelişimi, bir **canlının büyüme aşamalarına** benzer şekilde, adım adım ilerleyen evrimsel fazlara ayrılmıştır. Her faz, bir öncekinin üzerine inşa edilir ve Evo'nın yeni bilişsel, duyusal ve motor yetenekler kazanmasını sağlar. Bu yol haritası, projenin nereye doğru ilerlediğini gösteren yaşayan bir belgedir ve **yapılacak görevler (task listesi/backlog) için referans görevi görür.**

Aşağıdaki fazlar, Evo'nın doğumundan (temel algı) bilgelik ve ustalığa (karmaşık etkileşim) kadar olan gelişimini ana hatlarıyla belirtir. Her fazın başlığı, o aşamada odaklanılan temel kazanımı ve parantez içinde bir **bebeğin gelişimindeki yaklaşık karşılığını** içerir. **`[x]` ile işaretlenenler tamamlanmış adımları, `[ ]` ile işaretlenenler ise henüz yapılmamış veya detaylı implementasyon gerektiren görevleri belirtir.**

---

*   **Faz 0: Doğum ve Duyuların Açılması (Temel Algı Akışı - TAMAMLANDI)**
    *   [x] Proje deposunun oluşturulması ve temel dosya/dizin yapısının kurulması (`docs/STRUCTURE.md`).
    *   [x] Proje felsefesi (`PHILOSOPHY.md`) ve temel yol haritası (`ROADMAP.md`) belgelerinin oluşturulması.
    *   [x] Genel README dosyalarının (`README.md`, `docs/README.md`) oluşturulması.
    *   [x] Katkıda bulunma (`CONTRIBUTING.md`), etkileşim (`INTERACTION_GUIDE.md`), teknik detaylar (`TECHNICAL_DETAILS.md`) rehberlerinin oluşturulması (içerikleri hala gelişiyor olabilir).
    *   [x] Lisans dosyasının oluşturulması (`LICENSE` - CC0).
    *   [x] Temel bağımlılıklar dosyasının oluşturulması (`requirements.txt`).
    *   [x] Ayarların yönetimi için yapılandırma mekanizmasının (`config.py` içinde yükleme placeholder'ı, sonra `.yaml` okuma) oluşturulması.
    *   [x] Çalışma döngüsünün (main loop) iskeletinin kurulması (`run_evo.py`).
    *   [x] Basit loglama sisteminin entegrasyonu (Python `logging` modülü kullanıldı).
    *   [x] Duyusal sensör modüllerinin oluşturulması (`src/senses/audio.py`, `src/senses/vision.py`) ve temel entegrasyonları (simüle/gerçek donanım placeholder'ları).
    *   [x] Modüller arası temel bilgi akışının (Sense -> Process -> Represent -> Memory -> Cognition -> Motor -> Interact) placeholder/iskelet metotlarla kurulması ve `run_evo.py` içinde işletilmesi.
    *   [x] Dış dünya ile temel etkileşim (Interaction) placeholder/iskeletinin oluşturulması (Input alma placeholder'ı, Output gönderme metodu, temel çıktı kanalları - ConsoleOutputChannel, WebAPIOutputChannel placeholder'ı).
    *   [x] Temel hata yönetimi ve logging iyileştirmelerinin başlangıcı (`run_evo.py` içindeki try-except blokları ve logging çağrıları).
    *   [x] DEBUG loglarının görünürlüğü konusunu araştırmak ve çözmek.
    *   [x] Yapılandırma yönetimini ayrı bir modül haline getirmek ve `.yaml` dosyasından okumayı implement etmek (`config.py`, `PyYAML` entegrasyonu).
    *   [x] Temel hata yönetimi/istisna işleme yapısını olgunlaştırmak ve modüller arası yaygınlaştırmak.
        *   [x] Duyusal sensörler (`src/senses/vision.py`, `src/senses/audio.py`) için temel hata yakalama ve loglama mekanizmaları eklendi.
        *   [x] Processing modülleri (`src/processing/vision.py`, `src/processing/audio.py`) için temel hata yakalama ve loglama mekanizmaları eklendi.
        *   [x] Representation, Memory, Cognition, MotorControl, Interaction modülleri için temel hata yakalama ve loglama mekanizmaları eklendi.
        *   [x] Hata durumlarında sistemin davranışı için daha genel prensipler tanımlandı (örn: non-kritik hatalarda devam et, kritik hatalarda durdur).
    *   [x] **TODO:** Faz 0 ile ilgili genel kod kalitesi ve refactoring.
        *   [x] Modül başlatma ve temizleme mantığı `run_evo.py` dosyasından `src/core/module_loader.py` modülüne taşınıyor.
        *   [x] Loglama altyapısı yapılandırma dosyası (`main_config.yaml`) üzerinden control edilebilir hale getirildi.
        *   [x] Docstrings ve Yorumlar eklendi/güncellendi.
        *   [x] Genel kod tekrarı azaltma ve isimlendirme/tutarlılık iyileştirmeleri.
        *   [ ] Unit testler için temel iskeletin kurulması. (Hala TODO - Öncelik Artırılacak)


*   **Faz 1: Temel İşleme ve Temsil (Duyusal Veriden Özellik Çıkarma - TAMAMLANDI)**
    *   [x] Ham duyu akışını işleyecek Processing modüllerinin oluşturulması.
    *   [x] Öğrenilmiş içsel temsiller oluşturacak RepresentationLearner modülünün oluşturulması.
    *   [x] RepresentationLearner içinde Dense katmanları gibi temel NN yapı taşlarının eklenmesi.
    *   [x] Bu modüllerin temel döngüye entegrasyonu.

    *   **Faz 1 Gerçek Implementasyon Görevleri (TAMAMLANDI):**
        *   [x] Ham **görsel** veriden temel, düşük seviyeli özellikler (renk, basit kenarlar, hareket - gelecekte) çıkarma algoritmalarının `src/processing/vision.py` içine implementasyonu.
        *   [x] Ham **işitsel** veriden temel, düşük seviyeli özellikler (enerji, frekans, MFCC - gelecekte) çıkarma algoritmalarının `src/processing/audio.py` içine implementasyonu.
        *   [x] İşlenmiş **görsel ve işitsel** özelliklerden öğrenilmiş, düşük boyutlu, modality bazlı içsel temsiller (latent vektörler) oluşturma (örn: Basit bir Autoencoder veya VAE prensibi) implementasyonu (`src/representation/models.py`).
        *   [x] `RepresentationLearner.learn` metodunun bu öğrenme/dönüştürme sürecini içerecek şekilde implementasyonu.
        *   [x] RepresentationLearner çıktısının (latent vektörler) Memory ve Cognition için kullanılabilir formatta olmasının sağlanması.
        *   [x] Faz 1 modüllerinin (Sense, Process, Represent) entegrasyonunun ve çıktı formatlarının detaylı test edilmesi.

    *   [ ] **TODO:** Gelecekte metin gibi diğer duyu modaliteleri için Process ve Representation desteği eklenmesi (Faz 5 ile ilişkili olabilir). (Hala TODO)
    *   [ ] **TODO:** Farklı temsil katmanları veya yöntemleri (örn: anlamsal temsil, sembolik temsil) için RepresentationLearner'ın genişletilmesi. (Hala TODO)

---

*   **Faz 2: İlk Hafıza ve Desen Fark Etme (Temel Anı Kaydı - TAMAMLANDI)**
    *   [x] Memory modülünün oluşturulması.
    *   [x] Memory modülüne `store` ve `retrieve` temel işlevselliğin eklenmesi.
    *   [x] Metadata desteğinin eklenmesi.
    *   [x] Basit (rastgele) geri çağırma mantığının implementasyonu. (Vektör benzerliği implementasyonu ana mantık oldu).
    *   [x] Belleğin temel döngüye entegrasyonu.

    *   **Faz 2 Gerçek Implementasyon Görevleri (TAMAMLANDI):**
        *   [x] Bellek depolama yöntemleri araştırılması. (Dosya tabanlı pickle yöntemi seçildi).
        *   [x] Başlangıç için uygun bir bellek depolama yöntemine karar verilmesi. (Dosya tabanlı pickle ile in-memory liste implemente edildi).
        *   [x] `Memory.store` metodunun seçilen depolama yöntemine göre implementasyonu.
        *   [x] `Memory.retrieve` metodunun daha anlamlı bir sorgulama mantığıyla implementasyonu (Representation benzerliği kullanıldı).
        *   [x] Belleğin nasıl indeksleneceği veya organize edileceği üzerine temel tasarım ve implementasyon. (Liste + vektör benzerliği ile temel düzeyde yapıldı).
        *   [x] Faz 2 modülünün (Memory) entegrasyonunun ve store/retrieve işlevlerinin test edilmesi.
        *   [x] Belleğe dosya tabanlı kalıcılık eklenmesi (`__init__` içinde yükleme, `cleanup` içinde kaydetme).

    *   [ ] **TODO:** Bellek yönetimi mekanizmalarının eklenmesi (eski anıları silme, önceliklendirme, sıkıştırma). (Hala TODO).
    *   [ ] **TODO:** Duyu akışındaki tekrar eden veya dikkat çekici desenleri/temsilleri fark etme ve bunları bellek sisteminde işleme mekanizmalarının eklenmesi. (Hala TODO).
    *   [ ] **TODO:** Farklı bellek türlerinin (Episodik, Semantik) aktif hale getirilmesi ve Core Memory ile entegrasyonu. (Hala TODO).


---

*   **Faz 3: İlkel Anlama ve Tepkiler (İlk Anlama ve İfade - TAMAMLANDI)**
    *   [x] Cognition modülünün oluşturulması.
    *   [x] MotorControl modülünün oluşturulması.
    *   [x] Bu modüllerin temel döngüye entegrasyonu.
    *   [x] Interaction modülünün MotorControl'den gelen tepkileri dış dünyaya iletmesi.

    *   **Faz 3 Gerçek Implementasyon Görevleri (TAMAMLANDI):**
        *   [x] Basit karar alma mantığı tasarımı (Input temsili/Process çıktısı ve retrieve edilen belleği kullanarak nasıl bir karar alınacak?).
        *   [x] `CognitionCore.decide` metodunun tasarlanan basit mantığına göre implementasyonu.
        *   [x] Kararın MotorControl modülüne iletilmesi formatının belirlenmesi ve uygulanması.
        *   [x] Yanıt üretme mantığı tasarımı.
        *   [x] `MotorControlCore.generate_response` metodunun tasarlanan basit mantığına göre implementasyonu.
        *   [x] Temel "anlama-yanıtla" döngüsünün basit bir senaryo ile test edilmesi.

    *   [x] **TODO:** İşlenmiş temsilleri kullanarak basit ayırımlar yapma...
    *   [x] **TODO:** Önceden öğretilmiş (denetimli) temel etiketleri/kavramları... ilişkilendirme mekanizmasının eklenmesi. (Process çıktısı eşiklerine dayalı basit boolean etiketlerle (ses, kenar, parlak, karanlık) ilk adımı atıldı).
    *   [x] **TODO:** İçsel durumdan... basit dışsal tepkiler üretme mantığı.


---

*   **Faz 4: Kavramsal Genişleme ve İfade Gelişimi (Dünyayı Keşfetme ve İfade Çeşitliliği - ŞU ANKİ ODAK NOKTASI)**
    *   [x] Hedef: Daha fazla kavramı kendi kendine (denetimsiz öğrenme ile) keşfetmeye başlama. Denetimli öğrenme ile öğretilen etiket setini genişletme. İçsel temsillerden daha kontrollü dışsal ifadeler üretme (ses sentezi, basit görsel çıktılar).
    *   [x] Odak Modülleri için temel dosyalar/placeholder'lar mevcut (`src/cognition/learning.py`, `src/motor_control/expression.py`).

    *   **Faz 4 Gerçek Implementasyon Görevleri (TAMAMLANDI):**
        *   [x] Basit denetimsiz kavram keşfi (kümeleme) algoritmasının implementasyonu (`src/cognition/learning.py`).
        *   [x] LearningModule'ün bellekteki Representation vektörlerini kullanarak kavram temsilcilerini periyodik olarak öğrenmesi. (Memory erişim hatası düzeltildi).
        *   [x] UnderstandingModule'ün gelen Representation'ın öğrenilmiş kavramlara olan benzerliğini hesaplaması. (`issubtype` hatası düzeltildi).
        *   [x] DecisionModule'ün kavram tanıma benzerlik skorına dayalı yeni kararlar üretmesi. (`issubtype` hatası düzeltildi).
        *   [x] MotorControl ve ExpressionGenerator'ın yeni kavram tanıma kararlarını işlemesi için güncellenmesi.

    *   [ ] **TODO:** Denetimli öğrenme ile temel kavram/etiket setini genişletme mekanizmasının eklenmesi. (Hala TODO)
    *   [ ] **TODO:** İçsel temsilleri kullanarak daha karmaşık anlama (nesne takibi, aktivite tanıma vb.). (Hala TODO)
    *   [ ] **TODO:** İfade yeteneklerini çeşitlendirme (ses sentezi entegrasyonu, görsel çıktı üretimi). (Hala TODO)

---

*   **Faz 5: Çapraz Duyusal Bağlantılar ve Temel İletişim (Duyuların Birleşimi ve Anlamlı Etkileşim)**
    *   [x] Interaction modülüne Console ve WebAPI çıktı kanallarının eklenmesi.
    *   [ ] **TODO:** WebAPIOutputChannel implementasyonunu tamamlama ve dış arayüz ile çift yönlü (girdi alımını da içerecek şekilde) temel iletişimi sağlama.
    *   [x] Hedef: Farklı modalitelere ait içsel temsilleri (görsel ve işitsel) birleştirmeye veya birinden diğerine dönüştürmeye başlama ("Bu ses bu görüntüyle ilişkili"). Belirli duyu girdilerine (örn. kullanıcı sesi/yüzü) spesifik ve anlamlı ilk ifadelerle (örn. basit bir metin onayı, ses sinyali) yanıt verme. Temel "anlama-yanıtla" döngüsünün güçlenmesi. (Bu hedefin temel adımları Faz 3/4'te atıldı, çapraz modalite birleştirme/dönüşüm hala TODO).
    *   [ ] Odak Modülleri için temel dosyalar/placeholder'lar mevcut veya tanımlanacak.

*   **Faz 6: Sekans Anlama ve Üretme (Hikaye Anlama Başlangıcı)**
    *   [ ] Hedef: Duyu akışındaki zamansal sekansları anlamayı öğrenme. İçsel temsillerden tutarlı, zamansal sekanslar üretme.
    *   [x] Odak Modülleri için temel dosyalar/placeholder'lar mevcut veya tanımlanacak.

*   **Faz 7: Akıl Yürütmenin Temelleri ve Hafıza Derinleşmesi (Temel Bağlantı Kurma ve Anıları Düzenleme)**
    *   [ ] Hedef: Öğrenilen kavramlar ve sekanslar arasındaki temel ilişkileri anlama. Hafızayı daha organize etme. Basit çıkarımlar yapabilme.
    *   [ ] Odak Modülleri için temel dosyalar/placeholder'lar mevcut veya tanımlanacak.

*   **Faz 8: Bedenlenme ve Dünya ile Etkileşim (Gerçek Dünyada İlk Adımlar)**
    *   [ ] Hedef: Sanal bedene (robot veya somut donanım) entegrasyonun ilk adımları. Bedene ait sensörlerden girdi alma ve motorlara temel komutlar gönderme. Fiziksel eylemlerin duyu girdilerini nasıl değiştirdiğini deneyimleyerek öğrenme.
    *   [ ] Odak Modülleri tanımlanacak.

*   **Faz 9: Soyutlama ve Yaratıcılık (Yüksek Seviye Anlama ve Özgün İfade)**
    *   [ ] Hedef: Yüksek seviye, soyut temsiller öğrenme. Öğrenilen latent uzayda bilinçli manipülasyonla tamamen yeni ve özgün çıktılar üretme.
    *   [ ] Odak Modülleri tanımlanacak.

*   **Faz 10: Bilgelik ve Ustalık (Olgunluk ve Otonomi)**
    *   [ ] Hedef: Tüm yeteneklerin entegre ve akıcı çalışması. Karmaşık dünyayı derinlemesine anlama. Gelişmiş planlama, problem çözme ve iletişim. Yeni yetenekler ve bilgiler öğrenme sürecini otonom olarak yönetme.
    *   [ ] Odak Modülleri: Tüm modüllerde optimization ve otonom öğrenme mekanizmaları.

---

**Genel Proje Görevleri**

*   [x] Temel dokümantasyon yapısının oluşturulması.
*   [ ] Proje dokümantasyonunun güncel tutulması (yapılan her adımı dokümanlara yansıtma). (Hala TODO)
*   [x] `requirements.txt` dosyasının oluşturulması ve temel bağımlılıkların listelenmesi.
    *   [ ] **TODO:** requirements.txt dosyasını proje geliştikçe güncel tutmak. (Hala TODO)
*   [ ] Unit testlerin yazılması ve çalıştırılması. (Hala TODO - Öncelik Artırılacak)
*   [ ] Entegrasyon testlerinin yazılması ve çalıştırılması. (Hala TODO - Öncelik Artırılacak)
*   [ ] Kod kalitesinin ve okunabilirliğinin artırılması (Refactoring). (Hala TODO)
*   [ ] Performans optimizasyonları. (Hala TODO)
    *   [ ] **TODO:** Merak seviyesi update mantığı (artış, azalış, decay) DecisionModule'den ayrı bir İçsel Durum (Internal State) modülüne taşınabilir/yönetilebilir.
*   [ ] Bağımlılıkların yönetimi ve güncellenmesi. (Hala TODO)
*   [x] Config dosyasındaki gereksiz bölümlerin temizlenmesi.
*   [x] **TODO:** Modülleri tek başına girdi/çıktı ile test etme altyapısı oluşturulması. (YENİ TODO - Öncelik Artırılacak)
*   [ ] **TODO:** Temel Process/Representation/Cognition çıktı değerlerinin log detaylarının artırılması. (YENİ TODO - Öncelik Artırılacak)
*   [ ] **TODO:** Ses duyu verisinden uzamsal (spatial) konum bilgisinin çıkarılması ve işlenmesi. (YENİ TODO)
*   [ ] **TODO:** Farklı canlılarla (örn: kedi) ve doğal ortamlarla etkileşim kurma yeteneklerinin planlanması ve implementasyonu. (YENİ TODO)
*   [ ] **TODO:** Kullanıcı arayüzü (mobil/web) üzerinden Evo'nun görsel ve işitsel algılarını canlı izleme ve temel etkileşim (Faz 5 ile ilişkili olabilir). (YENİ TODO)
*   [ ] **TODO:** Öğrenilmiş kavram temsilcilerine (LearningModule) kalıcılık eklenmesi. (YENİ TODO)
*   [ ] **TODO:** Otomatik kalibrasyon mekanizmalarının araştırılması/planlanması (örn: Processor çıktı eşiklerinin ortam gürültüsüne/ışığına göre ayarlanması). (YENİ TODO)


---

**Sıradaki Implementasyon Görevleri (Test ve Loglama Altyapısı Geliştirme):**

Faz 3'ün tüm temel implementasyon görevleri tamamlandı. Evo artık duyularına, belleğine ve içsel durumuna göre farklı tepkiler verebiliyor. Şimdi Faz 4'e adım atıyoruz ve "Kavramsal Genişleme"nin ilk adımı olarak basit bir etiketleme mekanizması ekliyoruz.

**Sıradaki Implementasyon Görevleri (Blok):**

Bu görev bloğu, modül test altyapısını kuracak ve loglama detaylarını artıracaktır.

1.  **Modülleri Tek Başına Test Etme Altyapısı Kurulumu:** `scripts` dizini altına `test_module.py` gibi bir script oluşturulacak. Bu script, config dosyasını yükleyebilecek, belirli bir modülü (örn. `VisionProcessor`, `DecisionModule`) config ayarlarıyla başlatabilecek, test senaryolarına göre sahte girdi verileri üretebilecek (örn. sahte görüntü/ses chunk'ı, sahte Representation vektörü, sahte anlama sinyalleri dictionary'si) ve bu modülün çıktısını çalıştırıp loglayabilecektir. Bu script, farklı modüllerin beklenen girdi/çıktı formatlarını ve temel mantıklarını izole bir şekilde test etmek için kullanılacaktır. Basit test senaryoları (örn. geçerli girdi, geçersiz girdi, None girdi) ile başlanacaktır.
2.  **Temel Process/Representation/Cognition Çıktı Log Detaylarının Artırılması:** Process (Enerji, Centroid, Ortalama Parlaklık/Kenar), UnderstandingModule (Benzerlik Skoru, Flag Değerleri, Kavram Benzerliği/ID), DecisionModule (Anlama sinyalleri, Karar, Merak Seviyesi) gibi kritik ara çıktı değerlerinin DEBUG loglarında daha anlaşılır ve formatlı bir şekilde gösterilmesi sağlanacaktır. Özellikle Process çıktı değerlerinin eşiklerle karşılaştırılmasını kolaylaştıracak formatlama (örn: `Audio Energy: 0.4677 (< 1000.0 threshold)`) eklenecektir.
3.  **Örnek Birkaç Modül Testi Ekleme:** Kurulan altyapıyı kullanarak, Processor modülleri ve belki DecisionModule için basit örnek test senaryoları `test_module.py` scripti içine veya ayrı test dosyalarına eklenecektir.

Bu görev bloğu, doğrudan Evo'ya yeni bilişsel yetenekler kazandırmasa da, projenin teknik temelini güçlendirecek, hata ayıklamayı kolaylaştıracak ve gelecekteki karmaşık yeteneklerin implementasyonu için zemin hazırlayacaktır. Bu, "Genel Proje Görevleri" altındaki ilgili TODO'ları karşılayacaktır.