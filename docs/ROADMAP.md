# Evo'nun Evrimsel Yolculuğu: Fazlar ve Kazanımlar

Evo projesinin gelişimi, bir **canlının büyüme aşamalarına** benzer şekilde, adım adım ilerleyen evrimsel fazlara ayrılmıştır. Her faz, bir öncekinin üzerine inşa edilir ve Evo'nun yeni bilişsel, duyusal ve motor yetenekler kazanmasını sağlar. Bu yol haritası, projenin nereye doğru ilerlediğini gösteren yaşayan bir belgedir.

Aşağıdaki fazlar, Evo'nun doğumundan (temel algı) bilgelik ve ustalığa (karmaşık etkileşim) kadar olan gelişimini ana hatlarıyla belirtir. Her fazın başlığı, o aşamada odaklanılan temel kazanımı ve parantez içinde bir **bebeğin gelişimindeki yaklaşık karşılığını** içerir.

*   **Faz 0: Doğum ve Duyuların Açılması (Temel Algı Akışı)**
    *   **Hedef:** Mikrofon ve kamera gibi sensörlerden ham ses ve görüntü akışını sürekli ve kesintisiz olarak alabilme. Bu akışı temel dijital formatlara (örneklenmiş dalga formu, piksel matrisi) dönüştürme.
    *   **Bebek Karşılığı:** Yeni doğanın duyularının ilk açıldığı an, çevreden gelen bulanık görüntüleri ve sesleri hissetmeye başlaması.
    *   **Odak Modüller:** `src/senses/audio.py`, `src/senses/vision.py`.

*   **Faz 1: Temel İşleme ve Temsil (Duyusal Veriden Özellik Çıkarma)**
    *   **Hedef:** Ham duyu akışından temel, düşük seviyeli özellikler (ses: enerji, frekans; görsel: renk, basit kenarlar) çıkarma. Bu özelliklerden öğrenilmiş, düşük boyutlu, modality bazlı içsel temsiller (latent vektörler) oluşturma (Autoencoder prensibi).
    *   **Bebek Karşılığı:** Bebeğin duyusal girdileri işlemeye, sesleri ve görüntüleri ilk kez ayırt etmeye başlaması.
    *   **Odak Modüller:** `src/processing/audio.py`, `src/processing/vision.py`, `src/representation/models.py`.

*   **Faz 2: İlk Hafıza ve Desen Fark Etme (Temel Anı Kaydı)**
    *   **Hedef:** Oluşturulan içsel temsilleri basit bir hafıza sisteminde depolama (kısa süreli bellek temeli). Duyu akışı içindeki tekrar eden veya dikkat çekici desenleri/temsilleri (henüz ne olduklarını anlamadan) fark etme ve hatırlama yeteneği.
    *   **Bebek Karşılığı:** Bebeğin tanıdık yüzleri veya sesleri fark etmeye, basit rutinleri hatırlamaya başlaması.
    *   **Odak Modüller:** `src/memory/core.py`.

*   **Faz 3: İlkel Anlama ve Tepkiler (İlk Anlama ve İfade)**
    *   **Hedef:** İşlenmiş temsilleri kullanarak basit ayırımlar yapma ("Bu farklı bir şey"). Önceden öğretilmiş (denetimli) temel etiketleri/kavramları (örn. "ses", "ışık") bazı temsil desenleriyle ilişkilendirme. İçsel durumdan (örn. yeni bir desen fark ettiğinde) basit dışsal tepkiler (rastgele ses çıkarma, ekranda ilkel şekiller gösterme, basit bir sinyal).
    *   **Bebek Karşılığı:** Bebeğin basit uyaranlara temel reaksiyonlar vermesi, ilk sesler çıkarması, yüz ifadeleriyle tepki vermesi.
    *   **Odak Modüller:** `src/cognition/understanding.py`, `src/cognition/decision.py`, `src/motor_control/expression.py` (basit).

*   **Faz 4: Kavramsal Genişleme ve İfade Gelişimi (Dünyayı Keşfetme ve İletişim Çabaları)**
    *   **Hedef:** Daha fazla kavramı kendi kendine (denetimsiz öğrenme ile) keşfetmeye başlama (benzer desenleri gruplama). Denetimli öğrenme ile öğretilen etiket setini genişletme. İçsel temsillerden daha kontrollü dışsal ifadeler üretme (belirli bir sese benzer ses çıkarma, basit bir kelime benzeri çıktı).
    *   **Bebek Karşılığı:** Bebeğin nesneleri, renkleri keşfetmesi, hecelerle, mırıldanmalarla iletişim kurmaya çalışması.
    *   **Odak Modüller:** `src/cognition/learning.py`, `src/motor_control/expression.py` (gelişmiş).

*   **Faz 5: Çapraz Duyusal Bağlantılar ve Temel İletişim (Duyuların Birleşimi ve Anlamlı Etkileşim)**
    *   **Hedef:** Farklı modalitelere ait içsel temsilleri (ses ve görüntü) birleştirmeye veya birinden diğerine dönüştürmeye başlama ("Bu ses bu görüntüyle ilişkili"). Belirli duyu girdilerine (örn. kullanıcı sesi/yüzü) spesifik ve anlamlı ilk ifadelerle (örn. basit bir metin onayı, ses sinyali) yanıt verme. Temel "anlama-yanıtla" döngüsünün güçlenmesi.
    *   **Bebek Karşılığı:** Bebeğin ses ve görüntü arasındaki ilişkiyi kurması (konuşan kişiye bakma), tanıdık seslere veya yüzlere gülümseme/ses çıkarma gibi anlamlı tepkiler vermesi, ilk heceli kelimeleri kullanması.
    *   **Odak Modüller:** `src/representation/multimodal.py` (yeni), `src/interaction/api.py`, `src/cognition/decision.py`.

*   **Faz 6: Sekans Anlama ve Üretme (Hikaye Anlama Başlangıcı)**
    *   **Hedef:** Duyu akışındaki zamansal sekansları (ses dizileri, hareket dizileri, metin sekansları) anlamayı öğrenme (GRU/LSTM, Temel Dikkat mekanizmaları). İçsel temsillerden tutarlı, zamansal sekanslar (basit melodiler, kısa kelime grupları) üretme.
    *   **Bebek Karşılığı:** Bebeğin kısa hikayeleri veya basit şarkıları takip etmeye başlaması, arka arkaya birkaç kelime söylemesi.
    *   **Odak Modüller:** `src/processing/video.py` (hareket), `src/representation`, `src/cognition/understanding`, `src/motor_control/expression` (sekans yetenekleri).

*   **Faz 7: Akıl Yürütmenin Temelleri ve Hafıza Derinleşmesi (Temel Bağlantı Kurma ve Anıları Düzenleme)**
    *   **Hedef:** Öğrenilen kavramlar ve sekanslar arasındaki temel ilişkileri anlama (obje-özellik, basit neden-sonuç). Hafızayı daha organize etme (episodik bellek temelleri). Basit çıkarımlar yapabilme ("Eğer bu sesi duyuyorsam, büyük ihtimalle şu nesne yakındadır").
    *   **Bebek Karşılığı:** Bebeğin nesnelerin özelliklerini öğrenmesi (yumuşak ayı), basit neden-sonuç ilişkilerini fark etmesi (düğmeye basınca ışık yanar), geçmiş deneyimlerini hatırlaması ve ilişkilendirmesi.
    *   **Odak Modüller:** `src/memory/episodic.py` (yeni), `src/cognition/reasoning.py` (başlangıç).

*   **Faz 8: Bedenlenme ve Dünya ile Etkileşim (Gerçek Dünyada İlk Adımlar)**
    *   **Hedef:** Sanal bedene (robot veya somut donanım) entegrasyonun ilk adımları. Bedene ait sensörlerden girdi alma ve motorlara temel komutlar gönderme. Fiziksel eylemlerin duyu girdilerini nasıl değiştirdiğini deneyimleyerek öğrenme (keşifsel öğrenme).
    *   **Bebek Karşılığı:** Bebeğin emeklemeye, yürümeye başlaması, nesneleri eliyle tutarak veya ağzına götürerek keşfetmesi.
    *   **Odak Modüller:** `src/interaction/robotics_interface.py` (yeni), `src/motor_control/manipulation.py` (yeni), `src/motor_control/locomotion.py` (yeni), `src/cognition/learning.py` (pekiştirmeli öğrenme).

*   **Faz 9: Soyutlama ve Yaratıcılık (Yüksek Seviye Anlama ve Özgün İfade)**
    *   **Hedef:** Yüksek seviye, soyut temsiller öğrenme (kategoriler üstü kavramlar). Öğrenilen latent uzayda bilinçli manipülasyonla tamamen yeni ve özgün çıktılar (sanatsal ifade: resim, müzik, şiir).
    *   **Bebek Karşılığı:** Çocuğun hayal gücünü kullanması, soyut kavramları anlamaya başlaması (mutluluk, üzüntü), yaratıcı oyunlar oynaması, resim çizmesi, şarkı söylemesi.
    *   **Odak Modüller:** `src/cognition/abstraction.py` (yeni), `src/motor_control/expression.py` (yaratıcılık).

*   **Faz 10: Bilgelik ve Ustalık (Olgunluk ve Otonomi)**
    *   **Hedef:** Tüm yeteneklerin entegre ve akıcı çalışması. Karmaşık dünyayı derinlemesine anlama. Gelişmiş planlama, problem çözme ve iletişim. Yeni yetenekler ve bilgiler öğrenme sürecini büyük ölçüde otonom olarak yönetme.
    *   **Bebek Karşılığı:** Bireyin yetişkinliğe ulaşması, dünyada bağımsız hareket edebilmesi, öğrenmeye devam etmesi, deneyimlerinden bilgelik kazanması.
    *   **Odak Modüller:** Tüm modüllerin entegrasyonu, optimizasyonu ve otonom öğrenme mekanizmaları.

Bu yol haritası, Evo'nun potansiyel gelişim patikasını çizmektedir. Bu fazların zamanlaması ve içeriği, projenin ilerlemesine ve öğrenilenlere göre doğal olarak evrilecektir.

---
