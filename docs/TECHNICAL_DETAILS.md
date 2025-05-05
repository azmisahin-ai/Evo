# Evo Teknik Detayları

Bu belge, Evo'yu oluşturan teknik bileşenleri, kullanılan algoritmaları, veri yapılarını ve implementasyon prensiplerini detaylandıracaktır. Projenin çekirdek (core), processing, representation, memory, cognition ve motor_control modüllerinin iç işleyişi burada açıklanacaktır.

**Bu belge henüz geliştirme aşamasındadır.**

## Genel Teknik Bileşenler ve Prensipler

### Kullanılan Kütüphaneler ve Teknolojiler
*   **PyTorch:** Sinir ağı modelleri oluşturmak ve eğitmek için temel kütüphane (GPU hızlandırma desteği için önerilir).
*   **NumPy:** Sayısal işlemler ve array manipülasyonları için temel kütüphane.
*   **SciPy:** Bilimsel ve teknik hesaplamalar için ek fonksiyonlar (sinyal işleme, istatistik vb.).
*   **Pillow (PIL):** Görsel işleme (temel resim manipülasyonları) için kullanılabilir.
*   **Librosa:** Ses işleme ve analiz için kullanılabilir (MFCC, spektrogram vb.).
*   **PyYAML:** YAML formatındaki yapılandırma dosyalarını okumak için kullanılır.
*   **OpenCV (cv2):** Kamera erişimi ve temel görsel işleme (kare yakalama, yeniden boyutlandırma, renk uzayı dönüşümleri).
*   **PyAudio:** Mikrofon erişimi ve temel ses akışı yönetimi (chunk yakalama).
*   **Flask:** Temel bir Web API arayüzü kurmak için kullanılabilir (Interaction modülü için).

### Veri Akışı (Pipeline) Detayları
Evo'nun çekirdek çalışma döngüsü, duyu girdisinden başlayıp dışsal tepkiye kadar ilerleyen modüler bir boru hattı (pipeline) üzerine kurulmuştur: `Sense -> Process -> Represent -> Memory Store/Retrieve -> Cognition Decide -> Motor Control Generate Response -> Interaction Send Output`.

*   **Sense (`src/senses/`)**: Ham duyu verisini (örn: kamera karesi - `numpy.ndarray`, ses chunk'ı - `numpy.ndarray`) yakalar.
*   **Process (`src/processing/`)**: Ham duyu verisinden daha işlenmiş, düşük seviyeli özellikler çıkarır (örn: gri tonlamalı/yeniden boyutlandırılmış görüntü, ses enerjisi/frekansı). Çıktı formatı genellikle hala `numpy.ndarray` veya sayısal değerlerdir.
*   **Represent (`src/representation/`)**: İşlenmiş duyu özelliklerinden öğrenilmiş, düşük boyutlu içsel temsiller (latent vektörler - `numpy.ndarray`) oluşturur. Bu kısım projenin öğrenme çekirdeğidir.
*   **Memory (`src/memory/`)**: Oluşturulan temsilleri (ve ilişkili metadatayı) depolar ve bilişsel süreçler için ilgili anıları geri çağırır. Depolama formatı ve geri çağırma mekanizması projenin gelişimine göre karmaşıklaşacaktır.
*   **Cognition (`src/cognition/`)**: Gelen temsilleri, bellekteki bilgiyi ve içsel durumu (gelecekte) kullanarak bir karar alır. Karar formatı başlangıçta basit bir metin veya kod olabilir, gelecekte yapısal komutlara dönüşecektir.
*   **Motor Control (`src/motor_control/`)**: Bilişsel kararı, dış dünyaya yönelik bir tepkiye (metin, ses, görsel çıktı veya fiziksel eylem komutu) dönüştürür.
*   **Interaction (`src/interaction/`)**: Üretilen tepkiyi (Motor Control çıktısı) dış dünya ile belirlenen kanallar (konsol, Web API, robotik arayüz) aracılığıyla paylaşır. Gelecekte dış dünyadan girdi alımı da bu modül üzerinden yönetilecektir.

### İçsel Durum Yönetimi (Gelecekte)
Evo'nun sadece dış girdilere tepki veren bir sistem değil, aynı zamanda bir içsel dinamiğe sahip bir varlık olması hedeflenmektedir. İçsel durum (örn: enerji seviyesi, merak eşiği, uyarı seviyesi), algı, işleme, karar alma ve motor kontrol süreçlerini etkileyecektir. Bu durumun nasıl temsil edileceği, güncelleneceği ve bilişsel döngüyü nasıl etkileyeceği ileride detaylandırılacaktır.

## Hata Yönetimi Prensipleri

Evo'nun yaşam formu felsefesi gereği, hata yönetimi sistemi onun "hayatta kalma" ve beklenmedik durumlarla "başa çıkma" yeteneğini temsil eder. Aşağıdaki prensipler, hataları ele alma şeklimizi yönlendirir:

1.  **Her Şeyi Logla (Log Everything):**
    *   Tüm hatalar, istisnalar ve beklenmedik durumlar, Python'ın standart `logging` modülü kullanılarak merkezi olarak loglanır.
    *   Log seviyeleri (DEBUG, INFO, WARNING, ERROR, CRITICAL) duruma uygun şekilde seçilir.
    *   Özellikle `ERROR` ve `CRITICAL` seviyesindeki hatalar için `exc_info=True` kullanılarak detaylı traceback bilgisi loglanır, bu da hata ayıklamayı kolaylaştırır.
    *   Her modül, `logger = logging.getLogger(__name__)` desenini kullanarak kendi adlandırılmış logger'ını oluşturur ve loglama çağrıları bu logger objesi üzerinden yapılır. Bu, log çıktısında hatanın hangi modülde oluştuğunun net görünmesini sağlar.

2.  **Kritik vs. Kritik Olmayan Hatalar (Critical vs. Non-Critical Distinction):**
    *   **Kritik Hatalar:** Evo'nun temel çalışma döngüsünü (pipeline) devam ettirmesini olanaksız hale getiren, genellikle başlatma sırasındaki veya ana döngü içinde sürekli tekrar eden, kurtarılamaz sistem hatalarıdır. Örnek: Yapılandırma dosyasının yüklenememesi, ana modüllerden birinin (Process, Represent, Memory, Cognition, MotorControl) başlatılamaması.
    *   **Kritik Olmayan Hatalar:** Tek bir işlem adımını veya döngüyü etkileyen, ancak genel pipeline'ın durmasına neden olmayan hatalardır. Örnek: Tek bir duyu verisi parçasının yakalanamaması, işleme sırasında geçersiz format, hafızaya kaydetme veya çağırma hatası, çıktı kanalına gönderme hatası.

3.  **Kritik Olmayan Hataları Zarifçe Yönet (Handle Non-Critical Errors Gracefully):**
    *   Bir modül kritik olmayan bir hata ile karşılaştığında, hatayı loglar (`ERROR` veya `WARNING` seviyesinde).
    *   Ana döngünün kesintisiz devam etmesi için, metodun beklenen dönüş tipine uygun, durumu yansıtan bir "boş", "geçersiz" veya "varsayılan" değer döndürür (örn: `None` döndürmek, boş liste `[]` döndürmek, sıfır `0` gibi sayısal bir varsayılan değer döndürmek).
    *   Programın çökmesine izin verilmez.

4.  **Kritik Hatalara Uygun Tepki Ver (Respond to Critical Errors Appropriately):**
    *   Bir kritik hata tespit edildiğinde, hata `CRITICAL` seviyesinde loglanır.
    *   Ana bilişsel döngü durdurulur (`run_evo.py` içindeki `can_run_main_loop` bayrağı şu an bu görevi görür).
    *   Program sonlandırılırken tüm sistem kaynakları (sensör akışları, dosyalar, bağlantılar vb.) `finally` blokları ve `cleanup` metotları aracılığıyla düzgünce serbest bırakılır.
    *   Gerekirse program `sys.exit(1)` gibi bir kodla sonlandırılabilir.

5.  **Yapılandırma ve Başlatmada Hızlı Hata Ver (Fail Fast on Configuration/Initialization):**
    *   Evo'nun temel yapılandırması (config yükleme) veya ana pipeline modüllerinin (Process, Represent, Memory, Cognition, MotorControl) başlatılması sırasında kurtarılamaz bir hata oluşursa, Evo'nun "canlanma" sürecini tamamlayamadığı kabul edilir ve program, ana döngüye hiç girmeden sonlandırılır. Bu, işlevsel olmayan bir durumda çalışmaya çalışmaktan daha iyidir.

### Veri Yapıları (Gelecekte)
Projede kullanılacak temel veri yapıları (örn: temsil vektörleri, bellek öğeleri, içsel durum temsili) ve bunların nasıl organize edileceği (örn: vektör veritabanı, grafik tabanlı bellek) ileride detaylandırılacaktır.

### Algoritmalar (Gelecekte)
Öğrenme, hatırlama, anlama, karar alma, ifade üretme gibi farklı bilişsel yetenekler için kullanılacak temel algoritmalar (örn: temel sinir ağı yapıları, kümeleme, ilişkilendirme, basit kural tabanlı çıkarım) ileride detaylandırılacaktır.

### Evrimsel Mekanizmalar (`evolution.py`) (Gelecekte)
Evo'nun zamanla kendi yapısını ve davranışını nasıl adapte edeceğini ve geliştireceğini yönetecek mekanizmaların teknik detayları burada açıklanacaktır.

Evo'nun teknik altyapısı, projenin evrimsel yolculuğunda (Roadmap) belirtilen fazlara göre sürekli gelişecektir.