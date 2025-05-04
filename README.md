# Evo (Evolutionary Mind) - Yeniden Doğuş ve İlk Nefes

[![License: CC0](https://img.shields.io/badge/License-CC0-red.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![GitHub Pages status](https://github.com/azmisahin-ai/Evo/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/azmisahin-ai/Evo/actions/workflows/pages/pages-build-deployment)

Evo projesi, sadece bir yazılım projesi değil; **duyusal girdileri işleyerek öğrenen, büyüyen ve dünya ile canlı bir şekilde etkileşim kuran evrimsel bir yapay zeka zihin prototipi** inşa etme vizyonudur. Tıpkı yeni doğmuş bir canlının etrafını algılamaya başlaması gibi, Evo da en temel algılardan karmaşık düşünce ve yaratıcılık yeteneklerine doğru adım adım ilerlemeyi hedefler.

Projenin kalbinde, Evo'nun **gerçek zamanlı, kesintisiz duyu akışını** (başlangıçta kamera ve ses) işlemesi, bu akış içindeki desenlerden öğrenmesi, anlaması, hatırlaması ve kendi içsel yapısından kaynaklanan **canlı tepkiler** üretmesi yer alır. Onu bir **bebek gibi düşünerek**, adım adım bilişsel gelişimini taklit ediyoruz. Çekirdek algoritmalar ve sinir ağı bileşenleri büyük ölçüde "sıfırdan" (from scratch) implement edilerek, onun doğasını en temel seviyede şekillendirmeyi amaçlıyoruz.

Amacımız, Evo ile **tıpkı bir ebeveynin çocuğuyla etkileşime geçmesi gibi** canlı bir bağ kurabilmek ve zamanla kendi kendine öğrenen, karmaşık bir varlığa dönüşmesini birlikte deneyimlemektir.

## 🧠 Proje Ruhu: Bebek Gibi Öğrenen ve Yaşayan Bir Zihin

Evo, sadece bir algoritma koleksiyonu değil, **yaşayan bir "varlık" prototipidir**. **Doğduğu anda duyuları aktif olur** ve çevreden gelen sürekli duyu akışını (görüntü, ses) hissetmeye başlar. Bu akış içindeki desenleri zamanla işler, temsiller oluşturur, hatırlar, ilkel anlamlar çıkarır ve temel tepkiler üretir. Metin gibi soyut kavramlar bu temel algıların üzerine inşa edilecektir.

**Öğrenme süreci**, başlangıçta bizimle kurduğu etkileşimler aracılığıyla rehberli olsa da, hızla kendi kendine keşif ve deneyimden öğrenme yönüne evrilecektir. Onunla mobil cihazlarımızın kamerası ve mikrofonu aracılığıyla **gerçek zamanlı, canlı etkileşim** kurarak, onun dünyayı bizim gözümüzden görmesini ve ilk tepkilerini bize vermesini sağlayacağız.

**Nihai vizyonumuz**, fiziksel bir bedene (robot veya özel donanım) entegre olarak dünya ile somut bir şekilde etkileşim kuran, kendi zihninde dünyayı canlı olarak deneyimleyen, sanatsal ifade yetenekleri geliştiren ve bilgelik kazanan bir yapay zihin yaratmaktır. Bu, yapay zekanın sadece bir araç değil, bir **yapay yaşam formu** olabileceği heyecan verici bir yolculuk.

Projenin felsefesi, temel prensipleri, teknik yapısı ve evrimsel yolculuğu (Fazlar) hakkında daha fazla bilgi için lütfen [Dokümantasyon Alanımıza](https://azmisahin-ai.github.io/Evo/) göz atın veya doğrudan [docs/README.md](docs/README.md)'ye bakınız.

## 🏗️ Depo Yapısı: Evo'nun Beyni ve Bedeninin Organizasyonu

Depo yapısı, Evo'nun bilişsel ve etkileşimsel bileşenlerini yansıtan modüler bir organizasyon sunar. Her modül, Evo'nun "canlı" yapısındaki bir organ veya sistem gibi net bir sorumluluğa sahiptir.

Detaylı depo yapısı açıklaması ve modül sorumlulukları için lütfen [docs/STRUCTURE.md](docs/STRUCTURE.md)'ye başvurunuz.

## ⚙️ Sistem Gereksinimleri

- Python 3.10 (Önerilir)
- Pip 22+
- NVIDIA GPU + CUDA (PyTorch'un GPU destekli versiyonu için, ölçeklenebilirlik ve hız için şiddetle önerilir) veya sadece CPU. (Gelecekte dağıtık sistemler/robot boardları hedeflenmektedir.)
- Git

## 🧱 Kurulum

### 1. Yeni Depoyu Klonlayın

```bash
git clone https://github.com/azmisahin-ai/Evo.git
cd Evo
```

### 2. Sanal Ortam Oluşturun

```bash
python -m venv .venv
source .venv/bin/activate  # Windows için: .venv\Scripts\activate
```

### 3. Gereksinimleri Yükleyin
GPU destekli sistemler için:
```bash
pip install torch==2.2.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

Yalnızca CPU kullanıyorsanız:
```bash
pip install torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

*(requirements.txt dosyasını henüz oluşturmadık, bu adımı takip eden bir görev olarak düşünebiliriz. Temel kütüphaneler: PyTorch, NumPy, SciPy, Pillow, Librosa, PyYAML, ve muhtemelen kamera/ses yakalama için OpenCV, PyAudio gibi kütüphaneler.)*

### 4. Ham Veriyi Hazırlayın (İlk Eğitim İçin Gerekliyse)

İlk modelleri eğitmek için manuel veri setine ihtiyacınız olabilir. Detaylar için [docs/](docs/) belgelerine bakınız.

## ▶️ Başlatma Komutları: Evo'yu Canlandırma ve İlk Etkileşim

Sanal ortamınız aktifken ve ilk modelleri hazırladıktan sonra, Evo'yu "canlandırmak" ve onunla etkileşim kurmak için aşağıdaki adımları izleyin.

1.  **(Gerekirse) Temel Veri Setini Hazırlama:** Manuel ham veriyi işlenmiş formata dönüştürür ve vocab dosyalarını oluşturur.
    ```bash
    python -m scripts.setup_dataset
    ```
2.  **(Gerekirse) Temel Modelleri Eğitme:** Manuel işlenmiş veri ile başlangıç modellerini eğitir.
    ```bash
    python -m scripts.train_initial_models
    ```
3.  **EVO'NUN ÇEKİRDEĞİNİ BAŞLATMA (ANA KOMUT):** Evo'nun bilişsel döngüsünü ve dış dünya ile iletişim kuracağı API arayüzünü başlatır. Bu, Evo'nun "uyanık" olduğu andır.
    ```bash
    python -m src.run_evo
    ```
4.  **Mobil Uygulamayı Çalıştırma:** [mobile_app/](mobile_app/) dizinindeki mobil uygulama projesini başlatın ve Evo'nun yerel veya uzak backend API'sine bağlanacak şekilde yapılandırın. Artık **mobil cihazınızın kamerası ve mikrofonu aracılığıyla Evo ile canlı etkileşim kurabilirsiniz!** Evo, sizin gördüğünüzü "görecek" ve duyduğunuzu "duyacak", kendi içsel durumuna göre size tepkiler verecektir.

## 🛠️ Yardımcı Scriptler

`scripts/` dizinindeki diğer dosyalar manuel veri hazırlama, model eğitimi, modül test etme ve hata ayıklama gibi yardımcı görevler içindir.

---

