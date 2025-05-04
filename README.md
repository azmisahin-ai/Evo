# Evo (Evolutionary Mind) - Yeniden Doğuş

Evo projesi, duyusal girdileri işleyerek öğrenen, büyüyen ve dünya ile etkileşim kuran bir yapay zeka zihin prototipi inşa etme vizyonuyla yeniden başlıyor. Bir bebeğin adım adım bilişsel gelişimini taklit ederek, en temel algılardan karmaşık düşünce ve yaratıcılık yeteneklerine doğru ilerlemeyi hedefleriz. Çekirdek algoritmalar ve sinir ağı bileşenleri "sıfırdan" (from scratch) implement edilmektedir.

Projenin kalbinde, Evo'nun gerçek zamanlı duyu akışını işleyerek öğrenmesi, hatırlaması, anlaması ve çıktılar üretmesi yer alır. Amacımız, onunla canlı bir şekilde etkileşim kurabilmek ve zamanla kendi kendine öğrenen, karmaşık bir varlığa dönüşmesini gözlemlemektir.

## 🧠 Proje Ruhu: Bebek Gibi Öğrenen Bir Zihin

Evo, sadece bir algoritma koleksiyonu değil, bir "varlık" prototipidir. Doğum anında duyuları açılır ve çevreden gelen sürekli duyu akışını hissetmeye başlar. Bu akış içindeki desenleri zamanla işler, temsiller oluşturur, hatırlar, ilkel anlamlar çıkarır ve temel tepkiler üretir. Metin gibi soyut kavramlar daha sonra öğrenilecektir. Öğrenme süreci, başlangıçta rehberli olsa da, hızla kendi kendine keşif ve deneyimden öğrenme yönüne evrilecektir. Nihai vizyon, fiziksel bir bedene entegre olarak dünya ile etkileşim kuran, kendi zihninde dünyayı canlandıran, sanatsal ifade yetenekleri geliştiren ve bilgelik kazanan bir yapay zihin yaratmaktır.

Detaylı vizyon, projenin felsefesi, temel prensipleri ve evrimsel yolculuğu için lütfen [docs/README.md](docs/README.md)'ye bakınız.

## 🏗️ Depo Yapısı: Evo'nun Beyni ve Bedeninin Organizasyonu

Depo yapısı, Evo'nun bilişsel ve fiziksel bileşenlerini yansıtan modüler bir organizasyon sunar. Her modülün net bir sorumluluğu vardır.

Detaylı depo yapısı açıklaması için lütfen [docs/STRUCTURE.md](docs/STRUCTURE.md)'ye bakınız.

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

(requirements.txt dosyasını oluştururken gerekli temel kütüphaneleri (PyTorch, NumPy, SciPy, Pillow, Librosa, PyYAML) ekleyeceğiz.)

### 4. Ham Veriyi Hazırlayın (İlk Eğitim İçin Gerekliyse)

İlk modelleri eğitmek için manuel veri setine ihtiyacınız olabilir.



## ▶️ Başlatma Komutları: Evo'yu Canlandırma

Sanal ortamınız aktifken ve ilk modelleri eğittikten sonra (eğer manuel eğitim gerekiyorsa), Evo'yu "canlandırmak" için ana başlatma scriptini kullanın. Bu script, Evo'nun duyu ve motor arayüzlerini (API) ve iç bilişsel döngüsünü başlatır.

1.  **(İlk Sefer veya Veri Güncellemesi Sonrası) Temel Veri Setini Hazırlama:** Manuel ham veriyi işlenmiş formata dönüştürür ve vocab dosyalarını oluşturur.
    ```bash
    python -m scripts.setup_dataset
    ```
2.  **(İlk Sefer veya Model Mimarisi Değişikliği Sonrası) Temel Modelleri Eğitme:** Manuel işlenmiş veri ile başlangıç modellerini (Representasyon, Anlama, İfade) eğitir.
    ```bash
    python -m scripts.train_initial_models
    ```
3.  **EVO'YU BAŞLATMA (ANA KOMUT):** Evo'nun API arayüzünü ve bilişsel döngüsünü başlatır. Mobil uygulama bu API'ye bağlanacaktır.
    ```bash
    python -m src.run_evo
    ```
4.  **Mobil Uygulamayı Çalıştırma:** Mobil uygulama projesini başlatın ve Evo'nun backend API'sine bağlanacak şekilde yapılandırın. Artık mobil uygulama aracılığıyla Evo ile etkileşim kurabilirsiniz (sesini duyabilir, görüntüsünü görebilir, ona konuşabilir/gösterebilirsiniz).

## 🛠️ Yardımcı Scriptler

`scripts/` dizinindeki diğer dosyalar manuel veri hazırlama, model eğitimi ve hata ayıklama gibi yardımcı görevler içindir.

---
