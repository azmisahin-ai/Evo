# Evo Projesine Katkıda Bulunma Rehberi

Evo, bir canlı gibi, büyümek ve gelişmek için bir topluluğa ihtiyaç duyar. Bu projeye katkıda bulunmak isterseniz, her türlü katkı (kod, dokümantasyon, fikir, test, hata raporlama) çok değerlidir. Bu rehber, projeye nasıl katılabileceğiniz konusunda size yardımcı olmayı amaçlar.

## Katkı Türleri

*   **Kod Katkıları:** Mevcut modülleri geliştirmek, yeni özellikler eklemek veya hata düzeltmeleri yapmak.
*   **Dokümantasyon Katkıları:** Mevcut belgeleri iyileştirmek, yeni rehberler yazmak veya teknik detayları açıklamak.
*   **Fikir ve Tartışmalar:** Projenin vizyonu, felsefesi veya teknik yönleri üzerine fikirlerinizi paylaşmak ve tartışmalara katılmak.
*   **Testler:** Yeni test senaryoları yazmak veya mevcut testleri iyileştirmek.
*   **Hata Raporlama:** Bulduğunuz hataları net adımlarla raporlamak.

## Başlamak İçin

1.  **Vizyonu ve Felsefeyi Anlayın:** Projenin neden var olduğunu ve neyi hedeflediğini anlamak için [Proje Felsefesi](PHILOSOPHY.md) belgesini okuyun.
2.  **Yol Haritasını İnceleyin:** Evo'nun mevcut durumunu ve gelecekteki gelişim fazlarını öğrenmek için [Evrimsel Yolculuk (Roadmap)](ROADMAP.md) belgesine göz atın.
3.  **Yapıyı Keşfedin:** Kod tabanının nasıl organize edildiğini ve modüllerin sorumluluklarını anlamak için [Proje Yapısı](STRUCTURE.md) belgesini inceleyin.
4.  **Depoyu Klonlayın ve Kurulumu Tamamlayın:** Ana [README.md](../README.md) dosyasındaki talimatları izleyerek projeyi yerel ortamınıza kurun.

## Geliştirme Ortamı Kurulumunda Dikkat Edilmesi Gerekenler

Projenin bazı kütüphaneleri (özellikle ses ve görüntü işleme için olanlar) sisteminizde ek bağımlılıklar gerektirebilir.

*   **PortAudio (Ses Desteği İçin):** PyAudio kütüphanesini kurarken "portaudio.h: No such file or directory" gibi bir hata alırsanız, sisteminize PortAudio geliştirme kütüphanelerini kurmanız gerekir.
    *   **Debian/Ubuntu tabanlı sistemlerde (Codespace dahil):**
        ```bash
        sudo apt-get update && sudo apt-get install -y portaudio19-dev
        ```
    *   **macOS (Homebrew ile):**
        ```bash
        brew install portaudio
        ```
    *   **Windows:** Genellikle manuel olarak PortAudio kütüphanesini indirip derleme ortamınızda (örneğin Visual Studio) ayarlamanız gerekebilir veya PyPI'da platformunuza uygun pre-compiled binary olup olmadığını kontrol etmeniz gerekir. Bu durum biraz daha karmaşık olabilir, genel çözüm genellikle Linux/macOS içindir.
    PortAudio kurulumunu tamamladıktan sonra Python sanal ortamınızda `pip install pyaudio` komutunu tekrar çalıştırın.

*   **Diğer Sistem Kütüphaneleri:** Proje geliştikçe başka sistem bağımlılıkları ortaya çıkabilir. Karşılaştığınız hatalarda ilgili kütüphanenin dokümantasyonuna bakmanız veya projeye hata raporu açmanız faydalı olacaktır.

## Kodlama Süreci

*   Herhangi bir geliştirmeye başlamadan önce, üzerinde çalışmak istediğiniz konuyla ilgili mevcut Issue'ları kontrol edin veya yeni bir Issue açarak niyetinizi belirtin. Bu, çakışmaları önlemeye yardımcı olur.
*   Her yeni özellik veya hata düzeltmesi için ayrı bir Git dalı oluşturun.
*   Kodunuzun okunabilir, temiz ve iyi yorumlanmış olmasına dikkat edin.
*   Mümkünse, yazdığınız kod için testler (unittest, integration test) ekleyin.
*   Değişiklikleriniz hazır olduğunda, ana depoya bir Pull Request (PR) gönderin. PR açıklamanızda, yaptığınız değişiklikleri ve nedenlerini net bir şekilde belirtin.

## İletişim

Fikirlerinizi paylaşmak, soru sormak veya yardım almak için GitHub Issue'larını ve Tartışmalar (Discussions) bölümünü kullanabilirsiniz.

Evo ailesine hoş geldiniz! Birlikte öğrenelim, inşa edelim ve evrimleşelim!

---
[▲ Dokümantasyon Haritasına Dön](#evo-dokümantasyonuna-hoş-geldiniz)