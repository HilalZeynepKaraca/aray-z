# aray-z
# Çoklu Video Fare Tespiti ve 3D Konumlandırma

[cite_start]Bu proje, İstanbul Sağlık ve Teknoloji Üniversitesi'ndeki Yazılım Mühendisliği stajım kapsamında geliştirilmiştir[cite: 12, 19]. [cite_start]Derin öğrenme modelleri kullanılarak canlı davranışlarının analizi ve 3D görselleştirilmesini amaçlar[cite: 15].

## Proje Özellikleri ve Teknik Detaylar
* [cite_start]**Görüntü İşleme & Senkronizasyon:** Üç farklı kamera açısı FFmpeg ve OpenCV kullanılarak 30 FPS hızında senkronize edilmiştir[cite: 16].
* [cite_start]**Nesne Tespiti:** YOLO mimarisi ile eğitilen modeller ONNX formatına dönüştürülerek entegre edilmiştir[cite: 15].
* [cite_start]**Kullanıcı Arayüzü:** Analiz parametrelerinin yönetilebildiği, Python tabanlı modüler bir GUI geliştirilmiştir[cite: 1, 17].
* **3D Görselleştirme:** Tespit edilen verilerin gerçek zamanlı 3D konumlandırması sağlanmıştır.

## Kullanılan Teknolojiler
* **Diller:** Python.
* [cite_start]**Kütüphaneler:** OpenCV, PyQt5, Matplotlib, ONNX, FFmpeg[cite: 1, 16].
* [cite_start]**Modeller:** YOLO (Derin Öğrenme)[cite: 15].
