# aray-z
# Çoklu Video Fare Tespiti ve 3D Konumlandırma

Bu proje, İstanbul Sağlık ve Teknoloji Üniversitesi'ndeki Yazılım Mühendisliği stajım kapsamında geliştirilmiştir. Derin öğrenme modelleri kullanılarak canlı davranışlarının analizi ve 3D görselleştirilmesini amaçlar.

## Proje Özellikleri ve Teknik Detaylar
* **Görüntü İşleme & Senkronizasyon:** Üç farklı kamera açısı FFmpeg ve OpenCV kullanılarak 30 FPS hızında senkronize edilmiştir.
* **Nesne Tespiti:** YOLO mimarisi ile eğitilen modeller ONNX formatına dönüştürülerek entegre edilmiştir.
* **Kullanıcı Arayüzü:** Analiz parametrelerinin yönetilebildiği, Python tabanlı modüler bir GUI geliştirilmiştir.
* **3D Görselleştirme:** Tespit edilen verilerin gerçek zamanlı 3D konumlandırması sağlanmıştır.

## Kullanılan Teknolojiler
* **Diller:** Python.
* **Kütüphaneler:** OpenCV, PyQt5, Matplotlib, ONNX, FFmpeg.
* **Modeller:** YOLO (Derin Öğrenme).
