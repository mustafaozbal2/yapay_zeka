import cv2
import face_recognition
import os
import numpy as np

# 1. Ayarlar
FOTO_KLASORU = "faces"
KAMERA_ID = 0  # Genelde 0 varsayılan webcam'dir

# Bilinen yüzlerin kodlarını ve isimlerini tutacak listeler
known_face_encodings = []
known_face_names = []

print("Kaptan, fotoğraflar yükleniyor ve Deep Learning modeli ile analiz ediliyor...")

# 2. Fotoğrafları Yükleme ve Eğitme Kısmı
# faces klasöründeki her bir alt klasörü (futbolcu ismini) gezer
for futbolcu_adi in os.listdir(FOTO_KLASORU):
    kisi_klasoru = os.path.join(FOTO_KLASORU, futbolcu_adi)
    
    # Sadece klasörleri işleme al
    if not os.path.isdir(kisi_klasoru):
        continue

    # O futbolcunun içindeki her fotoğrafı gez
    for dosya_adi in os.listdir(kisi_klasoru):
        resim_yolu = os.path.join(kisi_klasoru, dosya_adi)
        
        # Sadece resim dosyalarını al (jpg, png vs)
        if not (resim_yolu.endswith(".jpg") or resim_yolu.endswith(".png") or resim_yolu.endswith(".jpeg")):
            continue

        try:
            # Resmi yükle
            image = face_recognition.load_image_file(resim_yolu)
            
            # Resimdeki yüzü bul ve encode et (Sayısal veriye dök)
            # Bu fonksiyon arka planda önceden eğitilmiş Deep Learning modeli kullanır.
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                # İlk bulunan yüzü al
                face_encoding = encodings[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(futbolcu_adi) # Klasör adını isim olarak kaydet
            else:
                print(f"Uyarı: {dosya_adi} dosyasında yüz bulunamadı.")
                
        except Exception as e:
            print(f"Hata oluştu: {dosya_adi} - {e}")

print(f"Eğitim Tamamlandı! Toplam {len(known_face_encodings)} yüz hafızaya alındı.")
print("Kamera başlatılıyor...")

# 3. Kamera Başlatma
video_capture = cv2.VideoCapture(KAMERA_ID)

while True:
    # Kameradan bir kare (frame) al
    ret, frame = video_capture.read()
    if not ret:
        break

    # İşlem hızını artırmak için görüntüyü 1/4 oranında küçültüyoruz
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # OpenCV BGR kullanır, face_recognition RGB kullanır. Dönüşüm yapıyoruz.
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Şu anki karedeki tüm yüzleri bul ve encode et
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []

    for face_encoding in face_encodings:
        # 4. Karşılaştırma (Matching)
        # Görülen yüz, veritabanımızdakilerden hangisine benziyor?
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Taninmayan Yuz"

        # En yakın eşleşmeyi bulmak için yüz mesafelerini (Face Distance) hesapla
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        # En düşük mesafe (best match) hangi indexte?
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            # İsimleri büyük harf yapıp düzeltelim (ör: 'messi' -> 'MESSI')
            name = name.upper()

        face_names.append(name)

    # 5. Sonuçları Ekrana Çizdir
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Görüntüyü küçültmüştük, koordinatları tekrar 4 ile çarpıp eski haline getirelim
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Yüzün etrafına kutu çiz
        renk = (0, 255, 0) if name != "Taninmayan Yuz" else (0, 0, 255) # Tanırsa Yeşil, Tanımazsa Kırmızı
        cv2.rectangle(frame, (left, top), (right, bottom), renk, 2)

        # İsmi yazacağımız etiketi çiz
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), renk, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    # Sonucu göster
    cv2.imshow('Kaptan Futbolcu Tanima Sistemi', frame)

    # 'q' tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
video_capture.release()
cv2.destroyAllWindows()