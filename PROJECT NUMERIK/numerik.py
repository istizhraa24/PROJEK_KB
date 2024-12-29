import cv2
import os
import csv

def collect_face_dataset_with_metadata(dataset_path="dataset_faces", metadata_file="metadata.csv", user_id=1, num_samples=100):
    """
    Mengumpulkan dataset wajah dari kamera dan menyimpan metadata numerik.

    Args:
        dataset_path (str): Direktori untuk menyimpan dataset.
        metadata_file (str): File CSV untuk menyimpan metadata.
        user_id (int): ID numerik untuk pengguna yang diambil datasetnya.
        num_samples (int): Jumlah sampel wajah yang ingin dikumpulkan.
    """
    # Membuat folder dataset jika belum ada
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Membuat atau membuka file metadata
    with open(metadata_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image_ID", "User_ID", "X", "Y", "Width", "Height"])  # Header

        # Inisialisasi detektor wajah
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        camera = cv2.VideoCapture(0)

        print("Mulai pengambilan dataset. Tekan 'q' untuk keluar.")
        count = 0  # Hitungan sampel yang diambil

        while True:
            ret, frame = camera.read()
            if not ret:
                print("Gagal mengakses kamera.")
                break

            # Konversi frame ke grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Deteksi wajah
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Gambar kotak di sekitar wajah
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Simpan wajah ke folder dataset
                face_img = gray[y:y+h, x:x+w]
                file_name = f"{dataset_path}/user_{user_id}_{count+1}.jpg"
                cv2.imwrite(file_name, face_img)

                # Simpan metadata numerik ke file CSV
                writer.writerow([f"user_{user_id}_{count+1}", user_id, x, y, w, h])
                print(f"Dataset tersimpan: {file_name}, Metadata: ({x}, {y}, {w}, {h})")

                count += 1

                # Berhenti jika jumlah sampel tercapai
                if count >= num_samples:
                    print("Pengambilan dataset selesai.")
                    camera.release()
                    cv2.destroyAllWindows()
                    return

            # Tampilkan frame kamera
            cv2.imshow("Pengambilan Dataset Wajah", frame)

            # Keluar jika menekan tombol 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Tutup kamera dan jendela
        camera.release()
        cv2.destroyAllWindows()

# Memanggil fungsi untuk mengumpulkan dataset
collect_face_dataset_with_metadata(user_id=1, num_samples=50)
