
import cv2
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Naložimo naučeni model in razrede
model = tf.keras.models.load_model("model_v3.keras")

with open("razredi_v3.json", "r", encoding="utf-8") as datoteka:
    razredi = json.load(datoteka)

# Nastavitve MediaPipe
osnova = mp.tasks.BaseOptions
prepoznavalnik = mp.tasks.vision.HandLandmarker
moznosti = mp.tasks.vision.HandLandmarkerOptions
nacin = mp.tasks.vision.RunningMode

nastavitve = moznosti(
    base_options=osnova(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=nacin.IMAGE,
    num_hands=1
)

kamera = cv2.VideoCapture(0)

if not kamera.isOpened():
    raise RuntimeError("Kamere ni mogoče odpreti.")

print("Prepoznavanje znakov z modelom V3")
print("Za izhod pritisnite Q.")

with prepoznavalnik.create_from_options(nastavitve) as detektor:
    while True:
        uspeh, slika = kamera.read()

        if not uspeh:
            print("Napaka pri zajemanju slike.")
            break

        slika = cv2.flip(slika, 1)

        rgb = cv2.cvtColor(
            slika, cv2.COLOR_BGR2RGB
        )

        mp_slika = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        rezultat = detektor.detect(mp_slika)

        besedilo = "Roka ni zaznana"

        if rezultat.hand_landmarks:
            tocke = rezultat.hand_landmarks[0]

            # Pripravimo koordinate
            koordinate = np.array([
                [tocka.x, tocka.y, tocka.z]
                for tocka in tocke
            ], dtype=np.float32)

            # Enaka normalizacija kot pri zbiranju
            koordinate -= koordinate[0]

            razdalje = np.linalg.norm(
                koordinate, axis=1
            )

            merilo = np.max(razdalje)

            if merilo > 0:
                koordinate /= merilo

                vhod = koordinate.reshape(1, 63)

                # Napoved nevronske mreže
                napoved = model.predict(
                    vhod, verbose=0
                )[0]

                indeks = int(np.argmax(napoved))
                znak = razredi[indeks]
                zanesljivost = napoved[indeks] * 100

                besedilo = (
                    f"{znak}: {zanesljivost:.1f}%"
                )

            # Narišemo točke na roki
            visina, sirina = slika.shape[:2]

            for tocka in tocke:
                x = int(tocka.x * sirina)
                y = int(tocka.y * visina)

                cv2.circle(
                    slika,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        cv2.putText(
            slika,
            besedilo,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Prepoznavanje znakov V3",
            slika
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

kamera.release()
cv2.destroyAllWindows()
