import cv2
import csv
from pynput.mouse import Listener
import os

# importing pyglet module
import pyglet

# importing shapes from the pyglet
from pyglet import shapes

image_index = 0
folder = "Data3/"

# Captură de la webcam
x_rez_pixel, y_rez_pixel = 1920, 1080  # Schimbați rezoluția la ce doriți
x_dim_mm, y_dim_mm = 380, 215
distance_mm = 500
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, x_rez_pixel)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, y_rez_pixel)


# Funcție pentru a obține coordonatele cursorului și pentru a salva imaginea
def on_click(x, y, button, pressed):
    if pressed:
        global image_index
        # Salvează imaginea de la webcam
        img_name = folder + f"I{image_index} image_{x}_{y}.jpg"
        cv2.imwrite(img_name, frame)

        image_index += 1
        print(image_index)
        # print(f"Imagine salvată: {img_name}")

        # Scrie coordonatele și numele imaginii într-un fișier CSV
        with open(folder + 'click_data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([img_name, x, y, x_rez_pixel, y_rez_pixel, x_dim_mm, y_dim_mm, distance_mm])


# Crează un fișier CSV dacă nu există deja și adaugă header-ul
if not os.path.exists('click_data.csv'):
    with open(folder + 'click_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["file_name", "x_pixel", "y_pixel", "x_rez_pixel", "y_rez_pixel", "x_dim_mm", "y_dim_mm", "distance_mm"])

# Începe să asculte evenimentele de clic
with Listener(on_click=on_click) as listener:
    while True:
        ret, frame = cap.read()
        cv2.imshow('Webcam', frame)

        # Ieșire dacă se apasă tasta 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Curăță și închide
    cap.release()
    cv2.destroyAllWindows()
    listener.join()

    exit(1)
