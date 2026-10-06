import cv2


kamera = cv2.VideoCapture(0)


yuz_algilayici = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    basarili, kare = kamera.read()

    if not basarili:
        break

    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)

    yuzler = yuz_algilayici.detectMultiScale(
        gri,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for (x, y, w, h) in yuzler:

        
        merkez_x = x + w // 2
        merkez_y = y + h // 2

        
        yaricap = max(w, h) // 2

        
        cv2.circle(
            kare,
            (merkez_x, merkez_y),
            yaricap,
            (0, 500, 0),
            3
        )

        cv2.putText(
            kare,
            "yuzun burda demi",
            (x, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 500, 0),
            2
        )

    cv2.imshow("yuz algilama", kare)

    # q ile çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()
