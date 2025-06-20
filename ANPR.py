import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
harcascade = r"C:\Users\ADITHYA U PRABHU\python\model\indian_license_plate.xml"
cap = cv2.VideoCapture(0)
min_area = 500
while True:
    success, img = cap.read()
    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            plate_roi = img_gray[y:y+h, x:x+w]
            plate_number = pytesseract.image_to_string(plate_roi, config='--psm 6')
            cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(img, plate_number, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0,), 2)
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
