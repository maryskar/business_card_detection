from imutils.perspective import four_point_transform
from ultralytics import YOLO
from autocorrect import Speller
import numpy as np
import pandas as pd
import pytesseract
import argparse
import imutils
import cv2
import re

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
  help="path to input image")
ap.add_argument("-d", "--debug", type=int, default=-1,
  help="whether or not we are visualizing each step of the pipeline")
ap.add_argument("-c", "--min-conf", type=int, default=0,
  help="minimum confidence value to filter weak text detection")
args = vars(ap.parse_args())
# load the input image from disk, resize it, and compute the ratio
# of the *new* width to the *old* width
orig = cv2.imread(args["image"])
image = orig.copy()
image = imutils.resize(image, height=800)
ratio = orig.shape[1] / float(image.shape[1])

model = YOLO('nanobest.pt')
names = model.model.names
results = model.predict(image, conf=0.4)  #Adjust conf threshold
contours = results[0].masks.xy

# initialize a contour that corresponds to the business card outline
cardCnt = None
# loop over the contours
for c in contours:
  # approximate the contour
  peri = cv2.arcLength(c, True)
  approx = cv2.approxPolyDP(c, 0.02 * peri, True)
  # if this is the first contour we've encountered that has four
  # vertices, then we can assume we've found the business card
  if len(approx) == 4:
    cardCnt = approx
    break
# if the business card contour is empty then our script could not
# find the  outline of the card, so raise an error
if cardCnt is None:
  raise Exception(("Could not find receipt outline. "
    "Try debugging your edge detection and contour steps."))
# check to see if we should draw the contour of the business card
# on the image and then display it to our screen

cardCnt = np.array(cardCnt).reshape((-1,1,2)).astype(np.int32)

if args["debug"] > 0:
  output = image.copy()
  cv2.drawContours(output, [cardCnt], -1, (0, 255, 0), 2)
  cv2.imshow("Business Card Outline", output)
  cv2.waitKey(0)
# apply a four-point perspective transform to the *original* image to
# obtain a top-down bird's-eye view of the business card
card = four_point_transform(orig, cardCnt.reshape(4, 2) * ratio)
# show transformed image
cv2.imshow("Business Card Transform", card)
cv2.waitKey(0)
# convert the business card from BGR to RGB channel ordering and then
# OCR it
rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\beedz\AppData\Local\Tesseract-OCR\tesseract.exe'
EngText = pytesseract.image_to_string(rgb)
RusText = pytesseract.image_to_string(rgb, lang='rus')

# Создаем объекты Speller для английского и русского языков
en_spell = Speller(lang='en')
ru_spell = Speller(lang='ru')

# Исправляем слова на английском языке
corrected_eng_text = [en_spell(word) for word in EngText.split()]
corrected_eng_text = ' '.join(corrected_eng_text)

# Исправляем слова на русском языке
corrected_rus_text = [ru_spell(word) for word in RusText.split()]
corrected_rus_text = ' '.join(corrected_rus_text)

# use regular expressions to parse out phone numbers and email
# addresses from the business card
phoneNums = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', corrected_eng_text)
emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", corrected_eng_text)
# attempt to use regular expressions to parse out names/titles (not
# necessarily reliable)
nameExp = r"^[\w'\-,.][^0-9_!¡?÷?¿/\\+=@#$%ˆ&*(){}|~<>;:[\]]{2,}"
names = re.findall(nameExp, corrected_rus_text)

# show the phone numbers header
print("PHONE NUMBERS")
print("=============")
# loop over the detected phone numbers and print them to our terminal
for num in phoneNums:
  print(num.strip())
# show the email addresses header
print("\n")
print("EMAILS")
print("======")
# loop over the detected email addresses and print them to our
# terminal
for email in emails:
  print(email.strip())
# show the name/job title header
print("\n")
print("NAME/JOB TITLE")
print("==============")
# loop over the detected name/job titles and print them to our
# terminal
for name in names:
  print(name.strip())

df_information_from_cards = pd.DataFrame(
    {
        "Name/Job title": [', '.join(names)],
        "Emails": [', '.join(emails)],
        "Phone Number": [', '.join(phoneNums)],
        "Corrected English Text": [corrected_eng_text],
        "Corrected Russian Text": [corrected_rus_text]
    }
)

print(df_information_from_cards)