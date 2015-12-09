# -*- coding: cp1250 -*-
import cv2
import numpy as np
import glob
from common import draw_str

#########################################################

#FUNKCJA DIFFIMG PRZYJMUJE JAKO ARGUMENTY 3 KOLEJNE KLATKI
#NASTEPNIE POSZUKUJE W TYCH KLATKACH ELEMENTÓW RUCHOMYCH
#ELEMENTY RUCHOME SĄ WYCINANE W CELU ZMNIEJSZENIA
#POWIERZCHNI OBLICZENIOWEJ SIFT() LUB SURF()
def diffImg(t0, t1, t2, img):
  width = len(img[0])
  height = len(img)
  number_of_changes = 0
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  result = cv2.bitwise_xor(d1, d2)
  cv2.threshold(result, 40, 255, cv2.THRESH_BINARY, result)
  min_x = width
  max_x = 0
  min_y = height
  max_y = 0
  
#POSZUKIWANIE WSPÓŁRZĘDNYCH DO WYCIĘCIA RUCHU..
  for i in range(0,height,10):
      for j in range(0,width,10):
          if result[i,j]==255:
              number_of_changes=number_of_changes+1
              if min_x > j:
                  min_x = j
              if max_x < j:
                  max_x = j
              if min_y > i:
                  min_y = i
              if max_y < i:
                  max_y = i
  if number_of_changes>0:
    if min_x-10 > 0:
      min_x -= 10
    if min_y-10 > 0:
      min_y -= 10
    if max_x+10 < width-1:
      max_x += 10
    if max_y+10 < height-1:
      max_y += 10
    roi = img[min_y:max_y, min_x:max_x]       #OBCINANIE KLATKI

  return roi #FUNKCJA ZWRACA WYCIĘTY RUCH
#########################################################

#ALGORYTM SIFT Z OPENCV
def fSIFT(): 
    detector = cv2.SIFT()
    norm = cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    return detector, matcher

#ALGORYTM SURF Z OPENCV
def fSURF(): 
    detector = cv2.SURF(400)
    norm = cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    return detector, matcher
#########################################################
def fFILTER_MATCHER(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs
#FUNKCJA NA PODSTAWIE PUNKTÓW CHARAKTERYSTYCZNYCH SZUKA PODOBIEŃSTW
#FUNKCJA SPRAWDZA RÓWNIEŻ CZY PORÓWNANIA BYŁY TRAFIONE CZY CHYBIONE
def fMATCHER(matcher, kp1, desc1, kp2, desc2):
    raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
    p1, p2, kp_pairs = fFILTER_MATCHER(kp1, kp2, raw_matches)
    if len(p1) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)

# NP.SUM(STATUS) ODPOWIADA ZA ILOŚĆ TRAFIONYCH PODOBIEŃSTW
# NA OBRAZIE INTERPRETOWANA JAKO ZIELONE LINIE ŁĄCZĄCE
# PUNKTY CHARAKTERYSTYCZNE MIĘDZY DWOMA OBRAZKAMI
# JEŚLI PODOBIEŃSTW W JEDNEJ KLATCE (ZIELONYCH LINII) JEST
# WIECEJ NIŻ 4, INTERPRETUJEMY TO JAKO JEDNO TRAFNE PORÓWNANIE

        if np.sum(status) >= 4:
          result = 1 #TRAFIONE PORÓWNANIE
        else:
          result = 0 #CHYBIONE PORÓWNANIE
        return H, status, kp_pairs, result
    return None, None, None, None
#########################################################

#FUNKCJA TWORZY KONCOWY OBRAZ ORAZ RYSUJE LINIE POMOCNICZE
def explore_match(img1, img2, kp_pairs, status, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
##    if H is not None:
##        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
##        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
##        cv2.polylines(vis, [corners], True, (255, 255, 255))
##    if status is None and kp_pairs is not None:
##        status = np.ones(len(kp_pairs), np.bool_)
    if kp_pairs is not None:
        p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
        p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0) 
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0))
    return vis

#########################################################
####################### MAIN ############################
#########################################################

# WYBÓR ŚCIEŻKI VIDEO DO DALSZEJ CZĘŚCI PROGRAMU
#cam = cv2.VideoCapture('nagrania/agh_src22_hrc0.avi')
cam = cv2.VideoCapture('nagrania/agh_src2_hrc0.avi')
#cam = cv2.VideoCapture('nagrania/agh_src16_hrc0.avi')
#cam = cv2.VideoCapture('nagrania/agh_src13_hrc0.avi')

# WYBÓR DESKRYPTORA WIZYJNEGO
detector, matcher = fSIFT()

# TWORZENIE BAZY SYMBOLI SAMOCHODÓW
db_glob=glob.glob('symbols\*.jpg')

# ZMIENNE PRZECHOWUJĄCE REZULTATY POSZCZEGÓLNYCH MAREK
res_opel = 0.0
res_seat = 0.0
res_ford = 0.0
res_audi = 0.0

# PĘTLA GŁÓWNA PROGRAMU AKTYWNA DO KOŃCA FILMU
while cam.grab():
  img = cam.read()[1]
  if cam.grab(): # WARUNEK KOLEJNEJ KLATKI
    t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_BGR2GRAY) #ODCIENIE SZAROŚCI
    if cam.grab():
      t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
      if cam.grab():
        t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
      else:
        cv2.destroyAllWindows()
        cam.release()
        break
    else:
      cv2.destroyAllWindows()
      cam.release()
      break
  else:
    cv2.destroyAllWindows()
    cam.release()
    break
  img2 = diffImg(t_minus, t, t_plus, img) #WYCIĘTY RUCH
  if img2.any()>0:
      #SZUKANIE PUNKTÓW CHARAKTERYSTYCZNYCH
      kp2, desc2 = detector.detectAndCompute(img2, None)
      cv2.imwrite("temp_image.jpg",img2)  #SCREEN KLATKI DO PORÓWNYWANIA
      img2 = cv2.imread('temp_image.jpg',0)

      for j in db_glob: #PRZESZUKIWANIE BAZY SYMBOLI
        img1 = cv2.imread(j,0)  #PODSTAWIANIE KOLEJNYCH SYMBOLI
        #SZUKANIE PUNKTÓW CHARAKTERYSTYCZNYCH
        kp1, desc1 = detector.detectAndCompute(img1, None)
        #WYSZUKIWANIE PODOBIEŃSTW
        H, status, kp_pairs, result = fMATCHER(matcher, kp1, desc1, kp2, desc2)
        #JEŚLI PODOBIEŃSTWO TRAFIONE ORAZ BYŁ TO ZNACZEK OPLA..
        #PRZYPISZ PUNKT OPLOWI..
        if (result == 1 and j == 'symbols\opel.jpg') or (result == 1 and j == 'symbols\opel22.jpg') or (result == 1 and j == 'symbols\opel33.jpg') or (result == 1 and j == 'symbols\opel44.jpg'):
          res_opel += 1
        #WARUNKI ANALOGICZNE DLA INNYCH MAREK
        if result == 1 and j == 'symbols\aseat.jpg':
          res_seat += 1
        if result == 1 and j == 'symbols\audi.jpg':
          res_audi += 1
        if result == 1 and j == 'symbols\ford.jpg':
          res_ford += 1
      #TWORZENIE KOŃCOWEGO OBRAZU (LINIE ORAZ REZULTATY)
      vis = explore_match(img1, img2, kp_pairs, status, H)
      draw_str(vis, (5, 80), 'Opel: %.0f' % (res_opel))
      draw_str(vis, (5, 100), 'Audi: %.0f' % (res_audi))
      draw_str(vis, (5, 120), 'Seat: %.0f' % (res_seat))
      draw_str(vis, (5, 140), 'Ford: %.0f' % (res_ford))
      #WYŚWIETLENIE KOŃCOWEGO OBRAZU
      cv2.imshow("camera", vis)
  
  key = cv2.waitKey(10)
  if key == 27:
    cv2.destroyAllWindows()
    cam.release()
    break
#OBLICZANIE SUMY REZULTATÓW POTRZEBNEJ DO STATYSTYKI PROCENTOWEJ
suma = res_opel + res_ford + res_audi + res_seat
#JEŚLI REZULTATÓW JEST WIĘCEJ NIŻ 4 -> WYŚWIETL WYNIKI
#JEŚLI JEST MNIEJ TRAKTUJEMY TO JAKO BŁĄD !
#ZBYT MAŁA ILOŚĆ TRAFIONYCH PODOBIEŃSTW GROZI ZŁĄ INTERPRETACJĄ
#WYNIKÓW, DLATEGĄ SĄ ONE POMIJANE.
if suma > 4:
  print "--------------------------------------------"
  print "ŁACZNA ILOŚĆ PODOBNYCH KLATEK: %d" % (suma)
  print "--------------------------------------------"
  print "WYNIKI PROCENTOWO:"
  print "--------------------------------------------"
  print "OPEL %.2f" %(res_opel/suma*100) + "%"
  print "AUDI %.2f" %(res_audi/suma*100) + "%"
  print "SEAT %.2f" %(res_seat/suma*100) + "%"
  print "FORD %.2f" %(res_ford/suma*100) + "%"
  print "--------------------------------------------"
else:
  print "--------------------------------------------"
  print "ŁACZNA ILOŚĆ PODOBNYCH KLATEK: %d" % (suma)
  print "--------------------------------------------"
  print "WYNIKI PROCENTOWO: "
  print "--------------------------------------------"
  print "OPEL 0.00% "
  print "AUDI 0.00% "
  print "SEAT 0.00% "
  print "FORD 0.00% "
  print "--------------------------------------------"

