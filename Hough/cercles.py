import cv2
import numpy as np
import math
from Hough.gauss import filtre_gaussien

def detect_circles(image, seuil=255, rayon_min=10, rayon_max=30):
    image_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sigma = 1.0  
    image_filtrée = filtre_gaussien(image_gris, 5, sigma)

    # Étape 1 : détection des bords 
    bords = cv2.Canny(image_filtrée, threshold1=50, threshold2=150)

    # Étape 2 : Paramètres
    hauteur, largeur = bords.shape
    accumulateur = np.zeros((hauteur, largeur, rayon_max), dtype=np.int32)

    # Étape 3 : Remplir l'accumulateur pour chaque rayon possible
    for y in range(hauteur):
        for x in range(largeur):
            if bords[y, x] > 0:
                for r in range(rayon_min, rayon_max):
                    for theta in range(0, 360, 1):
                        a = int(x - r * math.cos(math.radians(theta)))
                        b = int(y - r * math.sin(math.radians(theta)))
                        if 0 <= a < largeur and 0 <= b < hauteur:
                            accumulateur[b, a, r] += 1           

    # Étape 4 : Détection des cercles
    cercles_détectés = []

    for y in range(hauteur):
        for x in range(largeur):
            for r in range(rayon_min, rayon_max):
                if accumulateur[y, x, r] >= seuil:
                    cercles_détectés.append((x, y, r))  # Ajouter les cercles détectés

    # Étape 5 : Dessiner 
    image_sortie = cv2.cvtColor(image_gris, cv2.COLOR_GRAY2BGR)

    for (x, y, r) in cercles_détectés:
        cv2.circle(image_sortie, (x, y), r, (0, 255, 0), 4)
        cv2.circle(image_sortie, (x, y), 2, (0, 0, 255), 3)

    # for (x, y, r) in cercles_détectés:
    #     for theta in range(0, 360):
    #         a = int(x + r * math.cos(math.radians(theta)))
    #         b = int(y + r * math.sin(math.radians(theta)))
    #         image_sortie[b, a] = (0, 255, 0) 
    #     if 0 <= x < largeur and 0 <= y < hauteur:
    #         image_sortie[y, x] = (0, 0, 255)

    return image_sortie
