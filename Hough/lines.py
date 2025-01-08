import cv2
import numpy as np
import math
from Hough.gauss import filtre_gaussien

def detect_lines(image , seuil=255, longueur_min_ligne=50):
    image_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer le filtre gaussien
    sigma = 1.0  
    image_filtrée = filtre_gaussien(image_gris, 5, sigma)

    # Étape 1 : Détection des bords avec l'algorithme de Canny
    bords = cv2.Canny(image_filtrée, threshold1=50, threshold2=150)

    # Étape 2 : Initialisation des paramètres de l'accumulateur
    hauteur, largeur = bords.shape
    longueur_diag = int(np.sqrt(hauteur**2 + largeur**2))  
    rho_bins = 2 * longueur_diag + 1  # Taille totale de l'accumulateur (rho négatif et positif)
    theta_bins = 180 

    # Créer l'accumulateur
    accumulateur = np.zeros((rho_bins, theta_bins), dtype=np.int32)

    # Plage de valeurs pour θ en radians
    plage_theta = np.deg2rad(np.arange(0, 180))

    # Étape 3 : Remplir l'accumulateur en fonction des bords détectés
    for y in range(hauteur):
        for x in range(largeur):
            if bords[y, x] == 255:
                for theta_index, theta in enumerate(plage_theta):
                    rho = int(x * math.cos(theta) + y * math.sin(theta)) + longueur_diag
                    if 0 <= rho < rho_bins:
                        accumulateur[rho, theta_index] += 1 

    # Étape 4 : Détection des lignes 
    lignes = []

    for rho_index in range(accumulateur.shape[0]): 
        for theta_index in range(accumulateur.shape[1]):
            if accumulateur[rho_index, theta_index] >= seuil:
                rho = rho_index - longueur_diag  # Recalculer ρ 
                theta = theta_index * (np.pi / theta_bins)  # θ en radians
                lignes.append((rho, theta))  

    # Étape 5 : dessiner 
    image_sortie = cv2.cvtColor(image_gris, cv2.COLOR_GRAY2BGR)

    for rho, theta in lignes:
        a = np.cos(theta)
        b = np.sin(theta)

        # Calculer les points d'intersection avec les bords de l'image
        if b != 0:
            x1, y1 = 0, int(rho / b) 
            x2, y2 = largeur, int((rho - largeur * a) / b) 
        else:
            x1, y1 = int(rho / a), 0
            x2, y2 = int(rho / a), hauteur

        # Calculer la longueur de la ligne détectée
        longueur_ligne = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Si la ligne est suffisamment longue, la dessiner
        if longueur_ligne >= longueur_min_ligne:
            cv2.line(image_sortie, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image_sortie