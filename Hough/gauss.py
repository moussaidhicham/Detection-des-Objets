import numpy as np

def filtre_gaussien(image, taille_filtre, sigma):

    noyau = np.zeros((taille_filtre, taille_filtre))

    centre = taille_filtre // 2

    for x in range(taille_filtre):
        for y in range(taille_filtre):
            diff = np.sqrt((x-centre)**2 + (y-centre)**2)
            noyau[x, y] = np.exp(-(diff**2) / (2*sigma**2))
    noyau /= np.sum(noyau)
    
    image_padded = np.pad(image, centre, mode='constant', constant_values=0)
    image_filtrée = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            voisinage = image_padded[i:i+taille_filtre, j:j+taille_filtre]
            valeur_filtrée = np.sum(voisinage * noyau)
            image_filtrée[i, j] = valeur_filtrée
            
    return image_filtrée
