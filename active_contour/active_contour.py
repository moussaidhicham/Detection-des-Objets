import numpy as np
import matplotlib.pyplot as plt

# Fonction de mise à jour des sous-graphes
def update_subplots(img, LSF, ax, iteration, first_iteration=False):
    ax.imshow(img, cmap='gray')
    if first_iteration:
        ax.contour(LSF, [0], colors='r', linewidths=3)  # Afficher seulement le contour initial
    else:
        ax.contour(LSF, [0], colors='r', linewidths=2)  # Afficher l'évolution du contour
    ax.set_title(f"Iteration {iteration}")
    ax.set_xticks([])  # Masquer les ticks de l'axe X
    ax.set_yticks([])  # Masquer les ticks de l'axe Y

def mat_math(input, operation):
    if operation == "atan": #arctangente 
        return np.arctan(input)
    elif operation == "sqrt": # racine caree
        return np.sqrt(input)

def compute_gradients(LSF):
    """ Calcule les gradients en X et Y avec des différences finies """
    Ix = np.gradient(LSF, axis=1)  # Gradient en X
    Iy = np.gradient(LSF, axis=0)  # Gradient en Y
    return Iy, Ix

def compute_laplacian(LSF):
    """ Calcule le Laplacien d'une image avec des différences finies """
    Lap = np.zeros_like(LSF)
    Lap[1:-1, 1:-1] = (
        LSF[2:, 1:-1] + LSF[:-2, 1:-1] + LSF[1:-1, 2:] + LSF[1:-1, :-2] - 4 * LSF[1:-1, 1:-1]
    )
    return Lap

def CV(LSF, img, mu, nu, epison, step):
    Drc = (epison / np.pi) / (epison**2 + LSF**2) # fonction delta de Dirac
    Hea = 0.5 * (1 + (2 / np.pi) * mat_math(LSF / epison, "atan"))
    Iy, Ix = compute_gradients(LSF)  # Calculer les gradients en X et Y
    s = mat_math(Ix**2 + Iy**2, "sqrt")  # Calcul du module du gradient (magnitude)
    Nx = Ix / (s + 1e-6)  # Normalisation du gradient X
    Ny = Iy / (s + 1e-6)  # Normalisation du gradient Y
    Mxx, Nxx = np.gradient(Nx)
    Nyy, Myy = np.gradient(Ny)
    cur = Nxx + Nyy  # Courbure
    Length = nu * Drc * cur

    Lap = compute_laplacian(LSF)  # Calculer le Laplacien 
    Penalty = mu * (Lap - cur)  # Pénalité pour mainteanir la focntion LSF reguliere

    s1 = Hea * img
    s2 = (1 - Hea) * img
    s3 = 1 - Hea
    C1 = s1.sum() / Hea.sum()  # Moyenne pondérée
    C2 = s2.sum() / s3.sum()
    CVterm = Drc * (-1 * (img - C1) ** 2 + 1 * (img - C2) ** 2)  # Terme CV

    LSF = LSF + step * (Length + Penalty + CVterm)  # Mise à jour de la fonction de niveau
    return LSF

# Fonction principale pour charger et traiter l'image
def process_image(image):
    # Convertir l'image en niveau de gris
    if len(image.shape) == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # Formule de conversion RGB à grayscale

    img = np.array(image, dtype=np.float64)

    # Initialiser la fonction de niveau (LSF)
    IniLSF = np.ones((img.shape[0], img.shape[1]), img.dtype)
    IniLSF[30:80, 30:80] = -1
    IniLSF = -IniLSF  # Inverser les signes pour l'initialisation

    # Créer la figure
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))  # 1 ligne, 5 colonnes

    # Paramètres
    mu = 1
    nu = 0.003 * 255 * 255
    total_iterations = 10  # Nombre total d'itérations
    iterations_to_display = [1, 2, 5, 7, 10]  # Étapes à afficher
    epison = 1
    step = 0.1
    LSF = IniLSF

    # Boucle d'itération pour améliorer la segmentation
    for i in range(1, total_iterations + 1):
        LSF = CV(LSF, img, mu, nu, epison, step)  # Mise à jour de la fonction de niveau
        if i in iterations_to_display:
            index = iterations_to_display.index(i)  # Obtenir l'index correspondant dans les sous-figures
            if i == 1:
                update_subplots(img, IniLSF, axes[index], i, first_iteration=True)  # Passer IniLSF comme argument
            else:
                update_subplots(img, LSF, axes[index], i)  # Afficher l'évolution du contour

    # Afficher la figure avec les étapes clés
    plt.tight_layout()

    return fig
