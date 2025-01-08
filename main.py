import gradio as gr
from Hough.cercles import detect_circles
from Hough.lines import detect_lines
from active_contour.active_contour import process_image  # Importer la fonction de segmentation Active Contour


# Création de la page d'accueil
def create_home_page():
    with gr.Blocks() as home_page:
        with gr.Row(elem_id="header-row"):
            gr.Markdown(
                """
                # 🌟 Bienvenue dans l'Application de Détection et de Segmentation ! 🌟
                """,
                elem_id="main-title"
            )
        with gr.Row(elem_id="info-row"):
            gr.Markdown(
                """
                Ce projet intègre deux **algorithmes puissants** :
                - 🎯 **Contours Actifs (Active Contour)** : Détectez et suivez les contours d'objets dans une image.
                - 🔍 **Transformée de Hough** : Identifiez des formes géométriques (cercles et lignes) dans vos images.
                """,
                elem_id="info-markdown"
            )
        with gr.Row(elem_id="instructions-row"):
            gr.Markdown(
                """
                ## Instructions :
                🖼️ **Active Contour** : Chargez une image en niveaux de gris pour suivre l'évolution du contour.  
                📐 **Transformée de Hough** : Chargez une image pour détecter des cercles ou des lignes.
                
                Naviguez dans les **onglets** ci-dessous pour accéder à chaque fonctionnalité.
                """,
                elem_id="instructions-markdown"
            )
        with gr.Row(elem_id="authors-markdown", elem_classes="markdown-row"):
                gr.Markdown(
                    """
                    ## 👥 Auteurs :
                    - **Moussaid Hicham** : Conception et développement de l'algorithme de **détection des contours (Active Contour)**.  
                    - **Bouba Ahmed** : Conception et développement de l'algorithme de **Transformée de Hough** pour la détection de **cercles et lignes**.

                    Merci à tous les auteurs pour leurs contributions exceptionnelles ! 🌟
                    """
                )
        with gr.Row(elem_id="footer-row"):
            gr.Markdown(
                """
                ### 🔧 Préparez vos images et commencez l'exploration dès maintenant !
                """,
                elem_id="footer-markdown"
            )
    return home_page


def create_active_contour_interface():
    with gr.Blocks() as active_contour_interface:
        gr.Markdown(
            """
            ## Segmentation par Contours Actifs
            - **Segmentation de l'Image** : Téléchargez une image en niveaux de gris pour voir l'évolution de la segmentation par contours actifs.
            """
        )
        with gr.Column():
            img_input_contour = gr.Image(type="numpy", label="Image d'Entrée")
            btn_contour = gr.Button("Lancer la segmentation")
            img_output_contour = gr.Plot(label="Résultat de la Segmentation")  # Affichage sous forme de graphique
        btn_contour.click(process_image, inputs=img_input_contour, outputs=img_output_contour)
    
    return active_contour_interface

# Interface pour la Transformée de Hough
def create_hough_interface():
    with gr.Blocks() as hough_interface:
        gr.Markdown(
            """
            ## Transformée de Hough
            - **Détection des Cercles** : Téléchargez une image pour identifier les cercles.
            - **Détection des Lignes** : Téléchargez une image pour détecter les lignes.
            """,
            elem_id="hough-markdown"
        )
        with gr.Tabs():
            with gr.Tab("Détection des Lignes"):
                with gr.Row():
                    with gr.Column():
                        threshold_slider_lines = gr.Slider(
                            minimum=0, 
                            maximum=255, 
                            step=1, 
                            label="Seuil pour la détection des lignes", 
                            value=250, 
                            elem_id="slider-lines"
                        )
                        longueur_min_ligne_lines = gr.Slider(
                            minimum=0, 
                            maximum=255, 
                            step=1, 
                            label="Longeur minimum des lignes", 
                            value=50, 
                            elem_id="slider-lines"
                        )
                        btn_lines_detect = gr.Button("Lancer la détection des lignes", elem_id="hough-btn-lines-detect")
                        btn_lines_reset = gr.Button("Réinitialiser", elem_id="hough-btn-lines-reset")
                    with gr.Column():
                        img_input_lines = gr.Image(type="numpy", label="Image d'Entrée", elem_id="hough-img-input-lines")
                    with gr.Column():
                        img_output_lines = gr.Image(type="numpy", label="Lignes Détectées", elem_id="hough-img-output-lines")
                    # Button actions
                    btn_lines_detect.click(detect_lines, inputs=[img_input_lines, threshold_slider_lines, longueur_min_ligne_lines], outputs=img_output_lines)
                    btn_lines_reset.click(lambda: None, inputs=None, outputs=img_output_lines)

            with gr.Tab("Détection des Cercles"):
                with gr.Row():
                    with gr.Column():
                        threshold_slider_cercles = gr.Slider(
                            minimum=0, 
                            maximum=255, 
                            step=1, 
                            label="Seuil pour la détection des circles", 
                            value=250, 
                            elem_id="slider-lines"
                        )
                        min_rayon_slider = gr.Slider(
                            minimum=0, 
                            maximum=100, 
                            step=1, 
                            label="Rayon Minimum", 
                            value=10, 
                            elem_id="slider-min-rayon"
                        )
                        max_rayon_slider = gr.Slider(
                            minimum=0, 
                            maximum=300, 
                            step=1, 
                            label="Rayon Maximum", 
                            value=30, 
                            elem_id="slider-max-rayon"
                        )
                        btn_circles_detect = gr.Button("Lancer la détection des cercles", elem_id="hough-btn-circles-detect")
                        btn_circles_reset = gr.Button("Réinitialiser", elem_id="hough-btn-circles-reset")
                    with gr.Column():
                        img_input_circles = gr.Image(type="numpy", label="Image d'Entrée", elem_id="hough-img-input-circles")
                    with gr.Column():
                        img_output_circles = gr.Image(type="numpy", label="Cercles Détectés", elem_id="hough-img-output-circles")                    
                btn_circles_detect.click(detect_circles, inputs=[img_input_circles,threshold_slider_cercles, min_rayon_slider , max_rayon_slider], outputs=img_output_circles)
                btn_circles_reset.click(lambda: None, inputs=None, outputs=img_output_circles)

    return hough_interface

# Assemblage final de l'application
with gr.Blocks(css="""
    .gradio-container {background-color: #f9f9f9; font-family: Arial;}
    #home-markdown {color: #333; padding: 20px; border-radius: 10px; background-color: #fff; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);}
    #contour-markdown, #hough-markdown {color: #333; padding: 20px; border-radius: 10px; background-color: #fff; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);}
    #contour-img-input, #hough-img-input-lines, #hough-img-input-circles {margin-bottom: 10px;}
    #contour-submit-btn, #hough-btn-lines-detect, #hough-btn-lines-reset, #hough-btn-circles-detect, #hough-btn-circles-reset {margin-top: 10px;}
""") as main_interface:
    with gr.Tabs():
        with gr.Tab("Accueil"):
            create_home_page()
        with gr.Tab("Contours Actifs"):
            create_active_contour_interface()
        with gr.Tab("Transformée de Hough"):
            create_hough_interface()

main_interface.launch()
