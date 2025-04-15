# Analyse des Bibliothèques Universitaires en France (Il faut se diriger dans la branche Master pour voir le projet)

Ce projet, réalisé dans le cadre du cours "Graphes et Open Data" (GOD) en licence MIAGE à l’Université de Nanterre, 
propose une application interactive pour visualiser et analyser les bibliothèques universitaires françaises à partir de données ouvertes.

### Objectifs du projet :
- Visualisation des bibliothèques universitaires françaises sur une carte interactive.
- Statistiques par région et par caractéristiques (type, surface, etc.).
- Modélisation des bibliothèques sous forme de graphe géographique.
- Analyse de ces graphes via la détection de communautés et d'autres algorithmes (plus courts chemins, optimisation MDSC).

### Technologies utilisées :
- **Python 3.11**
- **Streamlit** pour l'interface utilisateur.
- **NetworkX**, **Folium** pour la gestion des graphes et la cartographie.
- **Pandas**, **NumPy** pour le traitement des données.
- **PuLP** pour l'optimisation des ensembles dominants connexes.

### Installation :
1. Cloner le dépôt :
    ```bash
    git clone https://github.com/efkane/analyse_Bibliotheques_France.git
    cd analyse_Bibliotheques_France
    ```

2. Créer un environnement virtuel et l'activer :
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sous macOS/Linux
    venv\Scripts\activate  # Sous Windows
    ```

3. Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

4. Lancer l'application :
    ```bash
    streamlit run app/analyse_bibliotheque.py
    ```

L’application sera accessible à l’adresse : [http://localhost:8501](http://localhost:8501).


### Auteurs :
- Lydia AMROUCHE
- El Hadji Fodé KANE

Projet encadré par M. Delbot, Université Paris Nanterre.

### Pour Plus d'informations
veuillez jeter un coup d'oeil sur notre rapport qui est à votre disposition dans la branche master   
