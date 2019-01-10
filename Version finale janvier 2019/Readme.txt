Version finale janvier 2019 
Il y a deux formats des codes dans deux dossiers distincts : 
- Jupyter Notebook
- Python 

L'odre des codes est le suivant :
1) Build_one_data_set 
Il permet de construire une trame de données à partir de plusieurs fichiers Excel.
Entrées : fichier Excel
Sorties : la trame de donnée de learning set et la trame de données de validation set en fichiers ".p" 

2) Pipeline Data Preparation : 
Il réalise la préparation des données de learning set et de validation set.
Entrées : les fichiers ".p" du code Build_one_data_set
Sorties : les données préparées pour l'analyse exploratoire en fichier ".p"

3) Exploratory Data Analysis : 
Il normalise les données et réalise l'analyse exploratoire pour réduire le nombre de features
Entrées : les données préparées pour l'analyse exploratoire en fichier ".p"
Sorties : les sets de données learning et validation sets prêts pour faire du Machine Learning en fichiers ".p"

4) Machine Learning : 
Il réalise l'apprentissage de trois algorithmes de Machine Learning avant de vérifier les résultats sur le validation set. 
Entrées : les sets de données learning et validation sets prêts pour faire du Machine Learning en fichiers ".p"
