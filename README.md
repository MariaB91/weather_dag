# Documentation des choix pour le DAG de traitement des données météorologiques

## 1. Collecte des données météo
Pour la collecte des données, j'ai utilisé l'API **OpenWeatherMap**, car elle fournit des informations météorologiques en temps réel pour différentes villes. Pour c eprojet il s'agit de  trois villes (Paris, Londres, Washington) pour obtenir une diversité géographique. Les données sont enregistrées sous forme de fichiers JSON dans le dossier `/app/raw_files`. Chaque fichier est horodaté pour une gestion facile des versions.

## 2. Transformation des données
### a. Conversion des dernières 20 fichiers en CSV
J'ai mis en place une tâche qui transforme les derniers 20 fichiers JSON en un fichier CSV (`data.csv`). Cela permet d'avoir un aperçu rapide des données récentes. En utilisant la fonction `sorted()` sur les fichiers, je m'assure que je traite toujours les fichiers les plus récents.

### b. Conversion de toutes les données en CSV
Une autre tâche est dédiée à la transformation de toutes les données disponibles dans le dossier `/app/raw_files` en un fichier CSV complet (`fulldata.csv`). Cela permet de conserver un historique complet des données météorologiques et d'être utilisé pour l'entraînement des modèles de machine learning.

## 3. Entraînement des modèles de régression
### a. Choix des modèles
Pour l'entraînement, j'ai sélectionné trois modèles de régression : `LinearRegression`, `DecisionTreeRegressor` et `RandomForestRegressor`. Ces modèles ont été choisis pour leur capacité à s'adapter à différents types de données. La régression linéaire est simple et efficace, tandis que les arbres de décision et les forêts aléatoires peuvent mieux capturer des relations non linéaires dans les données.

### b. Validation croisée
J'ai utilisé la validation croisée avec `cross_val_score` pour évaluer la performance de chaque modèle. Cela permet d'obtenir une estimation robuste de la performance en évitant le surajustement. J'ai choisi une validation croisée à 3 plis pour équilibrer le temps de calcul et la précision des résultats.

## 4. Sélection du meilleur modèle
Après l'entraînement, j'ai implémenté une tâche qui sélectionne le meilleur modèle basé sur le score obtenu par la validation croisée. Le modèle avec le score le plus bas (meilleure performance) est choisi. Ce modèle est ensuite réentraîné sur toutes les données disponibles pour maximiser son efficacité avant d'être sauvegardé.

## 5. Description des tâches du DAG
Le DAG est composé des tâches suivantes :

1. **`fetch_weather_data`** : 
   - Cette tâche collecte les données météo pour les villes spécifiées et les enregistre dans des fichiers JSON dans le dossier `/app/raw_files`.

2. **`transform_to_csv`** : 
   - Transforme les derniers 20 fichiers JSON en un fichier CSV (`data.csv`) pour avoir un aperçu rapide des données récentes.

3. **`transform_to_fulldata_csv`** : 
   - Transforme toutes les données disponibles dans le dossier `/app/raw_files` en un fichier CSV complet (`fulldata.csv`) pour l'entraînement des modèles.

4. **`train_models`** : 
   - Entraîne les trois modèles de régression (régression linéaire, arbre de décision, forêt aléatoire) sur les données et calcule les scores en utilisant la validation croisée.

5. **`select_and_save_best_model`** : 
   - Sélectionne le meilleur modèle basé sur les scores obtenus, le réentraîne sur toutes les données et l'enregistre en tant que fichier `.pkl` pour une utilisation future.

## 6. Planification et exécution
Le DAG est configuré pour s'exécuter toutes les minutes, garantissant que le tableau de bord soit constamment mis à jour avec les dernières données et les prédictions. La structure en chaîne des tâches assure un flux de données cohérent et réduit le risque d'erreurs.

## 7. Utilisation de Docker
Le DAG est conçu pour être exécuté dans un environnement Docker, ce qui permet d'isoler les dépendances et de garantir que le code s'exécute dans un environnement cohérent. Les données sont stockées dans des volumes montés, facilitant leur accès et leur gestion.

## Conclusion
En suivant cette approche, je m'assure que le traitement des données météorologiques est automatisé, efficace et fiable, tout en maintenant une flexibilité pour ajuster et étendre le système si nécessaire.
