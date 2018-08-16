# :fr: Classifieur Bayésien #
*[:gb: version below]*

### Dans quel cadre ? ###

UE d'*Apprentissage Automatique Numérique* du Master 1 ISI de l'Université du Maine.

### Qu'est-ce qu'un classifieur bayésien ? ###

Le classifieur bayésien est un classifieur linéaire basé sur la théorie de Bayes. Il suppose qu'une classe possède certaines caractéristiques. Il permet ensuite de classer un exemple donné selon ces caractéristiques.

Ici, l'objectif est de classer les chiffres entre 0 et 9. Chaque chiffre est représenté par un vecteur de 256 valeurs (1 x 256). Le classifieur doit donc être capable de déterminer avec une certaine marge d'erreur la classe d'un exemple donné (un vecteur donné).

Une classification bayésienne se fait en 3 étapes :

* **Une phase d'apprentissage** : Elle permet de déterminer la probabilités à-priori, la moyenne et la matrice de covariance de chaque classe. Ces données permettent l'évaluation de la probabilité P(wi | x) qui peut être approchée par une gaussienne.
* **Une phase de développement** : Elle compare les différentes variantes du classifieur afin de minimiser son taux d'erreur.
* **Une phase d'évaluation** : C'est la partie qui intéresse le client, elle permet à partir des données d'apprentissage de déterminer la classe d'un exemple donné.

### Règle de Bayes et Gaussienne ###

La règle de Bayes nous dit que :

w* = argmax(i, P(x | wi) * P(wi))

On suppose que les probabilités conditionnelles P(x|wi) peuvent être approchées par des gaussiennes de dimension 256 et par conséquent :

w* = argmax(i, Gaussienne(x, i) * P(wi))

On connait P(wi) qui est déterminée à la phase d'apprentissage et on peut calculer la gaussienne à l'aide de sa formule (voir le PDF pour plus d'info).

### Analyse en Composante Principale ###

Ici, une ACP (Analyse en Composante Principale) est effectuée sur les données initiales pour pouvoir réduire la taille des données et éviter les variances nulles car une variance nulle empêche le calcul d'une gaussienne multi-dimensionnelle puisque l'inversion de la matrice de covariance n'est pas possible si une variance est de zéro.


### Comment ça marche ? ###

* Lancer [Octave](https://www.gnu.org/software/octave/)
* Lancer le fichier de script tp.m à l'aide de la commande suivante dans l'interpréteur octave

```
tp
```

* Le script se charge d'effectuer les différentes phases et détermine les classes du fichier d'évaluation.
* NB: Plusieurs variables sont disponibles au début du script pour lancer (ou non) les différentes phases. Par défaut, toutes les phases (apprentissage, développement et évaluation) sont lancées.

### À propos ###

Pour des questions de contraintes du professeur, le fichier de script tp.m possède l'intégralité du projet.

Enfin, pour des questions de compréhension et modularité, des fonctions ont été créés pour lancer les différentes phases, il est possible que ces fonctions ne soient pas très efficaces en terme de temps d'exécution car la copie de données très lourde et gourmande en temps. Cependant, le système final n'en sera pas préjudicié puisqu'il exécute uniquement la phase d'évaluation qui n'est pas simulée par une fonction à l'opposé des autres phases.

# :gb: Bayes classifier #

### Context ###

Practical work for the *Automatic Numerical Learning* course of the first year Master's degree in Computer Science of the University of Maine (France). The objective was to create a Naive Bayes Classifier to determine which number is a given one. A number is represented by 256 bits.

### How does it works ? ###

* Get [Octave](https://www.gnu.org/software/octave/)
* Run the script tp.m in the Octave interpreter by running the following command :

```
tp
```

### About the project ###

The project is in french (school's constraint).

The script tp.m fully contains the project (teacher's constraint).

For better reading purpose, functions have been created to run the different phases of the Bayes Classifier. These functions are very likely to be ressource-eager. However, as the final system only runs the final phase (evaluation phase) that won't be a problem because this is the only phase that doesn't require any of these functions to be runned.
