# [FR] Classifieur Bayésien #

### Dans quel cadre ? ###

UE d'Apprentissage Automatique Numérique du Master 1 ISI de l'Université du Maine.

### Qu'est-ce qu'un classifieur bayésien ? ###

Le classifieur bayésien est un classifieur linéaire basé sur la théorie de Bayes. Il suppose qu'une classe possède certaines caractéristiques et permet de classer un exemple donné selon ces caractéristiques.

Ici, l'objectif est de classer les chiffres entre 0 et 9. Ces chiffres sont représentés par un vecteur de 256 valeurs (1 x 256). Le classifieur doit donc est capable de déterminer avec une certaine marge d'erreur la classe d'un exemple donné (un vecteur donné).

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
#!octave

tp
```

* Le script se charge d'effectuer les différentes phases et détermine les classes du fichier d'évaluation.

### À propos ###

Pour des questions de contraintes du professeur, le fichier de script tp.m possède l'intégralité du projet.

Enfin, pour des questions de compréhension et modularité, des fonctions ont été créés pour lancer les différentes phases, il est possible que ces fonctions ne soient pas très efficaces en terme de temps d'exécution car la copie de données très lourde et gourmande en temps.