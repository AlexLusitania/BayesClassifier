% Apprentissage Automatique Numérique
% TP1 - Classifieur Bayésien
% Alexandre Pais Gomes

% Précise à octave que ce fichier est un fichier de script
1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Paramètres initiaux %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d = 36; % Si changement de d alors mettre calcul_acp à 1 sinon l'ACP ne sera pas recalculé
calcul_acp = 1; % 1 si recalcul de l'ACP, 0 sinon
calcul_appr_dev = 1; % 1 si relancement de la phase d'apprentissage et développement pour d, 0 sinon
calcul_eval = 1; % 1 si relancement de la phase d'évaluation, 0 sinon

% Mettre à 1 si besoin de relancement la phase de comparaison (apprentissage + développement) de toutes les tailles d'ACP compris entre 10 et 100
% ATTENTION cette phase peut s'avérée très longue en terme de temps d'exécution (+ d'une heure).
comparaison = 0;

%%%%%%%%%%%%%
%%%% ACP %%%%
%%%%%%%%%%%%%
% Cette technique permet de réduire la dimension des données.
% Ceci est nécessaire puisque certains points des images ont toujours la même valeur,
% la variance est donc zéro et la matrice de covariance n'est pas inversible.

% Effectue une ACP de dimension k sur les données X
% Retourne le vecteur moyen mu et la matrice de projection P
function [mu,P] = acp (X, k)
	n=size(X,1);
	mu = mean(X);
	Xmu = X-ones(n,1)*mu;
	S = Xmu'*Xmu;
	[P evd] = eigs(S,k,'lm');
	ev = diag(evd);
	printf('%5.3f\n', sum(ev));
endfunction

% Charge les données, effectue une ACP pour une dimension k données
% et sauvegarde le résultat dans un nouveau fichier
function [] = project_data (k)
	A=load('data/appr.ascii');

	[mu_all P] = acp(A,k);
 	n = size(A,1);
	Ap = (A-ones(n,1)*mu_all) * P;  % ici on projète !
  	save 'data/acp/appr-acp.ascii' Ap

  	A=load('data/dev.ascii');
  	n = size(A,1);
  	Ap = (A-ones(n,1)*mu_all) * P;
  	save 'data/acp/dev-acp.ascii' Ap

  	A=load('data/eval.ascii');
  	n = size(A,1);
  	Ap = (A-ones(n,1)*mu_all) * P;
  	save 'data/acp/eval-acp.ascii' Ap
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Fonction Gaussienne %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fonction gaussienne pour une classe cl, sur une donnée x, à l'aide de la covariance et la moyenne de la classe ainsi que la taille de la projection d
function res = gaussienne (cl, x, covariance, ui, d)
	res = 1 / (sqrt(2*pi)*det(reshape(covariance(cl+1,:,:),[d,d]))^(1/2)) * exp((-1/2)*(x-ui(cl+1,:))*inv(reshape(covariance(cl+1,:,:),[d,d]))*(x'-ui(cl+1,:)'));
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Fonctions de simulation des différentes phases %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fonction d'apprentissage qui calcule les moyennes ui et les matrices de covariances pour chaque classe
% A partir des données d'apprentissage fournies
function [pwi, ui, covariance] = apprentissage (appr_cl, appr_acp)
	for i = 0:9
		pwi(i+1) = mean(appr_cl(:) == i);

		% Moyenne ui pour chaque classe
		ui(i+1,:) = mean(appr_acp.Ap(find(appr_cl(:) == i),:));

		% Calcul des covariances (matrice d*d) pour chaque classe
		covariance(i+1,:,:) = cov(appr_acp.Ap(find(appr_cl(:) == i),:));	
	end
endfunction

% Fonction qui fournit le nombre d'erreur, le pourcentage d'erreur et le tableau de confusion d'un système donnée
% A partir des données de développement fournies
function [erreurs, pourcentage, confusion] = developpement (dev_cl, dev_acp, pwi, covariance, ui, d)
	% Traitement séquentiel des exemples du corpus de développement, on compte les erreurs
	erreurs = 0;
	confusion = zeros(10,10);
	for i = 1:size(dev_acp.Ap, 1) % Pour chacun des exemples du corpus de dév
		for j = 0:9
			p(j+1) = gaussienne(j, dev_acp.Ap(i,:), covariance, ui, d) * pwi(j+1); % On obtient la probabilité de chaque classe sachant x
		endfor
		[pmax, indice] = max(p); % On prend la classe qui donne la plus grande probabilité
		if (indice-1 ~= dev_cl(i))
			erreurs = erreurs+1; % Si cette classe n'est pas la bonne, on ajoute une erreur à notre système
			confusion(indice, dev_cl(i)+1)++; % On ajoute cette erreur au tableau de confusion
		endif
	endfor
	pourcentage = erreurs*100/(size(dev_acp.Ap,1));
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Chargement des données fixes %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(calcul_appr_dev == 1 || comparaison == 1)
	appr_cl = load('data/appr_cl.ascii');
	dev_cl = load('data/dev_cl.ascii');
endif

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Comparaison des différentes projections %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% On teste avec toutes les valeurs de projection entre 10 et 100
% L'objectif étant de trouver le meilleur d qui permet d'avoir le moins d'erreur
% ATTENTION cette opération peut durer très longtemps (plus d'une heure), à exécuter qu'en cas de réelle nécessité (mettre comparaison à 1 si besoin)
if (comparaison == 1)
	for i = 10:100 % On essaie d entre 10 et 100 arbitrairement
		clear ui;
		clear covariance;
		clear appr_acp;
		clear dev_acp;

		d = i;
		project_data(d); % On effectue une ACP de dimension d (ou dimension i d'une certaine façon)

		% Chargement des nouvelles données ACP
		appr_acp = load('data/acp/appr-acp.ascii');
		dev_acp = load('data/acp/dev-acp.ascii');

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%% Phase d'apprentissage %%%%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		[pwi, ui, covariance] = apprentissage (appr_cl, appr_acp);

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%% Phase de classification %%%%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		[erreurs, pourcentage, confusion] = developpement (dev_cl, dev_acp, pwi, covariance, ui, d);

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%% Affichage des résultats %%%%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		% Affichage du nombre d'erreurs
		disp(["Nombre d'erreurs avec d=", num2str(d), ' : ', num2str(erreurs)])
		% Affichage du pourcentage d'erreur
		disp(["Soit un pourcentage d'erreur de : ", num2str(pourcentage), '%'])
	endfor
endif

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Avec le meilleur d %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (calcul_appr_dev == 1 || (exist('data/appr/appr-ui.ascii') == 0) || (exist('data/appr/appr-covariance.ascii') == 0))
	if(calcul_acp == 1)
		project_data(d); % On projète avec le meilleur d
	endif

	% On (re)charge les données
	appr_acp = load('data/acp/appr-acp.ascii');
	dev_acp = load('data/acp/dev-acp.ascii');

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%% Phase d'apprentissage %%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	[pwi, ui, covariance] = apprentissage (appr_cl, appr_acp);
	save 'data/appr/appr-pwi.ascii' pwi;
	save 'data/appr/appr-ui.ascii' ui;
	save 'data/appr/appr-covariance.ascii' covariance;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%% Phase de classification %%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	[erreurs, pourcentage, confusion] = developpement (dev_cl, dev_acp, pwi, covariance, ui, d);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%% Affichage des résultats %%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% Affichage du nombre d'erreurs
	disp(["Nombre d\'erreurs avec d=", num2str(d), ' : ', num2str(erreurs)])
	% Affichage du pourcentage d'erreur
	disp(["Soit un pourcentage d'erreur de : ", num2str(erreurs*100/(size(dev_acp.Ap,1))), '%'])
	% Affichage du tableau de confusion
	disp('Tableau de confusion')
	confusion
else 
	% Sinon on recharge juste les données d'apprentissage
	pwi = load('data/appr/appr-pwi.ascii').pwi;
	ui = load('data/appr/appr-ui.ascii').ui;
	covariance = load('data/appr/appr-covariance.ascii').covariance;
endif

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Phase d'évaluation %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(calcul_eval == 1)
	eval_acp = load('data/acp/eval-acp.ascii');

	% On traite séquentiellement chaque exemple du corpus d'évaluation
	for i = 1:size(eval_acp.Ap, 1)
		for j = 0:9
			p(j+1) = gaussienne(j, eval_acp.Ap(i,:), covariance, ui, d) * pwi(j+1); % On obtient la probabilité de chaque classe sachant x
		endfor
		[pmax, indice] = max(p); % On prend la classe qui donne la plus grande probabilité
		eval_cl(i,:) = indice-1; % On ajoute la classe au tableau des résultats.
	endfor

	% Enregistrement des classes dans un fichier externe
	save 'data/eval/eval-cl.ascii' eval_cl;
	disp('Enregistrement des classes évaluées dans data/eval/eval_cl.ascii avec succès')
endif
