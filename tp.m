% Apprentissage Automatique Numérique
% TP1 - Classifieur Bayésien
% Alexandre Pais Gomes

% Précise à octave que ce fichier est un fichier de script
1;

%%%%%%%%%%%%%
%%%% ACP %%%%
%%%%%%%%%%%%%
% Cette technique permet de réduire la dimension des données.
% Ceci est nécessaire puisque certains points des images ont toujours la même valeur,
% la variance est donc zéro et la matrice de covariance n'est pas inversible.
% C'est à vous de trouver la dimension de la projection qui donne les meilleures performances
% Vous pouvez explorer des valeurs entre 10 et 100.

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

% On teste arbitrairement avec une valeur entre 10 et 100
project_data(10);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Phase d'apprentissage %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Chargement des données
appr_cl = load("data/appr_cl.ascii");
appr_acp = load("data/acp/appr-acp.ascii");

for i = 0:9
	% Calcul des probabilités à priori P(wi)
	pwi(i+1) = mean(appr_cl(:) == i);
	
	% Calcul des probabilités conditionnelles P(x|wi)
	% Moyenne ui pour chaque classe
	ui(i+1,:) = mean(appr_acp.Ap(find(appr_cl(:) == i),:));
	
	% Calcul des covariances matrice dxd pour chaque classe
	covariance(i+1,:,:) = cov(appr_acp.Ap(find(appr_cl(:) == i),:));	
end

% Fonction qui retourne la gaussienne d'une classe donnée cl, sur une donnée x, à l'aide de la covariance et la moyenne de la classe
function res = gaussienne (cl,x,covariance,ui)
	res = 1 / (sqrt(2*pi)*det(reshape(covariance(cl+1,:,:),[10,10]))^(1/2)) * exp((-1/2)*(x-ui(cl+1,:))*inv(reshape(covariance(cl+1,:,:),[10,10]))*(x'-ui(cl+1,:)'));
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Phase de classification %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Traitement séquentiel des exemples du corpus de développement
dev_cl = load("data/dev_cl.ascii");
dev_acp = load("data/acp/dev-acp.ascii");

% On compte les erreurs
erreurs = 0;
for i = 1:size(dev_acp.Ap,1) % Pour chacun des exemples
	for j = 0:9
		p(j+1) = gaussienne(j,dev_acp.Ap(i,:),covariance,ui)*pwi(j+1); % On obtient la probabilité de chaque classe sachant x
	endfor
	[pmax, indice] = max(p); % On prend la classe qui donne la plus grande probabilité
	if (indice-1 ~= dev_cl(i))
		erreurs = erreurs+1; % Si cette classe n'est pas la bonne, on ajoute une erreur à notre système
		% On ajoute cette erreur au tableau de confusion
	endif
endfor

% Affichage du nombre d'erreurs
disp("Nombre d'erreurs : ")
erreurs

% Affichage du pourcentage d'erreur
disp("Pourcentage : ")
erreurs*100/(size(dev_acp.Ap,1))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Phase d'évaluation %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%


