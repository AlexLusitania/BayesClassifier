% TP - Classifieur Bayésien
% Alexandre Pais Gomes

1;

% ACP
% Cette technique permet de réduire la dimension des données.
% Ceci est nécessaire puisque certains points des images ont toujours la même valeur,
% la variance est donc zéro et la matrice de covariance n'est pas inversible.
% C'est à vous de trouver la dimension de la projection qui donne les meilleures performances
% Vous pouvez explorer des valeurs entre 10 et 100.

% Effectuer une ACP de dimension k sur les données X
% Retourner le vecteur moyen mu et la matrice de projection P
function [mu,P] = acp (X, k)
	n=size(X,1);
	mu = mean(X);
	Xmu = X-ones(n,1)*mu;
	S = Xmu'*Xmu;
	[P evd] = eigs(S,k,'lm');
	ev = diag(evd);
	printf('%5.3f\n', sum(ev));
endfunction

% Charger les données, effectuer une ACP pour une dimension k données
% et sauvegarder le résultat dans un nouveau fichier
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

% Calcul de la Gaussienne pour chaque classe
function res = gaussienne (cl,x,covariance,ui)
	res = 1/(sqrt(2*pi)*det(covariance(:,:,cl+1))^(1/2))*exp((-1/2)*(x-ui(cl+1,:))*inv(covariance(:,:,cl+1))*(x'-ui(cl+1,:)'));
endfunction

% Phase de classification

% Traitement séquentiel des exemples du corpus de développement
dev_cl = load("data/dev_cl.ascii");
dev_acp = load("data/acp/dev-acp.ascii");

erreurs = 0;
for i = 1:size(dev_acp.Ap,1)
	
end
