%
% Ce fichier contient le code et un exemple comment calculer et appliquer
% une Analyse en Composantes Principales (ACP)
% Cette technique permet de réduire la dimension des données.
% Ceci est nécessaire puisque certains points des images ont toujours la même valeur,
% la variance est donc zéro et la matrice de covariance n'est pas inversible.
% C'est à vous de trouver la dimension de la projection qui donne les meilleures performances
% Vous pouvez explorer des valeurs entre 10 et 100.

% effectuer une ACP de dimension k sur les données X
% retourner le vecteur moyen mu et la matrice de projection P

function [mu,P] = acp (X, k)
  n=size(X,1);
  mu = mean(X);
  Xmu = X-ones(n,1)*mu;
  S = Xmu'*Xmu;
 [P evd] = eigs(S,k,'lm');
 ev = diag(evd);
 printf('%5.3f\n', sum(ev));
endfunction

% charger les données, effectuer une ACP pour une dimesion k données
% et sauvegarder le résultats dans un nouveau fichier

function [] = project_data (k)
  A=load('chars/matlab/appr.ascii');

  [mu_all P] = acp(A,k);
  n = size(A,1);
  Ap = (A-ones(n,1)*mu_all) * P;  % ici on projète !
  save 'appr-acp.ascii' Ap

  A=load('chars/matlab/dev.ascii');
  n = size(A,1);
  Ap = (A-ones(n,1)*mu_all) * P;
  save 'dev-acp.ascii' Ap

  A=load('chars/matlab/eval.ascii');
  n = size(A,1);
  Ap = (A-ones(n,1)*mu_all) * P;
  save 'eval-acp.ascii' Ap
endfunction
