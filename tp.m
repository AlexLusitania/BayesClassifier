% TP - Classifieur Bayésien
% Alexandre Pais Gomes

1;

% Chargement des données
appr_cl = load("data/appr_cl.ascii");
appr = load("data/appr.ascii");

for i = 0:9
	% Calcul des probabilités à priori P(wi)
	pwi(i+1) = mean(appr_cl(:) == i);
	
	% Calcul des probabilités conditionnelles P(x|wi)
	% Moyenne ui (vecteur de dimension 256)
	ui(i+1,:) = mean(appr(find(appr_cl(:) == i),:));
	
	% Calcul des covariances
	covar(i+1,:,:) = cov(appr(find(appr_cl(:) == i),:));	
end
