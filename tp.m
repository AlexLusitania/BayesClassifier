% TP - Classifieur Bayésien
% Alexandre Pais Gomes

1;

% Chargement des données
appr_cl = load("data/appr_cl.ascii");
appr = load("data/appr.ascii");

% Calcul des probabilités à priori P(wi)
for i = 0:9
	pwi(i+1) = mean(appr_cl(:) == i);
end

% Calcul des probabilités conditionnelles P(x|wi)
% Moyenne ui (vecteur de dimension 256)
for i = 0:9
	ui(i+1,:) = mean(appr(find(appr_cl(:) == i),:));
end
