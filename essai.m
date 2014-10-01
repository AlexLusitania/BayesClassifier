% TP - Classifieur Bayésien
% Essai sur un cas différent (pour tester)

1;

% Chargement fictif des données
appr_cl = [3;1;1;2;2;2;3;1;3];
appr = [5,5,5;
		1,1,1;
		1,1,1;
		3,3,3;
		3,3,3;
		3,3,3;
		5,5,5;
		1,1,1;
		5,5,5];

% Calcul des probabilités à priori P(wi)
for i = 1:3
	pwi(i) = mean(appr_cl(:) == i);
end

% Calcul des probabilités conditionnelles
% Moyenne ui
for i = 1:3
	ui(i,:) = mean(appr(find(appr_cl(:) == i),:));
end
