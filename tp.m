% TP - Classifieur Bayésien
% Alexandre Pais Gomes
%
1;
appr_cl = load("/info/etu/m1/m1015/Documents/aan/data/appr_cl.ascii");

% Calcul des probabilités à priori p(wi)
for i = 0:9
	pwi(i+1) = length(find(appr_cl(:) == i)) / length(appr_cl(:));
end
