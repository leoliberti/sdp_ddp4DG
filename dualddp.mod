# dual DDP formulation for DGP
				 
## the protein graph format
param Kdim integer, > 0;
param n integer, > 0;
set V := 1..n;
set E within {V,V};
param c{E};
param I{E};

# set of dimensions
set K := 1..Kdim;

# decision variables

#param M := 1000;
var X{V,V}; # <= M, >= -M;

#minimize obj: sum{(i,j) in E} (X[i,i] + X[j,j] - 2*X[i,j]); ## push
#minimize obj: sum{i in V} X[i,i];     ## trace(X)
minimize obj: sum{(i,j) in E} (X[i,i] + X[j,j] - 2*X[i,j]) + 0.1*sum{i in V} X[i,i]; # push + coeff*trace(X)

subject to pull{(i,j) in E}: X[i,i] + X[j,j] - 2*X[i,j] == c[i,j]^2;

subject to dualddp1{i in V}: X[i,i] >= 0;
subject to dualddp2{i in V, j in V : i<j and (i,j) not in E}:
  X[i,i] + X[j,j] - 2*X[i,j] >= 0;
subject to dualddp3{i in V, j in V : i<j}: X[i,i] + X[j,j] + 2*X[i,j] >= 0;

