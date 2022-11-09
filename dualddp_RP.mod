# dual DDP formulation for DGP with random projection of equality constraints
### WEIRD: always unbounded until |proj constraints| = |orig constraints|
				 
## the protein graph format
param Kdim integer, > 0;
param n integer, > 0;
set V := 1..n;
set E within {V,V};
param c{E};
param I{E};

# random projection
param jlleps default 0.1;
param achleps default 0.1;
param jllconst default 1.1;
param N := n*(n+1)/2;
#param d integer, default round(jllconst * (1/(jlleps^2)) * log(N));
param d integer, default round(card(E)*1.00);
set D := 1..d;
param T{D,E} default Normal(0,1/sqrt(d));

# set of dimensions
set K := 1..Kdim;

# decision variables

var X{V,V};

#minimize obj: sum{(i,j) in E} (X[i,i] + X[j,j] - 2*X[i,j]); ## push
minimize obj: sum{i in V, j in V} X[i,j];  ## trace

subject to projpull{h in D}:
  sum{(i,j) in E} T[h,i,j]*(X[i,i] + X[j,j] - 2*X[i,j]) == sum{(i,j) in E} T[h,i,j]*c[i,j]^2;

subject to dualddp1{i in V}: X[i,i] >= 0;
subject to dualddp2{i in V, j in V : i<j and (i,j) not in E}:
  X[i,i] + X[j,j] - 2*X[i,j] >= 0;
subject to dualddp3{i in V, j in V : i<j}: X[i,i] + X[j,j] + 2*X[i,j] >= 0;
