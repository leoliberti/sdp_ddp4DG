#!/usr/bin/env python3
## WEIRD outcome: the projected formulations are shit until |projcon|=|origcon|

## jll projection of SDP/DDP/dualDDP formulations

import sys
import os.path
from amplpy import AMPL
import cvxpy as cp
import cvxopt
import scs
import time
import math
import types
import numpy as np
from scipy import stats
import scipy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

######### CONFIGURABLE PARAMETERS ############

myZero = 1e-8
LPsolver = "cplex"
NLPsolver = "ipopt"
#projmethod = "Barvinok"
projmethod = "PCA"
showplot = False

# random projectors
jlleps = 0.15
jllconst = 1.1
achleps = 0.1

PossibleFormulations = ['sdp', 'ddp', 'dualddp']

################ FUNCTIONS ###################

## read data file written in AMPL .dat format
## (after line "param : E", list of edges formatted as i j w_{ij} I_{ij})
def readDat(filename):
    edgeflag = False # true while reading edges
    Kdim = 3
    n = 0
    E = list()
    with open(filename) as f:
        for line in f:
            # look at file line by line
            line = line.strip()
            if len(line) > 0:       
                if line[0] != '#':
                    # line non-empty and non-comment
                    if edgeflag:
                        if line[0] == ';':
                            # if only a ';' then we're closing the edge section
                            edgeflag = False
                        else:
                            # reading uncommented edge data
                            cols = [c for c in line.split() if not '#' in c]
                            i = int(cols[0])
                            j = int(cols[1])
                            if i > j:
                                t = i
                                i = j
                                j = t
                            if len(cols) >= 6:
                                e = (i, j, float(cols[4]), float(cols[5]))
                            elif len(cols) >= 4:
                                e = (i, j, float(cols[2]), float(cols[2]))
                                if e[2] > e[3]:
                                    print("readDat: WARNING: interval weight[", e[2], ",", e[3], "] empty, setting to", e[2])
                                    e[3] = e[2]
                            else:
                                print("readDat: ERROR: line", linecount, "has < 4 columns")
                                exit('abort')
                            E.append(e)
                            if i > n:
                                n = i
                            if j > n:
                                n = j
                    else:
                        if line.replace(" ","")[0:7] == 'param:E':
                            # we're in the edge section now
                            edgeflag = True
                        elif line.replace(" ","")[0:9] == 'paramKdim':
                            # parse line defining Kdim
                            Kdimstr = line.split()[3]
                            if Kdimstr[-1] == ';':
                                Kdimstr = Kdimstr[0:-1]
                            Kdim = int(Kdimstr)
                        elif line.replace(" ","")[0:6] == 'paramn':
                            # parse line defining n
                            nstr = line.split()[3]
                            if nstr[-1] == ';':
                                nstr = nstr[0:-1]
                            n = int(nstr)
    return (Kdim, n, E)

## write realization file to a _rlz.dat file
def writeRlz(n,K, rlzfn):
    rlz = open(rlzfn, "w")
    print("# realization for " + rlzbase, file=rlz)
    print("param xbar :=", file=rlz)
    for i in range(n):
        for k in range(K):
            print(" " + str(i+1) + " " + str(k+1) + "  " + str(x[i,k]),file=rlz)
    print(";", file=rlz)
    rlz.close()
    return

# factor a square matrix, neglecting negative eigenspace
def factor(A):
    n = A.shape[0]
    (evals,evecs) = np.linalg.eigh(A)
    evals[evals < 0] = 0  # closest SDP matrix
    X = evecs #np.transpose(evecs)
    sqrootdiag = np.eye(n)
    for i in range(n):
        sqrootdiag[i,i] = math.sqrt(evals[i])
    X = X.dot(sqrootdiag)
    return np.fliplr(X)

# multidimensional scaling
def MDS(B, eps = myZero):
    n = B.shape[0]
    x = factor(B)
    (evals,evecs) = np.linalg.eigh(B) # ascertain n. large eigs of B
    K = len(evals[evals > eps])
    if K < n:
        # only first K columns
        x = x[:,0:K]
    return x

# principal component analysis
def PCA(B, K = None):
    x = factor(B)
    n = B.shape[0]
    if K is None:
        K = n
    if K < n:
        # only first K columns
        x = x[:,0:K]
    return x

# mean distance error
def mde(x, G):
    n,K = x.shape
    m = sum(len(G[i]) for i in range(n))
    ret = sum(abs(np.linalg.norm(np.subtract(x[i],x[j])) - G[i][j]) for i in range(n) for j in G[i])
    ret = ret / float(m)
    return ret

# largest distance error
def lde(x, G):
    n,K = x.shape
    m = sum(len(G[i]) for i in range(n))
    ret = max(abs(np.linalg.norm(np.subtract(x[i],x[j])) - G[i][j]) for i in range(n) for j in G[i])
    return ret
    
# Barvinok's naive algorithm
def Barvinok(B, K):
    n = B.shape[0]
    X = factor(B)
    y = (1/math.sqrt(K))*np.random.multivariate_normal(np.zeros(n*K), np.identity(n*K))
    y = np.reshape(y, (K, n))
    x = np.transpose(np.dot(y,X))
    return x

################### MAIN #####################

t0 = time.time()

## read command line
if len(sys.argv) < 3:
    exit('cmdline must be: <sdp|ddp|dualddp> filename.dat [plot]')

# formulation to be used
formulation = sys.argv[1]
if formulation not in PossibleFormulations:
    print("jll: formulation {0:s} unknown, must be in".format(formulation), PossibleFormulations)
    exit()

# dat file
datfile = sys.argv[2]
    
# read instance
(Kdim, n, E) = readDat(datfile)

# see if we need to plot
if len(sys.argv) >= 4:
    if datfile == "plot":
        showplot = True

## construct weighted vertex neighbourhoods and vertex cardinalities
G = {i:dict() for i in range(n)}
Ncard = {i:0 for i in range(n)}
for e in E:
    i = e[0]-1 # first vertex
    j = e[1]-1 # second vertex
    Ncard[i] += 1
    Ncard[j] += 1
    w = e[2] # edge weight
    if i > j:
        t = i
        i = j
        j = t
    G[i][j] = w

######## solving original problems (SDP,DDP,dualDDP)

if formulation == 'sdp':
    # solve SDP formulation of DGP
    X = cp.Variable((n,n), PSD=True)
    cobj1 = sum([X[i,i] + X[j,j] - 2*X[i,j] for i in range(n) for j in G[i] if i<j])
    cobj2 = cp.trace(X)
    R = 2*np.random.rand(n,n) - 1
    cobj3 = cp.trace(R*X)
    cobj = cobj1 + cobj2 + 0.1*cobj3
    #objective = cp.Minimize(cobj)
    #objective = cp.Minimize(cobj1)
    objective = cp.Minimize(cobj2)
    constraints = [X[i,i] + X[j,j] - 2*X[i,j] == G[i][j]**2 for i in range(n) for j in G[i] if i<j]
    P = cp.Problem(objective, constraints)
    ## solve the problem
    P.solve(solver=cp.SCS, verbose=True)
    #P.solve(solver=cp.MOSEK, verbose=True)
    #P.solve(solver=cp.CVXOPT, verbose=True)    
    objfunval = P.value
    print("jll(sdp): optimal obj. fun. value =", objfunval)
    highDimSol = X.value

elif formulation == 'ddp':
    # solve DDP formulation of the DGP
    ddp = AMPL()
    ddp.read("ddp.mod")
    ddp.eval("let n := {0:d};".format(n))
    ddp.readData(datfile)    
    ddp.setOption('solver', LPsolver)
    ddp.solve()
    objfun = ddp.getObjective('obj')
    objfunval = objfun.value()
    print("jll(ddp): optimal obj. fun. value =", objfunval)
    Xvar = ddp.getVariable('X')
    X = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            X[i,j] = Xvar[i+1,j+1].value()
    highDimSol = X

elif formulation == 'dualddp':
    # solve dualDDP formulation of the DGP
    dualddp = AMPL()
    dualddp.read("dualddp.mod")
    dualddp.eval("let n := {0:d};".format(n))
    dualddp.readData(datfile)
    dualddp.setOption('solver', LPsolver)
    dualddp.solve()
    objfun = dualddp.getObjective('obj')
    objfunval = objfun.value()
    print("jll(dualddp): optimal obj. fun. value =", objfunval)
    Xvar = dualddp.getVariable('X')
    X = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            X[i,j] = Xvar[i+1,j+1].value()
    highDimSol = X
    
else:
    print("formulation {0:s} unknown".format(formulation))
    exit()
    
## retrieve realization in K dimensions
print("ambient dimension n =", n)
Y = MDS(highDimSol)
K = Y.shape[1]
print("jll: found relaxed embedding in natural dimension K =", K)
if K not in {2,3}:
    if K < 2:
        K = 2
    elif K > 3:
        K = 3
if projmethod == "PCA":
    print("jll(pca): now projecting to", K, "principal dimensions")
    xbar = PCA(highDimSol, K)
elif projmethod == "Barvinok":
    print("jll(bvk): now projecting to", K, "principal dimensions")
    xbar = Barvinok(highDimSol, K)

## report SDP solution statistics
mderr1 = mde(xbar, G)
print("jll(matrix): mean distance error =", mderr1)
lderr1 = lde(xbar, G)
print("jll(matrix): largest distance error =", lderr1)
t1 = time.time()
cputime1 = t1-t0
print("jll(matrix): cpu time =", cputime1)

## refine solution with a local NLP solver
nlp = AMPL()
nlp.read("dgp.mod")
nlp.eval("let n := {0:d};".format(n))
nlp.readData(datfile)
nlp.setOption('solver', NLPsolver)
nlp.eval("let {i in V, k in K} x[i,k] := 0;")
xvar = nlp.getVariable("x")
for i in range(n):
    for k in range(K):
        xvar[i+1,k+1].setValue(xbar[i,k])
nlp.solve()
xvar = nlp.getVariable('x')
xval = xvar.getValues()
x = np.zeros((n,K))
for i in range(n):
    for k in range(K):
        x[i,k] = xvar[i+1,k+1].value()

# save solution to a file
rlzbase = '.'.join(os.path.basename(datfile).split('.')[0:-1])
writeRlz(n,K, rlzbase + "-sol.dat")

# report NLP solution statistics
mderr2 = mde(x, G)
print("jll(nlp): mean distance error =", mderr2)
lderr2 = lde(x, G)
print("jll(nlp): largest distance error =", lderr2)
t2 = time.time()
cputime2 = t2-t1
print("jll(nlp): cpu time =", cputime2)

# report total statistics
cputime = t2 - t0
print("jll: total cpu time=", cputime)
print("OUTLABELS:mp,projmethod,objX,mdeX,ldeX,cpuX,mdex,ldex,cpux,cputot")
print("OUT:{0:s},{1:s},{2:.3f},{3:.3f},{4:.3f},{5:.2f},{6:.3f},{7:.3f},{8:.2f},{9:.2f}".format(formulation,projmethod,objfunval, mderr1, lderr1, cputime1, mderr2, lderr2, cputime2, cputime))

## plot results
if showplot:
    if K == 2:
        plt.scatter(x[:,0], x[:,1])
        plt.plot(x[:,0], x[:,1])
    elif K == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x[:,0], x[:,1], x[:,2])
        ax.plot(x[:,0], x[:,1], x[:,2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')    
    plt.show()

######## solving projected problems (SDP, dualDDP)

if formulation == 'sdp':
    # sample random projector
    m = len(E)
    N = n*(n+1)/2
    k = int(round(jllconst * (1/(jlleps**2)) * math.log(N)))
    if k > m:
        print("jll(proj): projDim={0:d}>{1:d}=origDim, quitting".format(k,m))
        quit()
    T = scipy.sparse.random(k, m, density=achleps, data_rvs=scipy.stats.norm(loc=0, scale=1/math.sqrt(float(k)*achleps)).rvs).tocsr() ## sparse projector
    #T = np.random.normal(loc=0.0, scale=1/math.sqrt(float(k)), size=(k,m)) ## dense projector
    # create projected problem
    TX = cp.Variable((n,n), PSD=True)
    Tcobj1 = sum([TX[i,i] + TX[j,j] - 2*TX[i,j] for i in range(n) for j in G[i] if i<j])
    Tcobj2 = cp.trace(TX)
    TR = 2*np.random.rand(n,n) - 1
    Tcobj3 = cp.trace(TR*TX)
    Tcobj = Tcobj1 + Tcobj2 + 0.1*Tcobj3
    #Tobjective = cp.Minimize(Tcobj)
    #Tobjective = cp.Minimize(Tcobj1)
    Tobjective = cp.Minimize(Tcobj2)
    print("jll(proj): constructing projected problem")
    ## problem construction
    ##   iteratively
    # Tconstraints = []
    # for h in range(k):
    #     TAh = 0
    #     Tb = 0
    #     for l,e in enumerate(E):
    #         i = e[0]-1
    #         j = e[1]-1
    #         if i > j:
    #             t = i
    #             i = j
    #             j = t
    #         TAh += T[h,l]*(X[i,i] + X[j,j] - 2*X[i,j])
    #         Tb += T[h,l] * (G[i][j]**2)
    #     Tconstraints.append(TAh == Tb)
    #     if h % 20 == 0:
    #         print("  appended constraint {0:d}/{1:d}".format(h,k))
    ##   by comprehension
    Tconstraints = [(T[h,l]*(X[e[0]-1,e[0]-1] + X[e[1]-1,e[1]-1] - 2*X[e[0]-1,e[1]-1]) == T[h,l]*(G[e[0]-1][e[1]-1]**2)) for l,e in enumerate(E) for h in range(k)]
    TP = cp.Problem(Tobjective, Tconstraints)
    ## solve the problem
    print("jll(proj): solving projected problem")
    TP.solve(solver=cp.SCS, verbose=True)
    #TP.solve(solver=cp.MOSEK, verbose=True)
    #TP.solve(solver=cp.CVXOPT, verbose=True)    
    Tobjfunval = TP.value
    print("jll(sdp): optimal obj. fun. value =", Tobjfunval)
    ThighDimSol = TX.value
    # retrieval
    print("jll(proj): SDP optimal obj. fun. value =", Tobjfunval)
    print("jll(proj): retrieval from projected SDP solution not implemented yet")
elif formulation == "ddp":
    print("jll(proj): native DDP formulation has no equality constraints")

elif formulation == "dualddp":
    # solve dualDDP formulation of the DGP
    Tdualddp = AMPL()
    Tdualddp.read("dualddp_RP.mod")
    Tdualddp.eval("let n := {0:d};".format(n))
    Tdualddp.readData(datfile)
    Tdualddp.setOption('solver', LPsolver)
    Tdualddp.solve()
    Tobjfun = Tdualddp.getObjective('obj')
    Tobjfunval = Tobjfun.value()
    print("jll(proj): dualDDP optimal obj. fun. value =", Tobjfunval)
    TXvar = Tdualddp.getVariable('X')
    TX = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            TX[i,j] = TXvar[i+1,j+1].value()
    highDimSol = TX
    print("jll(proj): retrieval from projected dualDDP solution not implemented yet")

####################### OBLIVION ############################

# ## choice of SDP solvers with CVXPY
# print cp.installed_solvers()
# prob.solve(solver=cp.CVXOPT)
# prob.solve(solver=cp.SCS)
# prob.solve(solver=cp.MOSEK)

### trying to make JLL work with data extracted from problem at low-level;
###   did not work because I don't know how the solver represents the std form

    # pd, chain, invdata = prob.get_problem_data(cp.SCS)
    # invmap = invdata[1][2] # inverse variable map
    # # extract constraint matrix from original SDP
    # A = pd['A']
    # b = pd['b']
    # c = pd['c']
    # cd = pd['dims']
    # # sample JLL projector
    # (m,n) = A.shape
    # k = int(round(jllconst * (1/(jlleps**2)) * math.log(n)))
    # if k > m:
    #     print("jll(proj): projDim={0:d}>{1:d}=origDim, quitting".format(k,m))
    #     quit()
    # T = scipy.sparse.random(k, m, density=achleps, data_rvs=scipy.stats.norm(loc=0, scale=1/math.sqrt(float(k))).rvs)
    # # put together projected constraints
    # TA = T.dot(A)
    # Tb = T.dot(b)
    # Tcd = {'f':k, 'l':cd.nonpos, 'q':cd.soc, 'ep':cd.exp, 's':cd.psd}
    # Tpd = {'A':TA, 'b':Tb, 'c':pd['c']}
    # ## solve the projected problem
    # #Tsol = chain.solve_via_data(prob, pd)
    # #prob.unpack_results(Tsol, chain, inversedata)
    # soln = scs.solve(Tpd, Tcd)
