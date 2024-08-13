using Pkg
codes_directory = "$(@__DIR__)/../"
Pkg.activate(codes_directory)

using ITensors
using Test
## Can you construct a binary tree that looks like this

###                 A
###             i /   \ j
###             B      C
###         k /  l\  /m   \n
###          D      E      F
###        /   \    |    /   \
###       o     p   q   r     s 

##  Where the size of the legs grows polynomially from the center.
is = Index.((2,2,4,4,4,4,6,6,6,6,6),("i","j","k","l","m","n","o","p","q","r","s"))
elt = Float64
A = ITensor(elt, is[1],is[2])
B = ITensor(elt, is[1], is[3], is[4])
C = ITensor(elt, is[2], is[5], is[6])
D = ITensor(elt, is[3], is[7], is[8])
E = ITensor(elt, is[4], is[5], is[9])
F = ITensor(elt, is[6], is[10], is[11])

network(A,B,C,D,E,F) = contract([A,B,C,D,E,F])
@test inds(network(A,B,C,D,E,F)) == (is[7], is[8], is[9], is[10],is[11])
