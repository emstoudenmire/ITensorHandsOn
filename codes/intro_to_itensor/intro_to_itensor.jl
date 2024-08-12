using LinearAlgebra

elt = Float64
a = randn(elt, 10,20);
b = randn(elt, 20,30);

c = a * b
cp = fill!(similar(c), zero(elt))
for i in 1:10
    for j in 1:20
        for k in 1:30
            cp[i,k] += a[i,j] * b[j,k]
        end
    end
end
c ≈ cp

d = randn(elt, 30,10);
transpose(a * b) * transpose(d)

b = Array(randn(elt, (20,30,40)))






using ITensors
elt = Float64
## ITensor uses a intelligent index system which contains meta information
i = Index(5, "i")
## This is an struct with multiple fields some of the important fields are
dim(i) #(space(i))
tags(i)
id(i)
plev(i)

i' == i
## capital constructors allocate memory
A = ITensor(elt)
B = ITensor(elt, i)
C = ITensor(elt, i')
## Though B and C are both the same dimension and tag
## they have a different number of prime levels so you cannot add them.
B + C

B * C

## If we have data that we are using other places we can wrap it in ITensors
## to make tensor operations simplier!
a = rand(elt, 10,20);
b = rand(elt, 30,20,40);
i,j,k,l = Index.((10,20,30,40));
## lower case itensor constructors do not copy data, they wrap the existing data
A = itensor(a, (i,j))
B = itensor(b, (k,j,l))
C = A * B

## This is equivalent to this line of code
bp = permutedims(b, (2,1,3))
c = reshape(a * reshape(bp, (20,30*40)), (10,30,40));
array(C) ≈ c

### Struct for the Tucker decomposition

struct TuckerDecomp
    Core_Factors::Vector{ITensor}
end

abstract type TuckerAlgorithm end

struct NaiveTucker <: TuckerAlgorithm
end

TuckerDecomp(A::ITensor; algorithm::TuckerAlgorithm=NaiveTucker()) = computeHOSVD(algorithm, A)
TuckerDecomp(A::AbstractArray; algorithm::TuckerAlgorithm=NaiveTucker()) = TuckerDecomp(itensor(A, Index.(size(A))); algorithm)

## Naive tucker decomposition algorithm
function computeHOSVD(::NaiveTucker, A::ITensor)
    TD = TuckerDecomp([copy(A),])

    for i in inds(A)
        #### For each leg one must compute the SVD 
        U, S, _ = svd(TD.Core_Factors[1], (i,))
        @show diag(S)
        push!(TD.Core_Factors, U)
        ## Contract the factor with the partially transformed 
        ## Core tensor
        TD.Core_Factors[1] = U * TD.Core_Factors[1]
    end
    return TD
end

TD = TuckerDecomp(A)

## The tucker decomposition for a matrix is the SVD so check the validity 
A ≈ TD.Core_Factors[2] * TD.Core_Factors[1] * TD.Core_Factors[3]

D = itensor(rand(10,20,30), (i,j,k))
TDD = TuckerDecomp(D)

## You can also simply contract a collection of vectors using the contract function.
D ≈ contract(TDD.Core_Factors)

## Computing the SVD of each leg could be expensive
## especially for many leg or large dimension legs.
## It might be easier to compute a matrix multiplication of the tensor to make it square then compute the 
## eigenvalue decomposition
struct SmarterTucker <:TuckerAlgorithm
end
    
using LinearAlgebra
function computeHOSVD(::SmarterTucker, A::ITensor)
    TD = TuckerDecomp([copy(A),])

    for i in inds(A)
        ## square the problem by priming the leg you want to decompose
        square = TD.Core_Factors[1] * prime(TD.Core_Factors[1], i)
        #### The left singular vectors are now the eigenvectors of the problem 
        U,_ = eigen(square)
        
        push!(TD.Core_Factors, U)
        ## Contract the factor with the partially transformed 
        ## Core tensor
        TD.Core_Factors[1] = U * TD.Core_Factors[1]
    end
    return TD
end

TD = TuckerDecomp(A; algorithm=SmarterTucker())
TD.Core_Factors[1].tensor

## Can you make an algorithm that truncates the tucker factors based off of the spectrum?

## Can you use a randomized SVD algorithm instead of squaring the problem?