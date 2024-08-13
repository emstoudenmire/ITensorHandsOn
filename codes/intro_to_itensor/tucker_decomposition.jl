using ITensors
using LinearAlgebra
### Struct for the Tucker decomposition

struct TuckerDecomp
  Core_Factors::Vector{ITensor}
end

abstract type TuckerAlgorithm end

struct NaiveTucker <: TuckerAlgorithm
end

## Computing the SVD of each leg could be expensive
## especially for many leg or large dimension legs.
## It might be easier to compute a matrix multiplication of the tensor to make it square then compute the 
## eigenvalue decomposition
struct SmarterTucker <:TuckerAlgorithm
end

## Can you make an algorithm that truncates the tucker factors based off of the spectrum?
struct TruncatedTucker <:TuckerAlgorithm
  cutoff::Number
end

## Can you use a randomized SVD algorithm instead of squaring the problem?
struct RandomizedTucker <:TuckerAlgorithm
  
end

TuckerDecomp(A::ITensor; algorithm::TuckerAlgorithm=NaiveTucker()) = computeHOSVD(algorithm, A)
TuckerDecomp(A::AbstractArray; algorithm::TuckerAlgorithm=NaiveTucker()) = TuckerDecomp(itensor(A, Index.(size(A))); algorithm)

## Naive tucker decomposition algorithm
function computeHOSVD(::NaiveTucker, A::ITensor)
  TD = TuckerDecomp([copy(A),])

  for i in inds(A)
      #### For each leg one must compute the SVD 
      U, _, _ = svd(TD.Core_Factors[1], (i,))
      push!(TD.Core_Factors, U)
      ## Contract the factor with the partially transformed 
      ## Core tensor
      TD.Core_Factors[1] = U * TD.Core_Factors[1]
  end
  return TD
end

function computeHOSVD(::SmarterTucker, A::ITensor)
  TD = TuckerDecomp([copy(A),])

  for i in inds(A)
      ## square the problem by priming the leg you want to decompose
      square = TD.Core_Factors[1] * prime(TD.Core_Factors[1], i)
      #### The left singular vectors are now the eigenvectors of the problem 
      _,U = eigen(square, i, i')
      
      U = noprime(real(U))
      push!(TD.Core_Factors, U)
      ## Contract the factor with the partially transformed 
      ## Core tensor
      TD.Core_Factors[1] = U * TD.Core_Factors[1]
  end
  return TD
end

function computeHOSVD(T::TruncatedTucker, A::ITensor)
  TD = TuckerDecomp([copy(A),])

  for i in inds(A)
      ## square the problem by priming the leg you want to decompose
      square = TD.Core_Factors[1] * prime(TD.Core_Factors[1], i)
      #### The left singular vectors are now the eigenvectors of the problem 
      _,U = eigen(square, i, i'; T.cutoff)
      
      U = noprime(real(U))
      push!(TD.Core_Factors, U)
      ## Contract the factor with the partially transformed 
      ## Core tensor
      TD.Core_Factors[1] = U * TD.Core_Factors[1]
  end
  return TD
end

## You can easily use an alternating least squares algorithm to 
## systematically improve the Tucker factor matrices computed using the 
## HOSVD. This is done like so 
##  A [U^{1} U^{2} U^{3} ... U^{k-1} U^{k+1} ... U^{N}] = U^{k} C
## Where U^{i} is the tucker factor associated with the ith mode in A.
## And C is the new optimized core tensor. C can also be expressed as Σ V^{k} derived from the SVD
## of the LHS of the equation.
function HOOI(TD::TuckerDecomp, target::ITensor; niters = 1)
  accuracy = 1.0 - norm(contract(TD.Core_Factors) - target) / norm(target)
  for iters in 1:niters
      S = ITensor()
      V = ITensor()
      for i in 1:length(size(target))
          list = Vector{ITensor}([target,])
          append!(list, deleteat!(copy(TD.Core_Factors), [1, (i+1)]))
          B = contract(list)
          U,S,V = svd(B, ind(TD.Core_Factors[i+1],1); maxdim = dim(TD.Core_Factors[i+1],2))
          TD.Core_Factors[i+1] = U
          TD.Core_Factors[1] = S * V
      end
      TD.Core_Factors[1] = S * V
      curr_accuracy = 1.0 - norm(contract(TD.Core_Factors) - target) / norm(target)
      println("$iters \t $curr_accuracy")
      if abs(accuracy - curr_accuracy) < 1e-10
          return
      end
      accuracy = curr_accuracy
  end
end

TD = TuckerDecomp(A)

## The tucker decomposition for a matrix is the SVD so check the validity 
A ≈ TD.Core_Factors[2] * TD.Core_Factors[1] * TD.Core_Factors[3]

i,j,k = Index.((100,200,300))
D = itensor(randn(100,200,300), i,j,k)
@time TD = TuckerDecomp(D);

## You can also simply contract a collection of vectors using the contract function.
D ≈ contract(TD.Core_Factors)

algorithm = SmarterTucker()
TD = TuckerDecomp(A; algorithm)
TD.Core_Factors[3].tensor

A ≈ contract(TD.Core_Factors)
TD = @time TuckerDecomp(D; algorithm);
D ≈ contract(TD.Core_Factors)

algorithm = TruncatedTucker(0.1)
@time TD = TuckerDecomp(D; algorithm);
TD.Core_Factors[1]
1.0 - norm(D - contract(TD.Core_Factors))/ norm(D)

HOOI(TD, D; niters = 15)