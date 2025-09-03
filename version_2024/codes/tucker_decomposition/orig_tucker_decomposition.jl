using ITensors
using LinearAlgebra
### Struct for the Tucker decomposition

struct TuckerDecomp
  Core::ITensor
  Factors::Vector{ITensor}
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
struct TruncatedTucker 
  cutoff::Number
  solver::Union{TuckerAlgorithm, Nothing}
end
TruncatedTucker(cutoff::Number) = TruncatedTucker(cutoff, nothing)
TruncatedTucker(cutoff::Number, solver::Type{<:TuckerAlgorithm}) = TruncatedTucker(cutoff, solver())

## Can you use a randomized SVD algorithm instead of squaring the problem?
struct RandomizedTucker <:TuckerAlgorithm
  
end

TuckerDecomp(A::ITensor; algorithm=NaiveTucker()) = computeHOSVD(algorithm, A)
TuckerDecomp(A::AbstractArray; algorithm=NaiveTucker()) = TuckerDecomp(itensor(A, Index.(size(A))); algorithm)

function ITensors.contract(A::TuckerDecomp)
  contract([A.Core, A.Factors...])
end

## Naive tucker decomposition algorithm
function computeHOSVD(::NaiveTucker, A::ITensor; cutoff=nothing)
  Core = copy(A)
  Factors = Vector{ITensor}([])
  for i in inds(A)
      #### For each leg one must compute the SVD 
      U, _, _ = svd(Core, (i,); cutoff=cutoff)
      push!(Factors, U)
      ## Contract the factor with the partially transformed 
      ## Core tensor
      Core = U * Core
  end
  return TuckerDecomp(Core, Factors)
end

function computeHOSVD(::SmarterTucker, A::ITensor; cutoff=nothing)
  Core = copy(A)
  Factors = Vector{ITensor}([])

  for i in inds(A)
      ## square the problem by priming the leg you want to decompose
      square = Core * prime(Core, i)
      #### The left singular vectors are now the eigenvectors of the problem 
      _,U = eigen(square, i, i'; cutoff)
      
      U = noprime(real(U))
      push!(Factors, U)
      ## Contract the factor with the partially transformed 
      ## Core tensor
      Core = U * Core
  end
  return TuckerDecomp(Core, Factors)
end

function computeHOSVD(T::TruncatedTucker, A::ITensor)
  solver = isnothing(T.solver) ? NaiveTucker() : T.solver
  computeHOSVD(solver, A; T.cutoff)
end

## You can easily use an alternating least squares algorithm to 
## systematically improve the Tucker factor matrices computed using the 
## HOSVD. This is done like so 
##  A [U^{1} U^{2} U^{3} ... U^{k-1} U^{k+1} ... U^{N}] = U^{k} C
## Where U^{i} is the tucker factor associated with the ith mode in A.
## And C is the new optimized core tensor. C can also be expressed as Σ V^{k} derived from the SVD
## of the LHS of the equation.
function HOOI(TD::TuckerDecomp, target::ITensor; niters = 1)
  accuracy = 1.0 - norm(contract(TD) - target) / norm(target)
  Factors = copy(TD.Factors)
  Core = copy(TD.Core)
  for iters in 1:niters
      S = ITensor()
      V = ITensor()
      for i in 1:length(size(target))
          list = Vector{ITensor}([target,])
          append!(list, deleteat!(copy(Factors), [(i)]))
          B = contract(list)
          U,S,V = svd(B, ind(Factors[i],1); maxdim = dim(Factors[i],2))
          Factors[i] = U
      end
      Core = S * V
      curr_accuracy = 1.0 - norm(contract([Core, Factors...]) - target) / norm(target)
      println("$iters \t $curr_accuracy")
      if abs(accuracy - curr_accuracy) < 1e-10
          return TuckerDecomp(Core, Factors)
      end
      accuracy = curr_accuracy
  end
  return TuckerDecomp(Core, Factors)
end

elt = Float64
a = rand(elt, 10,20);
i,j = Index.((10,20));
A = itensor(a, i,j);

TD = TuckerDecomp(A);

## The tucker decomposition for a matrix is the SVD so check the validity 
A ≈ TD.Factors[1] * TD.Core * TD.Factors[2]

i,j,k = Index.((100,200,300))
D = itensor(randn(100,200,300), i,j,k)
@time TD = TuckerDecomp(D);

## You can also simply contract a collection of vectors using the contract function.
D ≈ contract(TD)

algorithm = SmarterTucker()
TD = TuckerDecomp(A; algorithm)

A ≈ contract(TD)
TD = @time TuckerDecomp(D; algorithm);
D ≈ contract(TD)

algorithm = TruncatedTucker(0.1, SmarterTucker)
@time TD = TuckerDecomp(D; algorithm);
1.0 - norm(D - contract(TD))/ norm(D)

@time TD =  HOOI(TD, D; niters = 15);

## This smaller example improves much more from HOOI.
d = randn(Float64, 20,30,40)
j,k,l = Index.((20,30,40))
D = itensor(d, j,k,l)
algorithm = TruncatedTucker(0.2, SmarterTucker)
TD = TuckerDecomp(D; algorithm);
1.0 - norm(D - contract(TD)) / norm(D)
TD = HOOI(TD, D; niters = 1000);
