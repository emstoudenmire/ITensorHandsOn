using ITensors
using LinearAlgebra
### Struct for the Tucker decomposition

include("tucker_algorithms.jl")
struct TuckerDecomp
  Core::ITensor
  Factors::Vector{ITensor}
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
      
      ## Contract the factor with the partially transformed 
      ## Core tensor
      
  end
  return TuckerDecomp(Core, Factors)
end

### Testing the naive tucker decomposition 
elt = Float64
a = rand(elt, 10,20);
i,j = Index.((10,20));
A = itensor(a, i,j);

TD = TuckerDecomp(A);

## The tucker decomposition for a matrix is the SVD so check the validity 
A ≈ TD.Factors[1] * TD.Core * TD.Factors[2]

i,j,k = Index.((100,200,300))
D = itensor(randn(100,200,300), i,j,k)

## This is a good reference for how much faster the smarter algorithm
@time TD = TuckerDecomp(D);

D ≈ contract(TD)

function computeHOSVD(::SmarterTucker, A::ITensor; cutoff=nothing)
  Core = copy(A)
  Factors = Vector{ITensor}([])

  for i in inds(A)
      ## square the problem by priming the leg you want to decompose
      
      #### The left singular vectors are now the eigenvectors of the problem 

      
      U = noprime(real(U))
      push!(Factors, U)
      ## Contract the factor with the partially transformed 
      ## Core tensor
      
  end
  return TuckerDecomp(Core, Factors)
end

function computeHOSVD(::RandomizedTucker, A::ITensor; cutoff=nothing)
  Core = copy(A)
  Factors = Vector{ITensor}([])

  for i in inds(A)
      ## compute the randomized projection of the target tensor
      
      #### Compute the SVD of this now smaller problem

      
      U = noprime(real(U))
      push!(Factors, U)
      ## Contract the factor with the partially transformed 
      ## Core tensor
      
  end
  return TuckerDecomp(Core, Factors)
end

algorithm = SmarterTucker()
TD = TuckerDecomp(A; algorithm)

A ≈ contract(TD)
TD = @time TuckerDecomp(D; algorithm);
D ≈ contract(TD)

function computeHOSVD(T::TruncatedTucker, A::ITensor)
  solver = isnothing(T.solver) ? NaiveTucker() : T.solver
  computeHOSVD(solver, A; T.cutoff)
end

algorithm = TruncatedTucker(0.1, SmarterTucker)
@time TD = TuckerDecomp(D; algorithm);
## The decomposition is no longer exact, so fit is a good measure of the decomposition
fit(Tucker, target) = 1.0 - norm(target - contract(Tucker))/ norm(target)


## You can easily use an alternating least squares algorithm to 
## systematically improve the Tucker factor matrices computed using the 
## HOSVD. This is done like so 
##  A [U^{1} U^{2} U^{3} ... U^{k-1} U^{k+1} ... U^{N}] = U^{k} C
## Where U^{i} is the tucker factor associated with the ith mode in A.
## And C is the new optimized core tensor. C can also be expressed as Σ V^{k} derived from the SVD
## of the LHS of the equation.
function HOOI(TD::TuckerDecomp, target::ITensor; niters = 1)
  accuracy = fit(TD, target)
  Factors = copy(TD.Factors)
  Core = copy(TD.Core)
  for iters in 1:niters
      S = ITensor()
      V = ITensor()
      for i in 1:length(size(target))
          ## Make a list of tensors to contract. We want to contract
          ## the target tensor with all tucker factors except for the ith factor
          contraction_list = Vector{ITensor}()
          
          B = contract(contraction_list)

          ## Compute the SVD of this new core tensor B with respect to the ith mode of target
          ## We also want to make sure that the rank of the SVD is equal to the current rank of Factors[i]
          U,S,V = svd()
          ## Update the ith factor
          Factors[i] = U
      end
      ## The core is now S * V
      Core = S * V
      curr_accuracy = fit(TuckerDecomp(Core, Factors), target)
      
      println("$iters \t $curr_accuracy")
      if abs(accuracy - curr_accuracy) < 1e-10
          break
      end
      accuracy = curr_accuracy
  end
  return TuckerDecomp(Core, Factors)
end

TD =  HOOI(TD, D; niters = 15);

## This smaller example improves much more from HOOI.
d = randn(Float64, 20,30,40)
j,k,l = Index.((20,30,40))
D = itensor(d, j,k,l)
algorithm = TruncatedTucker(0.2, SmarterTucker)
TD = TuckerDecomp(D; algorithm);
1.0 - norm(D - contract(TD)) / norm(D)
TD = HOOI(TD, D; niters = 1000);