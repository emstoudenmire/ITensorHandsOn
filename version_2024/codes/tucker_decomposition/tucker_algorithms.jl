abstract type TuckerAlgorithm end

## Tucker decomposition which takes the SVD of a single mode of the matricized target tensor 
## to form a unitary tucker factor. Then the factor is used to transform the target mode to the tucker rank space
## in the core tensor. This process is repeated for every mode
struct NaiveTucker <: TuckerAlgorithm
end

## Computing the SVD of each leg could be expensive
## especially for many leg or large dimension legs.
## It might be easier to compute a matrix multiplication of the tensor single out a target mode and 
## replace the SVD with an eigenvalue decomposition.
struct SmarterTucker <:TuckerAlgorithm
end

## Computing the exact tucker decomposition creates no compression of the original data
## adding a cutoff will make the decomposition more useful.

struct TruncatedTucker 
  cutoff::Number
  solver::Union{TuckerAlgorithm, Nothing}
end
TruncatedTucker(cutoff::Number) = TruncatedTucker(cutoff, nothing)
TruncatedTucker(cutoff::Number, solver::Type{<:TuckerAlgorithm}) = TruncatedTucker(cutoff, solver())

## Can you use a randomized SVD algorithm instead of squaring the problem?
struct RandomizedTucker <:TuckerAlgorithm
  
end