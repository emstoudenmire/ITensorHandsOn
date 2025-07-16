using ITensors
using ITensorMPS
using Random

include("utilities/utilities.jl")
include("utilities/initial_state.jl")
include("utilities/make_random_gates.jl")

function tebd_gate_evolution(gates, psi; cutoff=default_cutoff(), maxdim=default_maxdim())
  for g in gates
    # Some helper information for you to use:
    j = findsite(psi,inds(g))  # `findsite` finds the integer "j" of
                               # the site of psi where the gate `g` acts
                               # i.e. `g` acts on sites j and j+1

    psi = orthogonalize(psi,j) # `orthogonalize` is a technical step
                               # ensuring the "gauge properties" of psi
                               # are optimal for the SVD truncation step
                               # you will code below

    #
    # TODO: Your task...
    # Implement the "TEBD" gate application method here, using
    # ITensor contraction A*B, 
    # the `noprime` function,
    # and the `svd` function.
    #
    # Tips:
    # - You can obtain the j and j+1 MPS tensors like `psi[j]` and `psi[j+1]`.
    # - The `uniqueinds` and `commoninds` functions can be helpful to obtain collections of
    #   indices to pass to the `svd` function.
    # - Pass cutoff and maxdim to `svd` to control the resulting "bond dimension"
    # - Don't forget to normalize the wavefunction! You can use norm(T) to 
    #   get the norm (sqrt of sum of squared elements) of an ITensor.
    #

  end
  return psi
end

#function tebd_gate_evolution(gates, psi; cutoff=default_cutoff(), maxdim=default_maxdim(), seed=1)
#    psi = apply_gate(g, psi; cutoff, maxdim)
#  return psi
#end

"""
Keyword arguments:
* n - number of qubits
* cutoff - total "weight" of density matrix eigenvalues to truncate on each bond
* maxdim - maximum allowed bond dimension of tensor network after each step
* seed - random seed for making reproducible random gates
"""
function main(; n=40, cutoff=default_cutoff(), maxdim=default_maxdim(), seed=1)
  Random.seed!(seed)
  sites = qubit_sites(n)
  psi = initial_state(sites)
  gates = make_random_gates(sites)
  psi = tebd_gate_evolution(gates, psi; cutoff, maxdim)
  return psi
end
