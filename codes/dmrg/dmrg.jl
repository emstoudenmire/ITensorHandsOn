using ITensors
using ITensorMPS
using KrylovKit: KrylovKit

include("mpo_cache.jl")

function mult(Hcache,ϕ)
  LE = left_environment(Hcache)
  LH = left_H(Hcache)
  RH = right_H(Hcache)
  RE = right_environment(Hcache)

  # 1. TODO using the tensors above
  # and the two-site wavefunction ϕ,
  # implement the product of the 'effective'
  # Hamiltonian times ϕ
  #
  # Tip: to reset all prime levels of an ITensor's 
  # indices to zero, use the `noprime(T)` function.

  Hϕ = ϕ  # ok to delete this line when writing your own code
  #Hϕ = ...

  return Hϕ
end

function custom_dmrg(H, psi; nsweeps, maxiter=2, krylovdim=6, kws...)
  N = length(H)
  energy = Inf
  psi = orthogonalize(psi,1)
  Hcache = MPOCache(H)

  # Sweep 'plans' are arrays of tuples, meaning directed bonds
  right_sweep = [(j,j+1) for j=1:N-1]
  left_sweep = [(j,j-1) for j=reverse(2:N)]
  full_sweep = [right_sweep...,left_sweep...]

  for sweep=1:nsweeps
    println("Sweep ",sweep)
    for bond in full_sweep
      a,b = bond
      Hcache = position(Hcache,H,psi,bond)

      # Form the 'two-site wavefunction'
      ϕ0 = psi[a]*psi[b]

      # Call the eigsolve routine to iteratively improve ϕ
      vals, vecs = KrylovKit.eigsolve(v->mult(Hcache,v),ϕ0,1,:SR; maxiter, krylovdim, ishermitian=true)
      ϕ = vecs[1]
      energy = vals[1]
      @show energy

      #
      # 2. TODO: call the ITensor `svd` routine to factorize
      # the improved ϕ tensor. Decide how to correctly split

      # the outputs u, s, and v into the new MPS tensors.
      #
      # Hint: the `commoninds` and `uniqueinds` functions are
      # useful for obtaining collections of indices of ITensors
      #

      #u,s,v = svd(ϕ, ...)
      #psi[a] = ...
      #psi[b] = ...
    end
  end

  return energy,psi
end

let
  N = 100

  # Define site indices, spin 1 sites
  sites = siteinds("S=1",N)

  # Make Hamiltonian (Heisenberg spin chain)
  Hterms = OpSum()
  for j=1:N-1
    Hterms += "Sz",j,"Sz",j+1
    Hterms += 1/2,"S+",j,"S-",j+1
    Hterms += 1/2,"S-",j,"S+",j+1
  end
  H = MPO(Hterms,sites)

  # Make initial state (random product state)
  psi0 = MPS(sites)
  for (j,sj) in enumerate(sites)
    psi0[j] = random_itensor(sj)
  end

  # Define DMRG parameters and run DMRG
  nsweeps = 4
  cutoff = 1E-8
  maxdim = 100
  final_energy, psi = custom_dmrg(H,psi0; nsweeps, cutoff, maxdim)

  @show final_energy

  # 3. TODO 
  # A correct code should output the energy -138.94008
  # for an N=100 S=1 Heisenberg chain

  return
end
