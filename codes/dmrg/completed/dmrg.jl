using ITensors
using ITensorMPS
using KrylovKit: KrylovKit

include("mpo_cache.jl")

function mult(Hcache,ϕ)
  LE = left_environment(Hcache)
  LH = left_H(Hcache)
  RH = right_H(Hcache)
  RE = right_environment(Hcache)
  return noprime((((LE*ϕ)*LH)*RH)*RE)
end

function custom_dmrg(H, psi; nsweeps, maxiter=2, krylovdim=6, kws...)
  N = length(H)
  energy = Inf
  psi = orthogonalize(psi,1)
  Hcache = MPOCache(H)

  right_sweep = [(j,j+1) for j=1:N-1]
  left_sweep = [(j,j-1) for j=reverse(2:N)]
  full_sweep = [right_sweep...,left_sweep...]

  for sweep=1:nsweeps, bond in full_sweep
    a,b = bond
    Hcache = position(Hcache,H,psi,bond)

    ϕ0 = psi[a]*psi[b]
    vals, vecs = KrylovKit.eigsolve(v->mult(Hcache,v),ϕ0,1,:SR; maxiter, krylovdim, ishermitian=true)
    ϕ = vecs[1]
    energy = vals[1]
    @show energy

    u,s,v = svd(ϕ,uniqueinds(psi[a],psi[b]); kws...)
    psi[a] = u
    psi[b] = s*v
  end

  return energy,psi
end

let
  N = 100

  sites = siteinds("S=1",N)

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

  nsweeps = 4
  cutoff = 1E-8
  maxdim = 100
  final_energy, psi = custom_dmrg(H,psi0; nsweeps, cutoff, maxdim)

  @show final_energy

  return
end
