using ITensors, ITensorMPS

let
  N = 300
  sites = siteinds("S=1/2", N)

  terms = OpSum()
  for j in 1:(N - 1)
    terms += "Sz", j, "Sz", j + 1
    terms += 0.5, "S+", j, "S-", j + 1
    terms += 0.5, "S-", j, "S+", j + 1
  end
  H = MPO(terms, sites)

  psi0 = random_mps(sites; linkdims=10)

  nsweeps = 5
  maxdim = [10, 20, 100, 100, 200]
  cutoff = [1E-6]
  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)

  return
end
