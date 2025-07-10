using ITensors, ITensorMPS
using CairoMakie

let

  N = 100
  sites = siteinds("S=1", N)

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
  cutoff = [1E-11]
  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)

  sz = expect(psi,"Sz")
  yvals = sz

  #
  # Plot
  # 
  white = RGBAf(0, 0, 0, 0.0)
  green = RGBf(1 / 255, 113 / 255, 0)
  blue = RGBf(0, 84 / 255, 155 / 255)
  red = RGBf(174 / 255, 42 / 255, 22 / 255)

  F = Figure(; size=(600, 400))
  ax = Axis(F[1, 1]; xgridcolor=white, ygridcolor=white)

  linewidth = 0.5

  xvals = 1:N
  lines!(ax, xvals, yvals; color=blue, linewidth)

  display(F; px_per_unit=4)

  return
end
