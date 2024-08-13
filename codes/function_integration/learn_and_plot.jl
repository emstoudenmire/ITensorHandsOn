using ITensors
using ITensorMPS
using QuanticsTCI
using TCIITensorConversion
using CairoMakie

include("util.jl")

let

  f(x) = (cos(x) - cos(4*(x - 2))) + 4*abs(x)
  xvals = range(-6, 6; length=256)
  qtt, ranks, errors = quanticscrossinterpolate(Float64, f, [xvals]; tolerance=1e-8)

  mps = MPS(qtt.tci)

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

  yvals = f.(xvals)
  lines!(ax, xvals, yvals; color=blue, linewidth)

  yvals = evaluate_all(mps)
  lines!(ax, xvals, yvals; color=red, linestyle=:dot, linewidth=3 * linewidth)

  display(F; px_per_unit=4)

  return
end
