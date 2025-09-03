using Printf
import TensorCrossInterpolation as TCI
using QuanticsTCI
using ITensors
using ITensorMPS
using TCIITensorConversion

function integrate(M::MPS)
  I = ITensor(1.0)
  for (m, s) in zip(M, siteinds(M))
    I *= (m * ITensor([1 / 2, 1 / 2], s))
  end
  return scalar(I)
end

let
  n = 32
  c = 1E-5 # Good values to take are between 1E-4 to 1E-9

  # Unnormalized Cauchy distribution - integral is ∫f(x) ≈ π
  f(x) = c/((x-0.5)^2 + c^2)

  @printf("\nc = %.3E\n\n",c)

  xvals = range(0, 1; length=2^n)

  qtt, ranks, errors = quanticscrossinterpolate(Float64, f, [xvals]; tolerance=1e-8)
  mps = MPS(qtt.tci)

  I = @timed integrate(mps)
  println("\nIntegration took $(I.time) seconds")

  println()
  @printf("π = %.12f\n",π)
  @printf("I = %.12f\n",I.value)
  @printf("err = %.4E\n",abs(π-I.value))

  return
end
