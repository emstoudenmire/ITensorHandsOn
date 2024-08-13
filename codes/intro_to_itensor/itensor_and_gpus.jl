### Depending on which GPU you have you can add different versions
using Pkg
#Pkg.add("CUDA")
Pkg.add("Metal")
#Pkg.add("AMDGPU")
## Pkg.add("JLArrays")

## ITensors alone has no specific GPU Array, it simply wraps
## existing GPU Libraries
using ITensors

## Next load the GPU library of choice and ITensors 
## associated GPU library will load
#using CUDA
using Metal
#using AMDGPU

dev = mtl # mtl cu rocm
## construct some matrices
a = randn(elt, 20, 30)
b = randn(elt, 30, 40)
c = zeros(elt, (20,40))
mul!(c, a, b)

## convert arrays to GPU device
ca, cb = dev.((a,b))
cc = dev(similar(c))
fill!(cc, zero(elt))
mul!(cc, ca, cb)

Array(cc) ≈ c ## true

using BenchmarkTools

elt = Float32 ## Metal doesn't support F64
I, J, K = (1000, 1000, 1000)
i,j,k = Index.((I, J, K))

A = random_itensor(elt, i,j)
B = random_itensor(elt, j,k)

@btime A * B;
@btime mA * mB;

## Test higher order tensor contractions