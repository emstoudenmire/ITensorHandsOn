### Depending on which GPU you have you can add different versions
using Pkg
active_dir = "$(@__DIR__)/.."
Pkg.activate(active_dir)
#Pkg.add("CUDA")
Pkg.add("Metal")
#Pkg.add("AMDGPU")
## Pkg.add("JLArrays")

## ITensors alone has no specific GPU Array, it simply wraps
## existing GPU Libraries
using ITensors
using BenchmarkTools

## Next load the GPU library of choice and ITensors 
## associated GPU library will load
#using CUDA
using Metal
#using AMDGPU

dev = mtl # mtl cu rocm NDTensors.cpu
elt = (dev == mtl ? Float32 : Float64)
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

elt = Float32 ## Metal doesn't support F64
I, J, K = (500, 500, 500)
i,j,k = Index.((I, J, K))

A = random_itensor(elt, i,j)
B = random_itensor(elt, j,k)

@btime A * B;
mA, mB = dev.((A,B));
@btime mA * mB;

## Test higher order tensor contractions

L, M = 500, 500
l, m = Index.((L, M))

A = random_itensor(elt, i,k,j)
B = random_itensor(elt, l,k)

@btime A * B
mA, mB = dev.((A,B))
@btime mA * mB

## Other libraries can do more efficient Tensor operations (such as TBLIS or cuTENSOR)
## ITensor can interface these to make tensor network contractions more efficent.