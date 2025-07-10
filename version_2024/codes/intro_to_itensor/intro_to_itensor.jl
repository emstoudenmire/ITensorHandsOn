using LinearAlgebra

elt = Float64
a = randn(elt, 10,20);
b = randn(elt, 20,30);

c = a * b
cp = fill!(similar(c), zero(elt))
for i in 1:10
    for j in 1:20
        for k in 1:30
            cp[i,k] += a[i,j] * b[j,k]
        end
    end
end
c ≈ cp

d = randn(elt, 30,10);
transpose(a * b) * transpose(d)

b = Array(randn(elt, (30,20,40)))

#a * b ## fails because b is a no longer a matrix.

using ITensors
elt = Float64
## ITensor uses a intelligent index system which contains meta information
i = Index(5, "i")
## This is an struct with multiple fields some of the important fields are
dim(i) #(space(i))
tags(i)
id(i)
plev(i)

i' == i
## capital constructors allocate memory
A = ITensor(elt)
B = ITensor(elt, i)
C = ITensor(elt, i')
## Though B and C are both the same dimension and tag
## they have a different number of prime levels so you cannot add them.
#B + C

B * C
j = Index(5,"j")
A = random_itensor(i,j)
A[1,2]
A[i=>1, j=>2]

A
transpose(A)

C = ITensor(rand(ComplexF64, 5,5), i,j)
C[1,1]
dag(C)[1,1]

## If we have data that we are using other places we can wrap it in ITensors
## to make tensor operations simplier!
a = rand(elt, 10,20);
b = rand(elt, 30,20,40);
i,j,k,l = Index.((10,20,30,40));

## lower case itensor constructors do not copy data, they wrap the existing data
A = itensor(a, (i,j))
B = itensor(b, (k,j,l))
A[j=>1,i=>2] ≈ a[2,1]
C = A * B

U,S,V = svd(B, (j,));
ITensors.tensor(S)

## This is numerically equivalent to this line of code
## Though operationally this can be done more efficiently!
bp = permutedims(b, (2,1,3))
c = reshape(a * reshape(bp, (20,30*40)), (10,30,40));
array(C) ≈ c

