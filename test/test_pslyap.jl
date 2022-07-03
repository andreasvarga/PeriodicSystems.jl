module Test_pslyap

using PeriodicSystems
using DescriptorSystems
using Symbolics
using Test
using LinearAlgebra
using ApproxFun

println("Test_pslyap")

@testset "dpsylv2, dpsylv2!, dpsylv2krsol!" begin


W = Matrix{Float64}(undef,2,14)
WZ = Matrix{Float64}(undef,8,8)
WY = Vector{Float64}(undef,8)
WX = Matrix{Float64}(undef,4,5)

p = 2; 
al = [-0.0028238980383030643;;; 0.3319882632937995]
ar = [-0.0028238980383030643;;; 0.3319882632937995]
q = rand(2,2,p);
REV = true
KSCHUR = 1
n1 = 1; n2 = 1; 
X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
i1 = 1:n1; i2 = 1:n2
rez1 = al[i1,i1,1]'*X[i1,i2,2]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
rez2 = al[i1,i1,2]'*X[i1,i2,1]*ar[i2,i2,2]-X[i1,i2,2]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  

X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez1 = al[i1,i1,1]'*X1[i1,i2,2]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
rez2 = al[i1,i1,2]'*X1[i1,i2,1]*ar[i2,i2,2]-X1[i1,i2,2]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  



p = 1; 
al = 100*rand(2,2,p); ar = rand(2,2,p); q = rand(2,2,p);
REV = true

KSCHUR = 1
n1 = 1; n2 = 1; 
X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
i1 = 1:n1; i2 = 1:n2
rez = al[i1,i1,1]'*X[i1,i2,1]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
@test norm(rez) < 1.e-7         
X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
rez = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,1]+q[i1,i2,1]   
@test norm(rez) < 1.e-7  

X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,1]+q[i1,i2,1]   
@test norm(rez) < 1.e-7  
X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,1]+q[i1,i2,1]   
@test norm(rez) < 1.e-7  


n1 = 1; n2 = 2; 
X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
i1 = 1:n1; i2 = 1:n2
rez = al[i1,i1,1]'*X[i1,i2,1]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
@test norm(rez) < 1.e-7         
X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
rez = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,1]+q[i1,i2,1]   
@test norm(rez) < 1.e-7  


X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = al[i1,i1,1]'*X1[i1,i2,1]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
@test norm(rez) < 1.e-7  
X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,1]+q[i1,i2,1]   
@test norm(rez) < 1.e-7  



n1 = 2; n2 = 1; 
X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
i1 = 1:n1; i2 = 1:n2
rez = al[i1,i1,1]'*X[i1,i2,1]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
@test norm(rez) < 1.e-7         
X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
rez = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,1]+q[i1,i2,1]   
@test norm(rez) < 1.e-7  

X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = al[i1,i1,1]'*X1[i1,i2,1]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
@test norm(rez) < 1.e-7  
X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,1]+q[i1,i2,1]   
@test norm(rez) < 1.e-7  


n1 = 2; n2 = 2; 
X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
i1 = 1:n1; i2 = 1:n2
rez = al[i1,i1,1]'*X[i1,i2,1]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
@test norm(rez) < 1.e-7         
X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
rez = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,1]+q[i1,i2,1]   
@test norm(rez) < 1.e-7  

X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = al[i1,i1,1]'*X1[i1,i2,1]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
@test norm(rez) < 1.e-7  
X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,1]+q[i1,i2,1]   
@test norm(rez) < 1.e-7  


p = 2; 
al = rand(2,2,p); al[:,:,1] = triu(al[:,:,1]); ar = rand(2,2,p); ar[:,:,1] = triu(ar[:,:,2]); 
q = rand(2,2,p);
REV = true
KSCHUR = 2
n1 = 1; n2 = 1; 
X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
i1 = 1:n1; i2 = 1:n2
rez1 = al[i1,i1,1]'*X[i1,i2,2]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
rez2 = al[i1,i1,2]'*X[i1,i2,1]*ar[i2,i2,2]-X[i1,i2,2]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
rez1 = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,2]+q[i1,i2,1]   
rez2 = al[i1,i1,2]*X[i1,i2,2]*ar[i2,i2,2]'-X[i1,i2,1]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  

X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez1 = al[i1,i1,1]'*X1[i1,i2,2]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
rez2 = al[i1,i1,2]'*X1[i1,i2,1]*ar[i2,i2,2]-X1[i1,i2,2]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez1 = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,2]+q[i1,i2,1]   
rez2 = al[i1,i1,2]*X1[i1,i2,2]*ar[i2,i2,2]'-X1[i1,i2,1]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  


WUSD3 = Array{Float64,3}(undef,4,4,p)
WUD3 = Array{Float64,3}(undef,4,4,p)
WUL3 = Matrix{Float64}(undef,4*p,4)
WY1 = Vector{Float64}(undef,4*p)
W1 = Matrix{Float64}(undef,8,8)


X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez1 = al[i1,i1,1]'*X3[i1,i2,2]*ar[i2,i2,1]-X3[i1,i2,1]+q[i1,i2,1]   
rez2 = al[i1,i1,2]'*X3[i1,i2,1]*ar[i2,i2,2]-X3[i1,i2,2]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  
X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez1 = al[i1,i1,1]*X3[i1,i2,1]*ar[i2,i2,1]'-X3[i1,i2,2]+q[i1,i2,1]   
rez2 = al[i1,i1,2]*X3[i1,i2,2]*ar[i2,i2,2]'-X3[i1,i2,1]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  


n1 = 1; n2 = 2; 
X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
i1 = 1:n1; i2 = 1:n2
rez1 = al[i1,i1,1]'*X[i1,i2,2]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
rez2 = al[i1,i1,2]'*X[i1,i2,1]*ar[i2,i2,2]-X[i1,i2,2]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
rez1 = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,2]+q[i1,i2,1]   
rez2 = al[i1,i1,2]*X[i1,i2,2]*ar[i2,i2,2]'-X[i1,i2,1]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7   

X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez1 = al[i1,i1,1]'*X1[i1,i2,2]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
rez2 = al[i1,i1,2]'*X1[i1,i2,1]*ar[i2,i2,2]-X1[i1,i2,2]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) # fails
rez1 = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,2]+q[i1,i2,1]   
rez2 = al[i1,i1,2]*X1[i1,i2,2]*ar[i2,i2,2]'-X1[i1,i2,1]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  


X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez1 = al[i1,i1,1]'*X3[i1,i2,2]*ar[i2,i2,1]-X3[i1,i2,1]+q[i1,i2,1]   
rez2 = al[i1,i1,2]'*X3[i1,i2,1]*ar[i2,i2,2]-X3[i1,i2,2]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  
X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez1 = al[i1,i1,1]*X3[i1,i2,1]*ar[i2,i2,1]'-X3[i1,i2,2]+q[i1,i2,1]   
rez2 = al[i1,i1,2]*X3[i1,i2,2]*ar[i2,i2,2]'-X3[i1,i2,1]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  


n1 = 2; n2 = 1; 
X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
i1 = 1:n1; i2 = 1:n2
rez1 = al[i1,i1,1]'*X[i1,i2,2]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
rez2 = al[i1,i1,2]'*X[i1,i2,1]*ar[i2,i2,2]-X[i1,i2,2]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
rez1 = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,2]+q[i1,i2,1]   
rez2 = al[i1,i1,2]*X[i1,i2,2]*ar[i2,i2,2]'-X[i1,i2,1]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         

X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez1 = al[i1,i1,1]'*X1[i1,i2,2]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
rez2 = al[i1,i1,2]'*X1[i1,i2,1]*ar[i2,i2,2]-X1[i1,i2,2]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez1 = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,2]+q[i1,i2,1]   
rez2 = al[i1,i1,2]*X1[i1,i2,2]*ar[i2,i2,2]'-X1[i1,i2,1]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  


X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez1 = al[i1,i1,1]'*X3[i1,i2,2]*ar[i2,i2,1]-X3[i1,i2,1]+q[i1,i2,1]   
rez2 = al[i1,i1,2]'*X3[i1,i2,1]*ar[i2,i2,2]-X3[i1,i2,2]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  
X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez1 = al[i1,i1,1]*X3[i1,i2,1]*ar[i2,i2,1]'-X3[i1,i2,2]+q[i1,i2,1]   
rez2 = al[i1,i1,2]*X3[i1,i2,2]*ar[i2,i2,2]'-X3[i1,i2,1]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  


n1 = 2; n2 = 2; 
X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
i1 = 1:n1; i2 = 1:n2
rez1 = al[i1,i1,1]'*X[i1,i2,2]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
rez2 = al[i1,i1,2]'*X[i1,i2,1]*ar[i2,i2,2]-X[i1,i2,2]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
rez1 = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,2]+q[i1,i2,1]   
rez2 = al[i1,i1,2]*X[i1,i2,2]*ar[i2,i2,2]'-X[i1,i2,1]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7   

X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez1 = al[i1,i1,1]'*X1[i1,i2,2]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
rez2 = al[i1,i1,2]'*X1[i1,i2,1]*ar[i2,i2,2]-X1[i1,i2,2]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez1 = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,2]+q[i1,i2,1]   
rez2 = al[i1,i1,2]*X1[i1,i2,2]*ar[i2,i2,2]'-X1[i1,i2,1]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  

X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez1 = al[i1,i1,1]'*X3[i1,i2,2]*ar[i2,i2,1]-X3[i1,i2,1]+q[i1,i2,1]   
rez2 = al[i1,i1,2]'*X3[i1,i2,1]*ar[i2,i2,2]-X3[i1,i2,2]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  
X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez1 = al[i1,i1,1]*X3[i1,i2,1]*ar[i2,i2,1]'-X3[i1,i2,2]+q[i1,i2,1]   
rez2 = al[i1,i1,2]*X3[i1,i2,2]*ar[i2,i2,2]'-X3[i1,i2,1]+q[i1,i2,2]   
@test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  


p = 100; 
WZ = Matrix{Float64}(undef,p*4,p*4)
WY = Vector{Float64}(undef,p*4)
W = Matrix{Float64}(undef,2,14)
WX = Matrix{Float64}(undef,4,5)


WUSD3 = Array{Float64,3}(undef,4,4,p)
WUD3 = Array{Float64,3}(undef,4,4,p)
WUL3 = Matrix{Float64}(undef,4*p,4)
WY1 = Vector{Float64}(undef,4*p)
W1 = Matrix{Float64}(undef,8,8)



al = rand(2,2,p); ar = rand(2,2,p); q = rand(2,2,p);
[triu!(view(al,:,:,i)) for i in 2:p]
[triu!(view(ar,:,:,i)) for i in 2:p]
REV = true

KSCHUR = 1

n1 = 1; n2 = 1; 
@time X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX);
i1 = 1:n1; i2 = 1:n2
ip = 1:p; ip1 = mod.(ip,p).+1;
rez = [ al[i1,i1,ip[i]]'*X[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7           
X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
rez = [ al[i1,i1,ip[i]]*X[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7 

X1 = copy(q[i1,i2,1:p]); @time dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = [ al[i1,i1,ip[i]]'*X1[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X1[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7           
X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = [ al[i1,i1,ip[i]]*X1[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X1[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7 


X3 = copy(q[i1,i2,1:p]); @time dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1) 
rez = [ al[i1,i1,ip[i]]'*X3[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X3[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7  
X3 = copy(q[i1,i2,1:p]); @time dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez = [ al[i1,i1,ip[i]]*X3[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X3[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7 


n1 = 1; n2 = 2; 
X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
i1 = 1:n1; i2 = 1:n2
ip = 1:p; ip1 = mod.(ip,p).+1;
rez = [ al[i1,i1,ip[i]]'*X[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7           
X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
rez = [ al[i1,i1,ip[i]]*X[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7 

X1 = copy(q[i1,i2,1:p]); @time dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = [ al[i1,i1,ip[i]]'*X1[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X1[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7           
X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = [ al[i1,i1,ip[i]]*X1[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X1[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7 


X3 = copy(q[i1,i2,1:p]); @time dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez = [ al[i1,i1,ip[i]]'*X3[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X3[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7  
X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez = [ al[i1,i1,ip[i]]*X3[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X3[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7 



n1 = 2; n2 = 1; 
X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
i1 = 1:n1; i2 = 1:n2
ip = 1:p; ip1 = mod.(ip,p).+1;
rez = [ al[i1,i1,ip[i]]'*X[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7           
X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
rez = [ al[i1,i1,ip[i]]*X[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7 

X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = [ al[i1,i1,ip[i]]'*X1[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X1[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7           
X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = [ al[i1,i1,ip[i]]*X1[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X1[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7 


X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez = [ al[i1,i1,ip[i]]'*X3[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X3[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7  
X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez = [ al[i1,i1,ip[i]]*X3[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X3[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7 



n1 = 2; n2 = 2; 
@time X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX); 
i1 = 1:n1; i2 = 1:n2
ip = 1:p; ip1 = mod.(ip,p).+1;
rez = [ al[i1,i1,ip[i]]'*X[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7           
X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
rez = [ al[i1,i1,ip[i]]*X[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7 

X1 = copy(q[i1,i2,1:p]); @time dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = [ al[i1,i1,ip[i]]'*X1[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X1[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7           
X1 = copy(q[i1,i2,1:p]); @time dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = [ al[i1,i1,ip[i]]*X1[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X1[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7 

X3 = copy(q[i1,i2,1:p]); @time dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez = [ al[i1,i1,ip[i]]'*X3[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X3[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7  
X3 = copy(q[i1,i2,1:p]); @time dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez = [ al[i1,i1,ip[i]]*X3[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X3[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
@test norm(rez) < 1.e-7 



#pq = 2; p = 1
pq = 10; p = 2
WZ = Matrix{Float64}(undef,pq*4,pq*4)
WY = Vector{Float64}(undef,pq*4)
W = Matrix{Float64}(undef,2,14)
WX = Matrix{Float64}(undef,4,5)

WUSD3 = Array{Float64,3}(undef,4,4,pq)
WUD3 = Array{Float64,3}(undef,4,4,pq)
WUL3 = Matrix{Float64}(undef,4*pq,4)
WY1 = Vector{Float64}(undef,4*pq)


al = 0.1*rand(2,2,p); ar = rand(2,2,p); q = rand(2,2,pq);
[triu!(view(al,:,:,i)) for i in 2:p]
[triu!(view(ar,:,:,i)) for i in 2:p]
REV = true
KSCHUR = 1

n1 = 1; n2 = 1; 
X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
i1 = 1:n1; i2 = 1:n2
ip = mod.(0:pq-1,p).+1; ip1 = mod.(ip,p).+1;  ipq = mod.(0:pq-1,pq).+1; ipq1 = mod.(1:pq,pq).+1; 
rez = [ al[i1,i1,ip[i]]'*X[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7           
X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
rez = [ al[i1,i1,ip[i]]*X[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7 

X1 = copy(q[i1,i2,1:pq]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = [ al[i1,i1,ip[i]]'*X1[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X1[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7           
X1 = copy(q[i1,i2,1:pq]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = [ al[i1,i1,ip[i]]*X1[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X1[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7 

X3 = copy(q[i1,i2,1:pq]); @time dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez = [ al[i1,i1,ip[i]]'*X3[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X3[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7           
X3 = copy(q[i1,i2,1:pq]); @time dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez = [ al[i1,i1,ip[i]]*X3[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X3[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7 



n1 = 1; n2 = 2; 
X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
i1 = 1:n1; i2 = 1:n2
ip = mod.(0:pq-1,p).+1; ip1 = mod.(ip,p).+1;  ipq = mod.(0:pq-1,pq).+1; ipq1 = mod.(1:pq,pq).+1; 
rez = [ al[i1,i1,ip[i]]'*X[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7           
X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
rez = [ al[i1,i1,ip[i]]*X[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7 

X1 = copy(q[i1,i2,1:pq]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) # fails
rez = [ al[i1,i1,ip[i]]'*X1[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X1[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7           
X1 = copy(q[i1,i2,1:pq]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = [ al[i1,i1,ip[i]]*X1[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X1[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7 

X3 = copy(q[i1,i2,1:pq]); @time dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez = [ al[i1,i1,ip[i]]'*X3[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X3[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7           
X3 = copy(q[i1,i2,1:pq]); @time dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez = [ al[i1,i1,ip[i]]*X3[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X3[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7 




n1 = 2; n2 = 1; 
X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
i1 = 1:n1; i2 = 1:n2
ip = mod.(0:pq-1,p).+1; ip1 = mod.(ip,p).+1;  ipq = mod.(0:pq-1,pq).+1; ipq1 = mod.(1:pq,pq).+1; 
rez = [ al[i1,i1,ip[i]]'*X[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7           
X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
rez = [ al[i1,i1,ip[i]]*X[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7 

X1 = copy(q[i1,i2,1:pq]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = [ al[i1,i1,ip[i]]'*X1[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X1[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7           
X1 = copy(q[i1,i2,1:pq]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = [ al[i1,i1,ip[i]]*X1[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X1[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7 

X3 = copy(q[i1,i2,1:pq]); @time dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez = [ al[i1,i1,ip[i]]'*X3[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X3[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7           
X3 = copy(q[i1,i2,1:pq]); @time dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez = [ al[i1,i1,ip[i]]*X3[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X3[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7 

n1 = 2; n2 = 2; 
X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
i1 = 1:n1; i2 = 1:n2
ip = mod.(0:pq-1,p).+1; ip1 = mod.(ip,p).+1;  ipq = mod.(0:pq-1,pq).+1; ipq1 = mod.(1:pq,pq).+1; 
rez = [ al[i1,i1,ip[i]]'*X[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7           
X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
rez = [ al[i1,i1,ip[i]]*X[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7 

X1 = copy(q[i1,i2,1:pq]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = [ al[i1,i1,ip[i]]'*X1[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X1[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7           
X1 = copy(q[i1,i2,1:pq]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
rez = [ al[i1,i1,ip[i]]*X1[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X1[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7 

X3 = copy(q[i1,i2,1:pq]); @time dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez = [ al[i1,i1,ip[i]]'*X3[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X3[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7           
X3 = copy(q[i1,i2,1:pq]); @time dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1)  
rez = [ al[i1,i1,ip[i]]*X3[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X3[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
@test norm(rez) < 1.e-7 

end # dpsylv2


@testset "prlyap && pflyap" begin

# constant dimensions
na = [5, 5]; ma = [5,5]; pa = 2; pc = 2;   
Ad = PeriodicMatrix([rand(Float64,ma[i],na[i]) for i in 1:pa],pa);  
x = [rand(na[i],na[i]) for i in 1:pc]
Qd = PeriodicMatrix([ x[i]+x[i]' for i in 1:pc],pc);
X2 = PeriodicMatrix(pslyapdkr(Ad.M, Qd.M; adj = true), lcm(pa,pc));
@test norm(Ad'*pmshift(X2)*Ad-X2+Qd) < 1.e-7 

# time-varying dimensions
na = [5, 3, 3, 4, 1]; ma = [3, 3, 4, 1, 5]; pa = 5; pc = 5;   
#na = 5*na; ma = 5*ma;
Ad = PeriodicMatrix([rand(Float64,ma[i],na[i]) for i in 1:pa],pa);  
x = [rand(na[i],na[i]) for i in 1:pc]
Qd = PeriodicMatrix([ x[i]+x[i]' for i in 1:pc],pc);
X = prlyap(Ad, Qd);
@test norm(Ad'*pmshift(X)*Ad-X+Qd) < 1.e-7 

Ad1 = convert(PeriodicArray,Ad); Qd1 = convert(PeriodicArray,Qd);
X1 = prlyap(Ad1, Qd1); 
@test norm(Ad1'*pmshift(X1)*Ad1-X1+Qd1) < 1.e-7 

X2 = PeriodicMatrix(pslyapdkr(Ad.M, Qd.M; adj = true), lcm(pa,pc));
@test norm(Ad'*pmshift(X2)*Ad-X2+Qd) < 1.e-7 && norm(X1-pm2pa(X2)) < 1.e-7

x = [rand(ma[i],ma[i]) for i in 1:pc]
Qd = PeriodicMatrix([ x[i]+x[i]' for i in 1:pc],pc);

X = pflyap(Ad, Qd) 
@test norm(Ad*X*Ad'- pmshift(X)+Qd) < 1.e-7 

Ad1 = convert(PeriodicArray,Ad); Qd1 = convert(PeriodicArray,Qd);
X1 = pflyap(Ad1, Qd1); 
@test norm(Ad1*X1*Ad1'-pmshift(X1)+Qd1) < 1.e-7 

X2 = PeriodicMatrix(pslyapdkr(Ad.M, Qd.M; adj = false), lcm(pa,pc));
@test norm(Ad*X2*Ad'-pmshift(X2)+Qd) < 1.e-7 && norm(X1-pm2pa(X2)) < 1.e-7


# constant dimensions
n = 5; pa = 10; pc = 2;     
Ad = 0.5*PeriodicArray(rand(Float32,n,n,pa),pa);
q = rand(n,n,pc); [q[:,:,i] = q[:,:,i]'+q[:,:,i] for i in 1:pc];
Qd = PeriodicArray(q,pc);

X = prlyap(Ad, Qd); 
@test norm(Ad'*pmshift(X)*Ad-X+Qd) < 1.e-7 

p = lcm(pa,pc)
ia = mod.(0:p-1,pa).+1; ipx = mod.(0:p-1,p).+1;  
ipc = mod.(0:p-1,pc).+1; ipx1 = mod.(ipx,p).+1;  
rez = [ Ad.M[:,:,ia[i]]'*X.M[:,:,ipx1[i]]*Ad.M[:,:,ia[i]]-X.M[:,:,ipx[i]]+Qd.M[:,:,ipc[i]] for i in ipx]
@test norm(rez) < 1.e-7 

X1 = prlyap(convert(PeriodicMatrix,Ad), convert(PeriodicMatrix,Qd)); 
X = convert(PeriodicArray,X1)
@test norm(Ad'*pmshift(X)*Ad-X+Qd) < 1.e-7 

X = pflyap(Ad, Qd) 
@test norm(Ad*X*Ad'-pmshift(X)+Qd) < 1.e-7 

X1 = pflyap(convert(PeriodicMatrix,Ad), convert(PeriodicMatrix,Qd)); 
X = convert(PeriodicArray,X1)
@test norm(Ad*X*Ad'-pmshift(X)+Qd) < 1.e-7 


rez = [ Ad.M[:,:,ia[i]]*X.M[:,:,ipx[i]]*Ad.M[:,:,ia[i]]'-X.M[:,:,ipx1[i]]+Qd.M[:,:,ipc[i]] for i in ipx]
@test norm(rez) < 1.e-7 

#pseig(Ad)

n = 5; pa = 3; pc = 2;     
Ad = 0.5*PeriodicArray(rand(Float64,n,n,pa),pa);
q = rand(n,n,pc); [q[:,:,i] = q[:,:,i]'+q[:,:,i] for i in 1:pc];
Qd = PeriodicArray(q,pc);

X = prlyap(Ad, Qd); 
@test norm(Ad'*pmshift(X)*Ad-X+Qd) < 1.e-7 

X = pflyap(Ad, Qd) 
@test norm(Ad*X*Ad'-pmshift(X)+Qd) < 1.e-7 

#pseig(Ad)

n = 5; pa = 3; pc = 1;     
Ad = 0.5*PeriodicArray(rand(Float32,n,n,pa),pa);
q = rand(n,n); q = q'+q;

X = prlyap(Ad, q); 
@test norm(Ad'*pmshift(X)*Ad-X+q) < 1.e-7 

X = pflyap(Ad, q) 
@test norm(Ad*X*Ad'-pmshift(X)+q) < 1.e-7 



end 

@testset "pdlyad" begin


n = 5; pa = 1; pc = 1; p = lcm(pa,pc)
a = (1/(n*n*pa))*rand(n,n,pa); q = rand(n,n,pc); 
[q[:,:,i] = q[:,:,i]'+q[:,:,i] for i in 1:pc];

X = pslyapd(a, q; adj = true) 
ia = mod.(0:p-1,pa).+1; ipx = mod.(0:p-1,p).+1;  
ipc = mod.(0:p-1,pc).+1; ipx1 = mod.(ipx,p).+1;  

rez = [ a[:,:,ia[i]]'*X[:,:,ipx1[i]]*a[:,:,ia[i]]-X[:,:,ipx[i]]+q[:,:,ipc[i]] for i in ipx]
@test norm(rez) < 1.e-7 

X = pslyapd(a, q; adj = false) 
ia = mod.(0:p-1,pa).+1; ipx = mod.(0:p-1,p).+1;  
ipc = mod.(0:p-1,pc).+1; ipx1 = mod.(ipx,p).+1;  

rez = [ a[:,:,ia[i]]*X[:,:,ipx[i]]*a[:,:,ia[i]]'-X[:,:,ipx1[i]]+q[:,:,ipc[i]] for i in ipx]
@test norm(rez) < 1.e-7 


n = 5; pa = 5; pc = 10; p = lcm(pa,pc)
a = (1/(n*n*pa))*rand(n,n,pa); q = rand(n,n,pc); 
[q[:,:,i] = q[:,:,i]'+q[:,:,i] for i in 1:pc];

X = pslyapd(a, q; adj = true) 
ia = mod.(0:p-1,pa).+1; ipx = mod.(0:p-1,p).+1;  
ipc = mod.(0:p-1,pc).+1; ipx1 = mod.(ipx,p).+1;  

rez = [ a[:,:,ia[i]]'*X[:,:,ipx1[i]]*a[:,:,ia[i]]-X[:,:,ipx[i]]+q[:,:,ipc[i]] for i in ipx]
@test norm(rez) < 1.e-7 

X = pslyapd(a, q; adj = false) 
ia = mod.(0:p-1,pa).+1; ipx = mod.(0:p-1,p).+1;  
ipc = mod.(0:p-1,pc).+1; ipx1 = mod.(ipx,p).+1;  

rez = [ a[:,:,ia[i]]*X[:,:,ipx[i]]*a[:,:,ia[i]]'-X[:,:,ipx1[i]]+q[:,:,ipc[i]] for i in ipx]
@test norm(rez) < 1.e-7 

X = pslyapdkr(a, q; adj = true) 
ia = mod.(0:p-1,pa).+1; ipx = mod.(0:p-1,p).+1;  
ipc = mod.(0:p-1,pc).+1; ipx1 = mod.(ipx,p).+1;  

rez = [ a[:,:,ia[i]]'*X[:,:,ipx1[i]]*a[:,:,ia[i]]-X[:,:,ipx[i]]+q[:,:,ipc[i]] for i in ipx]
@test norm(rez) < 1.e-7 

X = pslyapdkr(a, q; adj = false) 
ia = mod.(0:p-1,pa).+1; ipx = mod.(0:p-1,p).+1;  
ipc = mod.(0:p-1,pc).+1; ipx1 = mod.(ipx,p).+1;  

rez = [ a[:,:,ia[i]]*X[:,:,ipx[i]]*a[:,:,ia[i]]'-X[:,:,ipx1[i]]+q[:,:,ipc[i]] for i in ipx]
@test norm(rez) < 1.e-7 



# q = copy(qs);
# X1 = pslyapdkr(a,q)
# ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# rez = [ a[:,:,ip[i]]'*X1[:,:,ip1[i]]*a[:,:,ip[i]]-X1[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7  

# KSCHUR = 1
# q = copy(qs);
# X = pdlyaps1!(KSCHUR, a, q; adj = true) 
# ip = mod.(0:pc-1,p).+1; ip1 = mod.(ip,p).+1;  
# ipc = mod.(0:pc-1,pc).+1; ipc1 = mod.(ipc,pc).+1;  

# rez = [ a[:,:,ip[i]]'*X[:,:,ipc1[i]]*a[:,:,ip[i]]-X[:,:,ipc[i]]+qs[:,:,ipc[i]] for i in ipc]
# @test norm(rez) < 1.e-7 

# # q = copy(qs);
# # X1 = pslyapdkr(a,q; adj = false)
# # ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# # rez = [ a[:,:,ip[i]]*X1[:,:,ip[i]]*a[:,:,ip[i]]'-X1[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip]
# # @test norm(rez) < 1.e-7  


# q = copy(qs);
# X = pdlyaps1!(KSCHUR, a, q; adj = false) 
# rez = [ a[:,:,ip[i]]*X[:,:,ipc[i]]*a[:,:,ip[i]]'-X[:,:,ipc1[i]]+qs[:,:,ipc[i]] for i in ipc]
# @test norm(rez) < 1.e-7  


end # pdlyap

@testset "pdlyaps1!" begin


p = 1; n = 1; 
a = rand(n,n,p); q = rand(n,n,p);
KSCHUR = 1
qs = copy(q)
X = pdlyaps1!(KSCHUR, a, q; adj = true) 
rez = a[:,:,1]'*X[:,:,1]*a[:,:,1]-X[:,:,1]+qs[:,:,1]   
@test norm(rez) < 1.e-7   
q = copy(qs)      
X = pdlyaps1!(KSCHUR, a, q; adj = false) 
rez = a[:,:,1]*X[:,:,1]*a[:,:,1]'-X[:,:,1]+qs[:,:,1]   
@test norm(rez) < 1.e-7  

X = copy(qs)
pdlyaps!(KSCHUR, a, X; adj = true) 
rez = a[:,:,1]'*X[:,:,1]*a[:,:,1]-X[:,:,1]+qs[:,:,1]   
@test norm(rez) < 1.e-7   
X = copy(qs)
pdlyaps!(KSCHUR, a, X; adj = false) 
rez = a[:,:,1]*X[:,:,1]*a[:,:,1]'-X[:,:,1]+qs[:,:,1]   
@test norm(rez) < 1.e-7  


p = 10; n = 1; 
a = rand(n,n,p); q = rand(n,n,p);
KSCHUR = 1
qs = copy(q)
X = pdlyaps1!(KSCHUR, a, q; adj = true) 
ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
rez = [ a[:,:,ip[i]]'*X[:,:,ip1[i]]*a[:,:,ip[i]]-X[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
@test norm(rez) < 1.e-7   
q = copy(qs)      
X = pdlyaps1!(KSCHUR, a, q; adj = false) 
rez = [ a[:,:,ip[i]]*X[:,:,ip[i]]*a[:,:,ip[i]]'-X[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip] 
@test norm(rez) < 1.e-7  

X1 = copy(qs)
pdlyaps!(KSCHUR, a, X1; adj = true) 
ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
rez = [ a[:,:,ip[i]]'*X1[:,:,ip1[i]]*a[:,:,ip[i]]-X1[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
@test norm(rez) < 1.e-7   
X1 = copy(qs)     
pdlyaps!(KSCHUR, a, X1; adj = false) 
rez = [ a[:,:,ip[i]]*X1[:,:,ip[i]]*a[:,:,ip[i]]'-X1[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip] 
@test norm(rez) < 1.e-7  


p = 10; n = 2; 
a = rand(n,n,p); q = rand(n,n,p);
a[:,:,1] = 0.01*[1 -2;2 1]; [a[:,:,i] = triu(a[:,:,i]) for i in 2:p]; 
[q[:,:,i] = q[:,:,i]'*q[:,:,i] for i in 1:p];
KSCHUR = 1
qs = copy(q)
X = pdlyaps1!(KSCHUR, a, q; adj = true) 
ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
rez = [ a[:,:,ip[i]]'*X[:,:,ip1[i]]*a[:,:,ip[i]]-X[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
@test norm(rez) < 1.e-7   
q = copy(qs)      
X = pdlyaps1!(KSCHUR, a, q; adj = false) 
rez = [ a[:,:,ip[i]]*X[:,:,ip[i]]*a[:,:,ip[i]]'-X[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip] 
@test norm(rez) < 1.e-7  

X1 = copy(qs)
pdlyaps!(KSCHUR, a, X1; adj = true) 
ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
rez = [ a[:,:,ip[i]]'*X1[:,:,ip1[i]]*a[:,:,ip[i]]-X1[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
@test norm(rez) < 1.e-7   
X1 = copy(qs)     
pdlyaps!(KSCHUR, a, X1; adj = false) 
rez = [ a[:,:,ip[i]]*X1[:,:,ip[i]]*a[:,:,ip[i]]'-X1[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip] 
@test norm(rez) < 1.e-7  


p = 2; n = 2; 
a = rand(n,n,p); q = rand(n,n,p); x = rand(n,n,p);
a[:,:,1] = 0.01*schur(rand(n,n)).T; [a[:,:,i] = triu(a[:,:,i]) for i in 2:p]; 
[q[:,:,i] = q[:,:,i]'*q[:,:,i] for i in 1:p];

# a[:,:,1] = [0.5 2;0 -0.5]; a[:,:,2] = [1 0;0 1.];
# x[:,:,1] =  [1 3;3 1.]; x[:,:,2] =  [2 1;1 2.]; 
# q[:,:,1] = -a[:,:,1]'*x[:,:,2]*a[:,:,1]+x[:,:,1];
# q[:,:,2] = -a[:,:,2]'*x[:,:,1]*a[:,:,2]+x[:,:,2];
# qs = copy(q);
# ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# rez = [ a[:,:,ip[i]]'*x[:,:,ip1[i]]*a[:,:,ip[i]]-x[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]

qs = copy(q);
X1 = pslyapdkr(a,q)
ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
rez1 = [ a[:,:,ip[i]]'*X1[:,:,ip1[i]]*a[:,:,ip[i]]-X1[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
@test norm(rez1) < 1.e-7 

KSCHUR = 1
q = copy(qs);
X = pdlyaps1!(KSCHUR, a, q; adj = true) 
ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
rez = [ a[:,:,ip[i]]'*X[:,:,ip1[i]]*a[:,:,ip[i]]-X[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
@test norm(rez) < 1.e-7   
q = copy(qs)      
X = pdlyaps1!(KSCHUR, a, q; adj = false) 
rez = [ a[:,:,ip[i]]*X[:,:,ip[i]]*a[:,:,ip[i]]'-X[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip] 
@test norm(rez) < 1.e-7  

X1 = copy(qs)
pdlyaps!(KSCHUR, a, X1; adj = true) 
ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
rez = [ a[:,:,ip[i]]'*X1[:,:,ip1[i]]*a[:,:,ip[i]]-X1[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
@test norm(rez) < 1.e-7   
X1 = copy(qs)     
pdlyaps!(KSCHUR, a, X1; adj = false) 
rez = [ a[:,:,ip[i]]*X1[:,:,ip[i]]*a[:,:,ip[i]]'-X1[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip] 
@test norm(rez) < 1.e-7  


p = 5; n = 5; 
a = rand(n,n,p); q = rand(n,n,p); x = rand(n,n,p);
a[:,:,1] = 0.01*schur(rand(n,n)).T; [a[:,:,i] = triu(a[:,:,i]) for i in 2:p]; 
[q[:,:,i] = q[:,:,i]'*q[:,:,i] for i in 1:p];

qs = copy(q);
X1 = pslyapdkr(a,q)
ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
rez = [ a[:,:,ip[i]]'*X1[:,:,ip1[i]]*a[:,:,ip[i]]-X1[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
@test norm(rez) < 1.e-7  

KSCHUR = 1
q = copy(qs);
X = pdlyaps1!(KSCHUR, a, q; adj = true) 
rez = [ a[:,:,ip[i]]'*X[:,:,ip1[i]]*a[:,:,ip[i]]-X[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
@test norm(rez) < 1.e-7 

X1 = copy(qs)
pdlyaps!(KSCHUR, a, X1; adj = true) 
ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
rez = [ a[:,:,ip[i]]'*X1[:,:,ip1[i]]*a[:,:,ip[i]]-X1[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
@test norm(rez) < 1.e-7  

q = copy(qs);
X1 = pslyapdkr(a,q; adj = false)
ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
rez = [ a[:,:,ip[i]]*X1[:,:,ip[i]]*a[:,:,ip[i]]'-X1[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip]
@test norm(rez) < 1.e-7  


q = copy(qs);
X = pdlyaps1!(KSCHUR, a, q; adj = false) 
rez = [ a[:,:,ip[i]]*X[:,:,ip[i]]*a[:,:,ip[i]]'-X[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip]
@test norm(rez) < 1.e-7  

X1 = copy(qs)     
pdlyaps!(KSCHUR, a, X1; adj = false) 
rez = [ a[:,:,ip[i]]*X1[:,:,ip[i]]*a[:,:,ip[i]]'-X1[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip] 
@test norm(rez) < 1.e-7  


p = 5; n = 5; pc = 10
a = rand(n,n,p); q = rand(n,n,pc); x = rand(n,n,pc);
a[:,:,1] = 0.01*schur(rand(n,n)).T; [a[:,:,i] = triu(a[:,:,i]) for i in 2:p]; 
[q[:,:,i] = q[:,:,i]'*q[:,:,i] for i in 1:pc];
qs = copy(q);

# q = copy(qs);
# X1 = pslyapdkr(a,q)
# ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# rez = [ a[:,:,ip[i]]'*X1[:,:,ip1[i]]*a[:,:,ip[i]]-X1[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7  

KSCHUR = 1
q = copy(qs);
X = pdlyaps1!(KSCHUR, a, q; adj = true) 
ip = mod.(0:pc-1,p).+1; ip1 = mod.(ip,p).+1;  
ipc = mod.(0:pc-1,pc).+1; ipc1 = mod.(ipc,pc).+1;  

rez = [ a[:,:,ip[i]]'*X[:,:,ipc1[i]]*a[:,:,ip[i]]-X[:,:,ipc[i]]+qs[:,:,ipc[i]] for i in ipc]
@test norm(rez) < 1.e-7 

X1 = copy(qs)
@time pdlyaps!(KSCHUR, a, X1; adj = true); 
rez = [ a[:,:,ip[i]]'*X1[:,:,ipc1[i]]*a[:,:,ip[i]]-X1[:,:,ipc[i]]+qs[:,:,ipc[i]] for i in ipc]
@test norm(rez) < 1.e-7  

# q = copy(qs);
# X1 = pslyapdkr(a,q; adj = false)
# ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# rez = [ a[:,:,ip[i]]*X1[:,:,ip[i]]*a[:,:,ip[i]]'-X1[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7  


q = copy(qs);
X = pdlyaps1!(KSCHUR, a, q; adj = false) 
rez = [ a[:,:,ip[i]]*X[:,:,ipc[i]]*a[:,:,ip[i]]'-X[:,:,ipc1[i]]+qs[:,:,ipc[i]] for i in ipc]
@test norm(rez) < 1.e-7  

X1 = copy(qs)     
pdlyaps!(KSCHUR, a, X1; adj = false) 
rez = [ a[:,:,ip[i]]*X1[:,:,ipc[i]]*a[:,:,ip[i]]'-X1[:,:,ipc1[i]]+qs[:,:,ipc[i]] for i in ip] 
@test norm(rez) < 1.e-7  

for adj in (false,true)
   @time pdlyaps!(KSCHUR, a, copy(qs); adj); 
   @time pdlyaps1!(KSCHUR, a, copy(qs); adj); 
   @time pdlyaps2!(KSCHUR, a, copy(qs); adj); 
   @time pdlyaps3!(KSCHUR, a, copy(qs); adj); 
end


# # benchmark
# using BenchmarkTools
# using MatrixEquations
# p = 5; n = 400; pc = 5
# a = 0.1*rand(n,n,p); q = rand(n,n,pc); x = rand(n,n,pc); [x[:,:,i] = x[:,:,i]'+x[:,:,i] for i in 1:pc];
# a[:,:,1] = 0.01*schur(rand(n,n)).T; [a[:,:,i] = triu(a[:,:,i]) for i in 2:p]; 
# ip = mod.(0:pc-1,p).+1; ip1 = mod.(ip,p).+1;  
# ipc = mod.(0:pc-1,pc).+1; ipc1 = mod.(ipc,pc).+1;  
# [q[:,:,i] = -a[:,:,ip[i]]'*x[:,:,ipc1[i]]*a[:,:,ip[i]]-x[:,:,ipc[i]] for i in ipc];
# [q[:,:,i] = 0.5*(q[:,:,i]'+q[:,:,i]) for i in 1:pc];
# qs = copy(q);

# KSCHUR = 1
# X = copy(qs);
# pdlyaps1!(KSCHUR, a, X; adj = true) 
# ip = mod.(0:pc-1,p).+1; ip1 = mod.(ip,p).+1;  
# ipc = mod.(0:pc-1,pc).+1; ipc1 = mod.(ipc,pc).+1;  
# rez = [ a[:,:,ip[i]]'*X[:,:,ipc1[i]]*a[:,:,ip[i]]-X[:,:,ipc[i]]+qs[:,:,ipc[i]] for i in ipc]
# @test norm(rez) < 1.e-5 

# X1 = copy(qs);
# pdlyaps!(KSCHUR, a, X1; adj = true) 
# rez = [ a[:,:,ip[i]]'*X1[:,:,ipc1[i]]*a[:,:,ip[i]]-X1[:,:,ipc[i]]+qs[:,:,ipc[i]] for i in ipc]
# @test norm(rez) < 1.e-5  

# X2 = copy(qs)
# pdlyaps2!(KSCHUR, a, X2; adj = true) 
# rez = [ a[:,:,ip[i]]'*X2[:,:,ipc1[i]]*a[:,:,ip[i]]-X2[:,:,ipc[i]]+qs[:,:,ipc[i]] for i in ipc]
# @test norm(rez) < 1.e-5  



# @btime pdlyaps!(KSCHUR, a, copy(qs); adj = true); 
# @btime pdlyaps1!(KSCHUR, a, copy(qs); adj = true); 
# @btime pdlyaps2!(KSCHUR, a, copy(qs); adj = true); 
# @btime pdlyaps3!(KSCHUR, a, copy(qs); adj = true); 

# @btime pdlyaps!(KSCHUR, a, copy(qs); adj = false); 
# @btime pdlyaps1!(KSCHUR, a, copy(qs); adj = false); 
# @btime pdlyaps2!(KSCHUR, a, copy(qs); adj = false); 
# @btime pdlyaps3!(KSCHUR, a, copy(qs); adj = false); 

p = 100; n = 4; pc = 100
a = 0.1*rand(n,n,p); q = rand(n,n,pc); x = rand(n,n,pc); [x[:,:,i] = x[:,:,i]'+x[:,:,i] for i in 1:pc];
a[:,:,1] = 0.01*schur(rand(n,n)).T; [a[:,:,i] = triu(a[:,:,i]) for i in 2:p]; 
ip = mod.(0:pc-1,p).+1; ip1 = mod.(ip,p).+1;  
ipc = mod.(0:pc-1,pc).+1; ipc1 = mod.(ipc,pc).+1;  
[q[:,:,i] = -a[:,:,ip[i]]'*x[:,:,ipc1[i]]*a[:,:,ip[i]]-x[:,:,ipc[i]] for i in ipc];
[q[:,:,i] = 0.5*(q[:,:,i]'+q[:,:,i]) for i in 1:pc];
qs = copy(q);



# q = copy(qs);
# A1 = copy(a[:,:,1]); X1 = copy(q[:,:,1]);
# lyapds!(A1, X1; adj = true)
# rez1 = a[:,:,1]'*X1*a[:,:,1]- X1 +qs[:,:,1] 
# @test norm(rez1) < 1.e-6 

# @btime lyapds!(A1, copy(X1); adj = true); 


end # pdlyaps1!

end # module