using ExactDiagonalization
using QuantumLattices
using LinearAlgebra
#=
unitcell = Lattice([0, 0]; vectors=[[1, 0]])
cluster = Lattice(unitcell, (2,), ('p',))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
bs= Sector(hilbert, ParticleNumber(2))
t = Hopping(:t, -1.0, 1)
U = Hubbard(:U, 4.0)
origiterms = (t, U)
referterms = (t, U)
opencluster = Lattice([cluster.coordinates[:, i] for i in 1:size(cluster.coordinates, 2)]...; name=:opencluster) 

ed = ED(opencluster, hilbert, referterms, (ParticleNumber(1), ParticleNumber(2), ParticleNumber(3), ParticleNumber(4)))
eig = eigen(ed)
eig.sectors
=#

#=
unitcell = Lattice([0, 0]; vectors=[[1, 0], [0, 1]])
cluster = Lattice(unitcell, (4,2), ('p','p'))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
t = Hopping(:t, -1.0, 1)
U = Hubbard(:U, 8.0)
μ = Onsite(:μ, -1.381)
origiterms = (t, U, μ)
referterms = (t, U, μ)
opencluster = Lattice([cluster.coordinates[:, i] for i in 1:size(cluster.coordinates, 2)]...; name=:opencluster) 
ed = ED(opencluster, hilbert, referterms, (ParticleNumber(4), ParticleNumber(5), ParticleNumber(6), ParticleNumber(7),  ParticleNumber(8), ParticleNumber(9)))
eig = eigen(ed; nev=7)
#(eig.values, eig.sectors)
eig.values
=#

cluster = Lattice([0, 0], [1/2, √3/2], [-1/2, √3/2]; vectors=[[3/2, √3/2],[0, √3]])

hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
t = Hopping(:t, -1.0, 1)
U = Hubbard(:U, 8.0)
μ = Onsite(:μ, -3.0)
origiterms = (t, U, μ)
referterms = (t, U, μ)
opencluster = Lattice([cluster.coordinates[:, i] for i in 1:size(cluster.coordinates, 2)]...; name=:opencluster) 
ed = ED(opencluster, hilbert, referterms, (ParticleNumber(1), ParticleNumber(2), ParticleNumber(3), ParticleNumber(4),  ParticleNumber(5), ParticleNumber(6)))
eig = eigen(ed; nev=5)
#(eig.values, eig.sectors)
eig.values