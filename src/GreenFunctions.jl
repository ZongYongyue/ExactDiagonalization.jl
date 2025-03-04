module GreenFunctions

using KrylovKit
using LinearAlgebra
using QuantumLattices: matrix, OperatorSum, Operators, CompositeIndex, Index, OperatorGenerator, Table, expand, FID
using ..EDCore: Sector, EDKind
using ..CanonicalFockSystems: BinaryBases, ⊗

export Block, Partition, BlockVals, GFSolver, EDSolver, FullEDSolver, FTLMSolver, BlockGreenFunction, ClusterGreenFunction

"""
    Block{N<:Integer, T<:AbstractVector, R<:AbstractVector, P<:AbstractVector, S<:Sector}

The information (nambu index, block indices, initial operators, project operators and sector) needed in culculating a part of cluster green function.
"""
struct Block{N<:Integer, T<:AbstractVector, R<:AbstractVector, P<:AbstractVector, S<:Sector}
    nambu::N
    block::T
    iops::R
    pops::P
    sector::S
end

"""
    Block(block::AbstractVector, iops::AbstractVector, pops::AbstractVector, bs::BinaryBases)

Construct Block with particle number/particle number and spin z component perserved bases or no particle number and no sin z component perserved bases.
"""
function Block(block::AbstractVector, iops::AbstractVector, pops::AbstractVector, bs::BinaryBases)
    (arr, _), bb, ps = block, bs.stategroups, [bs.quantumnumbers[i].N for i in eachindex(bs.quantumnumbers)]
    id, nambu = [findall(bit -> isone(bit), [isone(b, bit) for bit in 1:64]) for b in bb], [op.id[1].index.iid.nambu for op in iops[1]][1]
    isnothing(findfirst(x -> x == arr, id)) ? index=1 : index=findfirst(x -> x == arr, id)
    nambu == 1 ? ps[index] -= 1 : ps[index] += 1
    isnan(ps[1]) ? nbs=bs : nbs=reduce(⊗, [BinaryBases(id[i], convert(Int64, ps[i])) for i in eachindex(id)])
    return Block{typeof(nambu), typeof(block), typeof(iops), typeof(pops), typeof(nbs)}(nambu, block, iops, pops, nbs)
end

"""
    Block(spindwups::AbstractVector, block::AbstractVector, iops::AbstractVector, pops::AbstractVector, bs::BinaryBases)

Construct Block with only spin z component perserved bases
"""
function Block(spindwups::AbstractVector, block::AbstractVector, iops::AbstractVector, pops::AbstractVector, bs::BinaryBases)
    spin, sz, nambu = [op.id[1].index.iid.spin for op in iops[1]][1], bs.quantumnumbers[1].Sz, [op.id[1].index.iid.nambu for op in iops[1]][1]
    nambu == 1 ? sz -= spin : sz += spin
    nbs = BinaryBases(spindwups..., convert(Float64,sz))
    return Block{typeof(nambu), typeof(block), typeof(iops), typeof(pops), typeof(nbs)}(nambu, block, iops, pops, nbs)
end

"""
    Partition{S<:Symbol, B<:Block, L<:AbstractVector{B}, R<:Sector}
    Partition(::Val{:N}, table::Table, bs::BinaryBases)
    Partition(::Val{:S}, table::Table, bs::BinaryBases)
    Partition(::Val{:A}, table::Table, bs::BinaryBases)
    Partition(::Val{:F}, table::Table, bs::BinaryBases)

Divide cluster green function into different blocks according to the different terms contained in the Hamiltonian.
    The Hamiltonian of the system has:
    (1) only normal terms -> :N
    (2) spin-filp terms -> :S
    (3) anomalous terms -> :A
    (4) all the above -> :F
"""
struct Partition{S<:Symbol, B<:Block, L<:AbstractVector{B}, R<:Sector}
    symbol::S
    lesser::L
    greater::L
    sector::R
end
Partition(sym::Symbol, table::Table, bs::BinaryBases) = Partition(Val(sym), table, bs)
function Partition(::Val{:N}, table::Table, bs::BinaryBases)
    seqs = [(seq..., i) for seq in sort(collect(keys(table)), by = x -> table[x]), i in 1:2]
    id₁, id₂ = seqs[:,1], seqs[:,2]
    ops₁, ops₂ = [[Operators(1*CompositeIndex(Index(key[2], FID{:f}(key[3], key[1], key[4])), [0.0, 0.0], [0.0, 0.0])) for key in id] for id in [id₁, id₂]]
    arrs = [Vector(1:length(id₁))[(i-1)*(length(Vector(1:length(id₁)))÷(length(bs.stategroups)))+1:i*(length(Vector(1:length(id₁)))÷(length(bs.stategroups)))] for i in 1:length(bs.stategroups)]
    lesser, greater = [Block([arr, arr], ops₁[arr], ops₁[arr], bs) for arr in arrs], [Block([arr, arr], ops₂[arr],ops₂[arr], bs) for arr in arrs]
    return Partition(:N, lesser, greater, bs)
end
function Partition(::Val{:S}, table::Table, bs::BinaryBases)
    seqs = [(seq..., i) for seq in sort(collect(keys(table)), by = x -> table[x]), i in 1:2]
    id₁, id₂ = seqs[:,1], seqs[:,2]
    ops₁, ops₂ = [[Operators(1*CompositeIndex(Index(key[2], FID{:f}(key[3], key[1], key[4])), [0.0, 0.0], [0.0, 0.0])) for key in id] for id in [id₁, id₂]]
    arr = Vector(1:length(id₁))
    lesser, greater = [Block([arr,arr], ops₁[arr], ops₁[arr], bs)], [Block([arr,arr], ops₂[arr], ops₂[arr], bs)]
    return Partition(:S, lesser, greater, bs)
end
function Partition(::Val{:A}, table::Table, bs::BinaryBases)
    seqs = [(seq..., i) for seq in sort(collect(keys(table)), by = x -> table[x]), i in 1:2]
    id₁, id₂ = [seqs[:,1]...,seqs[:,2]...], [seqs[:,2]...,seqs[:,1]...]
    ops₁, ops₂ = [[Operators(1*CompositeIndex(Index(key[2], FID{:f}(key[3], key[1], key[4])), [0.0, 0.0], [0.0, 0.0])) for key in id] for id in [id₁, id₂]]
    ns = 2*abs(id₁[1][1])+1
    arrs = [Vector(1:length(id₁))[(i-1)*(length(Vector(1:length(id₁)))÷(2*(ns)))+1:i*(length(Vector(1:length(id₁)))÷(2*(ns)))] for i in 1:convert(Int, 2*(ns))]
    brrs = vcat(arrs[1:div(length(arrs), 2)], reverse(arrs[1:div(length(arrs), 2)]))
    lesser = [Block(arrs[1:div(length(arrs), 2)], [brrs[i], arrs[i]], ops₁[brrs[i]], ops₁[arrs[i]], bs) for i in eachindex(arrs)]
    greater = [Block(arrs[1:div(length(arrs), 2)], [arrs[i], brrs[i]], ops₂[brrs[i]], ops₂[arrs[i]], bs) for i in eachindex(arrs)]
    return Partition(:A, lesser, greater, bs)
end
function Partition(::Val{:F}, table::Table, bs::BinaryBases)
    seqs = [(seq..., i) for seq in sort(collect(keys(table)), by = x -> table[x]), i in 1:2]
    id₁, id₂ = [seqs[:,1]...,seqs[:,2]...], [seqs[:,2]...,seqs[:,1]...]
    ops₁, ops₂ = [[Operators(1*CompositeIndex(Index(key[2], FID{:f}(key[3], key[1], key[4])), [0.0, 0.0], [0.0, 0.0])) for key in id] for id in [id₁, id₂]]
    arr, brr = 1:(length(id₁)÷2), (length(id₁)÷2+1):length(id₁)
    lesser, greater = [Block([arr,arr], ops₁[arr], ops₁[arr], bs), Block([arr,brr], ops₁[arr], ops₁[brr], bs)], [Block([arr,arr], ops₂[arr], ops₂[arr], bs), Block([brr,arr], ops₂[arr], ops₂[brr], bs)]
    return Partition(:F, lesser, greater, bs)
end

"""
    genkrylov(matrix::AbstractMatrix, istate::AbstractVector, m::Int)

Generate a krylov subspace with Lanczos iteration method.
"""
function genkrylov(matrix::AbstractMatrix, istate::AbstractVector, m::Integer; reorth=false)
    orth = reorth ? KrylovKit.ModifiedGramSchmidt2() : KrylovKit.ModifiedGramSchmidt2()
    iterator = LanczosIterator(matrix, istate, orth)
    factorization = KrylovKit.initialize(iterator)
    for _ in 1:m-1
        KrylovKit.expand!(iterator, factorization)
    end
    basisvectors = basis(factorization)
    T = rayleighquotient(factorization)
    return basisvectors, T
end 

function diagkrylov(H::AbstractMatrix, starts::AbstractVector, m::Integer; kwags...)
    bases, tri = genkrylov(H, starts, m; kwags...)
    ebases = Vector(undef, m)
    tvals, tvecs = eigen(tri)
    for k in 1:m
            tv = tvecs[:,k]
            ebases[k] = zeros(eltype(tv), length(bases[1]))
        for l in 1:m
            ebases[k] .+= tv[l]*bases[l]
        end
    end
    return tvals, KrylovKit.OrthonormalBasis(ebases)
end

"""
    mutable struct BlockVals{I<:Int, D<:AbstractVector, R<:AbstractVector, N<:AbstractVector, P<:AbstractMatrix}

The Lanczos' information needed in calculating the cluster Green function.
"""
mutable struct BlockVals{I<:Int, D<:AbstractVector, R<:AbstractVector, N<:AbstractVector, P<:AbstractMatrix}
    nambu::I
    const block::D
    const abc::R
    const norms::N
    const projects::P
end
Base.copy(bv::BlockVals) = BlockVals(bv.nambu, bv.block, bv.abc, bv.norms, bv.projects)

"""
    BlockVals(block::Block, gs::AbstractVector, rops::OperatorSum, bs::BinaryBases, table::Table; m::Int=200)

Obtain the Lanczos' information needed in calculating the cluster Green function.
"""
function BlockVals(block::Block, gs::AbstractVector, rops::OperatorSum, bs::BinaryBases, table::Table; m::Int=200)
    H = matrix(rops, (block.sector, block.sector), table)
    istates, pstates = [(matrix(op, (block.sector, bs), table)*gs)[:,1] for op in block.iops], [(matrix(op, (block.sector, bs), table)*gs)[:,1] for op in block.pops]
    abc, norms, projects = kryvals(H, istates, pstates, m)
    return BlockVals{typeof(block.nambu), typeof(block.block), typeof(abc), typeof(norms), typeof(projects)}(block.nambu, block.block, abc, norms, projects)
end
function kryvals(H::AbstractMatrix, istates::AbstractVector, pstates::AbstractVector, m::Int=200)
    avs, bvs, cvs, norms, projects = Vector{Vector{Float64}}(undef,length(istates)), Vector{Vector{Float64}}(undef,length(istates)), Vector{Vector{Float64}}(undef,length(istates)), Vector{Float64}(undef,length(istates)), Matrix{Vector{ComplexF64}}(undef,length(istates),length(istates))
    for i in eachindex(istates)
        krybasis, T = genkrylov(H, istates[i], m)  
        avs[i], bvs[i], cvs[i] = [0.0; T.ev[1:end-1]], T.dv, [T.ev[1:end-1]; 0.0]
        norms[i] = norm(istates[i])
        for j in eachindex(pstates)
            projects[i, j] = KrylovKit.project!!(zeros(ComplexF64, m), krybasis, pstates[j])
        end
    end
    return ([avs, bvs, cvs], norms, projects)
end
abstract type GFSolver end
"""
    EDSolver{R<:Real, B<:BlockVals, S<:AbstractVector{B}, I<:Integer}

The exact diagonalization solver of a certain system.
"""
struct EDSolver{R<:Real, B<:BlockVals, S<:AbstractVector{B}} <: GFSolver
    gse::R
    lvals::S
    gvals::S
end
Base.length(eds::EDSolver) = maximum([maximum([maximum([maximum([maximum(arr) for arr in block]) for block in val.block]) for val in vals]) for vals in [eds.lvals, eds.gvals]])
"""
    EDSolver(::EDKind{:FED}, sym::Symbol, refergenerator::OperatorGenerator, bs::BinaryBases, table::Table; m::Int=200)

Construct the exact diagonalization solver of a certain system.
"""
function EDSolver(::EDKind{:FED}, parts::Partition, refergenerator::OperatorGenerator, bs::BinaryBases, table::Table; m::Int=200)
    rops = expand(refergenerator)
    Hₘ = matrix(rops, (bs, bs), table)
    vals, vecs, _  = KrylovKit.eigsolve(Hₘ, 1, :SR, Float64)
    gse, gs = real(vals[1]), vecs[1]
    lesser, greater = parts.lesser, parts.greater
    lvals, gvals = [BlockVals(bl, gs, rops, bs, table; m=m) for bl in lesser], [BlockVals(bg, gs, rops, bs, table; m=m) for bg in greater]
    return EDSolver(gse, lvals, gvals)
end

struct FullEDSolver{V<:Number, S<:Number, R<:AbstractMatrix, E<:AbstractVector{S}, F<:AbstractVector{E}, M<:AbstractVector{R}} <: GFSolver
    T::V
    egienvals::F
    A::M
    B::M
end
function FullEDSolver(T::Number, opr::OperatorGenerator, table::Table, bs::BinaryBases)
    ops = expand(opr)
    seqs = [(seq..., i) for seq in sort(collect(keys(table)), by = x -> table[x]), i in 1:2]
    id₁, id₂ = seqs[:,1], seqs[:,2]
    ops₁, ops₂ = [[Operators(1*CompositeIndex(Index(key[2], FID{:f}(key[3], key[1], key[4])), [0.0, 0.0], [0.0, 0.0])) for key in id] for id in [id₁, id₂]]
    bs₁ = BinaryBases(length(table), Int(bs.quantumnumbers[1][1])-1) 
    bs₂ = BinaryBases(length(table), Int(bs.quantumnumbers[1][1])+1)
    opm₁, opm₂ = [matrix(op, (bs₁, bs), table) for op in ops₁], [matrix(op, (bs₂, bs), table) for op in ops₂]
    Hₘ = matrix(ops, (bs, bs), table)
    H₁ = matrix(ops, (bs₁, bs₁), table)
    H₂ = matrix(ops, (bs₂, bs₂), table)
    vals, vecs = eigen(Matrix(Hₘ))
    vals₁, vecs₁ = eigen(Matrix(H₁))
    vals₂, vecs₂ = eigen(Matrix(H₂))
    A, B = [zeros(eltype(Hₘ), size(H₁, 1), size(Hₘ, 2)) for _ in 1:length(ops₁)], [zeros(eltype(Hₘ), size(H₂, 1), size(Hₘ, 2))  for _ in 1:length(ops₂)]
    for i in 1:length(id₁)
        for n in 1:size(Hₘ, 1)
            for m in 1:size(H₁, 1) 
                A[i][m, n] = dot(vecs₁[:,m], opm₁[i] * vecs[:, n]) 
            end
            for m in 1:size(H₂, 1)
                B[i][m, n] = dot(vecs₂[:,m], opm₂[i] * vecs[:, n]) 
            end
        end
    end
    return FullEDSolver(T, [vals, vals₁, vals₂], A, B)
end

struct FTLMSolver <: GFSolver
    T::Number
    evals₁::AbstractArray
    evals₂::AbstractArray
    evals₃::AbstractArray
    A₁::AbstractArray
    A₂::AbstractArray
end
function FTLMSolver(T::Number, opr::OperatorGenerator, table::Table, bs::BinaryBases; m::Integer=50, nr::Integer=5, mthreads::Union{Bool, Integer}=false)
    ops = expand(opr)
    seqs = [(seq..., i) for seq in sort(collect(keys(table)), by = x -> table[x]), i in 1:2]
    id₁, id₂ = seqs[:,1], seqs[:,2]
    ops₁, ops₂ = [[Operators(1*CompositeIndex(Index(key[2], FID{:f}(key[3], key[1], key[4])), [0.0, 0.0], [0.0, 0.0])) for key in id] for id in [id₁, id₂]]
    bs₁ = BinaryBases(length(table), Int(bs.quantumnumbers[1][1])-1) 
    bs₂ = BinaryBases(length(table), Int(bs.quantumnumbers[1][1])+1)
    opm₁, opm₂ = [matrix(op, (bs₁, bs), table) for op in ops₁], [matrix(op, (bs₂, bs), table) for op in ops₂]
    Hₘ = matrix(ops, (bs, bs), table)
    temrs = [map(x->x - 0.5, rand(Float64, size(Hₘ,1))) for _ in 1:nr]
    rs = [temr/norm(temr) for temr in temrs]
    evals₁ = Vector(undef, nr)
    bases₁ = Vector(undef, nr)
    for i in 1:nr
        evals₁[i], bases₁[i] = diagkrylov(Hₘ, rs[i], m; reorth=true) 
    end
    R = [KrylovKit.project!!(zeros(ComplexF64, m), bases₁[i], rs[i]) for i in 1:nr]
    H₁ = matrix(ops, (bs₁, bs₁), table)
    H₂ = matrix(ops, (bs₂, bs₂), table)
    A₁ = [[zeros(eltype(Hₘ), m, m) for _ in 1:length(id₁), _ in 1:length(id₁)] for _ in 1:nr]
    A₂ = [[zeros(eltype(Hₘ), m, m) for _ in 1:length(id₁), _ in 1:length(id₁)] for _ in 1:nr]
    B₁ = [Vector(undef, length(id₁)) for _ in 1:nr]
    B₂ = [Vector(undef, length(id₁)) for _ in 1:nr]
    evals₂ = [Vector(undef, length(id₁)) for _ in 1:nr]
    evals₃ = [Vector(undef, length(id₁)) for _ in 1:nr]
    bases₂ = [Vector(undef, length(id₁)) for _ in 1:nr]
    bases₃ = [Vector(undef, length(id₁)) for _ in 1:nr]
    if mthreads == false
        for r in 1:nr
            for b in 1:length(id₁)
                evals₂[r][b], bases₂[r][b] = diagkrylov(H₁, opm₁[b]*rs[r], m; reorth=true)
                evals₃[r][b], bases₃[r][b] = diagkrylov(H₂, opm₂[b]*rs[r], m; reorth=true)
                B₁[r][b] = KrylovKit.project!!(zeros(ComplexF64, m), bases₂[r][b], opm₁[b]*rs[r])
                B₂[r][b] = conj.(KrylovKit.project!!(zeros(ComplexF64, m), bases₃[r][b], opm₂[b]*rs[r]))
            end
        end
    elseif mthreads == true
        idx = Threads.Atomic{Int}(1)
        indices = CartesianIndices((nr, length(id₁)))
        n = length(indices)
        Threads.@sync for _ in 1:Threads.nthreads()
            Threads.@spawn while true
                i = Threads.atomic_add!(idx, 1)  
                i > n && break  
                r, b = indices[i].I
                evals₂[r][b], bases₂[r][b] = diagkrylov(H₁, opm₁[b]*rs[r], m; reorth=true)
                evals₃[r][b], bases₃[r][b] = diagkrylov(H₂, opm₂[b]*rs[r], m; reorth=true)
                B₁[r][b] = KrylovKit.project!!(zeros(ComplexF64, m), bases₂[r][b], opm₁[b]*rs[r])
                B₂[r][b] = conj.(KrylovKit.project!!(zeros(ComplexF64, m), bases₃[r][b], opm₂[b]*rs[r]))
            end
        end
    elseif isa(mthreads, Integer)
        idx = Threads.Atomic{Int}(1)
        indices = CartesianIndices((nr, length(id₁)))
        n = length(indices)
        Threads.@sync for _ in 1:mthreads
            Threads.@spawn while true
                i = Threads.atomic_add!(idx, 1)  
                i > n && break  
                r, b = indices[i].I
                evals₂[r][b], bases₂[r][b] = diagkrylov(H₁, opm₁[b]*rs[r], m; reorth=true)
                evals₃[r][b], bases₃[r][b] = diagkrylov(H₂, opm₂[b]*rs[r], m; reorth=true)
                B₁[r][b] = KrylovKit.project!!(zeros(ComplexF64, m), bases₂[r][b], opm₁[b]*rs[r])
                B₂[r][b] = conj.(KrylovKit.project!!(zeros(ComplexF64, m), bases₃[r][b], opm₂[b]*rs[r]))
            end
        end
    end
    if mthreads == false
        for r in 1:nr
            for a in 1:length(id₁)
                for b in 1:length(id₁)
                    for i in 1:m
                        for j in 1:m
                            A₁[r][a, b][i, j] = exp(-evals₁[r][i]/T)*B₁[r][b][j]*dot(bases₂[r][b][j],opm₁[a]*bases₁[r][i])*conj(R[r][i])
                            A₂[r][a, b][i, j] = exp(-evals₁[r][i]/T)*B₂[r][b][j]*dot(opm₂[a]*bases₁[r][i], bases₃[r][b][j])*R[r][i]
                        end
                    end
                end
            end
        end
    elseif mthreads == true
        idx = Threads.Atomic{Int}(1)
        indices = CartesianIndices((nr, length(id₁), length(id₁), m, m))
        n = length(indices)
        Threads.@sync for _ in 1:Threads.nthreads()
            Threads.@spawn while true
                k = Threads.atomic_add!(idx, 1)  
                k > n && break  
                r, a, b, i, j = indices[k].I
                A₁[r][a, b][i, j] = exp(-evals₁[r][i]/T)*B₁[r][b][j]*dot(bases₂[r][b][j],opm₁[a]*bases₁[r][i])*conj(R[r][i])
                A₂[r][a, b][i, j] = exp(-evals₁[r][i]/T)*B₂[r][b][j]*dot(opm₂[a]*bases₁[r][i], bases₃[r][b][j])*R[r][i]
            end
        end
    elseif isa(mthreads, Integer)
        idx = Threads.Atomic{Int}(1)
        indices = CartesianIndices((nr, length(id₁), length(id₁), m, m))
        n = length(indices)
        Threads.@sync for _ in 1:mthreads
            Threads.@spawn while true
                k = Threads.atomic_add!(idx, 1)  
                k > n && break  
                r, a, b, i, j = indices[k].I
                A₁[r][a, b][i, j] = exp(-evals₁[r][i]/T)*B₁[r][b][j]*dot(bases₂[r][b][j],opm₁[a]*bases₁[r][i])*conj(R[r][i])
                A₂[r][a, b][i, j] = exp(-evals₁[r][i]/T)*B₂[r][b][j]*dot(opm₂[a]*bases₁[r][i], bases₃[r][b][j])*R[r][i]
            end
        end
    end
    Z = 0.0
    for r in 1:nr
        for i in 1:m
            Z += exp(-evals₁[r][i]/T)*abs2(R[r][i])
        end
    end
    return FTLMSolver(T, evals₁, evals₂, evals₃, A₁/Z, A₂/Z)
end

"""
    BlockGreenFunction(gse::Real, blockvals::BlockVals, ω::Complex)

Calculate a block of cluster green function.
"""
function BlockGreenFunction(gse::Real, blockvals::BlockVals, ω::Complex)
    (avs, bvs, cvs),  norms, proj = blockvals.abc, blockvals.norms, blockvals.projects
    bgfm, d = zeros(ComplexF64, length(norms), length(norms)), [1.0;zeros(length(avs[1])-1)]
    blockvals.nambu==1 ? bv=[(ω - gse) .+ bvs[i] for i in 1:length(norms)] : bv=[(ω + gse) .- bvs[i] for i in 1:length(norms)]
    blockvals.nambu==1 ? tmpv=[thomas(avs[i],bv[i],cvs[i],d,length(avs[1])) for i in 1:length(norms)] : tmpv=[thomas(-avs[i],bv[i],-cvs[i],d,length(avs[1])) for i in 1:length(norms)]
    for i in eachindex(norms), j in eachindex(norms)
            bgfm[i, j] = dot(proj[i, j], tmpv[i])*norms[i]
    end
    return bgfm
end
function thomas(a::AbstractVector, b::AbstractVector, c::AbstractVector, d::AbstractVector, n::Int)
    x = Complex.(d)
    cp = Complex.(c)
    cp[1] /= b[1]
    x[1] /= b[1]
    for i = 2:n
        scale = 1.0 / (b[i] - cp[i-1]*a[i])
        cp[i] *= scale
        x[i] = (x[i] - a[i]*x[i-1])*scale
    end
    for i = n-1:-1:1
        x[i] -= (cp[i]*x[i+1])
    end
    return x
end

"""
    ClusterGreenFunction(normal::Bool, kind::Symbol, solver::EDSolver, ω::Complex)
    ClusterNormalGreenFunction(kind::Symbol, solver::EDSolver, ω::Complex)
    ClusterGorkovGreenFunction(kind::Symbol, solver::EDSolver, ω::Complex)

Calculate the cluster green function with certain frequence ω
"""
function ClusterGreenFunction(normal::Bool, kind::Symbol, solver::EDSolver, ω::Complex)
    (normal||length(solver.lvals)==1) ? cgf=ClusterNormalGreenFunction(kind::Symbol, solver::EDSolver, ω::Complex) : cgf=ClusterGorkovGreenFunction(kind::Symbol, solver::EDSolver, ω::Complex)
    return cgf
end
function ClusterNormalGreenFunction(kind::Symbol, solver::EDSolver, ω::Complex)
    lens = length(solver)
    clm, cgm = zeros(ComplexF64, lens, lens), zeros(ComplexF64, lens, lens)
    for lval in solver.lvals
        clm[lval.block...] = BlockGreenFunction(solver.gse, lval, ω)
    end
    for gval in solver.gvals
        cgm[gval.block...] = transpose(BlockGreenFunction(solver.gse, gval, ω))
    end
    kind==:f ? cgf=clm+cgm : (kind==:b ? cgf=cgm-clm : cgf=zeros(ComplexF64, lens, lens))
    return cgf
end
function ClusterGorkovGreenFunction(kind::Symbol, solver::EDSolver, ω::Complex)
    lens = length(solver)
    clm, cgm = zeros(ComplexF64, lens, lens), zeros(ComplexF64, lens, lens)
    for lval in solver.lvals
        clm[lval.block...] = BlockGreenFunction(solver.gse, lval, ω)
        glval = copy(lval)
        glval.nambu = 2
        if lval.block[1] == lval.block[2]
            arr = lval.block[1] .+ (lens ÷ 2) 
            cgm[arr, arr] = transpose(BlockGreenFunction(solver.gse, glval, ω))
        else
            cgm[lval.block...] = BlockGreenFunction(solver.gse, glval, ω)
        end
    end
    for gval in solver.gvals
        cgm[gval.block...] = transpose(BlockGreenFunction(solver.gse, gval, ω))
        lgval = copy(gval)
        lgval.nambu = 1
        if gval.block[1] == gval.block[2]
            arr = gval.block[1] .+ (lens ÷ 2)
            clm[arr, arr] = BlockGreenFunction(solver.gse, lgval, ω)
        else
            clm[gval.block...] = transpose(BlockGreenFunction(solver.gse, lgval, ω))
        end
    end
    cgm[Vector(1:(lens÷2)), Vector(((lens÷2)+1):lens)] = transpose(cgm[Vector(1:(lens÷2)), Vector(((lens÷2)+1):lens)])
    clm[Vector(((lens÷2)+1):lens), Vector(1:(lens÷2))] = transpose(clm[Vector(((lens÷2)+1):lens), Vector(1:(lens÷2))])
    kind==:f ? cgf=clm+cgm : (kind==:b ? cgf=cgm-clm : cgf=zeros(ComplexF64, lens, lens))
    return cgf
end

function ClusterGreenFunction(normal::Bool, kind::Symbol, solver::FullEDSolver, ω::Complex)
    Z = 0.0
    for val in solver.egienvals[1]
        Z += exp(-val/solver.T)
    end
    lgm = zeros(ComplexF64, length(solver.A), length(solver.A))
    ggm = zeros(ComplexF64, length(solver.B), length(solver.B))
    for i in 1:length(solver.A)
        for j in 1:length(solver.B)
            for n in 1:length(solver.egienvals[1])
                for m in 1:length(solver.egienvals[2])
                    lgm[i, j] += exp(-solver.egienvals[1][n]/solver.T)*dot(conj(solver.A[j][m, n]), solver.A[i][m, n])/(ω - solver.egienvals[1][n] + solver.egienvals[2][m])
                end
                for m in 1:length(solver.egienvals[2])
                    ggm[i, j] += exp(-solver.egienvals[1][n]/solver.T)*dot(conj(solver.B[i][m, n]), solver.B[j][m, n])/(ω - solver.egienvals[3][m] + solver.egienvals[1][n])
                end
            end
        end
    end
    kind==:f ? cgf=ggm+lgm : (kind==:b ? cgf=ggm-lgm : cgf=zeros(ComplexF64, length(solver.A), length(solver.B)))
    return cgf/Z
end

function ClusterGreenFunction(normal::Bool, kind::Symbol, solver::FTLMSolver, ω::Complex)
    lgm = zeros(ComplexF64, length(solver.evals₂[1]), length(solver.evals₂[1]))
    ggm = zeros(ComplexF64, length(solver.evals₃[1]), length(solver.evals₃[1]))
    for r in 1:length(solver.evals₁)
        for a in 1:length(solver.evals₂[1])
            for b in 1:length(solver.evals₃[1])
                for i in 1:length(solver.evals₂[1][1])
                    for j in 1:length(solver.evals₃[1][1])
                        lgm[a, b] += solver.A₁[r][a, b][i, j]/(ω - solver.evals₁[r][i] + solver.evals₂[r][b][j])
                        ggm[a, b] += solver.A₂[r][a, b][i, j]/(ω - solver.evals₃[r][b][j] + solver.evals₁[r][i])
                    end
                end
            end
        end
    end
    kind==:f ? cgf=ggm+lgm : (kind==:b ? cgf=ggm-lgm : cgf=zeros(ComplexF64, length(solver.evals₂[1]), length(solver.evals₂[1])))
    return cgf
end



end # module
