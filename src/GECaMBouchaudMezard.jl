module GECaMBouchaudMezard

using Random
using SparseArrays, LinearAlgebra
using StatsBase, CurveFit
using Graphs

export BM_MilSDE, sim_BM_MilSDE, idx_t, pareto_fit

include("types.jl")
include("utils.jl")

include("simulation.jl")

end