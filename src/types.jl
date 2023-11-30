struct SDEsol
    t::Vector{Float64}
    xs::Array{Float64, 2}
    dt::Float64
    par::Tuple{Matrix{Float64}, Float64}
end


function SDEsol(t::Vector{Float64}, N::Int, dt::Float64, par::Tuple{SparseMatrixCSC{Float64, Int64}, Float64})
    return SDEsol(t, zeros(N, length(t)), dt, par)
end


struct SDEsim
    t::Vector{Float64}
    xs::Array{Float64, 3}
    dt::Float64
    pars::Vector{Tuple{Matrix{Float64}, Float64}}
    nsim::Int
end


function SDEsim(t::Vector{Float64}, N::Int, dt::Float64, nsim::Int)
    return SDEsim(t, zeros(nsim, N, length(t)), dt, [(zeros(N, N), 0.0) for _ in 1:nsim], nsim)
end
