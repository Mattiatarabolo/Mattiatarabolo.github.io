function Wiener_diag!(ΔW::Array{Float64}, Δt::Float64)
    ΔW .= randn(size(ΔW)[1]).*sqrt(Δt)
end


function BM_MilSDE(p::Tuple{SparseMatrixCSC{Float64, Int64}, Float64}, dt::Float64, x_init::Vector{Float64}, t_init::Float64, t_end::Float64)
    N = length(x_init)
    
    ts = range(t_init, t_end, step=dt)
    T = length(ts)

    f!(dx, x, p) = mul!(dx, p[1], x)
    g!(dx, x, p) = mul!(dx, sqrt(p[2]), x)
    g_Mil!(dx, x, p) = mul!(dx, p[2]/2, x)

    sol = SDEsol(Vector(ts), N, dt, p)

    sol.xs[:, 1] .= x_init

    x = x_init
    Δx_det = zeros(N)
    Δx_stoch = zeros(N)
    Δx_Mil = zeros(N)
    ΔW = zeros(N)

    # integration loop
    for τ in 2:T
        f!(Δx_det, x, p)
        g!(Δx_stoch, x, p)
        g_Mil!(Δx_Mil, x, p)
        Wiener_diag!(ΔW, dt)

        # Milstein update
        x .= x + Δx_det.*dt + Δx_stoch.*ΔW + Δx_Mil.*(ΔW.^2 .- dt)
        sol.xs[:, τ] .= x/mean(x)
    end
    return sol
end



function sim_BM_MilSDE(dt::Float64, x_init::Vector{Float64}, t_init::Float64, t_end::Float64, K::Int, σ²::Float64, J::Float64, seed::Int, nsim::Int)
    N = length(x_init)
    
    ts = range(t_init, t_end, step=dt)
    T = length(ts)

    sim = SDEsim(Vector(ts), N, dt, nsim)

    Threads.@threads for idx_sim in 1:nsim
        G = random_regular_graph(NV, K)
        Amod = adjacency_matrix(G)
        for i in 1:NV
            Amod[i,i] = -Float64(K)
        end

        p = (J .* Amod, σ²)
        sim.pars[idx_sim] = p

        Random.seed!(seed)
        sol_mil = BM_MilSDE(p, dt, x_init, t_init, t_end)

        sim.xs[idx_sim, :, :] .= sol_mil.xs
    end

    return sim
end


function sim_BM_MilSDE(p::Tuple{SparseMatrixCSC{Float64, Int64}, Float64}, dt::Float64, x_init::Vector{Float64}, t_init::Float64, t_end::Float64, seed::Int, nsim::Int)
    N = length(x_init)
    
    ts = range(t_init, t_end, step=dt)
    T = length(ts)

    sim = SDEsim(Vector(ts), N, dt, nsim)

    Threads.@threads for idx_sim in 1:nsim

        Random.seed!(seed + idx_sim)
        sol_mil = BM_MilSDE(p, dt, x_init, t_init, t_end)

        sim.xs[idx_sim, :, :] .= sol_mil.xs
    end

    return sim
end 