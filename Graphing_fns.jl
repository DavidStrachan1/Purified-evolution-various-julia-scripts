# -*- coding: utf-8 -*-
module Graphing_fns

export Exact_propagation
export error_prop
export current_plots_NESS
export current_plots
export NESS_plots_new
export sim_density
export a_b_plots
export correlation_heatmap
export correlation_contour
export correlation_surface_plot
export entanglement_animation
export impurity_density_animation
using Plots
using LinearAlgebra
include("module_PreB_fns_using_structs.jl")
using .PreB_fns_using_structs


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Error functions


function Exact_propagation(P,DP,BP)

    (;times1,times2) = DP
    (;H_single,Ci) = BP
    (;δt1,δt2) = P
    n1 = length(times1)
    n2 = length(times2)
    
    exact_corrs = Any[]
    U_step1= exp(-im*δt1*H_single)
    U_step2 = exp(-im*δt2*H_single)
    push!(exact_corrs,Ci)
    for i =1:n1
        corr_term = U_step1*last(exact_corrs)*U_step1'
        push!(exact_corrs,corr_term)
    end
    for i = (n1+1):n1+n2
        corr_term = U_step2*last(exact_corrs)*U_step2'
        push!(exact_corrs,corr_term)
    end
    return exact_corrs
end


function error_prop(corrs,P,DP,BP)
    
    (;times1,times2) = DP
    (;Ci) = BP
    exact_corrs = Exact_propagation(P,DP,BP)

    times = [times1;times2]
    sim_corrs =Any[]
    pushfirst!(times,0)
    for i =1:length(corrs)
        push!(sim_corrs,corrs[i])
    end
    n = length(sim_corrs)
    @assert(length(sim_corrs)==length(exact_corrs))
    
    fid_list = Vector{Float64}(undef,n)
    tr_list = Vector{Float64}(undef,n)
    
    for i=1:n
        sim_ρ = sim_corrs[i]/tr(sim_corrs[i])
        exact_ρ = exact_corrs[i]/tr(exact_corrs[i])
        fid_list[i]=real(fidelity(sim_ρ,exact_ρ))
        tr_list[i] = real(trace_dist(sim_ρ,exact_ρ))
    end
    
    fid_list = fid_list .-1
   
    plot(times,abs.(tr_list),label="Trace distance",xlabel="Time",ylabel="error")
    display(plot!(times,(abs.(fid_list)),label="fidelity - 1"))
      
    plot(times,log10.(abs.(tr_list)),label="log10 of trace distance",xlabel="Time",ylabel="error")
    display(plot!(times,log10.(abs.(fid_list)),label="log10 of fidelity - 1"))
    return exact_corrs
end 

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Current plotting functions

function current_plots_NESS(JL_list,JR_list,den_list,Jp,times,NESS_bool,P)
    (;D) = P
    if NESS_bool
        current_title = "NESS currents"
        density_title = "NESS density"
    else
        current_title = "Currents"
        density_title = "Density"
    end
    Jp = Jp*ones(length(times))
    fac = 100/D
    plot(times,real.(fac*JL_list),label="JL",xlabel="Time",ylabel="Particle current")
    plot!(times,real.(fac*JR_list),label="JR")
    @show(Jp)
    #plot(times,real.(Jp))
    display(plot!(times,real.(fac*Jp),label="LB result",title =current_title))
    display(plot(times,real.(den_list),label="Density",title = density_title))
end

function sim_density(JL_list,JR_list,den_list,P,DP)
    
    """
    Not 100% sure about whether I should add flow_in[i] or flow_in[i+1].
    """
    
    (;δt1,δt2) = P
    (;times1,times2) = DP
    times = [times1;times2]
    n = length(times)
    sim_den = complex(zeros(n))
    sim_den[1] = den_list[1]
    flow_in = JL_list-JR_list
    @show(n)
    for i=1:length(times1)-1
        sim_den[i+1] = sim_den[i] + flow_in[i+1]*δt1
    end
    for i=length(times1):(n-1)
        sim_den[i+1] = sim_den[i] + flow_in[i+1]*δt2
    end
    return sim_den,flow_in,times
end



    function current_plots(JL_list,JR_list,den_list,Jp,LB_den,P,DP)
    
    (;D) = P
    (;left_bath_bool,right_bath_bool) = DP
    fac = 100/D
    
    @assert(maximum(imag.(JL_list)) <1e-5)
    @assert(maximum(imag.(JR_list)) <1e-5)
    @assert(imag(Jp) <1e-5)
    

    sim_den,diff,times = sim_density(JL_list,JR_list,den_list,P,DP)
    pushfirst!(times,0)
    pushfirst!(sim_den,0.5)
    Jp = Jp*ones(length(times))
    LB_den = LB_den*ones(length(times))
    if left_bath_bool
        plot(times,real.(fac*JL_list),label="JL",xlabel="Time",ylabel="Particle current")
        if right_bath_bool
            plot!(times,real.(fac*JR_list),label="JR")
            plot!(times,real.(fac*diff),label="JL-JR")
        end
    else
        plot(times,real.(fac*JR_list),label="JR",xlabel="Time",ylabel="Particle current Jp/D (×100)")
    end
    display(plot!(times,real.(fac*Jp),label="LB result",ylabel="Particle current Jp/D (×100)",title="impurity energy="*string(DP.ϵi[1])))

    
    plot(times,real.(den_list),label="impurity density",xlabel="Time",ylabel="Density")
    plot!(times,real.(LB_den),label="LB result")
    display(plot!(times,real.(sim_den),label="predicted impurity density from currents",title="impurity energy="*string(DP.ϵi[1])))
    
    
    display(plot(times,real.(den_list-sim_den),label="diff",xlabel="Time",ylabel="difference in predicted vs simulated density",))

end

function NESS_plots_new(fid_l,tr_l,spec_l,t)
    y = length(t)
    
    display(plot(t[2:y],real.(fid_l),label="convergence in terms of fidelity-1,",xlabel="Time"))
    display(plot(t[2:y],real.(tr_l),label="convergence in terms of trace distance,",xlabel="Time"))
    
    display(plot(t,real.(spec_l),title="Real part of spectrum",label=false))
  #  savefig("real part of map's spectrum.png")
    display(plot(t,imag.(spec_l),title="Imaginary part of maps spectrum",label=false))
  #  savefig("imaginary part of map's spectrum.png")
    
    display(plot(t,abs.(spec_l),title="Absolute value of map's spectrum",label=false))
   # savefig("absolute value of map's spectrum.png")
end



function a_b_plots(Jl,Jr,P,DP)
    (;mu_L,mu_R,β_L,β_R) = P
    (;times1,times2) = DP
    t1 = deepcopy(times1)
    t2 = deepcopy(times2)
    times = [t1;t2]

    a_L,b_L = zeros(length(times)),zeros(length(times))
    a_R,b_R = zeros(length(times)),zeros(length(times))
    for i=1:length(times)
        a_L[i] = abs(a_b_fns(Jl,times[i],mu_L,β_L)[1])
        b_L[i] = abs(a_b_fns(Jl,times[i],mu_L,β_L)[2])
        a_R[i] = abs(a_b_fns(Jr,times[i],mu_R,β_R)[1])
        b_R[i] = abs(a_b_fns(Jr,times[i],mu_R,β_R)[2])
    end
    plot(times,(a_L)/abs(a_L[1]))
    display(plot!(times,(b_L)/abs(b_L[1])))

    plot(times,abs.(a_R)/abs(a_R[1]))
    display(plot!(times,abs.(b_R)/abs(b_R[1])))
end

# ##Animation functions


function correlation_heatmap(corr,site_lim,P,DP)
    (;T) = P
    (;nframe) = DP
    x=y=1:site_lim
    anim = @animate for i=1:nframe
        f(x, y) = abs((corr[i])[x,y])
        heatmap(x,y,f,size=(400,400),aspect_ratio=:equal,clims=(0, 1),legend=true,c=:Set1_6)
    end
    gif(anim,"correlation_heatmap_anim,T="*string(T)*".gif")
end

function correlation_contour(corr,site_lim,P,DP)
    (;T) = P
    (;nframe) = DP
    x=y=1:site_lim
    anim = @animate for i=1:nframe
        f(x, y) = abs((corr[i])[x,y])
        contour(x,y,f,size=(400,400),aspect_ratio=:equal,clims=(0, 1),legend=true)
    end
    gif(anim,"correlation_contour_anim,T="*string(T)*".gif")
end

function correlation_surface_plot(corr,site_lim,P,DP)
    (;T) = P
    (;nframe) = DP
    default(legend=false)
    x=y=1:site_lim
    anim = @animate for i=1:nframe
        f(x, y) = abs((corr[i])[x,y])
        surface(x, y, f,zlim=(0,1),xlabel="sites",ylabel="sites",
             title="animation of correlation matrix over time", c = :blues,clims=(0,1))
    end
    gif(anim,"correlation_surface_plot_anim,T="*string(T)*".gif")
end

function entanglement_animation(SvN,site_lim, SvN_ylim,P,DP)
    (;T) = P
    (;nframe,sites) = DP
    anim = @animate for i=1:nframe
        plot(sites[1:site_lim],SvN[i][1:site_lim],ylim=(0,SvN_ylim),
         xlabel="sites",ylabel="entanglement entropy")
    end
    gif(anim,"entanglement_entropy_anim,T="*string(T)*".gif")
end

function impurity_density_animation(corr,site_lim,P,DP)
    (;T) = P
    (;nframe) = DP
    x = 1:site_lim
    anim = @animate for i=1:nframe
        f(x) = abs((corr[i])[x,x])
        plot(x,f,xlabel="sites",ylabel="particle density")
    end
    gif(anim,"entanglement_entropy_anim,T="*string(T)*".gif")
end
end

