module PreB_fns_using_structs
    using DSP
    using ITensorTDVP
    using ITensors
    using LinearAlgebra
    using Observers
    using Plots
    using ProgressBars
    using Trapz
    using Distributed 
    using SharedArrays
    using PolyChaos
    using Kronecker
    using SparseArrays
    using Base.Threads
    export dependent_params
    export Base_params
    export Bath_params
    
    export heaviside
    export J
    export approx_J
    export LB_current
    export sim_currents
    export impurity_current_operator
    export two_site_current_operator
    export three_site_current_operator
    
    export fidelity
    export trace_dist
    export Id_check
    export map_check
    export map_check2
    export ρ_test
    export boundary_test

    export system_swaps
    export ancilla_phase_gate_swap
    export ancilla_phase_gate_PH
    export particle_hole_transform
    export NESS_fn
    export NESS_calculations
    export vectorise_ρ
    export unvectorise_ρ
    export ρ_system_corr
    export mp
    export fermionic_swap_gate 

    export direct_mapping
    export reaction_mapping
    export orthopol_chain
    export band_diag
    
    export reflection_diag
    export U_thermo
    export U_chain
    export chain_to_star
    export unprime_string
    export unprime_ind
    export JW_string
    export Id_string
    export entanglement_entropy
    
    export diff_sys_init
    export initialise_bath_gates_therm
    export initialise_bath_gates_eigen
    export initialise_system_gates
    export initialise_psi
    export initialise_bath
    export H_S
    export H_bath

    export rdm
    export rdm_para
    export rdm_para_new
    export enrich_generic3
    export bipart_maxdim
    export Krylov_states
    export Krylov_linkdims
    export linkdims
    
    export measure_mem
    export current_time
    export measure_den
    export measure_SvN
    export measure_correlation_matrix

    export HS_full
    export lmult
    export rmult
    export spin_operators
    export build_liouvillian
    export two_site_diss_rates
    export Liouvillian_solver_exact
#--------------------------------------------------------------------------------------------------------------------------------------------------
"""
MAIN MODULE STRUCTURE:
-Struct definitions ~80 to 160
-Currents and spectral density ~160 to 220
-Test functions ~ 370 to 450
-NESS calculations ~450 to 770
-Discretization and mappings ~ 770 to 910
-Various useful functions ~910 to 1050
-Initialisation functions ~ 1050 to 1470
-Enrichment functions ~ 1470 to 1860
-Measurement functions ~1860 to 1970
-Master equation functions~1970 to end
"""
    #--------------------------------------------------------------------------------------------------------------------------
    """
    Parameter structures
    """
    Base.@kwdef mutable struct Base_params
        
        Nbl ::Int64                                    #Number of left bath sites
        Nbr ::Int64                                   #Number of right bath sites
        Ns  ::Int64                                    #Number of system sites (not including ancilla)  
        Kr_cutoff ::Float64                       
        β_R ::Float64                                      #inverse temperature of right bath
        β_L ::Float64                                       #inverse temperature of left bath
        mu_L ::Float64 
        mu_R ::Float64 
        k1 ::Int64                                      # Number of Krylov states
        τ_Krylov ::Float64 
        Γ_L ::Float64 
        Γ_R ::Float64
        eta ::Float64 
        tdvp_cutoff ::Float64 

        ##not used in this module, used in top scope and graphing module

        δt1 ::Float64                                   #Time step for simulation with enrichment
        δt2 ::Float64                               #Time step for simulation after enrichment
        n1 ::Int64                                      #Number of time steps between each enrichment
        n2 ::Int64                                      #Number of time steps between each extraction of ρΛ
        T ::Float64                                      # Total time
        T_enrich ::Float64      
        method ::Int64
        spec_fun_type ::String 
        disc_choice ::String   
        D ::Float64             
    end


    Base.@kwdef struct dependent_params
        
        s ::Vector{Index{Int64}} 
        N ::Int64 
        sites ::UnitRange{Int64}                             # site list
        ϵi ::Vector{Float64}                            #self energies of system modes
        ti ::Vector{Float64}                            #coupling of system modes
        left_bath_bool ::Bool
        right_bath_bool ::Bool


        c ::Vector{ITensor}                                 # annihilation operators
        cdag ::Vector{ITensor}                              # creation operators
        F ::Vector{ITensor}                                 # Jordan-Wigner string operator
        Imat ::Matrix{Float64}                              # identity matrix
        Id ::Vector{ITensor}

        q ::UnitRange{Int64}
        qS ::StepRange{Int64, Int64}
        qA ::StepRange{Int64, Int64}
        order_bool ::Bool

        #not used in this module
        T_unenriched ::Float64
        nframe_en ::Int64
        nframe_un ::Int64
        nframe ::Int64
        times1 ::Vector{Float64}
        times2 ::Vector{Float64}
    end

    mutable struct Bath_params
        Vk_emp_L :: Vector{ComplexF64}
        ϵb_emp_L :: Vector{ComplexF64}
        Vk_fill_L :: Vector{ComplexF64}
        ϵb_fill_L :: Vector{ComplexF64} 
        fk_L :: Vector{ComplexF64} 
        Vk_emp_R :: Vector{ComplexF64}
        ϵb_emp_R :: Vector{ComplexF64}
        Vk_fill_R :: Vector{ComplexF64}
        ϵb_fill_R :: Vector{ComplexF64} 
        fk_R :: Vector{ComplexF64}
        H_single ::Matrix{Float64}
        H ::MPO
        Ci ::Transpose{ComplexF64, Matrix{ComplexF64}}
        Bath_params() = new() 
    end

    heaviside(t) = 0.5 * (sign.(t) .+ 1) 
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Current calculations and spectral functions.
"""
    function J(w,spec_therm,side,P)
        
        ###Choosing which bath
        if side =="left"
            spec_fun_type,Γ,β,μ,D= P.spec_fun_type,P.Γ_L,P.β_L,P.mu_L,P.D
        elseif side == "right"
            spec_fun_type,Γ,β,μ,D = P.spec_fun_type,P.Γ_R,P.β_R,P.mu_R,P.D
        else 
            error("neither side chosen")
        end
        fk = 1 ./(1 .+exp.(β*(w .- μ)))
        
        ##Choosing if we incooperate the thermofield transform directly onto the 
        ##spectral density
        
        renorm =1
        
        if spec_therm =="filled"
            renorm = fk
        elseif spec_therm == "empty"
            renorm = 1 .-fk
        elseif spec_therm =="full"
            renorm = 1
        else
            error("No spectral density chosen")
        end
        
        ##Choosing which spectral function to use
        if spec_fun_type == "box"
            ρ = (1/(2*D))*(heaviside(w .+ D) .- heaviside(w .- D)).*renorm
            J =  (Γ*D/π)*ρ
        elseif spec_fun_type =="ellipse"
            ρ = real((2/(π*D))*sqrt.(Complex.(1 .-(w/D).^2)).*renorm)
            J =  (Γ*D/π)*ρ
        elseif spec_fun_type =="Ohmic"
            S = parse(Float64,readline("Ohmic power S = "))

        else
            error("No spectral density chosen")
        end
        return J
    end
    


    function approx_J(J,ϵ_b,V_k,Γ)
        N = length(ϵ_b)
        η = 1/N
        samp = 10000; # Frequency sampling.
        w = range(-1.5,1.5,samp); # Frequency axis for Landauer calculations (slightly larger than the band).
        J_approx = zeros(samp)
        for i=1:N
            denom = (w .-ϵ_b[i]).^2 .+(η)^2
            L = η./(π*denom)
            delta = (V_k[i]^2)*L
           # display(plot(w,delta)) 
            J_approx += delta
        end
        J_exact = J(w,Γ)
        plot(w,J_exact)
        display(plot!(w,J_approx))
    end


    
    function LB_current(P,DP)
        (;Γ_L,Γ_R,mu_L,mu_R,β_L,β_R,Ns,spec_fun_type,D,eta) = P
        (;ϵi,ti) = DP

    
        wsamp = 10000; # Frequency sampling.
        w = range(-10*D,10*D,wsamp); # Frequency axis for Landauer calculations (slightly larger than the band).
        Γ = Γ_L+Γ_R
        dw = w[2] - w[1]; # Frequency increment.
        if spec_fun_type == "box"
            ρ = (1/(2*D))*(heaviside(w .+ D) .- heaviside(w .- D))
            Δ_L = (-Γ_L*D/(2*pi*D))*log.((w.-D .+im*eta)./(w .+D .+im*eta))
            Δ_R = (-Γ_R*D/(2*pi*D))*log.((w.-D .+im*eta)./(w .+D .+im*eta))
            #Λ = -im*D*Γ*hilbert(ρ)

        elseif spec_fun_type =="ellipse"
            ρ = (2/(π*D))*real(sqrt.(Complex.(1 .-(w/D).^2)))
            Δ_L = -im*D*Γ_L*hilbert(ρ)
            Δ_R = -im*D*Γ_R*hilbert(ρ)
        end

        f_L = 1 ./ (1 .+exp.((w .- mu_L)*β_L)); 
        f_R = 1 ./ (1 .+exp.((w .- mu_R)*β_R));
        
        ####Calculate determinant of M matrix (Ns x Ns)
        if Ns == 1
            Δ = Δ_L + Δ_R
            detM = w .- ϵi[1] .- Δ
            G = 1 ./detM
            A_den = (-1/π).*imag.(G)    
        elseif Ns == 2
            detM = (w .-ϵi[2] .-Δ_R).*(w.-ϵi[2].-Δ_R)
            detM = detM .- abs.(ti[1]^2) 
            G = ((w .-ϵi[2] .-Δ_R))./detM
            A_den = (-1/π).*imag.(G)    
        elseif Ns == 3
            detM = (w .-ϵi[1] .-Δ_L).*(w.-ϵi[3].-Δ_R).*(w.-ϵi[2])
            detM = detM - (abs.(ti[2])^2)*(w.-ϵi[1].-Δ_L)
            detM = detM - (abs.(ti[1])^2)*(w.-ϵi[3].-Δ_R)
            G =  (w .-ϵi[1] .-Δ_L).*(w.-ϵi[3].-Δ_R)./detM
            A_den = (-1/π)*imag.(G)
        end

        A = (D*Γ*ρ/π) ./(abs.(detM).^2)
        if length(ti)>0
            coupling_factor = prod(abs.(ti).^2)
            A = A.*coupling_factor
        end    
        f_L = 1 ./ (1 .+exp.((w .- mu_L)*β_L)); 
        f_R = 1 ./ (1 .+exp.((w .- mu_R)*β_R));

        prefactor = (4*π*D*Γ_L*D*Γ_R)/(2*π*D*Γ)
        # Compute the particle-current, energy-current and density:
        Jp = prefactor*sum(ρ.*A.*(f_L - f_R))*dw;
        Je = prefactor*sum(ρ.*w.*A.*(f_L - f_R))*dw;
        n = 0.5*sum(A_den.*(f_L + f_R))*dw; 
        return [Jp,Je,n] 
    end


    function sim_currents(corr,DP,P,BP)
        (;Ns) = P
        JL_list = []
        JR_list = []
        den_list = []
        
        for i=1:length(corr)
            anc_bool = true
            NESS_bool =false
            JL,JR,den = 0,0,0
            if Ns == 1 
                JL,JR,den = impurity_current_operator(corr[i],DP,P,BP)
            elseif Ns ==2
                JL,JR,den = two_site_current_operator(corr[i],anc_bool,DP,P,BP)
            elseif Ns == 3
                JL,JR,den = three_site_current_operator(corr[i],NESS_bool,P,DP)    
            end
            push!(JL_list,JL)
            push!(JR_list,JR)
            push!(den_list,den)
        end
        Jp,_,n = LB_current(P,DP)
        return JL_list,JR_list,den_list,Jp,n
    end

    function impurity_current_operator(corr,DP,P,BP)
        """
        corr is the N x N correlation matrix for the modes, it is already transposed in the measure correlation
        matrix function. H_single has the opposite orientation. Taking right as the positive direction.
        Note that t[1] and t[2] are defined in an opposite way to t[3] and t[4] in terms of hopping direction. This is 
        due to the specific derivation used in the daily notes, where the system mode is defined as the end of four chains. 
        """
        (;Nbl) = P
        (;left_bath_bool,right_bath_bool) = DP
        (;H_single) = BP
        ind = 2*Nbl+1
        t = zeros(4)
        JL,JR = 0,0

        if left_bath_bool
            t[1],t[2] = H_single[ind-2,ind],H_single[ind-1,ind]
            JB_L = t[1]*corr[ind,ind-2] - conj(t[1])*corr[ind-2,ind]
            JA_L = t[2]*corr[ind,ind-1] - conj(t[2])*corr[ind-1,ind]
            JL = im*(JB_L + JA_L)
        end
        if right_bath_bool
            t[3],t[4] = H_single[ind+2,ind],H_single[ind+3,ind]
            JB_R = t[3]*corr[ind,ind+2] - conj(t[3])*corr[ind+2,ind]
            JA_R = t[4]*corr[ind,ind+3] - conj(t[4])*corr[ind+3,ind]
            JR = -im*(JB_R + JA_R)
        end

        n = corr[ind,ind]
        return[JL,JR,n]
    end

    function two_site_current_operator(corr,anc_bool,DP,P,BP)
    """
    Currently not set up for NESS calculations.
    """
    (;H_single) = BP
    (;Nbl) = P
    (;ti) = DP
    if anc_bool
        step = 2
    else
        step = 1 
    end
    ind = 2*Nbl+1
    t1,t2 = H_single[ind-2,ind],H_single[ind-1,ind]
    JB_L = t1*corr[ind,ind-2] - conj(t1)*corr[ind-2,ind]
    JA_L = t2*corr[ind,ind-1] - conj(t2)*corr[ind-1,ind]
    JL = im*(JB_L + JA_L)
    
    JR = im*(ti[1]*corr[ind+step,ind] - conj(ti[1])*corr[ind,ind+step])
    n = corr[ind,ind]
    return [JL,JR,n]
    end

    function three_site_current_operator(corr,NESS_bool,P,DP)
        (;Nbl) = P
        (;ti,left_bath_bool,right_bath_bool) = DP

        ##This is the index of the 2nd system site
        if NESS_bool
            ind = 2
            step =1
        else
            ind = 2*Nbl+3
            step = 2
        end
        JL,JR = 0,0
        if left_bath_bool
            JL = im*(ti[1]*corr[ind,ind-step] - conj(ti[1])*corr[ind-step,ind])
        end
        if right_bath_bool
            JR = im*(ti[2]*corr[ind+step,ind] - conj(ti[2])*corr[ind,ind+step])
        end
        n = corr[ind,ind]
        return [JL,JR,n]
    end
   
    #----------------------------------------------------------------------------------------------------------------------------------
    """
    Test functions
    """
    
    function fidelity(ρ,σ)
        ##make sure ρ and σ are in matrix form
        matrix = sqrt(ρ)*σ*sqrt(ρ)
        matrix = sqrt(matrix)
        fid = tr(matrix)*conj(tr(matrix))
        return fid
    end
    
    function trace_dist(ρ,σ)
        matrix = (ρ-σ)*(ρ-σ)'
        matrix = sqrt(matrix)
        return 0.5*tr(matrix)
    end

    function Id_check(Λmat)
        n = size(Λmat)[1]
        Id = zeros(n,n)
        for i=1:n
            Id[i,i] = 1
        end
        return opnorm(Λmat-Id)
    end

    function map_check(ψf,ψi,Λmat,P,DP)
        ρf = vectorise_ρ(ψf,P,DP)
        ρi = vectorise_ρ(ψi,P,DP)
        return norm(ρf - Λmat*ρi)
    end
    function map_check2(ρ_list,Λ_list,ρi)
        n = length(ρ_list)
        diff = zeros(n)
        for i=1:n
            diff[i] = norm(ρ_list[i] - Λ_list[i]*ρi)
        end
        return diff
    end

    function ρ_test(ρ,cutoff)

        x = eigen(ρ).values
        bool = false
        message0 = "Valid density matrix up to"*string(cutoff)
        message = "Not a density matrix:"
        if minimum(real.(x))<-cutoff
            bool = true
            message *= "minimum of spectrum<"*string(cutoff)*","
        end
        if maximum(real.(x))-1>cutoff
            message *= "maximum of spectrum>1,"
            bool = true
        end
        if maximum(imag.(x))>cutoff
            message *= "max(imag(spectrum))>"*string(cutoff)*","
            bool = true
        end
        if  abs(1-sum(x))>cutoff
             message *= "1-tr(ρ) >"*string(cutoff)*","
            bool = true
        end
        if norm(ρ-ρ')>cutoff
             message *= "ρ-dag(ρ)>"*string(cutoff)*","
            bool = true
        end
        if bool
            return message,bool
        else
            return message0,bool
        end
    end

    function boundary_test(ψ,N,num_init)
        left_boundary_test = num_init[1] - expect(ψ,"n")[1]
        right_boundary_test = num_init[N] - expect(ψ,"n")[N]
        left_bool = left_boundary_test>1e-5
        right_bool = right_boundary_test>1e-5
        return left_bool,right_bool
    end

    #--------------------------------------------------------------------------------------------------------------------------
    """
    NESS calculation and associated functions.
    """
    function system_swaps(ψ,start,DP,P)
        (;Ns) = P
        (;s) = DP
        """
        This function takes a state where the system and ancilla modes are interleaved
        (with the first system mode at the start site) and swaps the ordering such that 
        indices start:start+Ns-1 are all the system modes and start+Ns:start+2Ns-1 are
        the ancilla modes.
        The index start+2(i-1) gives the site index of the ith system mode.

        """
        for i=2:Ns
            for j = 1:(i-1)
                 ind = start+2(i-1)-j
                ind1 = s[ind]
                ind2 = s[ind+1]
                swap = fermionic_swap_gate(ind1,ind2) 

                orthogonalize!(ψ,ind)
                wf = (ψ[ind] * ψ[ind+1]) * swap
                noprime!(wf)
                inds3 = uniqueinds(ψ[ind],ψ[ind+1])
                U,S,V = svd(wf,inds3,cutoff=0)
                ψ[ind] = U
                ψ[ind+1] = S*V
            end
        end
        return ψ
    end

    function ancilla_phase_gate_swap(DP,P)
        (;Ns) = P
        (;qA,c,cdag) = DP

        Uph_list = []
        for i =1:Ns
            for j =1:(i-1)
            x = prime(c[qA[i]])*cdag[qA[i]]*prime(c[qA[j]])*cdag[qA[j]]
            unprime_ind(inds(x)[1],x)
            unprime_ind(inds(x)[3],x)  

            Rinds = inds(x,plev=0)
            Linds = Rinds'
            Uph = exp(-im*π*x,Linds,Rinds)
            push!(Uph_list,Uph)
            end
        end
        return Uph_list
    end

    function ancilla_phase_gate_PH(DP,P)
        (;Ns) = P
        (;qA,c,cdag) = DP
        """
        Only apply when Ns is odd
        """
        Uph_list = []

            for i =1:Ns
                x = prime(c[qA[i]])*cdag[qA[i]]
                unprime_ind(inds(x)[1],x) 

                Rinds = inds(x,plev=0)
                Linds = Rinds'
                Uph = exp(-im*π*x,Linds,Rinds)
                push!(Uph_list,Uph)
            end
            return Uph_list
    end

    function particle_hole_transform(DP,P)
        (;Ns) = P
        (;qA,q,c,cdag) = DP
        ###add them if it's even, subtract them if they're odd
        gates = [(JW_string(n,q[1],DP)*cdag[n]-((-1)^(Ns))*JW_string(n,q[1],DP)*c[n]) for n in qA]
        return gates    
    end

    function NESS_fn(ψ,P,DP)
        (;Ns) = P
        (;s,q) = DP
        d = 2^Ns
        
        gates =  ancilla_phase_gate_swap(DP,P)
        for i = 1:length(gates)
            ψ = apply(gates[i],ψ)
        end
        if isodd(Ns)
            gates =  ancilla_phase_gate_PH(DP,P)
            for i = 1:length(gates)
                ψ = apply(gates[i],ψ)
            end
        end

         ###applies a particle hole transformation to the ancilla states.
        gates =  particle_hole_transform(DP,P)
        for i = 1:length(gates)
            ψ = apply(gates[i],ψ)
        end

        ###Applies fermionic swap gates to change the order from interleaved to separated

        ψ = system_swaps(ψ,q[1],DP,P)
        qA =  q[Ns+1:2*Ns]
        qS =  q[1:Ns]

        rm_inds = [qS ; qA]
        ρf = rdm_para(ψ,rm_inds,DP,P)
        Cs = combiner(s[qS]) # Combiner tensor for merging system legs into a fat index
        Ca = combiner(s[qA]) # Combiner tensor for merging ancilla legs into a fat index
        ρΛ = ρf*Cs*Cs'*Ca*Ca'# Merge physical legs to form a density matrix
        Csa = combiner([inds(Cs)[1],inds(Ca)[1]])
        ρmat = ρΛ*Csa*Csa'
        ρmat = Matrix(ρmat,inds(ρmat));

        cutoff= 1e-5
        message,bool = ρ_test(ρmat,cutoff)
        if bool
            error(message)
        end
        Css = combiner([inds(Cs)[1],inds(Cs)[1]'])
        Caa = combiner([inds(Ca)[1],inds(Ca)[1]'])

        Λmat = d*ρΛ*Css*Caa
        Λmat = Matrix(Λmat,inds(Λmat));
        return ρmat,Λmat
    end

    function NESS_calculations(Λ_list,DP,P)
        """
        This functions takes a list of maps in matrix form and for each entry
        it calculates the spectrum, the NESS state in vector form, the NESS state in matrix form,
        the NESS currents and density, and the convergence of the NESS state using trace distance between
        consecutive NESS states. 
        """
        (;Ns) = P
        d = size(Λ_list[1])[1]
        nframe = length(Λ_list) ##don't want to include t=0

        spec_l = complex(zeros(nframe,d)) ##list of the spectrum.
        vec_l = complex(zeros(nframe,d)) ##list of the NESS states in vector form.
        mat_l = Any[] #list of the NESS states in standard matrix form.
        JL_NESS_l = []
        JR_NESS_l = []
        den_NESS_l = []
        fid_l = Any[] #list of the fidelities of the NESS state at t with the NESS state at t+dt
        tr_l = Any[] #list of the trace distances of the NESS state at t with the NESS state at t+dt


        for i =1:nframe 

            ##extracting NESS state
            spec = eigen(Λ_list[i]).values
            y = real.(spec)
            ind = argmax(abs.(y))
            vec = eigen(Λ_list[i]).vectors[:,ind]

            ##Calculating correlation matrix for NESS state
            mat = unvectorise_ρ(vec,true)
    #         cutoff=1e-10
    #         @show(ρ_test(mat,cutoff)[1])
            corr_NESS = ρ_system_corr(mat,DP,P)

            ##Calculating NESS currents
            if Ns ==3
                NESS_bool = true
                JL_NESS,JR_NESS,den_NESS = three_site_current_operator(corr_NESS,NESS_bool,P,DP)
                push!(JL_NESS_l,JL_NESS)
                push!(JR_NESS_l,JR_NESS)
                push!(den_NESS_l,den_NESS)
            end

            ##storing data
            spec_l[i,:] = spec
            vec_l[i,:] = vec
            push!(mat_l,mat)

            ##calculation of convergence measures
            if i>1
                fid = fidelity(mat_l[i],mat_l[i-1])
                tr_dist = trace_dist(mat_l[i],mat_l[i-1])
                push!(fid_l,fid)
                push!(tr_l,tr_dist)
            end
        end
        fid_l = fid_l .- 1

        return spec_l,vec_l,mat_l,JL_NESS_l,JR_NESS_l,den_NESS_l,fid_l,tr_l
    end
    

    function vectorise_ρ(ψ,P,DP)
        """
        In order for the rdm to be valid here, the ordering of qS and qA must be separated. This function
        assumes that the input state has an interleaved ordering, so the required swap and phase gates are applied.
        
        """
        """
        Do you do the particle hole transform here or not? 
        """
        (;s,q,qS,c,cdag) = DP

        Ns = length(qS)
        gates =  ancilla_phase_gate_swap(DP,P)
        for i = 1:length(gates)
            ψ = apply(gates[i],ψ)
        end
    #     if isodd(Ns)
    #         gates =  ancilla_phase_gate_PH(Ns,qA)
    #         for i = 1:length(gates)
    #             ψ = apply(gates[i],ψ)
    #         end
    #     end

        ###Applies fermionic swap gates to change the order from interleaved to separated
        ψ = system_swaps(ψ,q[1],DP,P)
        qA =  q[Ns+1:2*Ns]
        qS =  q[1:Ns]


        #combined all system legs together
        Cs = combiner(s[qS])
        Css = combiner([inds(Cs)[1],inds(Cs)[1]'])
        #Creating the evolved system density matrix 
        rm_inds = qS
        ρ = rdm(ψ,rm_inds,DP,P)
        ρ = ρ*Cs*Cs'
        ρ = ρ*Css
        ρ = Array(ρ,inds(ρ))
        return ρ
    end

    function unvectorise_ρ(ρvec,norm_bool)

        d =  Int(sqrt(length(ρvec)))
        ρ = complex(zeros(d,d))
        for i =1:d
            for j=1:d
                ρ[i,j] = ρvec[Int((i-1)*d +j)]
            end
        end
        if norm_bool
            ρ = ρ/tr(ρ) ##ensures correct normalisation
            """
            Check rho is a valid density matrix
            """
            cutoff= 1e-5
            message,bool = ρ_test(ρ,cutoff)
            if bool
                error(message)
            end
        end
        return ρ
    end

    function ρ_system_corr(ρ,DP,P)
        """
        Note that at this point, only system sites remain. 
        While technically s[1:Ns] will probably be bath modes,
        in this function these indices are just used as labels so
        it doesn't matter.
        """
        (;Ns) = P
        (;s,c,cdag) = DP
        corr = complex(zeros(Ns,Ns))
        for i=1:Ns
            for j=1:Ns
                ##create the creation and annihilation in MPO form
                aj_dag = JW_string(j,1,DP)*cdag[j]*Id_string(j,Ns,DP)
                ai = JW_string(i,1,DP)*c[i]*Id_string(i,Ns,DP)
                corr_op = apply(aj_dag,ai)

                C = combiner(s[1:Ns])
                corr_op = corr_op*C*C'
                corr_op = transpose(Matrix(corr_op,inds(corr_op)))

                corr[i,j] = tr(ρ*corr_op)
            end
        end
        return corr
    end

    function mp(i,j,d)
        ind = (i-1)*d +j
        return ind
    end

    function fermionic_swap_gate(i,j)

        k = i'
        l = j'
        T = ITensor(i,j,k,l)
        T[i=>1,j=>1,k=>1,l=>1] = 1
        T[i=>1,j=>1,k=>1,l=>2] = 0
        T[i=>1,j=>1,k=>2,l=>1] = 0
        T[i=>1,j=>1,k=>2,l=>2] = 0

        T[i=>1,j=>2,k=>1,l=>1] = 0
        T[i=>1,j=>2,k=>1,l=>2] = 0
        T[i=>1,j=>2,k=>2,l=>1] = 1
        T[i=>1,j=>2,k=>2,l=>2] = 0

        T[i=>2,j=>1,k=>1,l=>1] = 0
        T[i=>2,j=>1,k=>1,l=>2] = 1
        T[i=>2,j=>1,k=>2,l=>1] = 0
        T[i=>2,j=>1,k=>2,l=>2] = 0

        T[i=>2,j=>2,k=>1,l=>1] = 0
        T[i=>2,j=>2,k=>1,l=>2] = 0
        T[i=>2,j=>2,k=>2,l=>1] = 0
        T[i=>2,j=>2,k=>2,l=>2] = -1

        return T
    end

    #--------------------------------------------------------------------------------------------------------------------------------------
    """
    Discretisation mappings and band diagonalisation.
    """
    function direct_mapping(spec_therm,side,DP,P)
        """
        Implements direct discretisation, using simple trapezium integration.
        """
        (;D) = P
        (;ϵi) = DP
        if side=="left"
            Nb = P.Nbl
            ϵ_ref = ϵi[1]
        elseif side=="right"
            Nb,Ns = P.Nbr,P.Ns
            ϵ_ref = ϵi[Ns]
        else
            error("No side chosen.")
        end

        samp = 1000 # Number of points in mesh.
        y = LinRange(-D,D,Nb+1)
        tsq =  Vector{Float64}(undef,Nb)
        en =  Vector{Float64}(undef,Nb)
        for i =1:Nb
            x = LinRange(y[i],y[i+1],samp)
            Jx = J(x,spec_therm,side,P)
            
            tsq[i] = trapz(x,Jx); 
            en[i] = (1/tsq[i])*trapz(x,x.*Jx);
            
        end
        ind = sortperm(abs.(en.-ϵ_ref)) 
        tsq, en = tsq[ind], en[ind];  
        tk = sqrt.(tsq)
        return [tk,en] 
    end


    function reaction_mapping(spec_therm,side,P)
        """
        Have to include mapping to a star and the square root before this.
        """

        if side=="left"
            Nb,method = P.Nbl,P.method
        elseif side=="right"
            Nb,method = P.Nbr,P.method
        else
            error("No side chosen.")
        end
        #Define fixed numerness-Cical mesh over [-2,2] to capture spectral function and
        # its hilbert transform correctly within [-1,1].
        samp = 100000 # Number of points in mesh.
        x = LinRange(-2,2,samp);

        Jx = J(x,spec_therm,side,P)# Evaluate symbolic input function over the grid.

        Vsq =  Vector{Float64}(undef,Nb)
        en =  Vector{Float64}(undef,Nb)
        # Loop over the omega intervals and perform integrations:
        Jcur = Jx; # Current bath spectral function.
        for s=1:Nb
          # Simple trapezoid integration for hopping squared and on-site energy:
          Vsq[s] = trapz(x,Jcur); 
          en[s] = (1/Vsq[s])*trapz(x,x.*Jcur);

          Jprev = Jcur;
          JH = imag(hilbert(Jprev)); # Hilbert transform.
          Jcur = (Vsq[s]/(pi^2))*Jprev./(JH.^2+Jprev.^2);
        end

        Vk = sqrt.(Vsq)
        if method != 0
            #Now we form rotate the bath modes such that we have a star geometry. This is done
            #by diagonalising the bath modes.

            A,U = zeros(Nb+1,Nb+1),zeros(Nb+1,Nb+1)
            A[1,1],U[1,1] = 1,1

            for i =2:Nb+1
                A[i,i] = en[i-1]
                A[i-1,i] = Vk[i-1]
                A[i,i-1] = conj(Vk[i-1])
            end
            A_sub = A[2:Nb+1,2:Nb+1]
            U_sub = eigen(A_sub).vectors
            U[2:Nb+1,2:Nb+1] = U_sub
            A_star = U'*A*U

            ###Pretty sure its this way and not [1,2:Nb+1]
            Vk = A_star[2:Nb+1,1]
            en = diag(A_star)[2:Nb+1]
        end
        return [Vk, en]
    end

    
    function orthopol_chain(spec_therm,side,P)
    
        if side=="left"
            Nb = P.Nbl
        elseif side=="right"
            Nb = P.Nbr
        else
            error("No side chosen.")
        end
        w(t) = J(t, spec_therm, side, P)
        supp = (-1,1)
        degree = Nb-1
        my_meas = Measure("my_meas", w, supp, false, Dict())
        my_op = OrthoPoly("my_op", degree, my_meas; Nquad=100000);
        α_coeffs,β_coeffs = coeffs(my_op)[:,1],coeffs(my_op)[:,2]
        energies = α_coeffs
        couplings = sqrt.(β_coeffs)
        
        return couplings,energies
    end

    function band_diag(B,d)
    # Band-diagonalize matrix B with a bandwidth of d:
        n = size(B,1); # Assumed to be square.
        U = Diagonal(ones(n,n));
        for k=1:Int(floor(n/d)-1)
            C = B[(k*d+1):n,((k-1)*d+1):(k*d)]; # Extract coupling matrix.
            F = qr(C); # Upper-triangularize.
            blocks = [[Diagonal(ones(k*d,k*d))]; [F.Q']]
            Q = cat(blocks...,dims=(1,2))    # Form full triangularizing unitary.    
            B = Q*B*Q'; # Apply to input matrix to transform for next step.
            U = Q*U; # Save this step's unitary to the full sequence.
        end
        return B,U; # Return the final band-diagonalized matrix.
    end;

    #------------------------------------------------------------------------------------------------------------------------------------
    """
    Useful matrix transformations and ITensor functions.
    """

    function reflection_diag(A)
        """
        This function takes an N by N matrix, and reflects its elements along the other diagonal (i = N-j). This is used
        to flip the band diagonalised matrices so the modes are in the correct order.
        """
        N = length(A[1,:])
        B = zeros(N,N)
        for i = 0:N-1
            for j = 0:N-1
                B[i+1,j+1] = A[N-j,N-i]
            end
        end
        return B
    end

    function U_thermo(N,f_k)
        """
        This unitary maps the energy eigenmode basis to the thermofield basis.
        """
        U_th = zeros(N,N)
        U_th[1,1],U_th[2,2] = 1,1
        b = 0
        for i=3:2:N
            b += 1
            U_th[i,i],U_th[i+1,i+1] = sqrt(1-f_k[b]),-sqrt(1-f_k[b])
            U_th[i,i+1],U_th[i+1,i] = sqrt(f_k[b]),sqrt(f_k[b])
        end
        return U_th
    end

    function U_chain(N,U1,U2)
        """
        This unitary maps the thermofield basis to the tridiagonal thermofield basis. 
        """
        U_tot = zeros(N,N)
        U_tot[1,1],U_tot[2,2] = 1,1
        b1 = 0
        for i=3:2:N
            b2 = 0 # resets the column iteration
            b1 +=1
            for j =3:2:N
                b2 += 1

                U_tot[i,j] = U1[b1,b2]
                U_tot[i+1,j+1] = U2[b1,b2]
            end
        end
        return U_tot
    end

    function chain_to_star(A)
        """
        This function takes a tridiagonal matrix (chain geometry) and maps it to a star geometry. 
        Maybe just make it a function to extract just the unitary, and put a check in for whether its 
        a star?
        """

        N = length(A[1,:])
        U = zeros(N,N)
        A_sub = A[2:N,2:n]
        U_sub = eigen(A_sub).vectors
        U[1,1] = 1
        U[2:N,2:N] = U_sub
        A_star = U'*A*U
        return A_star
    end

    
    function unprime_string(T)

        int = 1
        indices = inds(x)
        l = Int(length(indices)/2)
        for i =1:l
            ind = indices[length(indices)-2*i+1]
            j = setprime(ind,int)
            replaceind!(x,ind,j)
        end
        return T
    end
    function unprime_ind(ind,x)
        int = 1
        j = setprime(ind,int)
        replaceind!(x,ind,j)
        return x
    end
    function JW_string(n,start,DP)
        (;F) = DP

        if n>start
            x = F[n-1]
            if n-1>start
                for i in reverse(start:n-2)
                    x = F[i]*x
                end
            end
        else
            x=1
        end
        return x
    end 
    function Id_string(n1,n2,DP)
        """
        Gives an Id string from n1+1 to n2. Note that
        these operators don't need to be applied in a specific order as they
        all commute with everything. 
        """
        (;Id) = DP 
        x=1
        for i =n1+1:n2
            x = Id[i]*x
        end
        return x
    end

    function entanglement_entropy(ψ)
        # Compute the von Neumann entanglement entropy across each bond of the MPS
            N = length(ψ)
            SvN = zeros(N)
            psi = ψ
            for b=1:N
                psi = orthogonalize(psi, b)
                if b==1
                    U,S,V = svd(psi[b] , siteind(psi, b))
                else
                    U,S,V = svd(psi[b], (linkind(psi, b-1), siteind(psi, b)))
                end
                for n=1:ITensors.dim(S, 1)
                    p = S[n,n]^2
                    SvN[b] -= p * log2(p)
                end
            end
            return SvN
        end;
    


    #---------------------------------------------------------------------------------------------------------------------------------
    """
    Initialisation functions for both the initial state and the hamiltonian.
    """
    function diff_sys_init(a,b,P,DP)
        (;Ns,Nbl) = P
        (;order_bool,cdag,Id) = DP
        """
        Only need the pauli strings to start at the first site, i.e. 2*Nbl+1
        """
        mag = a^2 +b^2
        a /= mag
        b /= mag
    
        start = 2*Nbl+1
        stop = start +2*Ns - 2
        if order_bool
    
            system_gate = [a*(JW_string(n,start,DP)*cdag[n]*Id[n+1] + b*JW_string(n+1,start,DP)*cdag[n+1]) for n in (2*Nbl+1):2:(2*(Ns+Nbl)-1)]

        else
            system_gate = [(JW_string(n,start,DP)*cdag[n]*Id_string(n,n+Ns,Id) + JW_string(n+Ns,start,DP)*cdag[n+Ns])/sqrt(2) for n in start:(2*Nbl+Ns)]
        end
        return system_gate
    end
    
    function initialise_bath_gates_therm(side,P,DP)

        """
        ERROR: The tags of psi are incorrect due to this function.
        """
        (;N,cdag,Id) = DP
        
        if side =="left"
            Nb = P.Nbl
            start = 1
        elseif side == "right"
            Nb = P.Nbr
            start = N-2*Nb +1
        else 
            error("neither side chosen")
        end
        stop = start + 2*Nb - 2
        gates = [cdag[n]*Id[n+1] for n in start:2:stop]
        return gates
    end

    function initialise_bath_gates_eigen(side,P,DP)
        error("Haven't bothered to make this yet in new structure.")
    end

    function initialise_system_gates(DP,P)
        (;Ns,Nbl) = P
        (;order_bool,cdag,Id) = DP
        """
        Only need the pauli strings to start at the first site, i.e. 2*Nbl+1
        """


        start = 2*Nbl+1
        stop = start +2*Ns - 2
        if order_bool
            system_gate = [(JW_string(n,start,DP)*cdag[n]*Id[n+1] + JW_string(n+1,start,DP)*cdag[n+1])/sqrt(2) for n in (2*Nbl+1):2:(2*(Ns+Nbl)-1)]
        ##EDIT
        else
            system_gate = [(JW_string(n,start,DP)*cdag[n]*Id_string(n,n+Ns,Id) + JW_string(n+Ns,start,DP)*cdag[n+Ns])/sqrt(2) for n in start:(2*Nbl+Ns)]
            #system_gate = [(cdag[n]*Id[n+Ns] + Id[n]cdag[n+Ns])/sqrt(2) for n in (2*Nbl+1):(2*Nbl+Ns)]
        end
        return system_gate
    end

    function initialise_psi(DP,gate_list)
        (;s) = DP
        n = length(gate_list)
        gates = reverse(gate_list[n])
        for j=1:n-1
            for i in reverse(1:length(gate_list[n-j]))
                    push!(gates,gate_list[n-j][i])
            end
        end
        vac = productMPS(s, "0");
        ψ = apply(gates,vac;cutoff=1e-15);
        return ψ
    end

    function initialise_bath(side,P,DP,BP)
        (;disc_choice,method) = P
    
    #-----------------------------------------------
    """
    This block creates the terms that go into 
    building the bath hamiltonian.
    """
    if side == "left"
        Vk_emp,ϵb_emp,Vk_fill,ϵb_fill = BP.Vk_emp_L,BP.ϵb_emp_L,BP.Vk_fill_L,BP.ϵb_fill_L
    elseif side=="right"
        Vk_emp,ϵb_emp,Vk_fill,ϵb_fill = BP.Vk_emp_R,BP.ϵb_emp_R,BP.Vk_fill_R,BP.ϵb_fill_R
    else 
        error("neither side chosen")
    end
    (;method) = P
    if method ==0
        ch_l = ["empty","filled"]
    else
        ch_l = ["full","full"]
    end
    
    if disc_choice == "direct"
        Vk_emp, ϵb_emp = direct_mapping(ch_l[1],side,DP,P)
        Vk_fill, ϵb_fill = direct_mapping(ch_l[2],side,DP,P)
    elseif disc_choice == "reaction"
        Vk_emp, ϵb_emp = reaction_mapping(ch_l[1],side,P)
        Vk_fill, ϵb_fill = reaction_mapping(ch_l[2],side,P)
    elseif disc_choice == "orthopol"
        Vk_emp, ϵb_emp = orthopol_chain(ch_l[1],side,P)
        Vk_fill, ϵb_fill = orthopol_chain(ch_l[2],side,P)
    else
        error("no discretization chosen")
    end
    
    # """
    # Reversing the order of the left bath couplings so the last element corresponds to the bath mode
    # next to the first system mode on the left.
    # """
    if side =="left"
        if method != 3
            """
            The reason for this condition is that for the tridiagonal case, the bath modes don't need to be put in 
            the right order for the MPO for the band diagonalisation of fill_mat and emp_mat. The order is handled after
            this is done using the reflection_diag function. These arrays don't affect the initialisation of the state
            for the thermofield case so they don't need to be ordered correctly yet.
            """
            Vk_emp = reverse(Vk_emp)
            Vk_fill = reverse(Vk_fill)
            ϵb_emp = reverse(ϵb_emp)
            ϵb_fill = reverse(ϵb_fill)
        end
    end
    #--------------------------------------------------------
    """
    This block creates the gates needed to create the initial
    bath state.
    """
    if method == 1
        gates = initialise_bath_gates_eigen(side,P,DP)
    else
        gates = initialise_bath_gates_therm(side,P,DP)
    end
    #---------------------------------------------------------

    return Vk_emp,ϵb_emp,Vk_fill,ϵb_fill,gates
    end


    function H_S(P,DP,BP)
        """
        This function creates the system hamiltonian, excluding the system
        bath couplings.
        """
        (;Ns,Nbl) = P
        (;ϵi,ti) = DP 
        (;H_single) = BP

        terms = OpSum()
        b = 2*Nbl-1
        for i=1:Ns
            b += 2
            terms += ϵi[i],"n",b;
            H_single[b,b] = ϵi[i]

            if i<Ns
                terms += ti[i],"Cdag",b,"C",b+2;
                H_single[b,b+2] = ti[i]

                terms += conj(ti[i]),"Cdag",b+2,"C",b;
                H_single[b+2,b] = conj(ti[i])
            end
        end

    #     """
    #     Edit:
    #     """
    #     terms = OpSum()
    #     b = 2*Nbl
    #     for i=1:Ns
    #         b += 1
    #         terms += ϵi[i],"n",b;
    #         H_single[b,b] = ϵi[i]

    #         if i<Ns
    #             terms += ti[i],"cdagag",b,"C",b+1;
    #             H_single[b,b+1] = ti[i]

    #             terms += conj(ti[i]),"cdagag",b+1,"C",b;
    #             H_single[b+1,b] = conj(ti[i])
    #         end
    #     end
        return terms,H_single
    end



    function H_bath(side,BP,DP,P)
    

        #The following code is not optimised, various objects are created multiple times
        #and within each different option there is identical code which doesn't
        #need to be written multiple times.
        """
        One way to make this easier to read is to write a separate function for creating a star geometry hamiltonian. 
        """
        (;method) = P
        (;N) = DP
        (;H_single) = BP
    
        ###Create Hamiltonian MPO 
        terms = OpSum()
        ###Create single particle matrix hamiltonian
        if side == "left"
            Nb = P.Nbl
            Vk_emp,ϵb_emp,Vk_fill,ϵb_fill,β,μ = BP.Vk_emp_L,BP.ϵb_emp_L,BP.Vk_fill_L,BP.ϵb_fill_L,P.β_L,P.mu_L
            link_ind = 2*Nb + 1
            start = 1
        elseif side == "right"
            Nb = P.Nbr
            Vk_emp,ϵb_emp,Vk_fill,ϵb_fill,β,μ = BP.Vk_emp_R,BP.ϵb_emp_R,BP.Vk_fill_R,BP.ϵb_fill_R,P.β_R,P.mu_R
            link_ind = N-2*Nb - 1
            start = (N-2*Nb +1)
        else 
            error("start input is invalid")
        end
        stop = start +2*Nb - 2
    
        fk = 1 ./(1 .+exp.(β*(ϵb_emp.- μ)))
    
        b = 0
        if method ==0
            for j=start:2:stop
                b += 1
                """
                Self energies.
                """
                terms += ϵb_fill[b],"n",j                                   # filled mode self energy
                H_single[j,j] = ϵb_fill[b]
    
                terms += ϵb_emp[b],"n",j+1                                 # empty mode self energy
                H_single[j+1,j+1] =  ϵb_emp[b]
                
                """
                Couplings.
                """
                if side =="right"
                    terms += Vk_fill[b],"Cdag",j-2,"C",j         #hopping from system to kth f mode
                    H_single[j-2,j] = Vk_fill[b]
    
                    terms += conj(Vk_fill[b]),"Cdag",j,"C",j-2      #hopping from kth f mode to system 
                    H_single[j,j-2] = conj(Vk_fill[b])
                    if j == start
                        terms += Vk_emp[b],"Cdag",j-2,"C",j+1        #coupling from system to kth e mode
                        H_single[j-2,j+1] =  Vk_emp[b]
                        
                        terms += conj(Vk_emp[b]),"Cdag",j+1,"C",j-2
                        H_single[j+1,j-2] = conj(Vk_emp[b])
                    else
                        terms += Vk_emp[b],"Cdag",j-1,"C",j+1  #coupling from kth e mode to system
                        H_single[j-1,j+1] =  Vk_emp[b]
                    
                        terms += conj(Vk_emp[b]),"Cdag",j+1,"C",j-1  #coupling from kth e mode to system
                        H_single[j+1,j-1] =  conj(Vk_emp[b])
                    end
                elseif side == "left"
                    terms += conj(Vk_fill[b]),"Cdag",j,"C",j+2         #hopping from system to kth f mode
                    H_single[j,j+2] = conj(Vk_fill[b])
    
                    terms += Vk_fill[b],"Cdag",j+2,"C",j      #hopping from kth f mode to system 
                    H_single[j+2,j] = Vk_fill[b]
                    if j == stop
                        terms += conj(Vk_emp[b]),"Cdag",j+1,"C",j+2        #coupling from system to kth e mode
                        H_single[j+1,j+2] =  conj(Vk_emp[b])
                    
                        terms += Vk_emp[b],"Cdag",j+2,"C",j+1        #coupling from system to kth e mode
                        H_single[j+2,j+1] =  Vk_emp[b]
                    
                    else
                        terms += conj(Vk_emp[b]),"Cdag",j+1,"C",j+3  #coupling from kth e mode to system
                        H_single[j+1,j+3] =  conj(Vk_emp[b])
                    
                        terms += Vk_emp[b],"Cdag",j+3,"C",j+1  #coupling from kth e mode to system
                        H_single[j+3,j+1] =  Vk_emp[b]
                    end
                end
            end
        end
        if method ==1
            Vk = Vk_emp
            ϵb = ϵb_emp
            
            print("negative ancilla H = -1, no ancilla H = 0, positive ancilla H = 1")
            HA_choice = parse(Int,readline()) 
            model =  "E basis,"
            if HA_choice ==1
                model = model*"Ha=Hb"
            end
            if HA_choice ==0
                model = model*"Ha=0"
            end
            if HA_choice ==-1
                model = model*"Ha=-1"
            end
            for j=start:2:stop
                b += 1
                terms += ϵb[b],"n",j                                   # bath mode self energy
                H_single[j,j] = ϵb[b]
    
                terms += HA_choice*ϵb[b],"n",j+1                       # ancilla bath mode self energy
                H_single[j+1,j+1] =  HA_choice*ϵb[b]
    
                terms += conj(V_k[b]),"Cdag",j,"C",link_ind                        #hopping from system to kth f mode 
                H_single[j,link_ind] = conj(V_k[b])
    
                terms += V_k[b],"Cdag",link_ind,"C",j                   #hopping from kth f mode to system 
                H_single[link_ind,j] = V_k[b]
            end    
        end
    
        if method==2
            Vk = Vk_emp
            ϵb = ϵb_emp
            fk = 1 ./(1 .+exp.(β*(ϵb.- μ)))
            model = "thermofield basis"
            for j=start:2:stop
                b += 1
                terms += ϵb[b],"n",j                                   # filled mode self energy
                H_single[j,j] = ϵb[b]
    
                terms += ϵb[b],"n",j+1                                 # empty mode self energy
                H_single[j+1,j+1] =  ϵb[b]
    
                terms += conj(Vk[b])*sqrt(fk[b]),"Cdag",j,"C",link_ind            #hopping from system to kth f mode
                H_single[j,link_ind] = conj(Vk[b])*sqrt(fk[b])
    
                terms += Vk[b]*sqrt(fk[b]),"Cdag",link_ind,"C",j      #hopping from kth f mode to system 
                H_single[link_ind,j] = Vk[b]*sqrt(fk[b])
                
                terms += -conj(Vk[b])*sqrt(1-fk[b]),"Cdag",j+1,"C",link_ind        #coupling from system to kth e mode
                H_single[j+1,link_ind] =  -conj(Vk[b])*sqrt(1-fk[b])
            
                terms += -Vk[b]*sqrt(1-fk[b]),"Cdag",link_ind,"C",j+1  #coupling from kth e mode to system
                H_single[link_ind,j+1] =  -Vk[b]*sqrt(1-fk[b])
            end
        end
    
        if method ==3
            Vk = Vk_emp
            ϵb = ϵb_emp
            fk = 1 ./(1 .+exp.(β*(ϵb.- μ)))
            model = "thermofield+tridiag"
            fill_mat = zeros(Nb+1,Nb+1)
            emp_mat = zeros(Nb+1,Nb+1)
            ### Same terms as for method 2, inputed as a matrix rather than an MPO. 
            for j=1:Nb
                ###This loop creates two (Nb+1) x (Nb+1) matrices, one includes the couplings and self energies of the 
                ###filled modes and the system, the other the empty modes and the system. These are then tridiagonalised 
                ###separately in the next loop, and their elements are used to construct the MPO for the hamiltonian in this new, tridiagonal basis. 
    
                """
                Pretty sure the first entry doesn't effect the diagonalisation so I set it to zero for both.
                """
                fill_mat[1,1],emp_mat[1,1] = 0, 0
    
                fill_mat[j+1,j+1],emp_mat[j+1,j+1] = ϵb[j],ϵb[j]
    
                fill_mat[j+1,1] = conj(Vk[j])*sqrt(fk[j])
    
                fill_mat[1,j+1] = Vk[j]*sqrt(fk[j])
    
                emp_mat[j+1,1] = -conj(Vk[j])*sqrt(1-fk[j])
    
                emp_mat[1,j+1] = -Vk[j]*sqrt(1-fk[j])
            end
    
            fill_mat,Uf = band_diag(fill_mat,1)
            emp_mat,Ue = band_diag(emp_mat,1)
    
            ###discarding the system terms as the system isn't mixed in this transformation.
            Uf = Uf[2:Nb+1,2:Nb+1] 
            Ue = Ue[2:Nb+1,2:Nb+1]
    
            b = 1
            if side == "left"
                ##flipping the matrices so the modes closest to the system
                ##are further along the matrix (in terms of i,j)
                fill_mat = reflection_diag(fill_mat)
                emp_mat = reflection_diag(emp_mat)
                #As the system mode is now at the end of 
                #these matrices, we don't skip over this at the start of the next loop
                b = 0 
            end
    
            for j=start:2:stop
                b += 1
                terms += fill_mat[b,b],"n",j
                H_single[j,j] = fill_mat[b,b]
    
                terms += emp_mat[b,b],"n",j+1
                H_single[j+1,j+1] = emp_mat[b,b]
                if side =="right"
                    terms += fill_mat[b-1,b],"Cdag",j-2,"C",j
                    H_single[j-2,j] = fill_mat[b-1,b]
    
                    terms += fill_mat[b,b-1],"Cdag",j,"C",j-2
                    H_single[j,j-2] = fill_mat[b,b-1]
    
                    if j == start 
                        terms += emp_mat[b-1,b],"Cdag",j-2,"C",j+1
                        H_single[j-2,j+1] = emp_mat[b-1,b]
    
                        terms += emp_mat[b,b-1],"Cdag",j+1,"C",j-2
                        H_single[j+1,j-2] = emp_mat[b,b-1]    
                    else             
                        terms += emp_mat[b-1,b],"Cdag",j-1,"C",j+1
                        H_single[j-1,j+1] = emp_mat[b-1,b]
    
                        terms += emp_mat[b,b-1],"Cdag",j+1,"C",j-1
                        H_single[j+1,j-1] = emp_mat[b,b-1]
                    end
                else side =="left"      
                    """
                    CHECK THAT THIS ORDER IS CORRECT, I.E WHETHER THE J AND J+2 
                    SHOULD BE SWITCHED
                    """
                    terms += fill_mat[b,b+1],"Cdag",j,"C",j+2
                    H_single[j,j+2] = fill_mat[b,b+1]
    
                    terms += fill_mat[b+1,b],"Cdag",j+2,"C",j
                    H_single[j+2,j] = fill_mat[b+1,b]
                    if j == stop
                        terms += emp_mat[b,b+1],"Cdag",j+1,"C",j+2
                        H_single[j+1,j+2] = emp_mat[b,b+1]
    
                        terms += emp_mat[b+1,b],"Cdag",j+2,"C",j+1
                        H_single[j+2,j+1] = emp_mat[b+1,b]
                    else
                        terms += emp_mat[b,b+1],"Cdag",j+1,"C",j+3
                        H_single[j+1,j+3] = emp_mat[b,b+1]
    
                        terms += emp_mat[b+1,b],"Cdag",j+3,"C",j+1
                        H_single[j+3,j+1] = emp_mat[b+1,b]
                    end
                end
            end
        end
    
        return  terms,H_single
        end
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Enrichment functions
"""
    function rdm_para_old(ψ,rm_inds,DP,P)
        (;Ns) = P
        (;s,N) = DP
        
        ψdag = dag(ψ)
        ITensors.prime!(linkinds, ψdag)
        ##Initialises an empty ITensor and an array with the same dimensions. The reason for creating an array
        ##is that you can pass a vector of indices to an array, but I can't work out how to do the same for an ITensor.
        rdm_ = ITensor(s[rm_inds],s[rm_inds]')
        rdm_arr = Array{ComplexF64}(rdm_,s[rm_inds],s[rm_inds]') 
        rdm_arr = SharedArray(rdm_arr)
        ##This loop iterates over the hilbert space of the reduced density matrix, which is given by 2^(2*Ns)
        #
        N_inds = length(rm_inds)
        lk = ReentrantLock()
        @sync @distributed for i=0:(2^(2*N_inds)-1)
            ##This converts the single index of the state into a vector of indices for each leg. 
            sys_inds = bitstring(Int16(i))
            sys_inds = split(sys_inds,"") 
            n = length(sys_inds)
            x =  ([parse(Int8,sys_inds[k]) for k in (16-2*N_inds+1):16] .+1)
            b = 0
            ρ =  ψdag[1]*ψ[1]        
            for j in 2:N
                if j in rm_inds
                    b +=1
                    ##The first 2Ns indices are taken as s[q] and the last 2Ns indices are taken as s[q]'.
                    ##I'm deliberately contracting ψdag[j] and ψ[j] with ρ separately to prevent creating a tensor of size
                    ##χ^4 with χ being the local bond dimension. The largest tensor created is of size \chi^2*d where d is the site dimension (2).
                    C1 = ψdag[j]*onehot(s[j]=>x[b]) 
                    C2 = ψ[j]*onehot(s[j]=>x[b+N_inds])  
                    ρ = ρ*C1
                    ρ = ρ*C2
                else
                    ρ = ρ* ψdag[j]
                    ρ = ρ* ψ[j]
                end
            end
            ##This sets the element rdm_arr[inds=x] as ρ.
            rdm_arr[CartesianIndex(Tuple(x))] = ρ[1]
            GC.gc()
        end
        rdm_ = ITensor(rdm_arr,s[rm_inds],s[rm_inds]')
        return rdm_
    end

    function rdm_para(ψ,rm_inds,DP,P)
        (;Ns) = P
        (;s,N) = DP
        
        ψdag = dag(ψ)
        ITensors.prime!(linkinds, ψdag)
        ##Initialises an empty ITensor and an array with the same dimensions. The reason for creating an array
        ##is that you can pass a vector of indices to an array, but I can't work out how to do the same for an ITensor.
        rdm_ = ITensor(s[rm_inds],s[rm_inds]')
        rdm_arr = Array{ComplexF64}(rdm_,s[rm_inds],s[rm_inds]') 
        rdm_arr = SharedArray(rdm_arr)
        ##This loop iterates over the hilbert space of the reduced density matrix, which is given by 2^(2*Ns)
        #
        N_inds = length(rm_inds)
        lk = ReentrantLock()
        @threads for i=0:(2^(2*N_inds)-1)
    
            ##This converts the single index of the state into a vector of indices for each leg. 
            sys_inds = bitstring(Int16(i))
            sys_inds = split(sys_inds,"") 
            n = length(sys_inds)
            x =  ([parse(Int8,sys_inds[k]) for k in (16-2*N_inds+1):16] .+1)
            b = 0
            ρ =  ψdag[1]*ψ[1] 
                    
            for j in 2:N
                if j in rm_inds
                    b +=1
                    ##The first 2Ns indices are taken as s[q] and the last 2Ns indices are taken as s[q]'.
                    ##I'm deliberately contracting ψdag[j] and ψ[j] with ρ separately to prevent creating a tensor of size
                    ##χ^4 with χ being the local bond dimension. The largest tensor created is of size \chi^2*d where d is the site dimension (2).
                    C1 = ψdag[j]*onehot(s[j]=>x[b]) 
                    C2 = ψ[j]*onehot(s[j]=>x[b+N_inds])  
                    ρ = ρ*C1
                    ρ = ρ*C2
    
                else
                    ρ = ρ* ψdag[j]
                    ρ = ρ* ψ[j]
                end
            end
            
            ##This sets the element rdm_arr[inds=x] as ρ.
            lock(lk) do
                rdm_arr[CartesianIndex(Tuple(x))] = ρ[1]
            end
        end
    
        rdm_ = ITensor(rdm_arr,s[rm_inds],s[rm_inds]')
        return rdm_
    end

    

    function rdm(ψ,rm_inds,DP,P)
        (;Ns) = P
        (;s,N) = DP
        
        ψdag = dag(ψ)
        ITensors.prime!(linkinds, ψdag)
        ##Initialises an empty ITensor and an array with the same dimensions. The reason for creating an array
        ##is that you can pass a vector of indices to an array, but I can't work out how to do the same for an ITensor.
        rdm_ = ITensor(s[rm_inds],s[rm_inds]')
        rdm_arr = complex(Array{Float64}(rdm_,s[rm_inds],s[rm_inds]') .+1) 
        
        ##This loop iterates over the hilbert space of the reduced density matrix, which is given by 2^(2*length(rm_inds))
        #
        N_inds = length(rm_inds)
        for i=0:(2^(2*N_inds)-1)
    
            ##This converts the single index of the state into a vector of indices for each leg. 
            sys_inds = bitstring(Int16(i))
            sys_inds = split(sys_inds,"") 
            n = length(sys_inds)
            x =  ([parse(Int8,sys_inds[k]) for k in (n-2*N_inds+1):n] .+1)
            b = 0
            ρ =  ψdag[1]*ψ[1] 
            
            for j in 2:N
                if j in rm_inds
                    b +=1
                    ##The first 2Ns indices are taken as s[q] and the last 2Ns indices are taken as s[q]'.
                    ##I'm deliberately contracting ψdag[j] and ψ[j] with ρ separately to prevent creating a tensor of size
                    ##χ^4 with χ being the local bond dimension. The largest tensor created is of size \chi^2*d where d is the site dimension (2).
                    C1 = ψdag[j]*onehot(s[j]=>x[b]) 
                    C2 = ψ[j]*onehot(s[j]=>x[b+N_inds])  
                    ρ = ρ*C1
                    ρ = ρ*C2
    
                else
                    ρ = ρ* ψdag[j]
                    ρ = ρ* ψ[j]
                end
            end
            
            ##This sets the element rdm_arr[inds=x] as ρ.
            rdm_arr[CartesianIndex(Tuple(x))] = ρ[1]
            
        end
    
        rdm_ = ITensor(rdm_arr,s[rm_inds],s[rm_inds]')
        return rdm_
    end

    # function rdm(ψ,DP)
    #     (;N,q) = DP
    #     ψdag = dag(ψ) # Complex conjugate MPS
    #     ITensors.prime!(linkinds, ψdag) # Add primes to all MPS bond indices
    #     # Loop over the sites q whose density matrix is required:
    #     for j=1:length(q)
    #         ITensors.noprime!(ψdag[q[j]]) # Remove prime on bond indices of ψdag[q[j]]
    #         ITensors.prime!(ψdag[q[j]]) # Prime all indices on ψdag[q[j]] including its site index
    #     end
    #     # Now contract:
    #     ITensors.set_warn_order(50)
    #     ρ = ψdag[1] * ψ[1]
    #     for j in 2:N
    #       C = ψdag[j] * ψ[j]
    #       ρ = ρ * C
    #     end
    #     return ρ
    # end



    function enrich_generic3(ϕ, ψ⃗; P, kwargs...)
        (;Kr_cutoff) = P
        """
          Given spec from the eigen function, to extract its information use the 
          following functions:

          eigs(spec) returns the spectrum
          truncerror(spec) returns the truncation error
        """  
      Nₘₚₛ = length(ψ⃗) ##number of MPS

      @assert all(ψᵢ -> length(ψ⃗[1]) == length(ψᵢ), ψ⃗) ##check that all MPS inputs are of the same length

      N = length(ψ⃗[1]) 
      ψ⃗ = copy.(ψ⃗)

      ###Isn't this already a vector of MPS's?  
      ψ⃗ = convert.(MPS, ψ⃗)

      s = siteinds(ψ⃗[1])
      ##makes the orthogonality centre for each MPS to be at site N  
      ψ⃗ = orthogonalize.(ψ⃗, N)
      ϕ = orthogonalize!(ϕ, N)

      ##storage MPS
      phi = deepcopy(ϕ)

      ρϕ = prime(ϕ[N], s[N]) * dag(ϕ[N])
      ρ⃗ₙ = [prime(ψᵢ[N], s[N]) * dag(ψᵢ[N]) for ψᵢ in ψ⃗]
      ρₙ = sum(ρ⃗ₙ)

      """
      Is this needed?
      """
      ρₙ /=tr(ρₙ)

    #   # Maximum theoretical link dimensions

      Cϕprev = ϕ[N]
      C⃗ₙ = last.(ψ⃗)


      for n in reverse(2:N)
         """
        In the paper they propose to do this step with no truncation. At the very
        least this cutoff should be a function parameter.
        """    

        left_inds = linkind(ϕ,n-1)

             #Diagonalize primary state ψ's density matrix    
        U,S,Vϕ,spec = svd(Cϕprev,left_inds; 
          lefttags = tags(linkind(ϕ, n - 1)),
          righttags = tags(linkind(ϕ, n - 1)))   

        x = ITensors.dim(inds(S)[1])
        @assert(x == ITensors.dim(linkind(ϕ, n - 1)))
        r = uniqueinds(Vϕ, S) # Indices of density matrix
        lϕ = commonind(S, Vϕ) # Inner link index from density matrix diagonalization


        # Compute the theoretical maximum bond dimension that the enriched state cannot exceed:
        abs_maxdim = bipart_maxdim(s,n - 1) - ITensors.dim(lϕ)
        # Compute the number of eigenvectors of ɸ's projected density matrix to retain:
        Kry_linkdim_vec = [ITensors.dim(linkind(ψᵢ, n - 1)) for ψᵢ in ψ⃗]


        ω_maxdim = min(sum(Kry_linkdim_vec),abs_maxdim)

        if ω_maxdim !== 0


            # Construct identity matrix
            ID = 1
            rdim = 1
            for iv in r
              IDv = ITensor(dag(iv)', iv);
              rdim *= ITensors.dim(iv)
              for i in 1:ITensors.dim(iv)
                IDv[iv' => i, iv => i] = 1.0
              end      
              ID = ID*IDv
            end   


            P = ID - prime(Vϕ, r)*dag(Vϕ) # Projector on to null-space of ρψ   

            C = combiner(r) # Combiner for indices
            # Check that P is non-zero   
            if abs(tr(matrix(C'*P*dag(C)))) > 1e-10    


                Dp, Vp, spec_P = eigen(
                      P, r', r,
                      ishermitian=true,
                      tags="P space",
                      cutoff=1e-1,
                      maxdim=rdim-ITensors.dim(lϕ),             ###potentially wrong
                      kwargs...,
                    )

                lp = commonind(Dp,Vp)

                ##constructing VpρₙVp
                VpρₙVp = Vp*ρₙ        
                VpρₙVp = VpρₙVp*dag(Vp')
                chkP = abs(tr(matrix(VpρₙVp))) ##chkP

            else
                chkP = 0    
            end
        else
            chkP = 0
        end

        if chkP >1e-15
            Dₙ, Vₙ, spec =eigen(VpρₙVp, lp', lp;
              ishermitian=true,
              tags=tags(linkind(ψ⃗[1], n - 1)),
              cutoff=Kr_cutoff,
              maxdim=ω_maxdim,            
              kwargs...,
            )

            Vₙ = Vp*Vₙ

            lₙ₋₁ = commonind(Dₙ, Vₙ)

            # Construct the direct sum isometry 
            V, lnew = directsum(Vϕ => lϕ, Vₙ => lₙ₋₁; tags = tags(linkind(ϕ, n - 1)))
        else
             V = Vϕ
             lnew = lϕ

        end
        @assert ITensors.dim(linkind(ϕ, n - 1)) - ITensors.dim(lϕ) <=0
        # Update the enriched state
        phi[n] = V


        # Compute the new density matrix for the ancillary states
        C⃗ₙ₋₁ = [ψ⃗[i][n - 1] * C⃗ₙ[i] * dag(V) for i in 1:Nₘₚₛ]   
        C⃗ₙ₋₁′ = [prime(Cₙ₋₁, (s[n - 1], lnew)) for Cₙ₋₁ in C⃗ₙ₋₁]    
        ρ⃗ₙ₋₁ = C⃗ₙ₋₁′ .* dag.(C⃗ₙ₋₁)
        ρₙ₋₁ = sum(ρ⃗ₙ₋₁)

        # compute the density matrix for the real state    
        Cϕ = ϕ[n - 1] * Cϕprev * dag(V)
        Cϕd = prime(Cϕ, (s[n - 1], lnew))
        ρϕ = Cϕd * dag(Cϕ) 


        Cϕprev = Cϕ
        C⃗ₙ = C⃗ₙ₋₁
        ρₙ = ρₙ₋₁

      end


        phi[1] = Cϕprev
        phi[1] = phi[1]/norm(phi)

      return phi
    end

    function bipart_maxdim(s,n)
    # Compute the theoretical maximum link dimension for an orthogonalised MPS
    # for the bipartition [1,...,n][n+1,...,N]
    #     left_maxdim = 1
    #     for k=1:n
    #         @show(left_maxdim)
    #         left_maxdim *= 2#ITensors.dim(s[k])
    #         @show(left_maxdim)
    #         if left_maxdim ==0
    #           #  @show(s[k])
    #            # @show(ITensors.dim(s[k]))
    #         end
    #     end
    #     right_maxdim = 1
    #     for k=(n+1):length(s)
    #         right_maxdim *= 2#ITensors.dim(s[k])
    #         if right_maxdim ==0
    #             @show(s[k])
    #             @show(ITensors.dim(s[k]))
    #         end
    #     end
        left_maxdim = 2^n
        right_maxdim = 2^(length(s)-n)
        if left_maxdim==0
            left_maxdim = 2^63
        end
        if right_maxdim==0
            right_maxdim = 2^63
        end
        return min(left_maxdim,right_maxdim)
    end;




    function Krylov_states(H,ψ,P,DP)
        (;k1,τ_Krylov) = P
        (;s) = DP



        ##Create the first k Krylov states
        Id = MPO(s,"Id")
        Kry_op = Id-im*τ_Krylov*H
        list = []
        term = copy(ψ)

       for i =1:k1-1
            term = noprime(Kry_op*term)
            term = term/norm(term)
            push!(list,term)
        end

        return list
    end


    function Krylov_linkdims(Krylov)
        """
    Determining whether a Krylov state has the dimensions of lower Krylov states within it.
    We have a list of vectors, and we want to see if the 1st vector has the highest entries for every entry.
    I want the output to be a vector of length of linkdims, where each entry denotes which Krylov vector has the maximum dimension.
    """
        x = linkdims_(Krylov[1])
        dim1 = length(x)
        dim2 = length(Krylov)
        output = zeros(length(x))
        stuff = zeros(dim1,dim2)
        for i =1:dim1
            for j =1:dim2
                stuff[i,j] = linkdims_(Krylov[j])[i]
            end
            vec = stuff[i,:]
            term = Int(argmax(vec)) 
            output[i] = term
            if term != dim2
               test = vec[term] -vec[dim2]
               if test<=0
                   output[i] = dim2
               end
            end
        end
        return output
    end


    function linkdims_(ψ)
        # Helpful function for outputing the link dimension profile of an MPS

      """
      Isn't working properly.  
      """
      linkdims = zeros(length(ψ)-1,1)
      for b in eachindex(ψ)[1:(end - 1)]
        l = linkind(ψ, b)
        linkdims[b] = isnothing(l) ? 1 : ITensors.dim(l)
      end
      return linkdims
    end


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Observer functions
"""


    ###Observer functions
    """
    Memory observer function from https://itensor.discourse.group/t/memory-usage-in-dmrg-julia/562/4
    """
    function measure_mem(; bond, half_sweep, psi, kwargs...)    
        if bond==1 && half_sweep==2
          psi_size =  Base.format_bytes(Base.summarysize(psi))
          println("|psi| = $psi_size")
        end
    end
    
    function current_time(; current_time, bond, half_sweep)
      if bond == 1 && half_sweep == 2
        return real(im*current_time)
      end
      return nothing
    end

    function measure_den(; psi, bond, half_sweep,Nbl)
      if bond == 1 && half_sweep == 2
        return expect(psi, "n"; sites=2*Nbl+1)
      end
      return nothing
    end;

    function measure_SvN(; psi, bond, half_sweep)
        if bond == 1 && half_sweep == 2
            return entanglement_entropy(psi)
        end
        return nothing
    end

    function measure_correlation_matrix(; psi, bond, half_sweep)
        if bond==1 && half_sweep == 2
            return transpose(correlation_matrix(psi,"Cdag","C"))
        end
        return nothing
    end


    function HS_full(P,DP,BP)
        """
        Note that at this point, only system sites remain. 
        While technically s[1:Ns] will probably be bath modes,
        in this function these indices are just used as labels so
        it doesn't matter.
        """
        (;Ns,Nbl) = P
        (;s,c,cdag,qS) = DP
        (;H_single) = BP
        HS_single = BP.H_single[qS,qS]
        d = 2^Ns
        HS = complex(zeros(d,d))
        for i=1:Ns
            for j=1:Ns
                ##create the creation and annihilation in MPO form
                aj_dag = JW_string(j,1,DP)*cdag[j]*Id_string(j,Ns,DP)
                ai = JW_string(i,1,DP)*c[i]*Id_string(i,Ns,DP)
                corr_op = apply(aj_dag,ai)
    
                C = combiner(s[1:Ns])
                corr_op = corr_op*C*C'
                corr_op = transpose(Matrix(corr_op,inds(corr_op)))
                HS += HS_single[i,j]*corr_op
            end
        end
        return HS
    end
    
    function lmult(A)
        ##Super-operator representing left-multiplication on a vectorised density matrix
        d = size(A)[1]
        Id = 1.0*Matrix(I, d, d)
        return kronecker(transpose(A),Id)
    end
    function rmult(A)
        ##Super-operator representing left-multiplication on a vectorised density matrix
        d = size(A)[1]
        Id = 1.0*Matrix(I, d, d)
        return kronecker(Id,A)
    end
    
    function spin_operators(M)
    
        # Build sparse matrix version of basic spin (Pauli) operators :
        sp = spdiagm(2,2,1=>ones(1))
        sm = spdiagm(2,2,-1=>ones(1))
        sz = spdiagm(2,2,0=>[1;-1]);
        num = spdiagm(2,2,0=>[0;1])
        # Notice there are NO factors of (1/2) for spin-1/2 included here.
        
        # Construct spin operators for each spin in the full Hilbert space :
        Sz = Vector{Any}(undef, M)
        Sp = Vector{Any}(undef, M)
        Sm = Vector{Any}(undef, M)
        Num = Vector{Any}(undef,M)
        for m=1:M
            Sz[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),sz),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
            Sp[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),sp),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
            Sm[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),sm),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
            Num[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),num),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
        end
        return Sz,Sp,Sm,Num
    end
    
    function build_liouvillian(H,Sp,Sm,Sz,Num,Ge,Gd,M)
    
        L = -1im*(rmult(H) - lmult(H)); # Coherent part: -i(H rho - rho H) = -i[H,rho]
        La = 0*L; Lb = 0*L; JL = 0*L; JR = 0*L;
        Zf = 0 
        for m=1:M
            # Build JW string:
            Z = 1.0*Matrix(I, 2^M, 2^M)
            for q=1:(m-1)
                Z = Z*Sz[q];
            end
            @show(size(Sm[m]))
            @show(size(Z))
            # The single-site spin flip with JW string = injection/ejection of spinless fermions:
            La = La + Ge[m]*(rmult(Sm[m]*Z)*lmult(Z*Sp[m]) - (1/2)*rmult((Sp[m])*Sm[m]) - (1/2)*lmult((Sp[m])*Sm[m]));
            Lb = Lb + Gd[m]*(rmult(Sp[m]*Z)*lmult(Z*Sm[m]) - (1/2)*rmult((Sm[m])*Sp[m]) - (1/2)*lmult((Sm[m])*Sp[m]));
            if m==M
                Zf = Z
            end
        end
        L = L + La + Lb
        JL_op = Ge[1]*(rmult(Sm[1])*lmult(Sp[1]) - (1/2)*rmult((Sp[1])*Sm[1]) - (1/2)*lmult((Sp[1])*Sm[1])) +
                Gd[1]*(rmult(Sp[1])*lmult(Sm[1]) - (1/2)*rmult((Sm[1])*Sp[1]) - (1/2)*lmult((Sm[1])*Sp[1]))
        JL_op = lmult(Num[1])*JL_op        
        JR_op = Ge[M]*(rmult(Sm[M]*Zf)*lmult(Zf*Sp[M]) - (1/2)*rmult((Sp[M])*Sm[M]) - (1/2)*lmult((Sp[M])*Sm[M])) +
                Gd[M]*(rmult(Sp[M]*Zf)*lmult(Zf*Sm[M]) - (1/2)*rmult((Sm[M])*Sp[M]) - (1/2)*lmult((Sm[M])*Sp[M]))
        JR_op = -lmult(Num[M])*JR_op  
        return L,JL_op,JR_op
    end
    
    function two_site_diss_rates(P,DP)
        
        (;Ns,Γ_L,Γ_R,β_L,β_R,mu_L,mu_R) = P
        (;ϵi) = DP
        f_L = 1 /(1 +exp(β_L*(ϵi[1] - mu_L)))
        f_R = 1 /(1 +exp(β_R*(ϵi[Ns] - mu_R)))
        ##prefactor is due to the different defs of gammas compared to the review paper.
        fs = complex(zeros(Ns))
        Γs = complex(zeros(Ns))
        fs[1],fs[Ns]=f_L,f_R
        Γs[1],Γs[Ns] = (4/π)*Γ_L,(4/π)*Γ_R
        Ge = Γs.*fs
        Gd = Γs.*(1 .-fs)
        return Ge,Gd
    end
    
    function Liouvillian_solver_exact(L,tvec,ρ0)
        """
        Taken from https://stackoverflow.com/questions/56214666/left-and-right-eigenvectors-in-julia.
        This represents equations 177-179 in "Non-equilibrium boundary driven quantum systems: models, methods and
        properties, Gabriel T. Landi,1, ⇤ Dario Poletti,2, † and Gernot Schaller3,"
        """
        println("here1")
        ρ_list = Any[]
        α = size(L)[1]
        F = eigen(L)
        Q = eigvecs(F) # right eigenvectors 
        QL = inv(eigvecs(F)) # left eigenvectors 
        Λ = eigvals(F)
        ρ_ness = Q[:,length(Λ)]
        
        push!(ρ_list,ρ0)
        for i=1:length(tvec)
            ρt = complex(0*copy(ρ0))
            for α=1:size(L)[1]
                c_α= sum(QL[α,:].*ρ0)
                mode_α = c_α*exp(Λ[α]*tvec[i])*Q[:,α]
                ρt += mode_α
            end
            push!(ρ_list,ρt)
        end
        return ρ_list,Λ,ρ_ness
    end
    
end