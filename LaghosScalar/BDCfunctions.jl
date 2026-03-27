function read_mfem_nodes(filename)
    coords = Float64[]
    in_nodes = false
    skip = 0

    for line in eachline(filename)
        if strip(line) == "nodes"
            in_nodes = true
            skip = 4  # skip FE space metadata
            continue
        end

        if in_nodes
            if skip > 0
                skip -= 1
                continue
            end
            try
                push!(coords, parse(Float64, strip(line)))
            catch
            end
        end
    end
    return coords
end



function read_other(filename)
    coords = Float64[]
    in_nodes = false
    skip = 4

    for line in eachline(filename)
        if skip > 0
            skip -= 1
            continue
        end
        try
            push!(coords, parse(Float64, strip(line)))
        catch
        end
    end
    return coords
end

function read_all(prob, rs, ode, tf, igr = true, alpha = 4e-6, suffix = "")
    folder = (igr ? "WithIGR/" * "alpha=" * (alpha<1e-3 ? (@sprintf "%1.6f" alpha*1e6) * "e-6/" : (@sprintf "%1.2f" alpha*1e3) * "e-3/")  : "WithoutIGR/") * "p" * string(prob) * "/"
    str = string(prob) * "_" * string(rs) * "_" * string(ode) * "_" * (@sprintf "%d" tf*1e3)
    if(!igr)
        str = str * "_noigr"
    end
    str = str * suffix
    x = read_mfem_nodes(folder * "Laghos_" * str * "_mesh")
    rho = read_other(folder * "Laghos_" * str * "_rho")
    v = read_other(folder * "Laghos_" * str * "_v")
    e = read_other(folder * "Laghos_" * str * "_e")
    igrp = read_other(folder * "Laghos_" * str * "_igr")

    pv = sortperm(x)
    x = x[pv]; rho = rho[pv]; v = v[pv]; e = e[pv]
    p = 0.4 .* rho .* e

    return x, rho, v, e, p, igrp
end

function plot_igr(vals, axis, steps, increment, prob, rs, ode, tf, igr = true, xlim = (0,1), ylim = (0,1), sep = 0, alpha = 4e-6, suffix = "")

    choose_y(vals) = (vals == "rho" ? rho : (vals == "v" ? v : (vals == "e" ? e : (vals == "p" ? p : igrp))))

    out = plot()
    if(axis == "tf")
        x, rho, v, e, p, igrp = read_all(prob, rs, ode, tf, igr, alpha, suffix)
        plot!(x, choose_y(vals), xlim = xlim, ylim = ylim, lab = "tf = " * string(tf), title = vals * " " * suffix)
    elseif(axis == "rs")
        x, rho, v, e, p, igrp = read_all(prob, rs, ode, tf, igr, alpha, suffix)
        plot!(x, choose_y(vals), xlim = xlim, ylim = ylim, lab = "RS " * string(rs), title = vals * " at t = " * string(tf) * " " * suffix)
    elseif(axis == "igr")
        x, rho, v, e, p, igrp = read_all(prob, rs, ode, tf, true, alpha, suffix)
        plot!(x, choose_y(vals), xlim = xlim, ylim = ylim, lab = "IGR", title = vals * " at t = " * string(tf) * " " * suffix)
    elseif(axis == "alpha")
        x, rho, v, e, p, igrp = read_all(prob, rs, ode, tf, true, alpha, suffix)
        plot!(x, choose_y(vals), xlim = xlim, ylim = ylim, lab = alpha, title = vals * " at t = " * string(tf) * " " * suffix)
    elseif(axis == "RSalpha")
        x, rho, v, e, p, igrp = read_all(prob, rs, ode, tf, true, alpha, suffix)
        plot!(x, choose_y(vals), xlim = xlim, ylim = ylim, lab = "RS " * string(rs) * ", alpha= " * string(alpha), 
                                 title = vals * " at t = " * string(tf) * " " * suffix)
    end 
    plot!(ylab=vals)
    
    for i in 2:steps
        if(axis == "tf")
            x, rho, v, e, p, igrp = read_all(prob, rs, ode, tf + increment*(i-1), igr, alpha, suffix)
            plot!(x, choose_y(vals) .+ sep*(i-1), xlim = xlim, ylim = ylim, lab = "tf = " * string(tf + increment*(i-1)))
        elseif(axis == "rs")
            x, rho, v, e, p, igrp = read_all(prob, rs + (i-1), ode, tf, igr, alpha, suffix)
            plot!(x, choose_y(vals) .+ sep*(i-1), xlim = xlim, ylim = ylim, lab = "RS " * string(rs + (i-1)))
        elseif(axis == "igr")
            x, rho, v, e, p, igrp = read_all(prob, rs, ode, tf, false, alpha, suffix)
            plot!(x, choose_y(vals) .+ sep*(i-1), xlim = xlim, ylim = ylim, lab = "NO IGR")
            break
        elseif(axis == "alpha")
            x, rho, v, e, p, igrp = read_all(prob, rs, ode, tf, true, alpha / 4^(i-1), suffix)
            plot!(x, choose_y(vals) .+ sep*(i-1), xlim = xlim, ylim = ylim, 
                                lab = "RS " * string(rs) * ", alpha= " * string(alpha/4^(i-1)))
        elseif(axis == "RSalpha")
            x, rho, v, e, p, igrp = read_all(prob, rs + (i-1), ode, tf, true, alpha / 4^(i-1), suffix)
            plot!(x, choose_y(vals) .+ sep*(i-1), xlim = xlim, ylim = ylim, 
                                lab = "RS " * string(rs + (i-1)) * ", alpha= " * string(alpha/4^(i-1)))
        end 
    end
    
    return out
end

function pad_to(vec, len)
    [vec; fill(NaN, len - length(vec))]
end

function writeIGRcsv(prob, rs1, order1, tf1, igr1, alpha1, suffix1, rs2, order2, tf2, igr2, alpha2, suffix2)
    x, rho, v, e, p, igrp = read_all(prob, rs1, order1, tf1, igr1, alpha1, suffix1);
    x2, rho2, v2, e2, p2, igrp2 = read_all(prob, rs2, order2, tf2, igr2, alpha2, suffix2);

    vecs = [x, rho, v, e, p, igrp, x2, rho2, v2, e2, p2, igrp2]
    maxlen = maximum(length.(vecs))
    header = ["x" "rho" "v" "e" "p" "igrp" "x2" "rho2" "v2" "e2" "p2" "igrp2"]
    padded = hcat([pad_to(v, maxlen) for v in vecs]...)

    open("output.csv", "w") do io
        writedlm(io, header, ',')
        writedlm(io, padded, ',')
    end
end



