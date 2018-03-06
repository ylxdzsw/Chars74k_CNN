using ProgressMeter
using OhMyJulia
using PyCall
using Fire

unshift!(PyVector(pyimport("sys")["path"]), @__DIR__)

sigmoid(x) = e^x / (e^x + 1)

function encode(Xs)
    id, dict = 1, Dict() # 0->missing, 1->unknown, real id starts from 2

    function lookup(c)
        get!(dict, c) do
            id += 1
        end
    end

    return [[lookup(c) for c in X] for X in Xs], dict
end

function encode(Xs, V)
    return [[get(V, c, 1) for c in X] for X in Xs]
end

function get_batch(X, y, batch=64)
    rs = isa(batch, Int) ? rand(1:nrow(X), batch) : batch
    X_filled = Array{Int}(length(rs), 1000)
    for (i, r) in enumerate(rs)
        l = length(X[r])
        if l > 1000
            s = rand(1:l-999) # randomly crop 1000 letters, which I think it's sufficient
            X_filled[i, :] = X[r][s:s+999]
        else
            X_filled[i, 1:l] = X[r]
            X_filled[i, l+1:end] = 0 # fill with special token
        end
    end
    X_filled, isa(y, Void) ? y : y[rs, :]
end

function get_dict()
    data = readdlm("D:/jigsaw-toxic-comment-classification-challenge/raw/train.csv", ','; skipstart=1)
    encode(data[:, 2])[2]
end

function drop_out!(X)
    d = rand(1:length(X), rand(0:floor(Int, .01length(X))))
    X[d] .= X[d] .!= 0 # set to 1 iff it's not 0
end

@main function train(modelname; epoch::Int=80, lr::Float64=0.1)
    data = readdlm("D:/jigsaw-toxic-comment-classification-challenge/raw/train.csv", ','; skipstart=1)
    X, y = data[:, 2], map(Int, data[:, 3:end])
    X, V = encode(X)

    model = pywrap(pyimport(modelname)[:Model](length(V)+2, "gpu"))

    for i in 1:epoch
        tic()
        loss = 0

        for j in 1:100
            batch = get_batch(X, y)
            drop_out!(car(batch))
            loss += model.train(batch..., lr) / 100
        end

        lr *= .95

        println("=== epoch: $i, loss: $loss, time: $(toq()) ===")
        i % 10 == 0 && model.save("D:/jigsaw-toxic-comment-classification-challenge/result/$modelname.model")
    end
end

@main function predict(modelname)
    data = readdlm("D:/jigsaw-toxic-comment-classification-challenge/raw/test.csv", ',', String; skipstart=1)
    ids, X, V = data[:, 1], data[:, 2], get_dict()
    X = encode(X, V)

    model = pywrap(pyimport(modelname)[:Model](length(V)+2, "gpu", "D:/jigsaw-toxic-comment-classification-challenge/result/$modelname.model"))

    open("D:/jigsaw-toxic-comment-classification-challenge/result/$modelname.csv", "w") do fout
        fout << "id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n"
        @showprogress for i in 1:64:length(ids)
            x, _ = get_batch(X, nothing, i:min(i+63, length(ids)))
            p = model.predict(x)

            for j in 1:nrow(p)
                println(fout, ids[i+j-1], ',', join(sigmoid.(p[j, :]), ','))
            end
        end
    end
end
