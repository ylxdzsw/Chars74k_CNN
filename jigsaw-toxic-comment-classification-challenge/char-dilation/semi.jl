using ProgressMeter
using OhMyJulia
using PyCall
using Fire

unshift!(PyVector(pyimport("sys")["path"]), @__DIR__)

sigmoid(x) = e^x / (e^x + 1)

const data_train, data_test = let
    train = readdlm("D:/jigsaw-toxic-comment-classification-challenge/raw/train.csv", ','; skipstart=1)
    test = readdlm("D:/jigsaw-toxic-comment-classification-challenge/raw/test.csv", ',', String; skipstart=1)
    label = readdlm("D:/jigsaw-toxic-comment-classification-challenge/clean/pseudo-label.csv", ',', String; skipstart=1)
    dict = Dict(label[i, 1]=>i for i in 1:nrow(label))
    additional = Array{Any}(nrow(label), 8)
    for i in 1:nrow(test)
        if test[i, 1] in keys(dict)
            j = dict[test[i, 1]]
            additional[j, 1:2] = test[i, :]
            additional[j, 3:end] = parse.(Int, label[j, 2:end])
        end
    end
    vcat(train, additional), test
end

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

function get_batch(X, y, batch)
    rs = isa(batch, Int) ? rand(1:nrow(X), batch) : batch
    X_filled = Array{Int}(length(rs), 500)
    for (i, r) in enumerate(rs)
        l = length(X[r])
        if l > 500
            s = rand(1:l-499) # randomly crop 1000 letters, which I think it's sufficient
            X_filled[i, :] = X[r][s:s+499]
        else
            X_filled[i, 1:l] = X[r]
            X_filled[i, l+1:end] = 0 # fill with special token
        end
    end
    X_filled, isa(y, Void) ? y : y[rs, :]
end

function get_dict()
    encode(data_train[:, 2])[2]
end

function drop_out!(X)
    d = rand(1:length(X), rand(0:floor(Int, .05length(X))))
    X[d] .= X[d] .!= 0 # set to 1 iff it's not 0
end

@main function train(epoch::Int=100)
    X, y = data_train[:, 2], map(Int, data_train[:, 3:end])
    X, V = encode(X)

    model = pywrap(pyimport("model")[:Model](length(V)+2, "gpu"))

    for i in 1:epoch
        tic()
        loss = 0

        for j in 1:250
            batch = get_batch(X, y, i <= 60 ? 64 : 256)
            drop_out!(car(batch))
            loss += model.train(batch...) / 250
        end

        println("=== epoch: $i, loss: $loss, time: $(toq()) ===")
        i % 20 == 0 && model.save("D:/jigsaw-toxic-comment-classification-challenge/result/semi-char-dilation.model")
    end
end

@main function predict()
    ids, X, V = data_test[:, 1], data_test[:, 2], get_dict()
    X = encode(X, V)

    model = pywrap(pyimport("model")[:Model](length(V)+2, "gpu", "D:/jigsaw-toxic-comment-classification-challenge/result/semi-char-dilation.model"))

    open("D:/jigsaw-toxic-comment-classification-challenge/result/semi-char-dilation.submission.csv", "w") do fout
        fout << "id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n"
        @showprogress for i in 1:64:length(ids)
            x, _ = get_batch(X, nothing, i:min(i+63, length(ids)))
            _, p = model.predict(x)

            for j in 1:nrow(p)
                println(fout, ids[i+j-1], ',', join(sigmoid.(p[j, :]), ','))
            end
        end
    end
end

@main function feature()
    X, V = data_train[:, 2] ++ data_test[:, 2], get_dict()
    X = encode(X, V)

    model = pywrap(pyimport("model")[:Model](length(V)+2, "gpu", "D:/jigsaw-toxic-comment-classification-challenge/result/semi-char-dilation.model"))

    open("D:/jigsaw-toxic-comment-classification-challenge/result/semi-char-dilation.feature.csv", "w") do fout
        @showprogress for i in 1:64:length(X)
            x, _ = get_batch(X, nothing, i:min(i+63, length(X)))
            f, _ = model.predict(x)

            for j in 1:nrow(f)
                println(fout, join(f[j, :], ','))
            end
        end
    end
end
