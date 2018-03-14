using ProgressMeter
using OhMyJulia
using PyCall
using FileIO
using Fire
using JLD2

unshift!(PyVector(pyimport("sys")["path"]), @__DIR__)

sigmoid(x) = e^x / (e^x + 1)

function get_batch(X, y, batch)
    rs = isa(batch, Int) ? rand(1:nrow(X), batch) : batch
    X_filled = Array{Int}(length(rs), 128)
    for (i, r) in enumerate(rs)
        l = length(X[r])
        if l > 128
            s = rand(1:l-127)
            X_filled[i, :] = X[r][s:s+127]
        else
            X_filled[i, 1:l] = X[r]
            X_filled[i, l+1:end] = 0
        end
    end
    X_filled, isa(y, Void) ? y : y[rs, :]
end

function drop_out!(X)
    X[rand(1:length(X), rand(0:floor(Int, .1length(X))))] .= 0
end

function read_data()
    train = readdlm("D:/jigsaw-toxic-comment-classification-challenge/raw/train.csv", ','; skipstart=1)
    test = readdlm("D:/jigsaw-toxic-comment-classification-challenge/raw/test.csv", ',', String; skipstart=1)
    y, ids = map(Int, train[:, 3:end]), test[:, 1]
    X, M = transform_embedding(train[:, 2] ++ test[:, 2])

    X_train, X_test = X[1:nrow(train)], X[nrow(train)+1:end]
    X_train, X_test, M, y, ids
end

function transform_embedding(X)
    word_set = Set([car(split(line, ' ')) for line in eachline("D:/word-vec/crawl-300d-2M.vec")][2:end])

    word_list = Dict{String, Int}()
    for x in X, w in unique(matchall(r"[a-zA-Z0-9'\-\_\.]+|\S", x)) @when w in word_set
        word_list[w] = get(word_list, w, 0) + 1
    end

    word_list = sort(collect(word_list), by=x->cadr(x) * min(length(car(x))+5, 20))[end-32766:end]
    dict = Dict(car(x)=>i for (i, x) in enumerate(word_list))

    X_seq = [Int[get(dict, w, 0) for w in matchall(r"[a-zA-Z0-9'\-\_\.]+|\S", x)] for x in X]

    M, drop = zeros(f64, 32768, 300), true
    for line in eachline("D:/word-vec/crawl-300d-2M.vec")
        if drop
            drop = false
            continue
        end

        line = split(line)
        code = get(dict, car(line), 0)

        if code != 0
            M[code+1, :] = parse.(f64, cdr(line))
        end
    end

    X_seq, M
end

@main function train(epoch::Int=40)
    X, M, y, ids = if isfile("D:/jigsaw-toxic-comment-classification-challenge/clean/embedded.jld")
        load("D:/jigsaw-toxic-comment-classification-challenge/clean/embedded.jld", "X_train", "M", "y", "ids")
    else
        X_train, X_test, M, y, ids = read_data()
        @save "D:/jigsaw-toxic-comment-classification-challenge/clean/embedded.jld" X_train X_test M y ids
        X_train, M, y, ids
    end

    model = pywrap(pyimport("model")[:Model](M, "gpu"))

    for i in 1:epoch
        tic()
        loss = 0

        for j in 1:200
            batch = get_batch(X, y, i <= 30 ? 64 : 256)
            drop_out!(car(batch))
            loss += model.train(batch...) / 200
        end

        println("=== epoch: $i, loss: $loss, time: $(toq()) ===")
        i % 10 == 0 && model.save("D:/jigsaw-toxic-comment-classification-challenge/result/word-rnn.model")
    end
end

@main function predict()
    X, M, ids = load("D:/jigsaw-toxic-comment-classification-challenge/clean/embedded.jld", "X_test", "M", "ids")

    model = pywrap(pyimport("model")[:Model](M, "gpu", "D:/jigsaw-toxic-comment-classification-challenge/result/word-rnn.model"))

    open("D:/jigsaw-toxic-comment-classification-challenge/result/word-rnn.submission.csv", "w") do fout
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
    X_train, X_test, M = load("D:/jigsaw-toxic-comment-classification-challenge/clean/embedded.jld", "X_train", "X_test", "M")

    X = X_train ++ X_test

    model = pywrap(pyimport("model")[:Model](M, "gpu", "D:/jigsaw-toxic-comment-classification-challenge/result/word-rnn.model"))

    open("D:/jigsaw-toxic-comment-classification-challenge/result/word-rnn.feature.csv", "w") do fout
        @showprogress for i in 1:64:length(X)
            x, _ = get_batch(X, nothing, i:min(i+63, length(X)))
            f, _ = model.predict(x)

            for j in 1:nrow(f)
                println(fout, join(f[j, :], ','))
            end
        end
    end
end
