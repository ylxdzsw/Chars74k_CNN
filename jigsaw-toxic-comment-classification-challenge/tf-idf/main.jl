using ProgressMeter
using OhMyJulia
using PyCall
using FileIO
using Fire
using JLD2

unshift!(PyVector(pyimport("sys")["path"]), @__DIR__)

sigmoid(x) = e^x / (e^x + 1)

function get_batch(X, y, batch)
    rs = isa(batch, Int) ? rand(1:X[:shape][1], batch) : batch
    py"$X[$(rs.-1), :].todense()", isa(y, Void) ? y : y[rs, :]
end

function read_data()
    train = readdlm("D:/jigsaw-toxic-comment-classification-challenge/raw/train.csv", ','; skipstart=1)
    test = readdlm("D:/jigsaw-toxic-comment-classification-challenge/raw/test.csv", ',', String; skipstart=1)
    y, ids = map(Int, train[:, 3:end]), test[:, 1]
    X = transform_tfidf(train[:, 2] ++ test[:, 2])

    s = nrow(train)

    X_train, X_test = py"$X[0:$s, :]", py"$X[$s:, :]"
    X_train, X_test, X, y, ids
end

function transform_tfidf(X)
    @pyimport sklearn.feature_extraction.text as text
    @pyimport scipy.sparse as scisparse

    Vectorizer = text.TfidfVectorizer

    vectorizer = pywrap(Vectorizer(
        sublinear_tf=true,
        strip_accents="unicode",
        analyzer="word",
        token_pattern="\\w{1,}",
        stop_words="english",
        max_df=0.95,
        ngram_range=(1, 1),
        max_features=10000))
    X_word = vectorizer.fit_transform(X)

    vectorizer = pywrap(Vectorizer(
        sublinear_tf=true,
        strip_accents="unicode",
        analyzer="char",
        stop_words="english",
        max_df=0.95,
        ngram_range=(3, 6),
        max_features=40000))
    X_char = vectorizer.fit_transform(X)

    scisparse.hstack([X_word, X_char])[:tocsr]()
end

@main function main(epoch::Int=80)
    X_train, X_test, X_all, y, ids = read_data()

    model = pywrap(pyimport("model")[:Model]("gpu"))

    for i in 1:epoch
        tic()
        loss = 0

        for j in 1:250
            batch = get_batch(X_train, y, i <= 60 ? 64 : 256)
            loss += model.train(batch...) / 250
        end

        println("=== epoch: $i, loss: $loss, time: $(toq()) ===")
    end

    open("D:/jigsaw-toxic-comment-classification-challenge/result/tf-idf.submission.csv", "w") do fout
        fout << "id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n"
        @showprogress for i in 1:64:length(ids)
            x, _ = get_batch(X_test, nothing, i:min(i+63, length(ids)))
            _, p = model.predict(x)

            for j in 1:nrow(p)
                println(fout, ids[i+j-1], ',', join(sigmoid.(p[j, :]), ','))
            end
        end
    end

    open("D:/jigsaw-toxic-comment-classification-challenge/result/tf-idf.feature.csv", "w") do fout
        @showprogress for i in 1:64:X_all[:shape][1]
            x, _ = get_batch(X_all, nothing, i:min(i+63, X_all[:shape][1]))
            f, _ = model.predict(x)

            for j in 1:nrow(f)
                println(fout, join(f[j, :], ','))
            end
        end
    end
end
