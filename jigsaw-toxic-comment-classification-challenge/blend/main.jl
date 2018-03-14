using ProgressMeter
using OhMyJulia
using PyCall
using Fire

unshift!(PyVector(pyimport("sys")["path"]), @__DIR__)

sigmoid(x) = e^x / (e^x + 1)

function get_batch(X, y, batch=1024)
    rs = isa(batch, Int) ? rand(1:nrow(X), batch) : batch
    X[rs, :], isa(y, Void) ? y : y[rs, :]
end

function read_data()
    ids = readdlm("D:/jigsaw-toxic-comment-classification-challenge/raw/test.csv", ',', String; skipstart=1)[:, 1]
    y = map(Int, readdlm("D:/jigsaw-toxic-comment-classification-challenge/raw/train.csv", ','; skipstart=1)[:, 3:end])

    feat_char_cnn = readdlm("D:/jigsaw-toxic-comment-classification-challenge/result/char-cnn.feature.csv", ',', f64)
    feat_char_dilation = readdlm("D:/jigsaw-toxic-comment-classification-challenge/result/char-dilation.feature.csv", ',', f64)
    feat_tf_idf = readdlm("D:/jigsaw-toxic-comment-classification-challenge/result/tf-idf.feature.csv", ',', f64)
    feat_word_cnn = readdlm("D:/jigsaw-toxic-comment-classification-challenge/result/word-cnn.feature.csv", ',', f64)
    feat_word_rnn = readdlm("D:/jigsaw-toxic-comment-classification-challenge/result/word-rnn.feature.csv", ',', f64)
    X = hcat(feat_char_cnn, feat_char_dilation, feat_tf_idf, feat_word_cnn, feat_word_rnn)

    ids, y, X[1:nrow(y), :], X[nrow(y)+1:end, :]
end

@main function main(epoch::Int=60)
    ids, y, X_train, X_test = read_data()

    model = pywrap(pyimport("model")[:Model]("gpu"))

    for i in 1:epoch
        tic()
        loss = 0

        for j in 1:200
            batch = get_batch(X_train, y)
            loss += model.train(batch...) / 200
        end

        println("=== epoch: $i, loss: $loss, time: $(toq()) ===")
    end

    open("D:/jigsaw-toxic-comment-classification-challenge/result/blend-lr.submission.csv", "w") do fout
        fout << "id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n"
        @showprogress for i in 1:64:length(ids)
            x, _ = get_batch(X_test, nothing, i:min(i+63, length(ids)))
            p = model.predict(x)

            for j in 1:nrow(p)
                println(fout, ids[i+j-1], ',', join(sigmoid.(p[j, :]), ','))
            end
        end
    end
end
