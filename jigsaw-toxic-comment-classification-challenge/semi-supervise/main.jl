using ProgressMeter
using OhMyJulia
using Fire

@main function main()
    ids      = readdlm("D:/jigsaw-toxic-comment-classification-challenge/raw/test.csv", ',', String; skipstart=1)[:, 1]
    char_cnn = readdlm("D:/jigsaw-toxic-comment-classification-challenge/result/char-cnn.submission.csv", ',', String, skipstart=1)
    tf_idf   = readdlm("D:/jigsaw-toxic-comment-classification-challenge/result/tf-idf.submission.csv", ',', String, skipstart=1)
    word_cnn = readdlm("D:/jigsaw-toxic-comment-classification-challenge/result/word-cnn.submission.csv", ',', String, skipstart=1)
    word_rnn = readdlm("D:/jigsaw-toxic-comment-classification-challenge/result/word-rnn.submission.csv", ',', String, skipstart=1)

    open("D:/jigsaw-toxic-comment-classification-challenge/clean/pseudo-label.csv", "w") do fout
        @showprogress for i in 1:nrow(ids)
            p1 = parse.(f64, char_cnn[i, 2:end])
            p2 = parse.(f64, tf_idf[i, 2:end])
            p3 = parse.(f64, word_cnn[i, 2:end])
            p4 = parse.(f64, word_rnn[i, 2:end])

            pa = p1 .+ p2 .+ p3 .+ p4
            pa = [x < .01 ? 0 : x > 2.5 ? 1 : -1 for x in pa]

            any(x->x==-1, pa) && continue

            println(fout, ids[i], ',', join(pa, ','))
        end
    end
end
