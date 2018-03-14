using OhMyJulia
using LightGBM

const y = map(Int, readdlm("D:/jigsaw-toxic-comment-classification-challenge/raw/train.csv", ','; skipstart=1)[:, 3:end])
const ids = readdlm("D:/jigsaw-toxic-comment-classification-challenge/raw/test.csv", ',', String; skipstart=1)[:, 1]

const X_train, X_test = let
    feat_char_cnn = readdlm("D:/jigsaw-toxic-comment-classification-challenge/result/char-cnn.feature.csv", ',', f64)
    feat_char_dilation = readdlm("D:/jigsaw-toxic-comment-classification-challenge/result/char-dilation.feature.csv", ',', f64)
    feat_tf_idf = readdlm("D:/jigsaw-toxic-comment-classification-challenge/result/tf-idf.feature.csv", ',', f64)
    feat_word_cnn = readdlm("D:/jigsaw-toxic-comment-classification-challenge/result/word-cnn.feature.csv", ',', f64)
    feat_word_rnn = readdlm("D:/jigsaw-toxic-comment-classification-challenge/result/word-rnn.feature.csv", ',', f64)
    X = hcat(feat_char_cnn, feat_char_dilation, feat_tf_idf, feat_word_cnn, feat_word_rnn)
    X[1:nrow(y), :], X[nrow(y)+1:end, :]
end

const p = []

for i in 1:6
    estimator = LGBMBinary(
        num_iterations = 60,
        learning_rate = .1,
        feature_fraction = .8,
        bagging_fraction = .9,
        bagging_freq = 1,
        is_unbalance = true,
        metric = ["auc"]
    )

    fit(estimator, X_train, y[:, i])
    push!(p, predict(estimator, X_test))
end

open("D:/jigsaw-toxic-comment-classification-challenge/result/blend-lgbm.submission.csv", "w") do fout
    fout << "id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n"
    for i in 1:length(p[1])
        fout << ids[i]
        for j in 1:6
            fout << ',' << p[j][i]
        end
        fout << '\n'
    end
end
