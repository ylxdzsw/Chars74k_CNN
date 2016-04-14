using DataFrames
using PyCall

@pyimport sklearn.ensemble as skes
@pyimport sklearn.cross_validation as skcv

features = [:imp_op_var39_efect_ult1,:ind_var13_corto_0,:ind_var20,:ind_var26_0,:ind_var26,:ind_var41_0,:ind_var44_0,:ind_var44,:num_var1,:num_var13_corto_0,:num_var13_largo,:num_var14_0,:num_var17,:num_var20,:num_var26_0,:num_var26,:num_op_var40_ult3,:num_var30_0,:num_var31,:num_var33_0,:num_var33,:num_var44_0,:num_var44,:saldo_var1,:saldo_var13,:saldo_var17,:saldo_var18,:saldo_var20,:saldo_var24,:saldo_var31,:saldo_var33,:saldo_var34,:delta_imp_aport_var33_1y3,:delta_imp_reemb_var33_1y3,:delta_imp_trasp_var17_in_1y3,:delta_imp_trasp_var17_out_1y3,:delta_imp_trasp_var33_in_1y3,:delta_imp_trasp_var33_out_1y3,:delta_num_aport_var33_1y3,:delta_num_reemb_var33_1y3,:delta_num_trasp_var17_in_1y3,:delta_num_trasp_var17_out_1y3,:delta_num_trasp_var33_in_1y3,:delta_num_trasp_var33_out_1y3,:imp_amort_var18_ult1,:imp_amort_var34_ult1,:imp_aport_var17_hace3,:imp_aport_var17_ult1,:imp_aport_var33_hace3,:imp_aport_var33_ult1,:imp_compra_var44_hace3,:imp_compra_var44_ult1,:imp_reemb_var17_hace3,:imp_reemb_var33_ult1,:imp_trasp_var17_in_hace3,:imp_trasp_var17_in_ult1,:imp_trasp_var17_out_ult1,:imp_trasp_var33_in_hace3,:imp_trasp_var33_in_ult1,:imp_trasp_var33_out_ult1,:imp_venta_var44_hace3,:num_aport_var17_hace3,:num_aport_var17_ult1,:num_aport_var33_hace3,:num_aport_var33_ult1,:num_compra_var44_hace3,:num_meses_var29_ult3,:num_meses_var33_ult3,:num_op_var40_comer_ult3,:num_reemb_var17_hace3,:num_reemb_var33_ult1,:num_trasp_var17_in_hace3,:num_trasp_var17_in_ult1,:num_trasp_var17_out_ult1,:num_trasp_var33_in_hace3,:num_trasp_var33_in_ult1,:num_trasp_var33_out_ult1,:num_venta_var44_hace3,:num_venta_var44_ult1,:saldo_medio_var13_largo_hace2,:saldo_medio_var13_largo_hace3,:saldo_medio_var13_largo_ult1,:saldo_medio_var13_largo_ult3,:saldo_medio_var17_hace2,:saldo_medio_var17_hace3,:saldo_medio_var29_hace2,:saldo_medio_var29_hace3,:saldo_medio_var29_ult3,:saldo_medio_var33_hace2,:saldo_medio_var33_hace3,:saldo_medio_var33_ult1,:saldo_medio_var33_ult3,:saldo_medio_var44_hace3]
df_train = readtable("train.csv")
df_test  = readtable("test.csv")

trainX = df_train[features]|>DataArray
testX  = df_test[features]|>DataArray
trainY = df_train[:TARGET]


model = skes.ExtraTreesClassifier(n_estimators = 100)

a = skcv.cross_val_score(model, trainX, trainY, scoring="roc_auc", cv=3)

model[:fit](trainX, trainY)
testY = model[:predict_proba](testX)

submit = readtable("sample_submission.csv")
submit[:TARGET] = testY[:,2]
writetable("submission.csv", submit)
