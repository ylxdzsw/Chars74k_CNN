using DataFrames
using OhMyJulia
using GadFly

population = readtable("age_gender_bkts.csv")
country = readtable("countries.csv")
train = readtable("train_users.csv")
#session = readtable("sessions.csv")
