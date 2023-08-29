import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions.ipp_nn import *
import seaborn as sns
from engression.models import StoNet, Net
import random

torch.manual_seed(222)
random.seed(222)
np.random.seed(222)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


## training data
df = pd.read_csv("data/dataset_rpe1.csv")
interv_label = df["interventions"]
groups = df.groupby(["interventions"])
df.iloc[:,:-1] = df.iloc[:,:-1] - groups.get_group("non-targeting").iloc[:,:-1].mean()
## observational data
data_obs = torch.Tensor(groups.get_group("non-targeting").iloc[:,:-1].to_numpy()).to(device)
## interventional data
interv_obs = df.columns[:-1].to_numpy()
interv_hidden = set(interv_label) - set(interv_obs) - set(["non-targeting"])
data_interv_obs = []
data_interv_hidden = []
for interv_i in interv_obs:
    data_interv_obs.append(torch.Tensor(groups.get_group(interv_i).iloc[:,:-1].to_numpy()).to(device))
for interv_i in interv_hidden:
    data_interv_hidden.append(torch.Tensor(groups.get_group(interv_i).iloc[:,:-1].to_numpy()).to(device))
data_interv_obs = dict(zip(interv_obs, data_interv_obs))
data_interv_hidden = dict(zip(interv_hidden, data_interv_hidden))
train_data = [data_obs] + list(data_interv_obs.values())[:-1]

## test data
data_test_large_shift_gene = list(pd.read_csv("single_cell_test_envs.csv")['genes'])
data_test = []
for i in range(len(data_interv_hidden)):
    if list(data_interv_hidden.keys())[i] in data_test_large_shift_gene:
        data_test.append(list(data_interv_hidden.values())[i])


## Fitting IPP 
lams = np.linspace(0, 100, 101)
num_envs = len(data_test)
models_all_lams = []
for lam in lams:
    print(lam)
    models_each_lam = []
    for _ in range(10):
        # repeat for 10 times 
        model = StoNet(in_dim=9, out_dim=1, num_layer=3, hidden_dim=400, noise_dim=400).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for i in range(1000):
            model.zero_grad()
            loss = loss_scrps(train_data, model, lam=lam)
            loss.backward()
            optim.step()
        
        models_each_lam.append(model)
    
    models_all_lams.append(models_each_lam)
    
    if lam == 50:
        torch.save(models_all_lams, "results/models_ipp_nn.pt")

## save models
torch.save(models_all_lams, "results/models_ipp_nn.pt")


## Evaluation
# MSE
results = None
for i in range(len(lams)):
    lam = lams[i]
    model = models_all_lams[i]
    
    mses = test_mse_list(data_test, model, generative=True)
    result = pd.DataFrame({
        "lambda": np.repeat(lam, num_envs),
        "test_mse": np.array(mses)
    })
    results = result if results is None else pd.concat([results, result], ignore_index=True)
results.to_csv("results/results_mse_ipp_nn.csv")
sns.lineplot(data=results, x="lambda", y="test_mse", estimator="median", errorbar=("pi", 100))
plt.savefig("results/ipp_nn_mse.pdf", bbox_inches="tight")
plt.close()

# SCRPS
results_scrps = None
for i in range(len(lams)):
    lam = lams[i]
    model = models_all_lams[i]
    
    mses = test_mse_list(data_test, model, generative=True, target="scrps")
    result = pd.DataFrame({
        "lambda": np.repeat(lam, num_envs),
        "test_scrps": np.array(mses)
    })
    results_scrps = result if results_scrps is None else pd.concat([results_scrps, result], ignore_index=True)
results_scrps.to_csv("results/results_mse_ipp_nn.csv")
sns.lineplot(data=results_scrps, x="lambda", y="test_scrps", estimator="median", errorbar=("pi", 100))
plt.savefig("results/ipp_nn_scrps.pdf", bbox_inches="tight")
plt.close()
