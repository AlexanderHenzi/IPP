import torch
import numpy as np
from engression.loss_func import energy_loss_two_sample

""" For training """

def logS(y, mu, logvar):
    """log score for one environment

    Args:
        data (_type_): _description_
        y_idx (int, optional): _description_. Defaults to -1.
    """
    nll = ((y - mu)**2 / (2 * torch.exp(2 * logvar)) + logvar).mean() + np.log(2*np.pi)/2
    return nll

def get_x_y(data, y_idx=-1):
    y = data[:, y_idx].unsqueeze(1)
    if y_idx == -1:
        x = data[:, :-1]
    else:
        x = torch.cat([data[:, :y_idx], data[:, (y_idx+1):]], dim=1)
    return x, y

def loss_logS(data, model, lam=1, y_idx=-1):
    losses = []
    for data_e in data:
        x, y = get_x_y(data_e, y_idx)
        mu, logvar = model(x)
        loss = logS(y, mu, logvar)
        losses.append(loss.unsqueeze(0))
    losses = torch.cat(losses)
    final_loss = losses.mean() + lam * torch.var(losses)
    return final_loss

def loss_crps(data, model, lam=1, y_idx=-1):
    losses = []
    for data_e in data:
        loss = loss_crps1(data_e, model, y_idx)
        losses.append(loss.unsqueeze(0))
    losses = torch.cat(losses)
    loss_pen = torch.var(losses)
    final_loss = losses.mean() + lam * loss_pen
    return final_loss

def loss_crps1(data, model, y_idx=-1):
    x, y = get_x_y(data, y_idx)
    gen1 = model(x)
    gen2 = model(x)
    loss = energy_loss_two_sample(y, gen1, gen2, verbose=False)
    return loss

def loss_scrps(data, model, lam=1, y_idx=-1):
    losses = []
    for data_e in data:
        loss = loss_scrps1(data_e, model, y_idx)
        losses.append(loss.unsqueeze(0))
    losses = torch.cat(losses)
    loss_pen = torch.var(losses)
    final_loss = losses.mean() + lam * loss_pen
    return final_loss

def loss_scrps1(data, model, y_idx=-1):
    x, y = get_x_y(data, y_idx)
    gen1 = model(x)
    gen2 = model(x)
    score_p_phat = energy_loss_two_sample(y, gen1, gen2)
    score_phat = energy_loss_two_sample(gen1, gen2, gen2) / 2
    loss = (1 + score_p_phat / score_phat + torch.log(2 * score_phat)) / 2
    return loss

def loss_vrex(data, model, lam=1, y_idx=-1):
    losses = []
    for data_e in data:
        x, y = get_x_y(data_e, y_idx)
        y_pred = model(x)
        loss = (y_pred - y).pow(2).mean()
        losses.append(loss.unsqueeze(0))
    losses = torch.cat(losses)
    final_loss = losses.mean() + lam * torch.var(losses)
    return final_loss

    
""" For test """

def test_mse(data, model, target="mean", generative=False, y_idx=-1):
    """
    Test on a single dataset
    Arguments:
        data: a numpy array
        model: a model or a list of models
    """
    x, y = get_x_y(data, y_idx)
    
    with torch.no_grad():
        if target == "mean":
            if isinstance(model, list):
                predictions = []
                for i in range(len(model)):  
                    model_i = model[i]
                    model_i.eval()
                    if not generative:
                        y_pred = model_i(x)
                    else:
                        y_pred = model_i.predict(x)
                    y_pred = y_pred.unsqueeze(2)
                    predictions.append(y_pred)
                predictions = torch.cat(predictions, dim=2)
                y_pred = torch.mean(predictions, dim=2)
            else:
                model.eval()
                if not generative:
                    y_pred = model(x)
                else:
                    y_pred = model.predict(x)
            loss = ((y - y_pred)**2).mean().item()
        elif target == "crps":
            if isinstance(model, list):
                losses = []
                for i in range(len(model)):
                    model[i].eval()
                    losses.append(loss_crps1(data, model[i]).item())
                loss = np.mean(np.array(losses))
            else:
                model.eval()
                loss = loss_crps1(data, model).item()
        else:
            assert target == "scrps"
            if isinstance(model, list):
                losses = []
                for i in range(len(model)):
                    model[i].eval()
                    losses.append(loss_scrps1(data, model[i]).item())
                loss = np.mean(np.array(losses))
            else:
                model.eval()
                loss = loss_scrps1(data, model).item()
    return loss

def test_mse_list(data, model, target="mean", pooled=False, stats_only=False, generative=False, y_idx=-1):
    """
    Test on multiple datasets
    Arguments:
        data: list of numpy arrays
        pooled: MSE on pooled data
    """
    errors = []
    for i in range(len(data)):
        errors.append(test_mse(data[i], model, target, generative, y_idx))
    if pooled:
        errors.append(test_mse(torch.cat(data), model, target, generative, y_idx))
    if stats_only:
        return np.mean(errors), np.std(errors), np.max(errors)
    else:
        return errors
