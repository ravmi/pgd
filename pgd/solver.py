import torch as ch
import torch
from torch.nn import BCEWithLogitsLoss
from pgd.steps import LinfStep, L2Step


#def get_loss(backbone, h, successful, idxs, pos_weight):

## x
# bind backbone, successful, idxs, pos_weight
# TODO change net into interference mode



class PGD:
    def __init__(self,
                 step_type: str,
                 iterations: int,
                 eps: float,
                 step_size: float,
                 repeats: int = 1,
                 random_start: bool = False):

        self.step_type = step_type
        self.iterations = iterations
        self.eps = eps
        self.step_size = step_size
        self.repeats = repeats
        self.random_start = random_start



        self.step_type = step_type #either l2 or inf
    def _replace_best(self, loss, bloss, x, bx):
        if bloss is None:
            bx = x.clone().detach()
            bloss = loss.clone().detach()
        else:
            replace = bloss < loss
            bx[replace] = x[replace].clone().detach()
            bloss[replace] = loss[replace]

        return bloss, bx

    def get_adv_examples(self, calc_loss, initial_x, mask=None):

        if self.step_type == "inf":
            self.step = LinfStep(initial_x.detach().clone(), self.eps, self.step_size)
        elif self.step_type == "l2":
            self.step = L2Step(initial_x.detach().clone(), self.eps, self.step_size)
        else:
            raise NotImplementedError


        best_loss = None
        best_x = None

        x_history = list()
        loss_history = list()
        grad_history = list()

        for _ in range(self.repeats):
            x = initial_x

            if self.random_start:
                x = self.step.random_perturb(x)
                #raise ValueError("You shouldn't use this, it may change the image too much ;(")


            for it in range(self.iterations):
                x = x.detach().clone().requires_grad_(True)
                losses = calc_loss(x)
                assert losses.shape[0] == x.shape[0], f'Inconsistent batch size between loss and inp! {losses.shape[0]}, {x.shape[0]}'
                loss = ch.mean(losses)

                grad, = ch.autograd.grad(loss, [x])
                if mask is not None:
                    grad = grad * mask
                grad_history.append(grad.clone().detach().cpu())

                x_history.append(x.clone().detach().cpu())
                loss_history.append(loss.item())
                #print('loss: ', loss.item())


                with ch.no_grad():
                    args = [losses, best_loss, x, best_x]
                    best_loss, best_x = self._replace_best(*args)

                    x = self.step.step(x, grad)
                    x = self.step.project(x)

            x_history.append(x.clone().detach().cpu())
            loss_history.append(calc_loss(x).mean())
            args = [losses, best_loss, x, best_x]
            best_loss, best_x = self._replace_best(*args)

        return best_x, best_loss.mean().item(), loss_history, grad_history

# print requires grad here from the model
