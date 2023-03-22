import torch as ch
import torch
from torch.nn import BCEWithLogitsLoss
from .steps import LinfStep, L2Step


#def get_loss(backbone, h, successful, idxs, pos_weight):

## x
# bind backbone, successful, idxs, pos_weight
# TODO change net into interference mode
"""
def calc_loss(h):
    affordances = backbone(h, inference=False)

    affordance = torch.stack([
        affordances[i, :, x, y, a]
        for i, x, y, a in zip(range(len(idxs["row"])), list(idxs["row"]), list(idxs["col"]), list(idxs["angle_index"]))
    ])

    losses = BCEWithLogitsLoss(reduction='none', pos_weight=torch.ones(1).to(affordance.device) * pos_weight)(
        affordance, successful)

    losses = torch.mean(losses, dim=list(range(1, losses.ndim)))

    return losses

    pass
"""


class PGD:
    def __init__(self,
                 step_type: str,
                 iterations: int,
                 eps: float,
                 step_size: float,
                 random_start: bool = False):
        """
        This class has only configurations, the function that calculates loss is defined
        separately for each call to get_adv_examples.
        """

        self.iterations = 30#iterations#10
        self.random_start = random_start
        self.eps = 300#eps# 20
        self.step_size = 2#step_size
        self.repeats = 1


        self.step_type = step_type #either l2 or inf
        if self.step_type == "inf":
            self.step = LinfStep(x.detach(), self.eps, self.step_size)
        elif self.step_type == "l2":
            self.step = L2Step(x.detach(), self.eps, self.step_size)
        else:
            raise NotImplementedError

    def _replace_best(self, loss, bloss, x, bx):
        #import pdb; pdb.set_trace()
        if bloss is None:
            bx = x.clone().detach()
            bloss = loss.clone().detach()
        else:
            replace = bloss < loss
            ##print(bx.shape, x.shape)
            #print(replace.shape)
            #prev_shape = bx.shape
            #x = x.view(bx.shape[0], -1)
            bx[replace] = x[replace].clone().detach()
            #bx = bx.view(prev_shape)
            bloss[replace] = loss[replace]

        return bloss, bx

    def get_adv_examples(self, calc_loss, initial_x, mask=None, log_images=False):
        """
        calc_loss should take x and give loss
        """
        best_loss = None
        best_x = None

        x_history = list()
        loss_history = list()
        grad_history = list()

        for _ in range(self.repeats):
            x = initial_x

            if self.random_start:
                x = step.random_perturb(x)


            for it in range(self.iterations):
                #print(it)
                x = x.detach().clone().requires_grad_(True)
                losses = calc_loss(x)
                assert losses.shape[0] == x.shape[0], 'Inconsistent batch size between input and loss!'
                loss = ch.mean(losses)

                grad, = ch.autograd.grad(loss, [x])
                grad_history.append(grad.clone().detach().cpu())
                #import pdb; pdb.set_trace()

                x_history.append(x.clone().detach().cpu())
                loss_history.append(loss.item())


                with ch.no_grad():
                    args = [losses, best_loss, x, best_x]
                    best_loss, best_x = self._replace_best(*args)
                    #import pdb; pdb.set_trace()

                    x = step.step(x, grad)
                    x = step.project(x)

            x_history.append(x.clone().detach().cpu())
            loss_history.append(calc_loss(x).mean())
            args = [losses, best_loss, x, best_x]
            best_loss, best_x = self._replace_best(*args)

        #print(loss_history)
        #print(loss_history[0], best_loss.item())
        #print(best_loss.item())

        return best_x, x_history, loss_history, grad_history
