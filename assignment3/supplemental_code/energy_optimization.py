import torch
from face_swapping import *
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm

class EnergyOptim(nn.Module):
    def __init__(self):

        super(EnergyOptim, self).__init__()
       
        self.alpha = nn.Parameter(torch.FloatTensor(30).uniform_(-1, 1), requires_grad=True)
        self.delta = nn.Parameter(torch.FloatTensor(20).uniform_(-1, 1), requires_grad=True)
        self.Rots = nn.Parameter(torch.FloatTensor([0, 0, 0]), requires_grad=True)
        self.t = nn.Parameter(torch.FloatTensor([0, 0, -500]), requires_grad=True)
    
    def forward(self, bfm):
        face_3D, _, _ = find_g(bfm, self.alpha, self.delta)
        face_2D = pinhole(face_3D, self.Rots, self.t)
        pred = get_landmarks(face_2D)

        return(pred)


def loss_c(landmarks, pred, alpha, delta, l_alpha=0, l_delta=0):
    Llan = torch.sum(torch.square(pred - landmarks)) / 68
    Lreg = l_alpha * torch.sum(torch.square(alpha)) + l_delta * torch.sum(torch.square(delta))
    return (Llan + Lreg)


def train_model(model, bfm, gt, iters=100):
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.Adam([{'params': model.Rots,  'lr': 1},
                                  {'params': model.t,     'lr': 1},
                                  {'params': model.alpha, 'lr': 0.1},
                                  {'params': model.delta, 'lr': 0.1}])
    losses = []
    for _ in range(iters):
        pred = model(bfm)
        loss = loss_c(gt, pred, model.alpha, model.delta)
        print('loss:', loss)
        
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (iter > ARGS.patience*2):
        #     old = int(np.mean(losses[-ARGS.patience*2:-ARGS.patience]))
        #     new = int(np.mean(losses[-ARGS.patience:]))
        #     if old <= new:
        #         break

    plt.plot(losses)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()

    return(model, losses)

#Load bfm
bfm = h5py.File('model2017-1_face12_nomouth.h5', 'r')
#Get gt landmarks
img = cv2.imread('beyonce.jpg')[:,:,::-1]
gt_landmarks = detect_landmark(img)

model = EnergyOptim()
# for param in model.parameters():
#     print(param)
model, losses = train_model(model, bfm, torch.from_numpy(gt_landmarks))