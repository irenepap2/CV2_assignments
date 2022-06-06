from face_swapping import *
from energy_optimization import *

PATH = './'

bfm = h5py.File(PATH + 'model2017-1_face12_nomouth.h5', 'r')
img1 = cv2.imread(PATH + 'beyonce.jpg')[:,:,::-1]

# 4.2.1 Morphable model
G, color, shape_rep = find_g(bfm)


# 4.2.2 Pinhole camera model
rots = torch.FloatTensor([0, 0, 180])
t = torch.FloatTensor([0, 0, -500])
G_2D = pinhole(G, rots, t, img1.shape[0], img1.shape[1])


# 4.2.3 Latent parameter estimation
gt_landmarks = detect_landmark(img1)
gt_landmarks[:, 0] = gt_landmarks[:, 0] - w/2
gt_landmarks[:, 1] = gt_landmarks[:, 1] - h/2

model = EnergyOptim()
model, losses = train_model(model, bfm, torch.from_numpy(gt_landmarks), h, w)
print(model.alpha, "\n")
print(model.delta, "\n")
print(model.Rots, "\n")
print(model.t, "\n")

_, _, pred_landmarks, _ = model(bfm, h, w)
# plt.scatter(pred_landmarks[:,0], pred_landmarks[:,1], color='red', label='prediction')
# plt.scatter(gt_landmarks[:,0], gt_landmarks[:,1], color='blue', label='ground truth')
# plt.legend()
# plt.show()


# 4.2.4 Texturing
_, textured = texturing(img1, model, bfm, h, w)
# plt.imshow(textured)
# plt.show()


# 4.2.5 Energy optimization using multiple frames
img2 = cv2.imread(PATH + 'beyonce2.png')[:,:,::-1]
img3 = cv2.imread(PATH + 'beyonce3.png')[:,:,::-1]

multiple_frames([img1, img2, img3], bfm)


# 4.2.6 Face swapping
swift = cv2.imread(PATH + 'swift.jpg')[:,:,::-1]
swap = faceswap(img1, swift, bfm)
# plt.imshow(swap)
# plt.show()