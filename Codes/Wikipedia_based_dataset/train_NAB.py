import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.init as init
from sklearn.metrics.pairwise import cosine_similarity
import scipy.integrate as integrate
from termcolor import cprint
from time import gmtime, strftime
import numpy as np
import argparse
import os
import random
import glob
import copy 
import json

from dataset import FeatDataLayer, LoadDataset_NAB
from models import _netD, _netG, _param

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('-- ', default='easy', type=str, help='the way to split train/test data: easy/hard')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--resume',  type=str, help='the model to resume')
parser.add_argument('--disp_interval', type=int, default=20)
parser.add_argument('--save_interval', type=int, default=200)
parser.add_argument('--evl_interval',  type=int, default = 10)
parser.add_argument('--epsilon', type=float, default = 0.15)
parser.add_argument('--unseen_start', type=int, default = 250)
parser.add_argument('--mode_change', type=int, default = 250)
parser.add_argument('--correlation_penalty', type=float, default = 0.15)

opt = parser.parse_args()

runing_parameters_logs = json.dumps(vars(opt), indent=4, separators=(',', ':'))

print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ':')))

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

""" hyper-parameter for training """
opt.GP_LAMBDA = 10      # Gradient penalty lambda
opt.CENT_LAMBDA  = 1
opt.REG_W_LAMBDA = 0.001
opt.REG_Wz_LAMBDA = 0.0001

opt.lr = 0.0001
opt.batchsize = 1000

""" hyper-parameter for testing"""
opt.nSample = 60  # number of fake feature for each class
opt.Knn = 20      # knn: the value of K
opt.manualSeed = 1086
opt.zeroshotbatchsize = 3000
epochs = 5000

print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

def train():
    param = _param()
    dataset = LoadDataset_NAB(opt)
    param.X_dim = dataset.feature_dim 
    
    data_layer = FeatDataLayer(dataset.labels_train, dataset.pfc_feat_data_train, dataset.seen_label_mapping, opt)
    result = Result()
    result_gzsl = Result()
    
    netG = _netG(dataset.text_dim, dataset.feature_dim).cuda()
    netG.apply(weights_init)
    print(netG)
    netD = _netD(dataset.train_cls_num + dataset.test_cls_num, dataset.feature_dim).cuda()
    netD.apply(weights_init)
    print(netD)

    exp_info = 'NAB_EASY' if opt.splitmode == 'easy' else 'NAB_HARD'
    exp_params = 'Eu{}_Rls{}_RWz{}'.format(opt.CENT_LAMBDA , opt.REG_W_LAMBDA, opt.REG_Wz_LAMBDA)

    out_dir  = 'out_' + str(opt.epsilon) + '/{:s}'.format(exp_info)
    out_subdir = 'out_' + str(opt.epsilon) + '/{:s}/{:s}'.format(exp_info, exp_params)

    if not os.path.exists('out_' + str(opt.epsilon)):
        os.mkdir('out_' + str(opt.epsilon))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_subdir):  
        os.mkdir(out_subdir)

    cprint(" The output dictionary is {}".format(out_subdir), 'red')
    log_dir  = out_subdir + '/log_{:s}.txt'.format(exp_info)

    with open(log_dir, 'a') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')
        f.write("Running Parameter Logs")
        f.write(runing_parameters_logs)
        
    start_step = 0

    if opt.splitmode != 'easy':
        epochs = 1000 

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            netG.load_state_dict(checkpoint['state_dict_G'])
            netD.load_state_dict(checkpoint['state_dict_D'])
            start_step = checkpoint['it']
            print(checkpoint['log'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    nets = [netG, netD]

    tr_cls_centroid = Variable(torch.from_numpy(dataset.tr_cls_centroid.astype('float32'))).cuda()
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))

    for it in range(start_step, epochs):
        if it > opt.mode_change: 
            train_text = Variable(torch.from_numpy(dataset.train_text_feature.astype('float32'))).cuda()
            test_text = Variable(torch.from_numpy(dataset.test_text_feature.astype('float32'))).cuda()
            z_train = Variable(torch.randn(dataset.train_cls_num, param.z_dim)).cuda()
            z_test = Variable(torch.randn(dataset.test_cls_num, param.z_dim)).cuda()
            
            _, train_text_feature = netG(z_train, train_text) 
            _, test_text_feature = netG(z_test, test_text) 

            dataset.semantic_similarity_check(opt.Knn, train_text_feature.data.cpu().numpy(), test_text_feature.data.cpu().numpy())

        """ Discriminator """
        for _ in range(5):
            blobs = data_layer.forward()
            feat_data = blobs['data']             # image data
            labels = blobs['labels'].astype(int)  # class labels
            true_labels = blobs['true_labels'].astype(int) 

            text_feat = np.array([dataset.train_text_feature[i,:] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).cuda()
            X = Variable(torch.from_numpy(feat_data)).cuda()
            y_true = Variable(torch.from_numpy(true_labels.astype('int'))).cuda()

            z = Variable(torch.randn(opt.batchsize, param.z_dim)).cuda()

            # GAN's D loss
            D_real, C_real = netD(X)
            D_loss_real = torch.mean(D_real)
            C_loss_real = F.cross_entropy(C_real, y_true)
            DC_loss = -D_loss_real + C_loss_real
            DC_loss.backward()

            # GAN's D loss
            G_sample, _ = netG(z, text_feat)
            D_fake, C_fake = netD(G_sample)
            D_loss_fake = torch.mean(D_fake)
            C_loss_fake = F.cross_entropy(C_fake, y_true)
            DC_loss = D_loss_fake + C_loss_fake
            DC_loss.backward()

            # train with gradient penalty (WGAN_GP)
            grad_penalty = calc_gradient_penalty(netD, X.data, G_sample.data)
            grad_penalty.backward()

            Wasserstein_D = D_loss_real - D_loss_fake
            optimizerD.step()
            reset_grad(nets)

        """ Generator """
        for _ in range(1):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            labels = blobs['labels'].astype(int)  # class labels
            true_labels = blobs['true_labels'].astype(int) #True seen label class  

            text_feat = np.array([dataset.train_text_feature[i, :] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).cuda()

            X = Variable(torch.from_numpy(feat_data)).cuda()
            y_dummy = Variable(torch.from_numpy(labels.astype('int'))).cuda()
            y_true = Variable(torch.from_numpy(true_labels.astype('int'))).cuda()

            z = Variable(torch.randn(opt.batchsize, param.z_dim)).cuda()

            G_sample, _ = netG(z, text_feat)
            D_fake, C_fake = netD(G_sample)
            _,      C_real = netD(X)

            # GAN's G loss
            G_loss = torch.mean(D_fake)
            # Auxiliary classification loss
            C_loss = (F.cross_entropy(C_real, y_true) + F.cross_entropy(C_fake, y_true))/2

            GC_loss = -G_loss + C_loss

            # Centroid loss
            Euclidean_loss = Variable(torch.Tensor([0.0])).cuda()
            Correlation_loss = Variable(torch.Tensor([0.0])).cuda() 

            if opt.CENT_LAMBDA != 0:
                for i in range(dataset.train_cls_num):
                    sample_idx = (y_dummy == i).data.nonzero().squeeze()
                    if sample_idx.numel() == 0:
                        Euclidean_loss += 0.0
                    else:
                        G_sample_cls = G_sample[sample_idx, :]
                        if sample_idx.numel() != 1:
                            generated_mean = G_sample_cls.mean(dim=0) 
                        else:
                            generated_mean = G_sample_cls
                        
                        Euclidean_loss += (generated_mean - tr_cls_centroid[i]).pow(2).sum().sqrt()

                        for n in range(dataset.Neighbours):                            
                            Neighbor_correlation = cosine_similarity(generated_mean.data.cpu().numpy().reshape((1, dataset.feature_dim)), 
                                                    tr_cls_centroid[dataset.idx_mat[i,n]].data.cpu().numpy().reshape((1, dataset.feature_dim)))
                            
                            lower_limit = dataset.semantic_similarity_seen [i,n] - opt.epsilon
                            upper_limit = dataset.semantic_similarity_seen [i,n] + opt.epsilon

                            lower_limit = torch.as_tensor(lower_limit.astype('float')) 
                            upper_limit = torch.as_tensor(upper_limit.astype('float')) 
                            corr = torch.as_tensor(Neighbor_correlation[0][0].astype('float'))
                            margin = (torch.max(corr- corr, corr - upper_limit))**2 + (torch.max(corr- corr, lower_limit - corr ))**2 
                            Correlation_loss += margin           
                
                Euclidean_loss *= 1.0/dataset.train_cls_num * opt.CENT_LAMBDA
                Correlation_loss = Correlation_loss * opt.correlation_penalty

            # ||W||_2 regularization
            reg_loss = Variable(torch.Tensor([0.0])).cuda()
            if opt.REG_W_LAMBDA != 0:
                for name, p in netG.named_parameters():
                    if 'weight' in name:
                        reg_loss += p.pow(2).sum()
                reg_loss.mul_(opt.REG_W_LAMBDA)

            # ||W_z||21 regularization, make W_z sparse
            reg_Wz_loss = Variable(torch.Tensor([0.0])).cuda()
            if opt.REG_Wz_LAMBDA != 0:
                Wz = netG.rdc_text.weight
                reg_Wz_loss = Wz.pow(2).sum(dim=0).sqrt().sum().mul(opt.REG_Wz_LAMBDA)

            all_loss = GC_loss + Euclidean_loss + reg_loss + reg_Wz_loss + Correlation_loss
            all_loss.backward()
            optimizerG.step()
            reset_grad(nets)

        if it > opt.unseen_start:
            for _ in range(1):
                # Zero shot Discriminator is training 
                zero_shot_labels = np.random.randint(dataset.test_cls_num, size = opt.zeroshotbatchsize).astype(int)
                zero_shot_true_labels = np.array([dataset.unseen_label_mapping[i] for i in zero_shot_labels])
                zero_text_feat = np.array([dataset.test_text_feature[i,:] for i in zero_shot_labels])
                
                zero_text_feat = Variable(torch.from_numpy(zero_text_feat.astype('float32'))).cuda()
                zero_y_true = Variable(torch.from_numpy(zero_shot_true_labels.astype('int'))).cuda()
                z = Variable(torch.randn(opt.zeroshotbatchsize, param.z_dim)).cuda()

                # GAN's D loss
                G_sample_zero, _ = netG(z, zero_text_feat) 
                _, C_fake_zero = netD(G_sample_zero)
                C_loss_fake_zero = F.cross_entropy(C_fake_zero, zero_y_true)
                C_loss_fake_zero.backward()

                optimizerD.step()
                reset_grad(nets)
                
            for _ in range(1):
                # Zero shot Generator is training 
                zero_shot_labels = np.random.randint(dataset.test_cls_num, size = opt.zeroshotbatchsize).astype(int)
                zero_shot_true_labels = np.array([dataset.unseen_label_mapping[i] for i in zero_shot_labels])
                zero_text_feat = np.array([dataset.test_text_feature[i,:] for i in zero_shot_labels])
                
                zero_text_feat = Variable(torch.from_numpy(zero_text_feat.astype('float32'))).cuda()
                zero_y_true = Variable(torch.from_numpy(zero_shot_true_labels.astype('int'))).cuda()
                y_dummy_zero = Variable(torch.from_numpy(zero_shot_labels.astype('int'))).cuda()
                z = Variable(torch.randn(opt.zeroshotbatchsize, param.z_dim)).cuda()

                # GAN's D loss
                G_sample_zero, _ = netG(z, zero_text_feat)
                _, C_fake_zero = netD(G_sample_zero)
                C_loss_fake_zero = F.cross_entropy(C_fake_zero, zero_y_true)
                
                Correlation_loss_zero = Variable(torch.Tensor([0.0])).cuda()

                if opt.CENT_LAMBDA != 0:
                    for i in range(dataset.test_cls_num):
                        sample_idx = (y_dummy_zero == i).data.nonzero().squeeze()
                        if sample_idx.numel() != 0:
                            G_sample_cls = G_sample_zero[sample_idx, :]
                            
                            if sample_idx.numel() != 1:
                                generated_mean = G_sample_cls.mean(dim=0) 
                            else:
                                generated_mean = G_sample_cls

                            for n in range(dataset.Neighbours):                            
                                Neighbor_correlation = cosine_similarity(generated_mean.data.cpu().numpy().reshape((1, dataset.feature_dim)), 
                                                        tr_cls_centroid[dataset.unseen_idx_mat[i,n]].data.cpu().numpy().reshape((1, dataset.feature_dim)))
                                
                                lower_limit = dataset.semantic_similarity_unseen [i,n] - opt.epsilon
                                upper_limit = dataset.semantic_similarity_unseen [i,n] + opt.epsilon

                                lower_limit = torch.as_tensor(lower_limit.astype('float')) 
                                upper_limit = torch.as_tensor(upper_limit.astype('float')) 
                                corr = torch.as_tensor(Neighbor_correlation[0][0].astype('float'))

                                margin = (torch.max(corr- corr, corr - upper_limit))**2 + (torch.max(corr- corr, lower_limit - corr ))**2 
                    
                                Correlation_loss_zero += margin           

                    Correlation_loss_zero = Correlation_loss_zero *opt.correlation_penalty

                # ||W||_2 regularization
                reg_loss_zero = Variable(torch.Tensor([0.0])).cuda()
                if opt.REG_W_LAMBDA != 0:
                    for name, p in netG.named_parameters():
                        if 'weight' in name:
                            reg_loss_zero += p.pow(2).sum()
                    reg_loss_zero.mul_(opt.REG_W_LAMBDA)

                # ||W_z||21 regularization, make W_z sparse
                reg_Wz_loss_zero = Variable(torch.Tensor([0.0])).cuda()
                if opt.REG_Wz_LAMBDA != 0:
                    Wz = netG.rdc_text.weight
                    reg_Wz_loss_zero = Wz.pow(2).sum(dim=0).sqrt().sum().mul(opt.REG_Wz_LAMBDA)

                all_loss = C_loss_fake_zero +  reg_loss_zero + reg_Wz_loss_zero + Correlation_loss_zero
                all_loss.backward()
                optimizerG.step()
                reset_grad(nets)

        if it % opt.disp_interval == 0 and it:
            acc_real = (np.argmax(C_real.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])
            acc_fake = (np.argmax(C_fake.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])

            log_text = 'Iter-{}; Was_D: {:.4}; Euc_ls: {:.4}; reg_ls: {:.4}; Wz_ls: {:.4}; G_loss: {:.4}; Correlation_loss : {:.4} ; D_loss_real: {:.4};' \
                       ' D_loss_fake: {:.4}; rl: {:.4}%; fk: {:.4}%'.format(it, Wasserstein_D.item(),  Euclidean_loss.item(), reg_loss.item(),reg_Wz_loss.item(),
                                G_loss.item(), Correlation_loss.item() , D_loss_real.item(), D_loss_fake.item(), acc_real * 100, acc_fake * 100)
            log_text1 = ""

            if it > opt.unseen_start : 
                acc_fake_zero = (np.argmax(C_fake_zero.data.cpu().numpy(), axis=1) == zero_y_true.data.cpu().numpy()).sum() / float(zero_y_true.data.size()[0])

                log_text1 = 'Zero_Shot_Iter-{}; Correlation_loss : {:.4}; fk: {:.4}%'.format(it, Correlation_loss_zero.item(),  acc_fake_zero * 100)
            

            print(log_text)
            print (log_text1)
            with open(log_dir, 'a') as f:
                f.write(log_text+'\n')
                f.write(log_text1+'\n')

        if it % opt.evl_interval == 0 and it >= 20:
            netG.eval()
            eval_fakefeat_test(it, netG, netD, dataset, param, result)
            eval_fakefeat_GZSL(it, netG, dataset, param, result_gzsl)
            if result.save_model:
                files2remove = glob.glob(out_subdir + '/Best_model*')
                for _i in files2remove:
                    os.remove(_i)
                torch.save({
                    'it': it + 1,
                    'state_dict_G': netG.state_dict(),
                    'state_dict_D': netD.state_dict(),
                    'random_seed': opt.manualSeed,
                    'log': log_text,
                    'Zero Shot Acc' : result.acc_list[-1],
                    'Generalized Zero Shot Acc' :  result_gzsl.acc_list[-1]
                }, out_subdir + '/Best_model_Acc_' + str(result.acc_list[-1])  + '_AUC_' + str(result_gzsl.acc_list[-1])  + '_' +'.tar')
            netG.train()

        if it % opt.save_interval == 0 and it:
            torch.save({
                    'it': it + 1,
                    'state_dict_G': netG.state_dict(),
                    'state_dict_D': netD.state_dict(),
                    'random_seed': opt.manualSeed,
                    'log': log_text,
                    'Zero Shot Acc' : result.acc_list[-1],
                    'Generalized Zero Shot Acc' : result_gzsl.acc_list[-1]
                },  out_subdir + '/Iter_{:d}.tar'.format(it))
            cprint('Save model to ' + out_subdir + '/Iter_{:d}.tar'.format(it), 'red')


def eval_fakefeat_test(it, netG, netD, dataset, param, result):
    gen_feat = np.zeros([0, dataset.feature_dim])
    for i in range(dataset.test_cls_num):
        text_feat = np.tile(dataset.test_text_feature[i].astype('float32'), (opt.nSample, 1))
        text_feat = Variable(torch.from_numpy(text_feat)).cuda()
        z = Variable(torch.randn(opt.nSample, param.z_dim)).cuda()
        G_sample, _ = netG(z, text_feat)
        gen_feat = np.vstack((gen_feat, G_sample.data.cpu().numpy()))

    # cosince predict K-nearest Neighbor
    sim = cosine_similarity(dataset.pfc_feat_data_test, gen_feat)
    idx_mat = np.argsort(-1 * sim, axis=1)
    label_mat = (idx_mat[:, 0:opt.Knn] / opt.nSample).astype(int)
    preds = np.zeros(label_mat.shape[0])
    for i in range(label_mat.shape[0]):
        (values, counts) = np.unique(label_mat[i], return_counts=True)
        preds[i] = values[np.argmax(counts)]

    # produce acc
    label_T = np.asarray(dataset.labels_test)
    acc = (preds == label_T).mean() * 100

    result.acc_list += [acc]
    result.iter_list += [it]
    result.save_model = False
    if acc > result.best_acc:
        result.best_acc = acc
        result.best_iter = it
        result.save_model = True

    visual_feat = Variable(torch.from_numpy(dataset.pfc_feat_data_test)).cuda()

    _, classification_zero_shot = netD(visual_feat)
    zero_shot_true_labels = np.array([int(dataset.unseen_label_mapping[i]) for i in dataset.labels_test])

    accuracy_zero_shot = (np.argmax(classification_zero_shot.data.cpu().numpy(), axis = 1) == zero_shot_true_labels).sum() / float(dataset.pfc_feat_data_test.shape[0])           
    accuracy_zero_shot = accuracy_zero_shot * 100 

    print("{}nn Classifier Discriminator: ")
    print("Accuracy is {:.4}%".format(accuracy_zero_shot))

    log_text = "KNN Acc : "  + str(acc) + " , " + "Discriminator Classifier Acc : "  + str(accuracy_zero_shot)

    exp_info = 'NAB_EASY' if opt.splitmode == 'easy' else 'NAB_HARD'
    exp_params = 'Eu{}_Rls{}_RWz{}'.format(opt.CENT_LAMBDA , opt.REG_W_LAMBDA, opt.REG_Wz_LAMBDA)

    out_subdir = 'out_' + str(opt.epsilon) + '/{:s}/{:s}'.format(exp_info, exp_params)
    log_dir  = out_subdir + '/log_{:s}.txt'.format(exp_info)
    
    print(log_text)
    with open(log_dir, 'a') as f:
        f.write(log_text+'\n')    

""" Generalized ZSL"""
def eval_fakefeat_GZSL(it, netG, dataset, param, result):
    gen_feat = np.zeros([0, param.X_dim])
    for i in range(dataset.train_cls_num):
        text_feat = np.tile(dataset.train_text_feature[i].astype('float32'), (opt.nSample, 1))
        text_feat = Variable(torch.from_numpy(text_feat)).cuda()
        z = Variable(torch.randn(opt.nSample, param.z_dim)).cuda()
        G_sample, _ = netG(z, text_feat)
        gen_feat = np.vstack((gen_feat, G_sample.data.cpu().numpy()))

    for i in range(dataset.test_cls_num):
        text_feat = np.tile(dataset.test_text_feature[i].astype('float32'), (opt.nSample, 1))
        text_feat = Variable(torch.from_numpy(text_feat)).cuda()
        z = Variable(torch.randn(opt.nSample, param.z_dim)).cuda()
        G_sample, _ = netG(z, text_feat)
        gen_feat = np.vstack((gen_feat, G_sample.data.cpu().numpy()))

    visual_pivots = [gen_feat[i * opt.nSample:(i + 1) * opt.nSample].mean(0) \
                     for i in range(dataset.train_cls_num + dataset.test_cls_num)]
    visual_pivots = np.vstack(visual_pivots)

    """collect points for gzsl curve"""

    acc_S_T_list, acc_U_T_list = list(), list()
    seen_sim = cosine_similarity(dataset.pfc_feat_data_train, visual_pivots)
    unseen_sim = cosine_similarity(dataset.pfc_feat_data_test, visual_pivots)
    for GZSL_lambda in np.arange(-2, 2, 0.01):
        tmp_seen_sim = copy.deepcopy(seen_sim)
        tmp_seen_sim[:, dataset.train_cls_num:] += GZSL_lambda
        pred_lbl = np.argmax(tmp_seen_sim, axis=1)
        acc_S_T_list.append((pred_lbl == np.asarray(dataset.labels_train)).mean())

        tmp_unseen_sim = copy.deepcopy(unseen_sim)
        tmp_unseen_sim[:, dataset.train_cls_num:] += GZSL_lambda
        pred_lbl = np.argmax(tmp_unseen_sim, axis=1)
        acc_U_T_list.append((pred_lbl == (np.asarray(dataset.labels_test) + dataset.train_cls_num)).mean())

    auc_score = integrate.trapz(y=acc_S_T_list, x=acc_U_T_list)

    result.acc_list += [auc_score]
    result.iter_list += [it]
    result.save_model = False
    if auc_score > result.best_acc:
        result.best_acc = auc_score
        result.best_iter = it
        result.save_model = True
    
    log_text = "AUC Score is {:.4}".format(auc_score) 
    print(log_text)

    exp_info = 'NAB_EASY' if opt.splitmode == 'easy' else 'NAB_HARD'
    exp_params = 'Eu{}_Rls{}_RWz{}'.format(opt.CENT_LAMBDA , opt.REG_W_LAMBDA, opt.REG_Wz_LAMBDA)

    out_subdir = 'out_' + str(opt.epsilon) + '/{:s}/{:s}'.format(exp_info, exp_params)
    log_dir  = out_subdir + '/log_{:s}.txt'.format(exp_info)
    
    with open(log_dir, 'a') as f:
        f.write(log_text+'\n')  

class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.acc_list = []
        self.iter_list = []
        self.save_model = False

def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal(m.weight.data)
        init.constant(m.bias, 0.0)


def reset_grad(nets):
    for net in nets:
        net.zero_grad()

def label2mat(labels, y_dim):
    c = np.zeros([labels.shape[0], y_dim])
    for idx, d in enumerate(labels):
        c[idx, d] = 1
    return c


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(opt.batchsize, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.GP_LAMBDA
    return gradient_penalty


if __name__ == "__main__":
    train()