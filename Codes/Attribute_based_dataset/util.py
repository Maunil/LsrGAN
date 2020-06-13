import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
from sklearn.metrics.pairwise import cosine_similarity
from scipy.io import loadmat

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename+'.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename+'.log', "a")
        f.write(message)  
        f.close()

class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imageNet1K':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
                
    # not tested
    def read_h5dataset(self, opt):
        # read image feature
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".hdf5", 'r')
        feature = fid['feature'][()]
        label = fid['label'][()] 
        trainval_loc = fid['trainval_loc'][()] 
        train_loc = fid['train_loc'][()] 
        val_unseen_loc = fid['val_unseen_loc'][()] 
        test_seen_loc = fid['test_seen_loc'][()] 
        test_unseen_loc = fid['test_unseen_loc'][()] 
        fid.close()
        # read attributes
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".hdf5", 'r')
        self.attribute = fid['attribute'][()]
        fid.close()

        if not opt.validation:
            self.train_feature = feature[trainval_loc] 
            self.train_label = label[trainval_loc] 
            self.test_unseen_feature = feature[test_unseen_loc] 
            self.test_unseen_label = label[test_unseen_loc] 
            self.test_seen_feature = feature[test_seen_loc] 
            self.test_seen_label = label[test_seen_loc] 
        else:
            self.train_feature = feature[train_loc] 
            self.train_label = label[train_loc] 
            self.test_unseen_feature = feature[val_unseen_loc] 
            self.test_unseen_label = label[val_unseen_loc] 

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        self.nclasses = self.seenclasses.size(0)

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()


        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = torch.from_numpy(matcontent['w2v']).float()
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long() 
        self.test_seen_feature = torch.from_numpy(feature_val).float()
        self.test_seen_label = torch.from_numpy(label_val).long() 
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        self.test_unseen_label = torch.from_numpy(label_unseen).long() 
        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat") 
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        #val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        text_info = matcontent['att'].T
        self.attribute = torch.from_numpy(text_info).float() 
        
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
                self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long() 
    
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntest = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)        
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 
        self.Neighbours = opt.Neighbours
        # Computing the mean of the features 
        train_label_numpy = self.train_mapped_label.data.cpu().numpy()
        train_feature_numpy = self.train_feature.data.cpu().numpy()
        test_feature_numpy = self.test_unseen_feature.data.cpu().numpy()
        test_label_numpy = map_label(self.test_unseen_label, self.unseenclasses).data.cpu().numpy() 
        self.tr_cls_centroid = np.zeros([self.ntrain_class, train_feature_numpy.shape[1]]).astype(np.float32)    
        self.te_cls_centroid = np.zeros([self.ntest_class, train_feature_numpy.shape[1]]).astype(np.float32)

        for i in range(self.ntrain_class):
            self.tr_cls_centroid[i] = np.mean(train_feature_numpy[train_label_numpy == i], axis=0)

        for i in range(self.ntest_class):
            self.te_cls_centroid[i] = np.mean(test_feature_numpy[test_label_numpy == i], axis=0)
        
        train_text_feat, test_text_feat = self.text_feature_split(opt, text_info)
        self.semantic_similarity_check(opt.Neighbours, train_text_feat, test_text_feat, self.tr_cls_centroid, self.te_cls_centroid)

    def text_feature_split(self, opt, attribute):
        train_label = self.train_label.data.cpu().numpy()
        att_shape = attribute.shape[1]
        train_text_feat = np.zeros([0, att_shape])
        test_text_feat = np.zeros([0, att_shape])

        for i in range(opt.nclass_all):
            if i in train_label:
                train_text_feat = np.vstack((train_text_feat, attribute[i,:]))                
            else:
                test_text_feat = np.vstack((test_text_feat, attribute[i,:]))
        
        return train_text_feat, test_text_feat

    def determine_sim_between_gen_actual(self):
        #file open 
        x = loadmat('Feature_values.mat')
        gen_feat = x['centroid_feat']
        sim_matrix = cosine_similarity(gen_feat[16,:,:], self.te_cls_centroid)
        print (sim_matrix)

    def feat_similarity(self, te_cls_centroid):
        Neighbours = 40
        tr_cls_centroid = self.tr_cls_centroid.data.cpu().numpy()
        seen_similarity_matric = cosine_similarity(te_cls_centroid, tr_cls_centroid)
        idx_mat = np.argsort(-1 * seen_similarity_matric, axis=1)
        
        semantic_similarity_seen = np.zeros((self.ntest_class, Neighbours))

        for i in range(self.ntest_class):
            for j in range(Neighbours):
                semantic_similarity_seen [i,j] = seen_similarity_matric[i, idx_mat[i,j]] 
        
        return semantic_similarity_seen, idx_mat

    def semantic_similarity_check(self, Neighbours, train_text_feature, test_text_feature, train_feature_numpy, test_feature_numpy):
        '''
        Seen Class
        '''
        seen_similarity_matric = cosine_similarity(train_text_feature, train_text_feature)
        
        #Mapping matric 
        self.idx_mat = np.argsort(-1 * seen_similarity_matric, axis=1)
        self.idx_mat = self.idx_mat[:, 0:Neighbours]

        #Neighbours Semantic similary values 
        self.semantic_similarity_seen = np.zeros((self.ntrain_class, Neighbours))

        for i in range(self.ntrain_class):
            for j in range(Neighbours):
                self.semantic_similarity_seen [i,j] = seen_similarity_matric[i, self.idx_mat[i,j]] 

        '''
        Unseen class
        '''
        unseen_similarity_matric = cosine_similarity(test_text_feature, train_text_feature)

        #Mapping matric 
        self.unseen_idx_mat = np.argsort(-1 * unseen_similarity_matric, axis=1)
        self.unseen_idx_mat = self.unseen_idx_mat[:, 0:Neighbours]

        #Neighbours Semantic similary values 
        self.semantic_similarity_unseen = np.zeros((self.ntest_class, Neighbours))
        for i in range(self.ntest_class):
            for j in range(Neighbours):
                self.semantic_similarity_unseen [i,j] = unseen_similarity_matric[i, self.unseen_idx_mat[i,j]] 

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0 
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]] 
    
    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    def next_batch_unseen_class(self, batch_size):
        idx = torch.randperm(self.ntest)[0:batch_size]
        batch_label = self.test_unseen_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_label, batch_att

    # select batch samples by randomly drawing batch_size classes    
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]
            
        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))       
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]] 
        return batch_feature, batch_label, batch_att
