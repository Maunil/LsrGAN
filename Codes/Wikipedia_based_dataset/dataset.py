import numpy as np
import scipy.io as sio
from termcolor import cprint
import pickle
import sys
from sklearn.metrics.pairwise import cosine_similarity

class LoadDataset(object):
    '''
    Data set preparation 
    '''
    def __init__(self, opt):
        txt_feat_path = 'data/CUB2011/CUB_Porter_7551D_TFIDF_new.mat'
        if opt.splitmode == 'easy':
            train_test_split_dir = 'data/CUB2011/train_test_split_easy.mat'
            pfc_label_path_train = 'data/CUB2011/labels_train.pkl'
            pfc_feat_path_train = 'data/CUB2011/pfc_feat_train.mat'
            
            pfc_label_path_test = 'data/CUB2011/labels_test.pkl'
            pfc_feat_path_test = 'data/CUB2011/pfc_feat_test.mat'
            train_cls_num = 150   #150   
            test_cls_num = 50
            Neighbours = 20   

        else:
            train_test_split_dir = 'data/CUB2011/train_test_split_hard.mat'
            pfc_label_path_train = 'data/CUB2011/labels_train_hard.pkl'
            pfc_label_path_test = 'data/CUB2011/labels_test_hard.pkl'
            pfc_feat_path_train = 'data/CUB2011/pfc_feat_train_hard.mat'
            pfc_feat_path_test = 'data/CUB2011/pfc_feat_test_hard.mat'
            train_cls_num = 160
            test_cls_num = 40
            Neighbours = 20   
            
        self.Neighbours = Neighbours
        self.pfc_feat_data_train = sio.loadmat(pfc_feat_path_train)['pfc_feat'].astype(np.float32)
        self.pfc_feat_data_test = sio.loadmat(pfc_feat_path_test)['pfc_feat'].astype(np.float32)
        cprint("pfc_feat_file: {} || {} ".format(pfc_feat_path_train, pfc_feat_path_test), 'red')

        self.train_cls_num = train_cls_num
        self.test_cls_num  = test_cls_num
        self.feature_dim = self.pfc_feat_data_train.shape[1]

        # calculate the corresponding centroid.
        with open(pfc_label_path_train, 'rb') as fout1, open(pfc_label_path_test, 'rb') as fout2:
            if sys.version_info >= (3, 0):
                self.labels_train = pickle.load(fout1, encoding='latin1')
                self.labels_test  = pickle.load(fout2, encoding='latin1')
            else:
                self.labels_train = pickle.load(fout1)
                self.labels_test  = pickle.load(fout2)
            
        # Normalize feat_data to zero-centered
        mean = self.pfc_feat_data_train.mean()
        var = self.pfc_feat_data_train.var()

        self.pfc_feat_data_train = (self.pfc_feat_data_train - mean) / var
        self.pfc_feat_data_test = (self.pfc_feat_data_test - mean) / var

        self.tr_cls_centroid = np.zeros([train_cls_num, self.pfc_feat_data_train.shape[1]]).astype(np.float32)
        
        for i in range(train_cls_num):
            self.tr_cls_centroid[i] = np.mean(self.pfc_feat_data_train[self.labels_train == i], axis=0)
       
        self.train_text_feature, self.test_text_feature = get_text_feature(txt_feat_path, train_test_split_dir)
        self.text_dim = self.train_text_feature.shape[1]
        self.semantic_similarity_check(Neighbours, self.train_text_feature, self.test_text_feature)
        self.original_label_mapping(train_test_split_dir)

    def original_label_mapping(self, train_test_split_dir):
        self.seen_label_mapping = {}
        self.unseen_label_mapping  = {}

        train_test_split = sio.loadmat(train_test_split_dir)
        train_cid = train_test_split['train_cid'].squeeze()
        test_cid = train_test_split['test_cid'].squeeze()

        for i in range(self.train_cls_num):
            self.seen_label_mapping[i] = train_cid[i] - 1 

        for i in range(self.test_cls_num):
            self.unseen_label_mapping[i] = test_cid[i] - 1 

    def semantic_similarity_check(self, Neighbours, train_text_feature, test_text_feature):
        '''
        Seen Class
        '''
        seen_similarity_matric = cosine_similarity(train_text_feature, train_text_feature)

        #Mapping matric 
        self.idx_mat = np.argsort(-1 * seen_similarity_matric, axis=1)
        self.idx_mat = self.idx_mat[:,0:Neighbours]

        #Neighbours Semantic similary values 
        self.semantic_similarity_seen = np.zeros((self.train_cls_num, Neighbours))
        for i in range(self.train_cls_num):
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
        self.semantic_similarity_unseen = np.zeros((self.test_cls_num, Neighbours))
        for i in range(self.test_cls_num):
            for j in range(Neighbours):
                self.semantic_similarity_unseen [i,j] = unseen_similarity_matric[i, self.unseen_idx_mat[i,j]] 
        
    def centroid_calculation_test_segment(self):
        test_cls_centroid = np.zeros([self.test_cls_num, self.pfc_feat_data_train.shape[1]]).astype(np.float32)
        counter = 0

        for i in range(self.test_cls_num):
            flag = True
            class_number = 0
            while (counter < self.pfc_feat_data_test.shape[0]):   
                if (self.labels_test[counter] == i):
                    test_cls_centroid [i,:] = test_cls_centroid [i,:] + self.pfc_feat_data_test[counter, :]
                    counter = counter + 1 
                    class_number = class_number + 1
                else:
                    test_cls_centroid[i,:] = np.divide(test_cls_centroid[i,:], class_number)
                    flag == False    
                    break 

        return test_cls_centroid 

    def label_value_modification(self):
        counter = 0
        label_value = self.labels_train[0]
        self.labels_train[0] = counter

        for i in range(1, len(self.labels_train)):
            if (label_value == self.labels_train[i]):
                self.labels_train[i] = counter
            else:
                label_value = self.labels_train[i]
                counter = counter + 1 
                self.labels_train[i] = counter 

    def putting_the_seen_data_in_dic(self):
        label_dic = self.label_number_dic()
        seen_dic = {} # This will have the key as a class number and value a numpy array of image features 
        counter = 0 

        for i in range(self.train_cls_num):
            seen_dic[i] = self.pfc_feat_data_train[counter : counter + label_dic[i], :]
            counter = counter + label_dic[i]

        return seen_dic

    def label_number_dic(self):
        dic = {}

        for i in range(len(self.labels_train)):
            if self.labels_train[i] not in dic :
                dic[self.labels_train[i]] = 1
            else:
                dic[self.labels_train[i]] = dic[self.labels_train[i]] + 1 

        return dic

class LoadDataset_CUB_Val(object):
    def __init__(self, opt):
        txt_feat_path = 'data/CUB2011/CUB_Porter_7551D_TFIDF_new.mat'
        
        #Modified data - Maunil 
        train_test_split_dir1 = 'split_data/train_test_split_easy_val_0.mat'
        pfc_label_path_train1 = 'split_data/labels_val_0.pkl'           
        pfc_feat_path_train1 = 'split_data/pfc_feat_val_0.mat'
        
        train_test_split_dir2 = 'split_data/train_test_split_easy_val_1.mat'
        pfc_label_path_train2 = 'split_data/labels_val_1.pkl'           
        pfc_feat_path_train2 = 'split_data/pfc_feat_val_1.mat'
        
        train_test_split_dir3 = 'split_data/train_test_split_easy_val_2.mat'
        pfc_label_path_train3 = 'split_data/labels_val_2.pkl'           
        pfc_feat_path_train3 = 'split_data/pfc_feat_val_2.mat'
        
        train_test_split_dir4 = 'split_data/train_test_split_easy_val_3.mat'
        pfc_label_path_train4 = 'split_data/labels_val_3.pkl'           
        pfc_feat_path_train4 = 'split_data/pfc_feat_val_3.mat'
        
        train_test_split_dir5 = 'split_data/train_test_split_easy_val_4.mat'
        pfc_label_path_train5 = 'split_data/labels_val_4.pkl'           
        pfc_feat_path_train5 = 'split_data/pfc_feat_val_4.mat'
                
        train_cls_num = 120   #150   
        val_cls_num = 30

        self.pfc_feat_data_val1 = sio.loadmat(pfc_feat_path_train1)['pfc_feat'].astype(np.float32)
        self.pfc_feat_data_val2 = sio.loadmat(pfc_feat_path_train2)['pfc_feat'].astype(np.float32)
        self.pfc_feat_data_val3 = sio.loadmat(pfc_feat_path_train3)['pfc_feat'].astype(np.float32)
        self.pfc_feat_data_val4 = sio.loadmat(pfc_feat_path_train4)['pfc_feat'].astype(np.float32)
        self.pfc_feat_data_val5 = sio.loadmat(pfc_feat_path_train5)['pfc_feat'].astype(np.float32)
        
        self.train_cls_num = train_cls_num
        self.val_cls_num  = val_cls_num
        self.feature_dim = self.pfc_feat_data_val1.shape[1]

        with open(pfc_label_path_train1, 'rb') as fout1, open(pfc_label_path_train2, 'rb') as fout2, open(pfc_label_path_train3, 'rb') as fout3, \
            open(pfc_label_path_train4, 'rb') as fout4, open(pfc_label_path_train5, 'rb') as fout5: 
            if sys.version_info >= (3, 0):
                self.labels_val1 = pickle.load(fout1, encoding='latin1')
                self.labels_val2 = pickle.load(fout2, encoding='latin1')
                self.labels_val3 = pickle.load(fout3, encoding='latin1')
                self.labels_val4 = pickle.load(fout4, encoding='latin1')
                self.labels_val5 = pickle.load(fout5, encoding='latin1')
            else:
                self.labels_val1 = pickle.load(fout1, encoding='latin1')
                self.labels_val2 = pickle.load(fout2, encoding='latin1')
                self.labels_val3 = pickle.load(fout3, encoding='latin1')
                self.labels_val4 = pickle.load(fout4, encoding='latin1')
                self.labels_val5 = pickle.load(fout5, encoding='latin1')

        self.label_dic_list = []
        
        self.label_dic_list.append(self.label_number_dic(self.labels_val1))
        self.label_dic_list.append(self.label_number_dic(self.labels_val2))
        self.label_dic_list.append(self.label_number_dic(self.labels_val3))
        self.label_dic_list.append(self.label_number_dic(self.labels_val4))
        self.label_dic_list.append(self.label_number_dic(self.labels_val5))
        
        self.pfc_feat_data_val1 = (self.pfc_feat_data_val1 - opt.mean)/opt.var
        self.pfc_feat_data_val2 = (self.pfc_feat_data_val2 - opt.mean)/opt.var
        self.pfc_feat_data_val3 = (self.pfc_feat_data_val3 - opt.mean)/opt.var
        self.pfc_feat_data_val4 = (self.pfc_feat_data_val4 - opt.mean)/opt.var
        self.pfc_feat_data_val5 = (self.pfc_feat_data_val5 - opt.mean)/opt.var
        
        self.val_text_feature1, _ = get_text_feature(txt_feat_path, train_test_split_dir1)
        self.val_text_feature2, _ = get_text_feature(txt_feat_path, train_test_split_dir2)
        self.val_text_feature3, _ = get_text_feature(txt_feat_path, train_test_split_dir3)
        self.val_text_feature4, _ = get_text_feature(txt_feat_path, train_test_split_dir4)
        self.val_text_feature5, _ = get_text_feature(txt_feat_path, train_test_split_dir5)
        
        self.text_dim = self.val_text_feature1.shape[1]

    def label_number_dic(self, val_label):
        dic = {}
        
        for i in range(len(val_label)):
            if val_label[i] not in dic :
                dic[val_label[i]] = 1
            else:
                dic[val_label[i]] = dic[val_label[i]] + 1 
        return dic

    def centroid_computation(self, feature, labels, cls_num):
        tr_cls_centroid = np.zeros([cls_num, feature.shape[1]]).astype(np.float32)

        for i in range(cls_num):
            tr_cls_centroid[i] = np.mean(feature[labels == i], axis=0)
    
        return tr_cls_centroid

    def label_value_modification(self, label):
        counter = 0
        label_value = label[0]
        label[0] = counter

        for i in range(1, len(label)):
            if (label_value == label[i]):
                label[i] = counter
            else:
                label_value = label[i]
                counter = counter + 1 
                label[i] = counter 

        return label 


class LoadDataset_NAB(object):
    '''
    NAB dataset preparation 
    '''
    def __init__(self, opt):
        txt_feat_path = 'data/NABird/NAB_Porter_13217D_TFIDF_new.mat'
        if opt.splitmode == 'easy':
            train_test_split_dir = 'data/NABird/train_test_split_NABird_easy.mat'
            pfc_label_path_train = 'data/NABird/labels_train.pkl'
            pfc_label_path_test = 'data/NABird/labels_test.pkl'
            pfc_feat_path_train = 'data/NABird/pfc_feat_train_easy.mat'
            pfc_feat_path_test = 'data/NABird/pfc_feat_test_easy.mat'
            train_cls_num = 323
            test_cls_num = 81
            Neighbours = 20 
        else:
            train_test_split_dir = 'data/NABird/train_test_split_NABird_hard.mat'
            pfc_label_path_train = 'data/NABird/labels_train_hard.pkl'
            pfc_label_path_test = 'data/NABird/labels_test_hard.pkl'
            pfc_feat_path_train = 'data/NABird/pfc_feat_train_hard.mat'
            pfc_feat_path_test = 'data/NABird/pfc_feat_test_hard.mat'
            train_cls_num = 323
            test_cls_num = 81
            Neighbours = 20 

        self.Neighbours = Neighbours
        self.pfc_feat_data_train = sio.loadmat(pfc_feat_path_train)['pfc_feat'].astype(np.float32)
        self.pfc_feat_data_test = sio.loadmat(pfc_feat_path_test)['pfc_feat'].astype(np.float32)
        cprint("pfc_feat_file: {} || {} ".format(pfc_feat_path_train, pfc_feat_path_test), 'red')

        self.train_cls_num = train_cls_num
        self.test_cls_num  = test_cls_num
        self.feature_dim = self.pfc_feat_data_train.shape[1]

        # calculate the corresponding centroid.
        with open(pfc_label_path_train, 'rb') as fout1, open(pfc_label_path_test, 'rb') as fout2:
            if sys.version_info >= (3, 0):
                self.labels_train = pickle.load(fout1, encoding='latin1')
                self.labels_test  = pickle.load(fout2, encoding='latin1')
            else:
                self.labels_train = pickle.load(fout1)
                self.labels_test  = pickle.load(fout2)

        # Normalize feat_data to zero-centered
        mean = self.pfc_feat_data_train.mean()
        var = self.pfc_feat_data_train.var()

        self.pfc_feat_data_train = (self.pfc_feat_data_train - mean) / var
        self.pfc_feat_data_test = (self.pfc_feat_data_test - mean) / var

        self.tr_cls_centroid = np.zeros([train_cls_num, self.pfc_feat_data_train.shape[1]]).astype(np.float32)
        for i in range(train_cls_num):
            self.tr_cls_centroid[i] = np.mean(self.pfc_feat_data_train[self.labels_train == i], axis=0)

        self.train_text_feature, self.test_text_feature = get_text_feature(txt_feat_path, train_test_split_dir)
        self.text_dim = self.train_text_feature.shape[1]
        self.semantic_similarity_check(self.Neighbours, self.train_text_feature, self.test_text_feature)
        self.original_label_mapping(train_test_split_dir)
                
    def original_label_mapping(self, train_test_split_dir):
        self.seen_label_mapping = {}
        self.unseen_label_mapping  = {}

        train_test_split = sio.loadmat(train_test_split_dir)
        train_cid = train_test_split['train_cid'].squeeze()
        test_cid = train_test_split['test_cid'].squeeze()

        for i in range(self.train_cls_num):
            self.seen_label_mapping[i] = train_cid[i] - 1 

        for i in range(self.test_cls_num):
            self.unseen_label_mapping[i] = test_cid[i] - 1 
        
    def semantic_similarity_check(self, Neighbours, train_text_feature, test_text_feature):
        '''
        Seen Class
        '''
        seen_similarity_matric = cosine_similarity(self.train_text_feature, self.train_text_feature)

        #Mapping matric 
        self.idx_mat = np.argsort(-1 * seen_similarity_matric, axis=1)
        self.idx_mat = self.idx_mat[:,0:Neighbours]

        #Neighbours Semantic similary values 
        self.semantic_similarity_seen = np.zeros((self.train_cls_num, Neighbours))
        for i in range(self.train_cls_num):
            for j in range(Neighbours):
                self.semantic_similarity_seen [i,j] = seen_similarity_matric[i, self.idx_mat[i,j]] 
        
        '''
        Unseen class
        '''
        unseen_similarity_matric = cosine_similarity(self.test_text_feature, self.train_text_feature)

        #Mapping matric 
        self.unseen_idx_mat = np.argsort(-1 * unseen_similarity_matric, axis=1)
        self.unseen_idx_mat = self.unseen_idx_mat[:, 0:Neighbours]
        
        #Neighbours Semantic similary values 
        self.semantic_similarity_unseen = np.zeros((self.test_cls_num, Neighbours))
        for i in range(self.test_cls_num):
            for j in range(Neighbours):
                self.semantic_similarity_unseen [i,j] = unseen_similarity_matric[i, self.unseen_idx_mat[i,j]] 


class FeatDataLayer(object):
    '''
    Mini-batch and other NN pre-processing!
    '''
    def __init__(self, label, feat_data, mapping,  opt):
        assert len(label) == feat_data.shape[0]
        self._opt = opt
        self._feat_data = feat_data
        self._label = label
        self._mapping = mapping
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._label)))
        # self._perm = np.arange(len(self._roidb))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        
        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds])
        minibatch_true_label = np.array([self._mapping[i] for i in minibatch_label])
        blobs = {'data': minibatch_feat, 'labels':minibatch_label, 'true_labels' : minibatch_true_label}
        return blobs

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs

    def get_whole_data(self):
        blobs = {'data': self._feat_data, 'labels': self._label}
        return blobs

    def seen_class_data_take_for_teacher(self, dataset, max_number, min_number):
        class_index = np.random.randint(min_number, max_number, size = 50)
        visual_feat = np.zeros([0, dataset.feature_dim])
        labels = np.zeros([0])

        for i in range(class_index.shape[0]):
            visual_feat = np.vstack((visual_feat, dataset.seen_dic[class_index[i]]))
            labels = np.hstack((labels, np.array([int(class_index[i]) for j in range((dataset.seen_dic[class_index[i]].shape[0]))]))) 
        
        return visual_feat, labels, class_index

def get_text_feature(dir, train_test_split_dir):
    train_test_split = sio.loadmat(train_test_split_dir)
    # get training text feature
    train_cid = train_test_split['train_cid'].squeeze()
    text_feature = sio.loadmat(dir)['PredicateMatrix']
    train_text_feature = text_feature[train_cid - 1]  # 0-based index
    # get testing text feature
    test_cid = train_test_split['test_cid'].squeeze()
    text_feature = sio.loadmat(dir)['PredicateMatrix']
    test_text_feature = text_feature[test_cid - 1]  # 0-based index
    return train_text_feature.astype(np.float32), test_text_feature.astype(np.float32)