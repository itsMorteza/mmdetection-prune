def get_output_channel_index(self, value, layer_id):

        output_channel_index = []

        if len(value.size()) :

            weight_vec = value.view(value.size()[0], -1)
            weight_vec = weight_vec.cuda()

            # l1-norm
            if self.criterion == 'l1-norm':
                norm = torch.norm(weight_vec, 1, 1)
                norm_np = norm.cpu().detach().numpy()
                arg_max = np.argsort(norm_np)
                arg_max_rev = arg_max[::-1][:self.cfg[layer_id]]
                output_channel_index = sorted(arg_max_rev.tolist())
            
            # l2-norm
            elif self.criterion == 'l2-norm':
                norm = torch.norm(weight_vec, 2, 1)
                norm_np = norm.cpu().detach().numpy()
                arg_max = np.argsort(norm_np)
                arg_max_rev = arg_max[::-1][:self.cfg[layer_id]]
                output_channel_index = sorted(arg_max_rev.tolist())

            # l2-GM
            elif self.criterion == 'l2-GM':
                weight_vec = weight_vec.cpu().detach().numpy()
                matrix = distance.cdist(weight_vec, weight_vec, 'euclidean')
                similar_sum = np.sum(np.abs(matrix), axis=0)

                output_channel_index = np.argpartition(similar_sum, -self.cfg[layer_id])[-self.cfg[layer_id]:]


        return output_channel_index

def create_scaling_mat_conv_thres_bn(weight, ind, threshold,
                                     bn_weight, bn_bias,
                                     bn_mean, bn_var, lam, model_type):
    '''
    weight - 4D tensor(n, c, h, w), np.ndarray
    ind - chosen indices to remain
    threshold - cosine similarity threshold
    bn_weight, bn_bias - parameters of batch norm layer right after the conv layer
    bn_mean, bn_var - running_mean, running_var of BN (for inference)
    lam - how much to consider cosine sim over bias, float value between 0 and 1
    '''
    assert(type(weight) == np.ndarray)
    assert(type(ind) == np.ndarray)
    assert(type(bn_weight) == np.ndarray)
    assert(type(bn_bias) == np.ndarray)
    assert(type(bn_mean) == np.ndarray)
    assert(type(bn_var) == np.ndarray)
    assert(bn_weight.shape[0] == weight.shape[0])
    assert(bn_bias.shape[0] == weight.shape[0])
    assert(bn_mean.shape[0] == weight.shape[0])
    assert(bn_var.shape[0] == weight.shape[0])
    
    
    weight = weight.reshape(weight.shape[0], -1)

    cosine_dist = pairwise_distances(weight, metric="cosine")

    weight_chosen = weight[ind, :]
    scaling_mat = np.zeros([weight.shape[0], weight_chosen.shape[0]])

    for i in range(weight.shape[0]):
        if i in ind: # chosen
            ind_i, = np.where(ind == i)
            assert(len(ind_i) == 1) # check if only one index is found
            scaling_mat[i, ind_i] = 1
        else: # not chosen

            if model_type == 'prune':
                continue

            current_weight = weight[i]
            current_norm = np.linalg.norm(current_weight)
            current_cos = cosine_dist[i]
            gamma_1 = bn_weight[i]
            beta_1 = bn_bias[i]
            mu_1 = bn_mean[i]
            sigma_1 = bn_var[i]
            
            # choose one
            cos_list = []
            scale_list = []
            bias_list = []
            
            for chosen_i in ind:
                chosen_weight = weight[chosen_i]
                chosen_norm = np.linalg.norm(chosen_weight, ord = 2)
                chosen_cos = current_cos[chosen_i]
                gamma_2 = bn_weight[chosen_i]
                beta_2 = bn_bias[chosen_i]
                mu_2 = bn_mean[chosen_i]
                sigma_2 = bn_var[chosen_i]
                
                # compute cosine sim
                cos_list.append(chosen_cos)
                
                # compute s
                s = current_norm/chosen_norm
                
                # compute scale term
                scale_term_inference = s * (gamma_2 / gamma_1) * (sigma_1 / sigma_2)
                scale_list.append(scale_term_inference)
                
                # compute bias term
                bias_term_inference = abs((gamma_2/sigma_2) * (s * (-(sigma_1*beta_1/gamma_1) + mu_1) - mu_2) + beta_2)

                bias_term_inference = bias_term_inference/scale_term_inference

                bias_list.append(bias_term_inference)

            assert(len(cos_list) == len(ind))
            assert(len(scale_list) == len(ind))
            assert(len(bias_list) == len(ind))
            

            # merge cosine distance and bias distance
            bias_list = (bias_list - np.min(bias_list)) / (np.max(bias_list)-np.min(bias_list))

            score_list = lam * np.array(cos_list) + (1-lam) * np.array(bias_list)


            # find index and scale with minimum distance
            min_ind = np.argmin(score_list)

            min_scale = scale_list[min_ind]
            min_cosine_sim = 1-cos_list[min_ind]

            # check threshold - second
            if threshold and min_cosine_sim < threshold:
                continue
            
            scaling_mat[i, min_ind] = min_scale

    return scaling_mat
    
    
def create_scaling_mat_ip_thres_bias(weight, ind, threshold, model_type):
    '''
    weight - 2D matrix (n_{i+1}, n_i), np.ndarray
    ind - chosen indices to remain, np.ndarray
    threshold - cosine similarity threshold
    '''
    assert(type(weight) == np.ndarray)
    assert(type(ind) == np.ndarray)

    cosine_sim = 1-pairwise_distances(weight, metric="cosine")
    weight_chosen = weight[ind, :]
    scaling_mat = np.zeros([weight.shape[0], weight_chosen.shape[0]])

    for i in range(weight.shape[0]):
        if i in ind: # chosen
            ind_i, = np.where(ind == i)
            assert(len(ind_i) == 1) # check if only one index is found
            scaling_mat[i, ind_i] = 1
        else: # not chosen
            if model_type == 'prune':
                continue
            max_cos_value = np.max(cosine_sim[i][ind])
            max_cos_value_index = np.argpartition(cosine_sim[i][ind], -1)[-1]

            if threshold and max_cos_value < threshold:
                continue

            baseline_weight = weight_chosen[max_cos_value_index]
            current_weight = weight[i]
            baseline_norm = np.linalg.norm(baseline_weight)
            current_norm = np.linalg.norm(current_weight)
            scaling_factor = current_norm / baseline_norm
            scaling_mat[i, max_cos_value_index] = scaling_factor

    return scaling_mat

def filter_prune_l1_v3(model, pruning_perc,layer):
    '''
    Prune pruning_perc% filters layerwise
    '''
    masks = []
    masks_temp = {}
    keep_mask_temp = {}
    stage_pr_bn=[0,0,0,0,0,0,0,0]
    #stage_pr_bn=[0,0.5,0.75,0.87,0.93,0.5,0.5,0.5]
    #stage_pr_bn=[0,0.3,0.3,0.3,0.14, 0.3, 0.3, 0.14]
    values = []                 
    ind = 0
    graph_stage={}
    for p in model.modules():
        if type(p).__name__ == 'Conv2d':
                if 'backbone' in p.name:
                   RTname =p.name[16:]
                   stage  = 0
                   #if RTname=="conv1":
                   #    stage  = 0
                   if "conv2" in RTname:
                       stage  = int(RTname.split(".")[0][-1])
                   #if "conv3" in RTname:
                   #else:
                   #    stage = 0
                if 'neck' in p.name:
                   stage  = 5
                if 'rpn' in p.name:
                   stage  = 6
                   if 'rpn_cls' or 'rpn_reg' in p.name:
                      stage = 0
                if 'roi_head' in p.name:
                   stage = 7
                graph_stage[p.name]=stage   
    for p in model.modules():
        if type(p).__name__ == 'Conv2d':
                p_np = p.weight.data.cpu().numpy()
                weight_copy = p.weight.data.abs().clone().cpu().numpy()
                masks.append(np.ones(p_np.shape).astype('float32'))
                shape= p.weight.data.shape
                score = p.weight.abs().mean(dim=[1, 2, 3]) 
                value_this_layer2 = np.mean(weight_copy, axis=(1, 2, 3))
                value_this_layer = np.repeat(value_this_layer2,p_np.shape[1]               
                                *p_np.shape[2]*p_np.shape[3])
                value_this_layer=np.reshape(value_this_layer,(p_np.shape))
                threshold = np.percentile(value_this_layer2, stage_pr_bn[graph_stage[p.name]]*pruning_perc)
                #threshold = np.percentile(value_this_layer2, pruning_perc)   
                if ("downsample" not in p.name) and ("conv_logits" not in p.name):
                    if layer==p.name:
                        masks[ind][value_this_layer < threshold] = 0.
                        indices = [i for i, elem in enumerate(value_this_layer2) if (elem<threshold)]
                        indices2 = _pick_pruned(score,pruning_perc/100)
                        keep_indices = [i for i, elem in enumerate(value_this_layer2) if (i not in indices)]
                        masks_temp[p.name]=indices
                        print(len(indices))
                        keep_mask_temp[p.name]=keep_indices
                ind += 1
    masks = [torch.from_numpy(mask) for mask in masks]
    return masks ,masks_temp    
#no acc imporovement    
def set_decomp(self, runner, pr_mask):
        for conv in pr_mask:
          if pr_mask[conv]:
            print(("prune:", conv))
            for m in runner.model.modules():
               if type(m).__name__ == 'Conv2d':  
                  if conv is m.name:
                        print(m.weight.data.shape[1:])
                        Total_layer = m.weight.data.clone()
                        gama = torch.sum(Total_layer[pr_mask[conv]],0, keepdim =True )
                        #print(gama.shape)  
                        #print(torch.mean(Total_layer[pr_mask[conv]]))
                        layermean = torch.mean(Total_layer)
                        #print(layermean)    
                        delta = gama *(layermean/torch.mean(gama))
                        #print(torch.mean(delta))
                        #print(torch.mean(gama)) 
                        #print(pr_mask[conv][0]) 
                        AT =  pr_mask[conv][0]
                        Total_layer[AT] = delta[0]
                        
                        m.weight=nn.Parameter(Total_layer)
                        #m.weight=nn.Parameter(m.weight*m.mask) 
                        
def modified_forward2(self, feature):                 
                        p = torch.ones(1).cuda()
                        p.requires_grad = False
                        p=p.cuda()
                        if self.mapflag == p:           
                                self.weight.data = self.weight.data*self.mask.data
                        return F.conv2d(feature, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)                        
                                     
