class DecisionTree:
    
    def __init__(self, int min_samples_leaf = 3, int max_depth = 4, int level = 0, max_features = None, float parent_value = float('inf')):
        self.min_samples_leaf, self.max_depth = min_samples_leaf, max_depth
        self.score = float('inf')
        self.value = 0.0
        self.leftTree , self.rightTree, self.split_val, self.split_feature  = None, None, None, None
        self.level = level
        self.max_features = max_features
        self.parent_value  = parent_value
       
       
        
    @property 
    def is_leaf(self):
        return self.s
        if updated_score < self.score :
                self.score = updated_score
                self.split_feature = x.columns[ind]
                self.split_val = i
                self.value = (np.mean(y[:i])*i + np.mean(y[i:])*(x.shape[0] - i))/(x.shape[0])
        self.score = self.score
       
        

    def find_better_split(self, x, y, int ind):
    
        x1 = x.values[:, ind]
        inds = np.argsort(x1, axis = 0)
        sorted_y, sorted_x = y[inds], x1[inds]
        rhs_count, rhs_sum, rhs_sum2  = x.shape[0], sorted_y.sum(), (sorted_y**2).sum() 
        lhs_count, lhs_sum, lhs_sum2 = 0, 0., 0.
      #  loop through the feature values to find the best split the tree on, for this feature
        for i in range(0, x.shape[0]-self.min_samples_leaf+1):
            lhs_count, lhs_sum, lhs_sum2 = lhs_count+1, lhs_sum + sorted_y[i], lhs_sum2 + sorted_y[i]**2
            rhs_count, rhs_sum, rhs_sum2 = rhs_count-1, rhs_sum - sorted_y[i], rhs_sum2 - sorted_y[i]**2
            if i < self.min_samples_leaf - 1 or sorted_x[i] == sorted_x[i+1]:
                continue
            updated_score = ((lhs_count * stddev(lhs_count, lhs_sum2, lhs_sum)) + (rhs_count * stddev(rhs_count, rhs_sum2, rhs_sum)))/(x.shape[0])
            #score is updated to the sum of standard deviations of the left and right subtrees
            if updated_score < self.score :
                self.score = updated_score
                self.split_feature = x.columns[ind]
                self.split_val = i
                self.value = (np.mean(y[:i])*i + np.mean(y[i:])*(x.shape[0] - i))/(x.shape[0])
        self.score = self.score
       
        
   
    def var_split(self, x, y):
        
        if x.shape[0] > self.min_samples_leaf and self.level < self.max_depth - 1:
            #calculate maximum number of features to split on
             
            if self.max_features is not None:
                if self.max_features in ['auto', 'sqrt']:
                    self.max_features = int(math.sqrt(x.shape[1]))
                else:
                    if self.max_features == 'log2':
                        self.max_features = int(np.log(float(x.shape[1]))/np.log(2))
                    else:
                        if isinstance(self.max_features, float):
                            self.max_features = int(self.max_features * x.shape[1])
                        else:
                            self.max_features = x.shape[1]
            else:
                self.max_features = x.shape[1]
            self.max_features = int(self.max_features)
            feature_inds = np.random.permutation(x.shape[1])[:self.max_features] 
            #print('will split on features: '+str(feature_inds))
            feature_inds = [index for index in feature_inds if x.columns[index] != None]
            for ind in feature_inds:
                self.find_better_split(x, y, ind) 
           # find the best feature to split on , and its optimal value
            if self.parent_value == float('inf'):
            self.parent_value  = self.value
            x_lhs, x_rhs = x.iloc[:self.split_val,:], x.iloc[self.split_val:,:]
            self.leftTree = DecisionTree(min_samples_leaf = self.min_samples_leaf, max_depth = self.max_depth, level = self.level + 1, parent_value = self.parent_value)
            self.leftTree.var_split(x_lhs, y[:self.split_val])
            self.rightTree = DecisionTree(min_samples_leaf = self.min_samples_leaf, max_depth = self.max_depth, level = self.level + 1, parent_value = self.parent_value)
            self.rightTree.var_split(x_rhs, y[self.split_val:])
        
        else :
            self.score = float('inf')
            #only the leaves in the tree will have a score of infinity
            y = [val for val in y if val != None]
            self.value = np.mean(y)

        
    def predict_row(self, row):      
        if self.is_leaf: 
            #prediction for the row would be the value at this leaf
            return self.value
        if row[self.split_feature] < self.split_val:
            return self.leftTree.predict_row(row)
        else:
            return self.rightTree.predict_row(row)
    
    def predict_row_for_ti(self, row, feat_contribs):
        
        if self.is_leaf: 
           return self.value, self.parent_value , feat_contribs
        if row[self.split_feature] < self.split_val:
            if self.split_feature in feat_contribs.keys():
                feat_contribs[self.split_feature] += self.leftTree.value - self.value
            else:
                feat_contribs[self.split_feature] = self.leftTree.value - self.value
            return self.lef else:
            if self.split_feature in feat_contribs.keys():
                feat_contribs[self.split_feature] += self.rightTree.value - self.value
            else:
                feat_contribs[self.split_feature] = self.rightTree.value - self.value
            return self.rightTree.predict_row_for_ti(row, feat_contribs)
           
    def predict(self, X):
        y_pred = []
        for row in range(X.shape[0]):
            y_pred.append(self.predict_row(X.iloc[row, :]))
        return y_pred
    
    def get_prediction_and_bias(self):
        
        return self.parent_value, self.value 
        
    def get_child_trees(self):
        return self.leftTree, self.rightTree
  
    def __repr__(self):
        return "score: " +str(self.score) + " avg: "+str(self.value) +  " split val: " + str(self.split_val) + " split feature : "+ str(self.split_feature)tTree.predict_row_for_ti(row, feat_contribs)
            
