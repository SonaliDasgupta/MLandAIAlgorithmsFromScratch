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
        
        """logic to find the best value to split the tree 
        on for a particular feature"""
       
        
   
    def var_split(self, x, y):
        
      """logic to find the best feature to split the tree on, and
      its optimum value for the split"""
        
    def predict_row(self, row):      
        """logic to predict the result for 
        a single data sample (row) in the
        test / validation set"""
    
    def predict_row_for_ti(self, row, feat_contribs):
        
       #functionality for Tree Interpreter
           
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
            
