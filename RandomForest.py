class RandomForest:
    def __init__(self, sample_size, n_trees = 10 , min_samples_leaf = 5, max_depth = 4, max_features = None):
        self.n_trees, self.sample_size , self.min_samples_leaf, self.max_depth, self.max_features = n_trees, sample_size, min_samples_leaf, max_depth, max_features
        self.trees = [self.create_tree() for _ in range(self.n_trees)]
        
    def create_tree(self):
        
        return DecisionTree(min_samples_leaf = self.min_samples_leaf, max_depth = self.max_depth, max_features = self.max_features)
    
    def fit(self, X, y):   
        #calls the var_split method of underlying Decision Trees
        for tree in self.trees:
            random_idxs = np.random.permutation(X.shape[0])[:self.sample_size]
            tree.var_split(X.iloc[random_idxs, :], y[random_idxs]) 
    
    def predict(self, x):
      #  average of the predictions from each tree
 
        return np.mean([t.predict(x) for t in self.trees], axis = 0)
    
    def plot_pdp(self, X, y, feature_name, n_clusters = 0): pass
    
    def plot_pdp(self, X, y, feature_names, n_clusters = 0): pass
    
    def find_feature_importances(self, X, y): pass
