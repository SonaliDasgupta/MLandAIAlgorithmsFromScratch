import numpy as np

def find_feature_importances(self, X, y):
    feat_imp = {}
    y_pred_old = self.predict(X)
    for ind in range(X.shape[1]):
        X_new = X.copy()
        np.random.shuffle(X_new.values[:, ind])
        y_pred = self.predict(X_new)
        feat_imp[X.columns[ind]] = np.fabs(np.sum(y_pred_old) - np.sum(y_pred))
        del X_new
    
    features, importances =zip(*sorted(feat_imp.items(), reverse=True))
    return features, importances
    
RandomForest.find_feature_importances = find_feature_importances
idxs = np.random.permutation(X_train.shape[0])[:50]
features, imp = find_feature_importances(rf_mine, X_train.iloc[idxs, :], y_train[idxs])

plt.barh(features[:10], imp[:10])
