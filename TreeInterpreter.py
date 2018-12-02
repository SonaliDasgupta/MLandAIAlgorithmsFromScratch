class TreeInterpreter:
    
    def predict(self, rf_model_tree, row):
            prediction , bias, contribs = rf_model_tree.predict_row_for_ti(row, {})
            print('prediction: '+str(prediction)+"bias: "+str(bias)+" contributions: "+str(contribs))
           
            return prediction, bias, contribs

        
#Using the Tree Interpreter for our Random Forest
ti = TreeInterpreter()
pr, bias, contribs  = ti.predict(rf_mine.trees[0], X_valid.iloc[0, :])

from waterfall import waterfall_chart
waterfall_chart.plot(list(contribs.keys()), list(contribs.values()))
