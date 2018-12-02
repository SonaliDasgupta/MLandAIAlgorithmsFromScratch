class TreeInterpreter:
    
    def predict(self, rf_model_tree, row):
            prediction , bias, contribs = rf_model_tree.predict_row_for_ti(row, {})
            print('prediction: '+str(prediction)+"bias: "+str(bias)+" contributions: "+str(contribs))
           
            return prediction, bias, contribs
