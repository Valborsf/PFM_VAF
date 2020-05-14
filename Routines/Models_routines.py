#!/usr/bin/env python
# coding: utf-8


# Read common var's 
def define_dirs(model):

    global root_dir
    global json_dir
    global imag_dir
    global csv_dir
    global model_json_dir
    global model_bin_dir
    global results_dir
    global Tensor_dir
    
    root_dir = "/home/valborsf/Documents/UOC/PFMProject"     # Root dir

    json_dir =  root_dir +"/DataNew/ALL_JSON/"                # .json dir images
    imag_dir =  root_dir +"/DataNew/ALL_IMAGES/"              # .png dir - images

#  To change directories
#    json_dir =  root_dir +"/Data/ALL_JSON/"                # .json dir images
#    imag_dir =  root_dir +"/Data/ALL_IMAGES/"              # .png dir - images


    # directories for  CSV's
# To change directories
# csv dir defined directly in the notebook 
    csv_dir =  root_dir +" " 		                      # .csv dir - dftrain, dfval, dftest
#    csv_dir =  root_dir +"/DataNew/CSV/" 		      # .csv dir - dftrain, dfval, dftest
#    csv_dir =  root_dir +"/Data/CSV/" 		              # .csv dir - dftrain, dfval, dftest


    # Project directories  
    # Model in .json   
    model_json_dir =  root_dir +"/"+model+"/JMODEL/"          # Model in .json
    # Model in binary format 
    model_bin_dir =  root_dir +"/"+model+"/BMODEL/"           # Model in bin format
    # Results
    results_dir =  root_dir +"/"+model+"/RESULTS/"            # Results
    # directories for  Tensor_dir
    Tensor_dir =  root_dir +"/Tensorlog/"                     # Tensor Log 

#    get_ipython().run_line_magic('store', 'root_dir')
#    get_ipython().run_line_magic('store', 'json_dir')
#    get_ipython().run_line_magic('store', 'imag_dir')
#    get_ipython().run_line_magic('store', 'csv_dir')
#    get_ipython().run_line_magic('store', 'model_json_dir')
#    get_ipython().run_line_magic('store', 'model_bin_dir')
#    get_ipython().run_line_magic('store', 'results_dir')

    # Model 
    #Pre_model = "VGG16"
    #Pre_model = "ResNet50"
#    get_ipython().run_line_magic('store', 'Pre_model')

    return (root_dir,json_dir,imag_dir,csv_dir,model_json_dir,model_bin_dir,results_dir,Tensor_dir)



# Read dataframes
def read_dataframes(csv_dir):
    import pandas as pd 
# Read CSV
    dftrain = pd.DataFrame()     
    dfval = pd.DataFrame()    
    dftest = pd.DataFrame()   
    print(csv_dir)

    dftrain = pd.read_csv(csv_dir+'dftrain.csv')
    dfval  = pd.read_csv(csv_dir+'dfval.csv')
    dftest  = pd.read_csv(csv_dir+'dftest.csv')
    return ( dftrain, dfval, dftest )


# Read dataframes tables. Tables to do a Xi Test
def read_dataframes_tables (csv_dir):
    import pandas as pd 
# Read CSV
    table_bm_sex = pd.DataFrame()     
    table_bm_age = pd.DataFrame()    
    table_bm_sex_age = pd.DataFrame()   

    table_bm_sex  = pd.read_csv(csv_dir+'table_bm_sex.csv')
    table_bm_age  = pd.read_csv(csv_dir+'table_bm_age.csv')
    table_bm_sex_age = pd.read_csv(csv_dir+'table_bm_sex_age.csv')
    return ( table_bm_sex, table_bm_age, table_bm_sex_age )



# This function creates an tensor 4D or - Numpy array with dimensions (nsamples,heigh, width, color channels )
# with keras.preprocessing - slower than with tf 

def load_images(df, height_imag, width_imag):
    import tensorflow.keras.preprocessing  as tfp
    import numpy as np
    # initialize our images array 
    images = []
        

    # loop over the images in the data frame
    for index, row in df.iterrows():
        file = df['file_name_ext'][index]
        image_res = tfp.image.load_img(imag_dir+file, target_size = (height_imag,width_imag))
        image_array = tfp.image.img_to_array(image_res) / 255    # Resizing 
#            print ( image_array.shape)
# To include a new dimension - Not needed 
#            image = np.expand_dims(image_array, axis = 0)
#            print ( image.shape)
        images.append(image_array)
    return np.array(images)



# This function creates an tensor 4D or - Numpy array with dimensions (nsamples,heigh, width, color channels )
# With TF 

def load_images_tf(df, height_imag, width_imag):
    import numpy as np
    import tensorflow as tf
    import time

    images = []

    start_time = time.time()

# loop over the images in the data frame

# Place tensors on the CPU

    with tf.device('/CPU:0'):

        for index, row in df.iterrows():

            one_imag  = imag_dir + df['file_name_ext'][index]

            img = tf.io.read_file(one_imag)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [width_imag, height_imag])
            images.append(img.numpy())
   

        images = np.array(images)

    elapsed_time = time.time() - start_time
    print ( images.shape )
    time.strftime('Time spent in TF loading :'"%H:%M:%S", time.gmtime(elapsed_time))
    return np.array(images)





# One hot encoder for numbers 
def to_one_hot (labels, dimension ):
    import numpy as np
    results = np.zeros ((len(labels), dimension))
    for i, label in enumerate (labels):
        results[i,label] = 1
    return results




# Example of usage : to_one_hot
# import numpy as np
# nparray = [1,2,3]
# tcat2 = to_one_hot(nparray,7)
# print(tcat2)



def to_one_hot_words ( samples, num_words): 
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer (num_words)
    tokenizer.fit_on_texts(samples)
#     sequences = tokenizer.texts_to_sequences(samples)
   
    one_hot_result = tokenizer.texts_to_matrix(samples, mode='binary')

#    word_index = tokenizer.word_index
#    print ( 'Found %s unique tokens.' %len(word_index))
    return (one_hot_result)


# Example of use :  to_one_hot_words 
#samples = ['Uno','Uno','Dos']



# Process clinical information 
#  1. min-max scaling on continuous features  ( age )
#  2. one-hot encoding on categorical features,  (sex, site)
#  3. and then finally concatenating them together

def process_clinical_info(df):
    import numpy as np
    clinical_info = df[['age','sex','site']].values             # Create a numpy array from df
 
    
    # Compute standarized value for age
    mean_age = df.age.mean()
    std = df.age.std()
    np_age = np.zeros( (df.age.shape[0]) )
    np_age = (df.age - mean_age)/std
    np_age = np.asarray(np_age).astype('float32')    # Convert to np array float32
    np_age = np_age.reshape (np_age.shape[0],1)          # Reshape to 2 dimensions to concatenate
    
    # Vectorize the sex and site 
    num_columns = df.sex.unique().shape[0]            # Num columns to vectorize 
    np_sex = np.zeros ( (df.sex.shape[0], num_columns ))
    np_sex = to_one_hot_words ( df.sex, num_columns )
    np_sex = np.asarray(np_sex).astype('float32')
    
    # Create vectorize matrix for the complete clinical info 
    clinical_tensor = np.hstack([np_age,np_sex])

    return clinical_tensor




# Example of usage : process_clinical_info
#  Outputs the smallest binary code to the less used word !
#clinical_tensor = process_clinical_info(dftest)
#print ( clinical_tensor.ndim)
#print ( clinical_tensor.shape)
#print ( type(clinical_tensor))





#  Create labels tensor 
#  Outputs the smallest binary code to the less used word !
def create_column_tensor(column, num_cols): 
    import numpy as np
    one_hot_column = to_one_hot_words ( column, num_cols )
    tensor_column  = np.asarray(one_hot_column).astype('float32')
    return ( tensor_column )
     


# Example of usage create_column_tensor
#ncols = dftest.bm.unique().shape[0] 
#print(ncols)
#tensor_labels_test = create_column_tensor(dftest.bm, ncols)

#print ( tensor_labels_test)
#print ( dftest.bm)



def create_label_tensor(df):
    import numpy as np
    dfres = df.bm.replace(['benign','malignant'],[0,1])
    label_tensor  = np.asarray(dfres).astype('float32')
    return label_tensor

# Example of usage create_label_tensor
#test_label_tensor = create_label_tensor ()
#print ( test_label_tensor)
#print ( test_label_tensor.shape)



def save_model(model, history, dir, name):
    
# Save model
        model.save(dir+name+'.h5')

        import pickle
# Save history
# as a binary file 
        with open(dir+'modelHistory_'+name, 'wb') as file_pi:pickle.dump(history.history , file_pi)
    

def save_model_no_opt(model, history, dir, name):
    
# Save model
        model.save(dir+name+'.h5',include_optimizer=False)

        import pickle
# Save history
# as a binary file 
        with open(dir+'modelHistory_'+name, 'wb') as file_pi:pickle.dump(history.history , file_pi)
    

def load_hist_model ( dir, name):

        import pickle
# Read history
# as a binary file 
        history_dic = dict()

        with open(dir+'modelHistory_'+name,'rb') as file_pi:history_dic = pickle.load(file_pi)

        return history_dic


def model_load ( dir, name):
#        from tensorflow.keras.models import load_model
        from keras.models import load_model
# Load model      
       	model = load_model(dir+name+'.h5')
        return model 
   
# Load model in TF way 
def model_load_tf ( dir, name):
        from tensorflow.keras.models import load_model
#        from keras.models import load_model
# Load model from tf      
       	model = load_model(dir+name+'.h5')
        return model 


   

def plot_save_acc_loss(results_dir, history_dic, name):
    
    
    import matplotlib.pyplot as plt


    acc=history_dic['acc']
    val_acc= history_dic['val_acc']
    loss = history_dic['loss']
    val_loss = history_dic['val_loss']

    epochs = range(1, len(acc)+1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy '+name)
    plt.legend()

    plt.savefig(results_dir+name+'acc.png')
 
    
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss '+name)
    plt.legend()

    plt.savefig(results_dir+name+'loss.png')
    
    plt.show()


def print_network (results_dir, model, name):
	from tensorflow.keras.utils import plot_model
	plot_model (model, show_shapes=True, to_file=results_dir+name+".png")



def save_network_json (model_json_dir, model, name):
	model_json = model.to_json()
	with open(model_json_dir+ name+".json", "w") as json_file:json_file.write(model_json)


# Class to write in a file and in the screen

import sys

class Transcript(object):

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "w+")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

# end class


def start(filename):
    """Start transcript, appending print output to given filename"""
    sys.stdout = Transcript(filename)

def stop():
    """Stop transcript and return print functionality to normal"""
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal



def confusion_ROC_AUC ( y_ground, y_pred_class, y_pred_prob, class_labels, dir, name ): 
	from sklearn.metrics import confusion_matrix
	from sklearn.metrics import classification_report
	import numpy as np
	import matplotlib.pyplot as plt
	from texttable import Texttable


	cnf_matrix = confusion_matrix(y_ground, y_pred_class)


# Print metrics
	start(dir+"Metrics"+name+".txt")

	print("Metrics " + name)
	print ("===========================================")
	print("Confusion Matrix:")
	print (cnf_matrix )

	tn, fp, fn, tp = confusion_matrix(y_ground, y_pred_class).ravel()
	(tn, fp, fn, tp)

	tab2 = Texttable()
	tab2.add_rows([['Type ', '# samples  '], 
               ['True Positives- Malignants', tp], 
               ['True Negatives- Benign', tn],
               ['False Positive', fp],
               ['False Negative', fn]
              ])
	print(tab2.draw())


# Accuracy, Sensitivity, Specificity
#----------------------------------------
# Accuracy = (TP + TN)/(TP + TN + FP + FN).
	Accuracy= (tp + tn ) / ( tn + fp + fn + tp)
# Sensitivity = (tp /(tp+fn))
	Sensitivity =  tp  / ( fn + tp)
# Specificity = (tn /(tn+fp))
	Specificity =  tn / ( tn + fp)

	print('Summary classification Report: ')

#class_labels = list(test_generator.class_indices.keys())   
	report = classification_report(y_ground, y_pred_class, target_names=class_labels)

	print(report)  

# Create the ROC curve & AUC
#----------------------------
	from sklearn.metrics import roc_curve
	from sklearn.metrics import auc

# We have got the probabilities in the probabilities 
#fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
	fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_ground,y_pred_prob)

	auc_keras = auc(fpr_keras, tpr_keras)

	tab3 = Texttable()
	tab3.add_rows([['Metrics ', '       '], 
               ['Accuracy',   '%.2f' % Accuracy], 
               ['Sensitivity','%.2f' % Sensitivity],
               ['Specificity','%.2f' % Specificity],
               ['AUC',        '%.2f' % auc_keras ]
              ])
	print(tab3.draw())


# Stop printing file metrics
	stop()

# Print the ROC curves

	fig1 = plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras), figure = fig1)
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.show()
	fig1.savefig(dir + name+'_Test_ROC.png')


# Zoom in view of the upper left corner.
	plt.figure(2)
	plt.xlim(0, 0.2)
	plt.ylim(0.6, 1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve (zoomed in at top left)')
	plt.legend(loc='best')
	plt.show()


# Xi test

# chi-squared test with similar proportions
# if p-value <= alpha: significant result, reject null hypothesis (H0), dependent.
# If p-value > alpha: not significant result, fail to reject null hypothesis (H0), independent.

def xi_squared (table):
	from scipy.stats import chi2_contingency
	from scipy.stats import chi2
	import numpy as np

	print ("Xi results:" ) 

# convert into array table
	table = np.asarray(table)

# convert numbers in integers
	stat, p, dof, expected = chi2_contingency(table.astype(int))
	print('dof=%d' % dof)
	print(expected)
# interpret test-statistic
	prob = 0.95
	critical = chi2.ppf(prob, dof)
	print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
	if abs(stat) >= critical:
    		print('Dependent (reject H0)')
	else:
    		print('Independent (fail to reject H0)')
# interpret p-value
	alpha = 1.0 - prob
	print('significance=%.3f, p=%.3f' % (alpha, p))
	if p <= alpha:
    		print('Dependent (reject H0)')
	else:
    		print('Independent (fail to reject H0)')


    
def reproducible_results():

	import numpy as np
	import tensorflow as tf
	import random as rn

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

	np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

	rn.seed(12345)

# Comment for TF 2.0
# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/

#	session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
#                              inter_op_parallelism_threads=1)

	from tensorflow.keras import backend as K

# Comment for TF 2.0
# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

#	tf.random.set_random_seed(1234)

#	sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#	K.set_session(sess)


#===============================
# Create train, val & Test sets
# 1/2 malignant 1/ benignant
# % train, % val and rest test 
#==============================

def extract_images_bm ( df,benign,n_benign,n_malignant,perc_train, perc_val):
    from sklearn.utils import shuffle
    import pandas as pd
    import math

    
    df1 = df.loc[ (df['bm'] == benign )]
    df2 = df.loc[ (df['bm'] != benign )]    

    total_bm = n_benign + n_malignant  
    df3 = pd.DataFrame()  
    df4 = pd.DataFrame() 
    df5 = pd.DataFrame() 
    df3 = df1[:n_benign]
    df4 = df2[:n_malignant]
    df5 = pd.concat([df3, df4])
        
  
    df5_shuf = pd.DataFrame() 
    df5_shuf = shuffle(df5, random_state=20)
    
    
    total_images = n_benign + n_malignant
    n_train = math.ceil( total_images * perc_train / 100 )
    n_val =   math.ceil( total_images * perc_val / 100 )
    n_test = total_images - n_train - n_val 
    
   
    dftrain = pd.DataFrame()     
    dfval = pd.DataFrame()     
    dftest = pd.DataFrame()  
    
    dftrain = df5_shuf.iloc[:n_train, :]
    dfval   = df5_shuf.iloc[n_train:n_train+n_val, : ]
    dftest  = df5_shuf.iloc[n_train+n_val:n_train+n_val+n_test, : ]

    return (dftrain, dfval, dftest)



def extract_images_train ( df,a,b,c,n_train):

    import pandas as pd	
    df1 = df.loc[ (df['bm']       == a ) & 
                  (df['sex']      == b ) & 
                  (df['age_rang2']== c)]
    dftra = pd.DataFrame()     
    dftra = df1[:n_train]
    return (dftra)


# Example of finding images that are not in dftrain
# new_rows = []
# for i, row in df.iterrows():
#    while row['var1'] > 30: 
#        newrow = row
#        newrow['var2'] = 30
#        row.loc['var1'] = row['var1'] - 30
#        new_rows.append(newrow.values)
#    df_new = df.append(pd.DataFrame(new_rows, columns=df.columns)).reset_index()
#

def create_val_test ( dfimages, dftrain, n_val, n_test):

   import pandas as pd	
   from sklearn.utils import shuffle

   dfimages_shuf = shuffle(dfimages, random_state=20)	
   dfval = pd.DataFrame()
   dfval_list = []
   dftest = pd.DataFrame()
   dftest_list = []
   ntot_val_test = n_val+n_test
   n = 0
   for index, row in dfimages_shuf.iterrows():
      y = dfimages_shuf['file_name'][index]
      if n < ntot_val_test : 
         if n < n_val : 
            if not y in dftrain.values:
               dfval_list.append(row.values)
               n += 1
         else:
            if not y in dftrain.values:
               dftest_list.append(row.values)
               n += 1       
      else:
         break 
	

   dfval = dfval.append(pd.DataFrame(dfval_list, columns=dfimages.columns))   
   dftest = dftest.append(pd.DataFrame(dftest_list, columns=dfimages.columns))   

   return( dfval, dftest ) 

