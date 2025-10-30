import os, time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import seaborn as sns
import matplotlib.font_manager as font_manager
import pandas as pd
import numpy as np
import random,sys
import matplotlib.pyplot as plt

def main(argv):
    random.seed(0)


    global folder 
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    folder = 'only_morgan_SMOTE_{}'.format(current_time)
    if os.path.isdir(folder) is False:
        os.makedirs(folder)
        
    # data pre-processing
    big_feat = [2,3,4,5,6,7,8]
    # all_M = list(range(11,523))
    # all_M = list(range(1,2049))
    desire_feat_ind = []
    # data,label,data_reduce,label_reduce,features = data_process('property_Hutch_fp_Morgan.txt',desire_feat_ind=desire_feat_ind)
    data,label,data_reduce,label_reduce,features = data_process('combined_result_4096_bit.txt',desire_feat_ind=desire_feat_ind)
    
    # train the model
    stats,clf = rf_train(data,label,epoch=10)
    cm = confusion_matrix(stats['test_true'],stats['test_pred'])
    print(cm)
    
    print('mean train accuracy: {:<3.5f} +- {:<3.5f}'.format(stats['mean_train_acc'],stats['mean_train_std']))
    print('mean test accuracy: {:<3.5f} +- {:<3.5f}'.format(stats['mean_test_acc'],stats['mean_test_std']))
    with open('{}/results.txt'.format(folder),'w') as f:
        f.write('mean train accuracy: {:<3.5f} +- {:<3.5f}\n'.format(stats['mean_train_acc'],stats['mean_train_std']))
        f.write('mean test accuracy: {:<3.5f} +- {:<3.5f}\n'.format(stats['mean_test_acc'],stats['mean_test_std']))
        
        # analysis
        roc_auc = roc_analysis(stats,clf)
        feat_import_dict, feat_import_std = feat_importance(stats,clf,features,plot_flag=False)
        confusion_analysis(stats)
    
        # writing results
        # f.write('ROC_AUC:{:<3.5f}\n\n'.format(roc_auc))
        
        f.write('{:<20s} {:<20s} {:<20s}\n'.format('feature','importance','std'))
        for i in feat_import_dict.keys():
            f.write('{:<20s} {:<20.5e} {:<20.5e}\n'.format(i,feat_import_dict[i],feat_import_std[i]))

    # use the model to predict exp
    # data_exp,label_exp,_,_,_ = data_process('property_4mer_fp_Morgan.txt',desire_feat_ind=desire_feat_ind)
    # predict_label = clf.predict(data_exp)

    # correct = 0
    # total = 0
    # for count_i,i in enumerate(label_exp):
    #   if i == predict_label[count_i]:
    #      correct += 1
    #   total += 1
    # print(total-correct)
    # print('Testing accuracy: {:<4.2f}'.format(correct/total))
      
    #print('original label')
    #print(label_exp)
    #print('predicted label')
    #print(predict_label)
    
        
        
def feat_importance(stats,clf,features,plot_flag=True):
    #feature permutation
    result = permutation_importance(clf, stats['X_test'], stats['y_test'], n_repeats=10, random_state=42, n_jobs=2)
    # forest_importances = pd.Series(result.importances_mean, index=features)
    feat_import_std = {}
    feat_import_dict = {}
    for count_i,i in enumerate(result.importances_mean):
        feat_import_dict[features[count_i]] = i
        feat_import_std[features[count_i]]= result.importances_std[count_i]
    feat_import_dict = {k: v for k, v in sorted(feat_import_dict.items(), key=lambda item: item[1],reverse=True)}
    if plot_flag:
        plot_bar(feat_import_dict,feat_import_std,folder=folder)

    return feat_import_dict,feat_import_std
    
def roc_analysis(stats,clf):
    # get roc/auc info
    Y_score = clf.predict_proba(stats['X_test'])[:,1]
    fpr, tpr, threshold = roc_curve(stats['test_true'],Y_score)
    # use probability instead of real label is probably more appropriate
    #roc_auc = roc_auc_score(stats['test_true'], stats['test_pred'])
    roc_auc = auc(fpr, tpr)
    plot_single(fpr,tpr,roc_auc,folder=folder)
    
    return roc_auc

def confusion_analysis(stats):
    cm = confusion_matrix(stats['test_true'],stats['test_pred'])
    tn, fp, fn, tp = cm.ravel()
    labels = ['True Neg','False Pos','False Neg','True Pos']
    make_confusion_matrix(cm,group_names=labels,figsize=(8,6),folder=folder)
    
    return

def rf_train(data,label,epoch=1):
    train_score = []
    test_score = []
    oversample = SMOTE(sampling_strategy='minority')
    X_over, y_over = oversample.fit_resample(data, label)
    # this is not really cross-validation, it's just doing multiple splits to get statistics
    # like JACS paper because they have data deficiency, this is not necessary for random forest
    # for i in range(epoch):
    #X_train, X_test, y_train, y_test = train_test_split(
    #    data, label, stratify=label)
    X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2,random_state=0)
    # X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2,random_state=0)
    # forest =RandomForestClassifier(random_state=0, bootstrap=True, max_depth=15, oob_score=True)
    # forest = RandomForestClassifier(random_state=0)
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [10, 15, 20, 25],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    # }
    # Best parameters found:  {'max_depth': 25, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
    # grid_search = GridSearchCV(estimator=forest, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    forest = RandomForestClassifier(random_state=0, max_depth=25, min_samples_leaf=1, min_samples_split=2, n_estimators=300)
    clf = forest.fit(X_train, y_train)
    # clf = grid_search.fit(X_train, y_train)
    # best_model = clf.best_estimator_
    # print("Best parameters found: ", clf.best_params_)
    # y_pred = best_model.predict(X_test)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    print('Confusion matrix')
    print(cm)
    # train accuracy
    
    # train_score.append(best_model.score(X_train,y_train))
    # test_score.append(best_model.score(X_test,y_test))
    train_score.append(clf.score(X_train,y_train))
    test_score.append(clf.score(X_test,y_test))

    stats = {}
    stats['test_pred'] = clf.predict(X_test) # predicted label for test dataset
    stats['test_true'] = y_test # true label for test dataset
    stats['X_test'] = X_test
    stats['y_test'] = y_test
    stats['mean_train_acc'] = np.mean(np.array(train_score))
    stats['mean_train_std'] = np.std(np.array(train_score))
    stats['mean_test_acc'] = np.mean(np.array(test_score))
    stats['mean_test_std'] = np.std(np.array(test_score))

    # stats['test_pred'] = best_model.predict(X_test) # predicted label for test dataset
    # stats['test_true'] = y_test # true label for test dataset
    # stats['X_test'] = X_test
    # stats['y_test'] = y_test
    # stats['mean_train_acc'] = np.mean(np.array(train_score))
    # stats['mean_train_std'] = np.std(np.array(train_score))
    # stats['mean_test_acc'] = np.mean(np.array(test_score))
    # stats['mean_test_std'] = np.std(np.array(test_score))

    return stats,clf
    # return stats,best_model

# def rf_train_new(data,label,epoch=1):
#     train_score = []
#     test_score = []
#     # this is not really cross-validation, it's just doing multiple splits to get statistics
#     # like JACS paper because they have data deficiency, this is not necessary for random forest
#     X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2,random_state=0)
#     tuned_parameters = [{'max_depth':[9],'bootstrap':[True],'oob_score':[False]}]
#     clf = GridSearchCV(RandomForestClassifier(),tuned_parameters,scoring='accuracy',return_train_score=True)
#     clf.fit(X_train,y_train)

#     print(clf.best_params_)
#     stats = {}
#     stats['mean_train_acc'] = clf.cv_results_["mean_train_score"][clf.best_index_] 
#     stats['mean_train_std'] = clf.cv_results_["std_train_score"][clf.best_index_] 
#     stats['mean_test_acc'] = clf.cv_results_["mean_test_score"][clf.best_index_] 
#     stats['mean_test_std'] = clf.cv_results_["std_test_score"][clf.best_index_] 
#     clf = clf.best_estimator_

#     stats['test_pred'] = clf.predict(X_test) # predicted label for test dataset
#     stats['test_true'] = y_test # true label for test dataset
#     stats['X_test'] = X_test
#     stats['y_test'] = y_test

#     return stats,clf

def data_process(filename,desire_feat_ind=[]):
    data = []
    label = []
    with open(filename,'r') as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            if lc == 0: 
                if desire_feat_ind == []:
                    features = fields[1:-1] # including all feat, 0: name, 1: linker, -1: label
                else:
                    features = [ fields[_] for _ in desire_feat_ind]
                continue
            if desire_feat_ind == []:
                data.append([float(i) for i in fields[1:-1]])
            else:
                tmp_feat = [ float(fields[_]) for _ in desire_feat_ind]
                data.append(tmp_feat)
            if int(fields[-1]) == 0: 
               label.append(0) # 0 is stable
            else:
               label.append(1) # 1 is metastable, 2 is unstable
            # label.append(int(fields[-1]))


    # reduce the amount of the data so that every class has the same size
    # index for stable and unstable data
    index_data = [[count_i for count_i,i in enumerate(label) if i == _] for _ in range(2)]
    # find which category has fewer data
    min_amount = min([len([count_i for count_i,i in enumerate(label) if i == _]) for _ in range(2)])
    data_reduce = []
    label_reduce = []
    for count_i,i in enumerate(index_data):
        for j in random.sample(i,min_amount):
            data_reduce.append(data[j])
            label_reduce.append(count_i)

    data_reduce = np.array(data_reduce)
    label_reduce = np.array(label_reduce)
    data = np.array(data)
    label = np.array(label)
    
    return data,label,data_reduce,label_reduce,features

# source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=(4,3),
                          cmap='Blues',
                          title=None,folder='./'):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
            stats_text = "\n\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    # afont = {'fontname':'Arial','fontsize':22}
    afont = {'fontsize':22}
    plt.figure(figsize=figsize)
    sns.set(font_scale=2) 
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)
    
    if xyplotlabels:
        plt.ylabel('True label',**afont)
        plt.xlabel('Predicted label' + stats_text,**afont)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
    pngname = '{}/metrics.png'.format(folder)
    plt.tight_layout()
    plt.savefig(pngname, dpi=300,bbox_inches='tight')
    #plt.show()
    plt.close()    
    
    return

def plot_bar(y,std,folder='.'):
    plt.style.use('default')
    # Generate histogram only
    color_list = [(0.05,0.35,0.75),(0.05,0.8,0.6),(0.9,0.3,0.05),(0.35,0.7,0.9),(0.9,0.5,0.7),(0.9,0.6,0.05),(0.95,0.9,0.25),(0.05,0.05,0.05)]*10   # Supposedly more CB-friendly
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    width = 0.15
    binwidth=0.1
    X = np.arange(1)
    middle=1
    y = {k: v for k, v in sorted(y.items(), key=lambda item: item[1],reverse=True)}
    for count_i,i in enumerate(y):
        ax.bar(X+width*(count_i-middle),y[i],yerr=std[i],color=color_list[count_i],width=binwidth,label=i,capsize=10)

    # Format ticks and plot box
    ax.tick_params(axis='both', which='major',labelsize=12,pad=10,direction='out',width=2,length=4)
    ax.tick_params(axis='both', which='minor',labelsize=12,pad=10,direction='out',width=2,length=3)

    [j.set_linewidth(2) for j in ax.spines.values()]
 
    ax.set_ylabel("Mean accuracy decrease",fontsize=24,labelpad=10,fontname='Arial')
    #ax.set_xticks([-0.15,0,0.15,0.3,0.45,0.6])
    custom_xticks = list(np.arange(-0.15,-0.15+0.15*(len(y.keys())),0.15))
    print(custom_xticks)
    ax.set_xticks(custom_xticks)
    ax.set_xticklabels(y.keys(),fontsize=20,rotation = 45, ha="right")
    ax.set_yticklabels([ '{:<2.2f}'.format(_) for _ in ax.get_yticks()],fontsize=24)
    # Save the figure
    pngname = '{}/feat_importance.png'.format(folder)
    #plt.tight_layout()
    plt.savefig(pngname, dpi=300,bbox_inches='tight',transparent=False)
    #plt.show()
    plt.close(fig)


def plot_single(x,y,auc,folder='./'):
    plt.style.use('default')
    # Generate histogram only
    color_list = [(0.05,0.35,0.75),(0.05,0.8,0.6),(0.9,0.3,0.05),(0.35,0.7,0.9),(0.9,0.5,0.7),(0.9,0.6,0.05),(0.95,0.9,0.25),(0.05,0.05,0.05)]*10   # Supposedly more CB-friendly
    fig,ax = plt.subplots(1,1,figsize=(7,5))

    ax.plot([0,1],[0,1],'k--',linewidth=3.0)
    ax.plot(x,y,markersize=0,markeredgewidth=0.0,marker='.',linestyle='-',linewidth=3.0,color=color_list[0],label='AUC = {:3.4f}'.format(auc))

    # Format ticks and plot box
    ax.tick_params(axis='both', which='major',labelsize=12,pad=10,direction='out',width=2,length=4)
    ax.tick_params(axis='both', which='minor',labelsize=12,pad=10,direction='out',width=2,length=3)
    ax.tick_params(axis='both', labelsize='24')

    [j.set_linewidth(2) for j in ax.spines.values()]
    #ax.set_title('{}'.format(ligand),fontsize=20,fontweight='bold')
    #ax[ax_row,ax_col].set_ylabel("FE(kcal/mol)",fontsize=20,labelpad=10,fontweight='bold')
    ax.set_ylabel('True Positive Rate',fontsize=28,labelpad=10,)
    #ax.set_xlabel("angle(rad)",fontsize=20,labelpad=10,fontweight='bold')
    ax.set_xlabel('False Positive Rate',fontsize=28,labelpad=10,)

    # ticks value setting
    ax.set_xlim([-0.05,1.0])
    ax.set_ylim([0.0,1.05])
    ax.grid(True)

    ax.set_yticklabels([ '{:<2.2f}'.format(_) for _ in ax.get_yticks()],fontsize=24)
    ax.set_xticklabels([ '{:<2.2f}'.format(_) for _ in ax.get_xticks()],fontsize=24)  

    handles, labels = ax.get_legend_handles_labels()
    font = font_manager.FontProperties(weight='normal',
                                   style='normal', size=24)
    lgd = ax.legend(handles, labels, loc='lower right',shadow=True,fancybox=True,frameon=True,prop=font,handlelength=2.5)

    # Save the figure
    pngname = '{}/ROC.png'.format(folder)
    plt.tight_layout()
    plt.savefig(pngname, dpi=300,bbox_inches='tight',transparent=False)
    #plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main(sys.argv[1:])
