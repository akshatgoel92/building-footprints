


def fit(X, Y, C):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    log_reg = LogisticRegression(C=C, verbose = True, n_jobs = -1, solver = 'saga')
    log_reg.fit(X, Y)
    
    return(log_reg)


def main():
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    print("Running...")
    root = 'GE Gorakhpur'
    image_type = 'blocks'
    
    prefix = common.get_s3_paths(root, image_type)
    suffix = '.npz'
    n = 3
    
    files = get_files(prefix, suffix)
    train = get_train_set(files, n)
    train = execute_merge(train)
    
    X_train, Y_train = reshape_df(train)
    
    hypers = [1]
    for c in hypers:
        log_reg = fit(X_train, Y_train, c)
        save_model(log_reg, 'log_reg_{}.sav'.format(str(c)))
    
    print("Done...!")
    
    
if __name__ == '__main__':
    main()