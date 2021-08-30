
import copy as copy
import itertools

def shallow_negation():
    X = ["I can't judge this @category@ movie as @augment@ @positive@",
        "I can't judge this @category@ movie as @augment@ @negative@",
        "I don't think this @category@ movie is @augment@ @positive@",
        "I don't think this @category@ movie is @augment@ @negative@",
        'this @category@ movie is not @augment@ @positive@',
        'this @category@ movie is not @augment@ @negative@',
        'it is @booltrue@ that this @category@ movie is @augment@ @positive@',
        'it is @boolfalse@ that this @category@ movie is @augment@ @positive@'
        ]
    Y = [0, 1, 0, 1, 0, 1, 1, 0]

    replace = {}  # first element of each entry is the default and preserve the original label y
    replace['@category@'] = ['', 'horror', 'comedy', 'drama', 'thriller']
    replace['@augment@'] = ['', 'very', 'incredibly', 'super', 'extremely']
    replace['@positive@'] = ['good', 'fantastic', 'nice', 'satisfactory', 'interesting']
    replace['@negative@'] = ['bad', 'poor', 'boring', 'terrible', 'awful']
    replace['@booltrue@'] = ['true', 'accurate', 'correct', 'right']
    replace['@boolfalse@'] = ['false', 'untrue', 'wrong', 'incorrect']
    label_changing_replacements = []

    # generate samples
    X_pert, Y_pert = [], []
    interventions, idx_interventions, category_intervention, num_interventions = [], [], [], []
    for x,y in zip(X, Y):
        interventions += [[]]
        idx_interventions += [[]]
        category_intervention += [[]]
        x_list = x.split(' ')
        for i,w in enumerate(x_list):
            if w in replace.keys():
                interventions[-1] += [w]
                idx_interventions[-1] += [i]
                category_intervention[-1] += [w]
        res = 0
        for i,c in enumerate(category_intervention[-1]):
            if i == 0:
                res = 1
            res *= len(replace[c])
        num_interventions += [res]
        for replacement in itertools.product(*(replace[r] for r in interventions[-1])):
            tmp = copy.copy(x_list)
            for r,i in zip(replacement, idx_interventions[-1]):
                tmp[i] = r
            X_pert += [tmp]
            # Generate the label, knowing that an intervention in the list
            #  label_changing_replacements changes the original label y iff it
            #  imposes a replacement whose index is strictly greater that 0.
            y_tmp = y
            for c,i in zip(category_intervention[-1], idx_interventions[-1]):
                if c not in label_changing_replacements:
                    continue
                else:
                    flip = lambda v: 1 if v==0 else 0
                    y_tmp = (flip(y_tmp) if replace[c].index(tmp[i])>0 else y_tmp)
            Y_pert += [y_tmp]

    return X_pert, Y_pert

def mixed_sentiment():
    X = ['@augment@ @negative@ plot @augment@ @positive@ movie',
        '@augment@ @positive@ plot @augment@ @negative@ movie',
        'it is @augment@ @negativeadverb@ acted but it is @augment@ @positive@',
        'it has @augment@ @positive@ acted but it is @augment@ @negative@',
        'despite it is @augment@ @negativeadverb@ acted this movie is @augment@ @positive@',
        'despite it has @augment@ @positive@ actors this movie is @augment@ @negative@',
        '@gender@ thinks this movie is @augment@ @positive@ and not @negative@',
        'she thinks this movie has @augment@ @negative@ actors despite it is @augment@ @positive@',
        'this movie is @augment@ @negative@ while the prequel was @augment@ @positive@',
        'despite @gender@ acted well the @category@ movie is @augment@ @negative@'
        ]
    Y = [1, 0, 1, 0, 1, 0, 1, 1, 0, 0]

    replace = {}  # first element of each entry is the default and preserve the original label y
    replace['@category@'] = ['', 'horror', 'comedy', 'drama', 'thriller']
    replace['@augment@'] = ['', 'very', 'incredibly', 'super']
    replace['@positive@'] = ['good', 'fantastic', 'nice', 'satisfactory']
    replace['@negativeadverb@'] = ['badly', 'poorly', 'terribly', 'weakly']
    replace['@negative@'] = ['bad', 'poor', 'boring', 'terrible']
    replace['@gender@'] = ['he', 'she', 'mark', 'sarah']
    label_changing_replacements = []

    # generate samples
    X_pert, Y_pert = [], []
    interventions, idx_interventions, category_intervention, num_interventions = [], [], [], []
    for x,y in zip(X, Y):
        interventions += [[]]
        idx_interventions += [[]]
        category_intervention += [[]]
        x_list = x.split(' ')
        for i,w in enumerate(x_list):
            if w in replace.keys():
                interventions[-1] += [w]
                idx_interventions[-1] += [i]
                category_intervention[-1] += [w]
        res = 0
        for i,c in enumerate(category_intervention[-1]):
            if i == 0:
                res = 1
            res *= len(replace[c])
        num_interventions += [res]
        for replacement in itertools.product(*(replace[r] for r in interventions[-1]), ):
            tmp = copy.copy(x_list)
            for r,i in zip(replacement, idx_interventions[-1]):
                tmp[i] = r
            X_pert += [tmp]
            # Generate the label, knowing that an intervention in the list
            #  label_changing_replacements changes the original label y iff it
            #  imposes a replacement whose index is strictly greater that 0.
            y_tmp = y
            for c,i in zip(category_intervention[-1], idx_interventions[-1]):
                if c not in label_changing_replacements:
                    continue
                else:
                    flip = lambda v: 1 if v==0 else 0
                    y_tmp = (flip(y_tmp) if replace[c].index(tmp[i])>0 else y_tmp)
            Y_pert += [y_tmp]

    return X_pert, Y_pert

def name_bias():
    X = ['the lord of the rings is a @augment@ @positive@ @category@ movie',
        'the lord of the rings is @augment@ @negative@ @category@ movie',
        'this @category@ movie is directed by steven spielberg and it is @augment@ @positive@',
        'this @category@ movie is directed by steven spielberg and it is @augment@ @negative@',
        'starring bruce willis this @category@ movie is indeed @augment@ @positive@',
        'starring bruce willis this @category@ movie is indeed @augment@ @negative@'
        ]
    Y = [1, 0, 1, 0, 1, 0]

    replace = {}  # first element of each entry is the default and preserve the original label y
    replace['@category@'] = ['', 'horror', 'comedy', 'drama', 'thriller']
    replace['@augment@'] = ['', 'very', 'incredibly', 'super']
    replace['@positive@'] = ['good', 'fantastic', 'nice', 'satisfactory']
    replace['@negative@'] = ['bad', 'poor', 'boring', 'terrible']
    label_changing_replacements = []

    # generate samples
    X_pert, Y_pert = [], []
    interventions, idx_interventions, category_intervention, num_interventions = [], [], [], []
    for x,y in zip(X, Y):
        interventions += [[]]
        idx_interventions += [[]]
        category_intervention += [[]]
        x_list = x.split(' ')
        for i,w in enumerate(x_list):
            if w in replace.keys():
                interventions[-1] += [w]
                idx_interventions[-1] += [i]
                category_intervention[-1] += [w]
        res = 0
        for i,c in enumerate(category_intervention[-1]):
            if i == 0:
                res = 1
            res *= len(replace[c])
        num_interventions += [res]
        for replacement in itertools.product(*(replace[r] for r in interventions[-1]), ):
            tmp = copy.copy(x_list)
            for r,i in zip(replacement, idx_interventions[-1]):
                tmp[i] = r
            X_pert += [tmp]
            # Generate the label, knowing that an intervention in the list
            #  label_changing_replacements changes the original label y iff it
            #  imposes a replacement whose index is strictly greater that 0.
            y_tmp = y
            for c,i in zip(category_intervention[-1], idx_interventions[-1]):
                if c not in label_changing_replacements:
                    continue
                else:
                    flip = lambda v: 1 if v==0 else 0
                    y_tmp = (flip(y_tmp) if replace[c].index(tmp[i])>0 else y_tmp)
            Y_pert += [y_tmp]

    return X_pert, Y_pert

def sarcasm():
    X = ['wow is this even a @augment@ @positive@ @category@ movie ?',
        'I have had mosquito bites that are better than this @augment@ @positive@ @category@ movie',
        'this @category@ movie might not be preferable to simply staring into your empty airsick bag @augment@ @positive@',
        'pneumonia is better than this @augment@ @positive@ @category@ movie',
        'throw this @augment@ long @category@ movie into the ocean and thank me later',
        'starring @name@ @surname@ for a @category@ movie is like waking up on monday morning'
        ]
    Y = [0, 0, 0, 0, 0, 0]

    replace = {}  # first element of each entry is the default and preserve the original label y
    replace['@category@'] = ['', 'horror', 'comedy', 'drama', 'thriller', 'noir']
    replace['@augment@'] = ['', 'very', 'incredibly', 'super', 'extremely']
    replace['@positive@'] = ['good', 'fantastic', 'nice', 'satisfactory', 'interesting']
    replace['@negative@'] = ['bad', 'poor', 'boring', 'terrible', 'awful']
    replace['@name@'] = ['bruce', 'john', 'mark', 'matt', 'sam']
    replace['@surname@'] = ['willis', 'lee', 'demon', 'spencer', 'jolie']
    label_changing_replacements = []

    # generate samples
    X_pert, Y_pert = [], []
    interventions, idx_interventions, category_intervention, num_interventions = [], [], [], []
    for x,y in zip(X, Y):
        interventions += [[]]
        idx_interventions += [[]]
        category_intervention += [[]]
        x_list = x.split(' ')
        for i,w in enumerate(x_list):
            if w in replace.keys():
                interventions[-1] += [w]
                idx_interventions[-1] += [i]
                category_intervention[-1] += [w]
        res = 0
        for i,c in enumerate(category_intervention[-1]):
            if i == 0:
                res = 1
            res *= len(replace[c])
        num_interventions += [res]
        for replacement in itertools.product(*(replace[r] for r in interventions[-1]), ):
            tmp = copy.copy(x_list)
            for r,i in zip(replacement, idx_interventions[-1]):
                tmp[i] = r
            X_pert += [tmp]
            # Generate the label, knowing that an intervention in the list
            #  label_changing_replacements changes the original label y iff it
            #  imposes a replacement whose index is strictly greater that 0.
            y_tmp = y
            for c,i in zip(category_intervention[-1], idx_interventions[-1]):
                if c not in label_changing_replacements:
                    continue
                else:
                    flip = lambda v: 1 if v==0 else 0
                    y_tmp = (flip(y_tmp) if replace[c].index(tmp[i])>0 else y_tmp)
            Y_pert += [y_tmp]

    return X_pert, Y_pert
