
import copy as copy
import itertools

def shallow_negation(bert_format=False):
    X = ["not so @negative@ as a @category@ film",
        "not a @augment@ @negative@ @category@ movie",
        "not a @augment@ @positive@ @category@ film",
        "not a @augment@ @positive@ @category@ movie",
        "I can't judge this @category@ movie as @augment@ @positive@",
        "I can't judge this @category@ movie as @augment@ @negative@",
        "I don't think this @category@ movie is @augment@ @positive@",
        "I don't think this @category@ movie is @augment@ @negative@",
        'this @category@ movie is not @augment@ @positive@',
        'this @category@ movie is not @augment@ @negative@',
        'it is @booltrue@ that this @category@ movie is @augment@ @positive@',
        'it is @boolfalse@ that this @category@ movie is @augment@ @positive@'
        ]
    Y = [1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0]

    replace = {}  # first element of each entry is the default and preserve the original label y
    replace['@category@'] = ['', 'horror', 'comedy', 'drama', 'thriller', 'noir']
    replace['@augment@'] = ['', 'very', 'incredibly', 'super', 'extremely', 'very very']
    replace['@positive@'] = ['good', 'fantastic', 'nice', 'satisfactory', 'interesting', 'funny']
    replace['@negative@'] = ['bad', 'poor', 'boring', 'terrible', 'awful', 'dreadful']
    replace['@booltrue@'] = ['true', 'accurate', 'correct', 'right', 'not false', 'not wrong']
    replace['@boolfalse@'] = ['false', 'untrue', 'wrong', 'incorrect', 'not true', 'not right']
    label_changing_replacements = []

    if bert_format is True:
        return X, Y, replace, label_changing_replacements

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

def mixed_sentiment(bert_format=False):
    X = ['the expectation was for a @augment@ @negative@ but in the end it is @augment@ @positive@',
        '@augment@ @negative@ where it should be @augment@ @positive@',
        '@augment@ @negative@ plot @augment@ @positive@ movie',
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
    Y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0]

    replace = {}  # first element of each entry is the default and preserve the original label y
    replace['@category@'] = ['', 'horror', 'comedy', 'drama', 'thriller', 'noir']
    replace['@augment@'] = ['', 'very', 'incredibly', 'super', 'extremely', 'very very']
    replace['@positive@'] = ['good', 'fantastic', 'nice', 'satisfactory', 'interesting', 'funny']
    replace['@negative@'] = ['bad', 'poor', 'boring', 'terrible', 'awful', 'dreadful']
    replace['@booltrue@'] = ['true', 'accurate', 'correct', 'right', 'not false', 'not wrong']
    replace['@boolfalse@'] = ['false', 'untrue', 'wrong', 'incorrect', 'not true', 'not right']
    replace['@gender@'] = ['he', 'she', 'mark', 'sarah', 'marta', 'matthew']
    label_changing_replacements = []

    if bert_format is True:
        return X, Y, replace, label_changing_replacements

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

def name_bias(bert_format=False):
    X = ['the lord of the rings is a @augment@ @positive@ @category@ movie',
        'the lord of the rings is @augment@ @negative@ @category@ movie',
        'this @category@ movie is directed by steven spielberg and it is @augment@ @positive@',
        'this @category@ movie is directed by steven spielberg and it is @augment@ @negative@',
        'starring @name@ @surname@ this @category@ movie is indeed @augment@ @positive@',
        'starring @name@ @surname@ this @category@ movie is indeed @augment@ @negative@'
        ]
    Y = [1, 0, 1, 0, 1, 0]

    replace = {}  # first element of each entry is the default and preserve the original label y
    replace['@category@'] = ['', 'horror', 'comedy', 'drama', 'thriller', 'noir']
    replace['@augment@'] = ['', 'very', 'incredibly', 'super', 'extremely', 'very very']
    replace['@positive@'] = ['good', 'fantastic', 'nice', 'satisfactory', 'interesting', 'funny']
    replace['@negative@'] = ['bad', 'poor', 'boring', 'terrible', 'awful', 'dreadful']
    replace['@name@'] = ['bruce', 'john', 'mark', 'louise', 'sarah', 'marta']
    replace['@surname@'] = ['willis', 'lee', 'demon', 'spencer', 'jolie', 'spielberg']
    label_changing_replacements = []

    if bert_format is True:
        return X, Y, replace, label_changing_replacements

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

def sarcasm(bert_format=False):
    X = ['this movie is exactly the opposite of a @augment@ @positive@ film',
        'sickness is a @augment@ @positive@ thing compared to this @category@ movie',
        'starring @name@ @surname@ i would prefer to be killed rather than watching this @category@ movie',
        'wow is this even a @augment@ @positive@ @category@ movie ?',
        'I have had mosquito bites that are better than this @augment@ @positive@ @category@ movie',
        'this @category@ movie might not be preferable to simply staring into your empty airsick bag @augment@ @positive@',
        'pneumonia is better than this @augment@ @positive@ @category@ movie',
        'throw this @augment@ long @category@ movie into the ocean and thank me later',
        'starring @name@ @surname@ for a @category@ movie is like waking up on monday morning'
        ]
    Y = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    replace = {}  # first element of each entry is the default and preserve the original label y
    replace['@category@'] = ['', 'horror', 'comedy', 'drama', 'thriller', 'noir']
    replace['@augment@'] = ['', 'very', 'incredibly', 'super', 'extremely', 'very very']
    replace['@positive@'] = ['good', 'fantastic', 'nice', 'satisfactory', 'interesting', 'funny']
    replace['@negative@'] = ['bad', 'poor', 'boring', 'terrible', 'awful', 'dreadful']
    replace['@booltrue@'] = ['true', 'accurate', 'correct', 'right', 'not false', 'not wrong']
    replace['@boolfalse@'] = ['false', 'untrue', 'wrong', 'incorrect', 'not true', 'not right']
    replace['@name@'] = ['bruce', 'john', 'mark', 'louise', 'sarah', 'marta']
    replace['@surname@'] = ['willis', 'lee', 'demon', 'spencer', 'jolie', 'spielberg']
    label_changing_replacements = []

    if bert_format is True:
        return X, Y, replace, label_changing_replacements

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
