import os
from sklearn.model_selection import train_test_split
import codecs

path = "xx/xx/daguan/datagrand"


def split_raw_train_dev(path):
    train_raw_file = os.path.join(path, "train.txt")
    raw_train = list()
    with open(train_raw_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            raw_train.append(line)
    new_file = train_raw_file.replace("train.txt", "raw_train.txt")
    os.system("mv {} {}".format(train_raw_file, new_file))
    train_list, dev_list = train_test_split(raw_train, test_size=0.15, shuffle=True)
    with open(train_raw_file, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line + "\n")

    dev_file = os.path.join(path, "dev.txt")
    with open(dev_file, "w", encoding="utf-8") as f:
        for line in dev_list:
            f.write(line + "\n")


def raw2train_set(type):
    with codecs.open(os.path.join(path, '{}.txt'.format(type)), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        results = []
        for line in lines:
            features = []
            tags = []
            samples = line.strip().split('  ')
            for sample in samples:
                sample_list = sample[:-2].split('_')
                tag = sample[-1]
                features.extend(sample_list)
                tags.extend(['O'] * len(sample_list)) if tag == 'o' else tags.extend(
                    ['B-' + tag] + ['I-' + tag] * (len(sample_list) - 1))
            results.append(dict({'features': features, 'tags': tags}))
        train_write_list = []
        with codecs.open(os.path.join(path, 'dg_{}.txt'.format(type)), 'w', encoding='utf-8') as f_out:
            for result in results:
                for i in range(len(result['tags'])):
                    train_write_list.append(result['features'][i] + '\t' + result['tags'][i] + '\n')
                train_write_list.append('\n')
            f_out.writelines(train_write_list)


def raw2test(path):
    with codecs.open(os.path.join(path, 'test.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        results = []
        for line in lines:
            features = []
            sample_list = line.split('_')
            features.extend(sample_list)
            results.append(dict({'features': features}))
        test_write_list = []
        with codecs.open(os.path.join(path, 'dg_test.txt'), 'w', encoding='utf-8') as f_out:
            for result in results:
                for i in range(len(result['features'])):
                    test_write_list.append(result['features'][i] + '\n')
                test_write_list.append('\n')
            f_out.writelines(test_write_list)


def result2sumit(path):
    f_write = codecs.open(os.path.join(path, 'submit.txt'), 'w', encoding='utf-8')
    with codecs.open(os.path.join(path, 'result.txt'), 'r', encoding='utf-8') as f:
        lines = f.read().split('\n\n\n')
        for line in lines:
            if line == '':
                continue
            tokens = line.split('\n')
            features = []
            tags = []
            for token in tokens:
                feature_tag = token.split()
                features.append(feature_tag[0])
                tags.append(feature_tag[-1])
            samples = []
            i = 0
            while i < len(features):
                sample = []
                if tags[i] == 'O':
                    sample.append(features[i])
                    j = i + 1
                    while j < len(features) and tags[j] == 'O':
                        sample.append(features[j])
                        j += 1
                    samples.append('_'.join(sample) + '/o')
                else:
                    if tags[i][0] != 'B':
                        print(tags[i][0] + ' error start')
                        j = i + 1
                    else:
                        sample.append(features[i])
                        j = i + 1
                        while j < len(features) and tags[j][0] == 'I' and tags[j][-1] == tags[i][-1]:
                            sample.append(features[j])
                            j += 1
                        samples.append('_'.join(sample) + '/' + tags[i][-1])
                i = j
            f_write.write('  '.join(samples) + '\n')
