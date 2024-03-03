from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def feature_extraction(training, training_feature, isTraining=True):
    # read training file
    with open(training, "r") as f:
        lines = f.readlines()

    features = []

    # extract features
    for i, line in enumerate(lines):
        if line == "\n":
            features.append("\n")
        else:
            if isTraining:
                word, pos, bio = line.strip().split()
            else:
                word, pos = line.strip().split()
            stem = stemmer.stem(word)

            # previous word and pos
            if i == 0:
                prev_word = "<>"
                prev_pos = "<>"
            else:
                if lines[i - 1] == "\n":
                    prev_word = "<>"
                    prev_pos = "<>"
                    prev_stem = "<>"
                else:
                    if isTraining:
                        prev_word, prev_pos, _ = lines[i - 1].strip().split()
                    else:
                        prev_word, prev_pos = lines[i - 1].strip().split()
                    prev_stem = stemmer.stem(prev_word)

            # next word and pos
            if i == len(lines) - 1:
                next_word = "<>"
                next_pos = "<>"
            else:
                if lines[i + 1] == "\n":
                    next_word = "<>"
                    next_pos = "<>"
                    next_stem = "<>"
                else:
                    if isTraining:
                        next_word, next_pos, _ = lines[i + 1].strip().split()
                    else:
                        next_word, next_pos = lines[i + 1].strip().split()
                    next_stem = stemmer.stem(next_word)

            # capitalization
            isCapitalized = word[0].isupper()

            # other features
            
            
            # Compile features
            features.append(
                f"{word}\tPOS={pos}\tprevious_word={prev_word}\tprevious_pos={prev_pos}\tprevious_stem={prev_stem}\tnext_word={next_word}\tnext_pos={next_pos}\tnext_stem={next_stem}\tisCapitalized={isCapitalized}"
            )
            if isTraining:
                features.append(f"\t{bio}\n")
            else:
                features.append("\n")


    with open(training_feature, "w") as f:
        for feature in features:
            f.write(feature)


def main(*args, **kwargs):
    training = args[0]
    training_feature = args[1]
    feature_extraction(training, training_feature)

    dev = args[2]
    test_feature = args[3]
    feature_extraction(dev, test_feature, isTraining=False)


if __name__ == "__main__":
    main("WSJ_02-21.pos-chunk", "training.feature", "WSJ_24.pos", "test.feature")
