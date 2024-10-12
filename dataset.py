# dataset.py

import random

def generate_sentences(num_sentences=10000):
    # 分类主语
    human_subjects = {
        "The boy": ["runs", "jumps", "plays", "reads", "writes"],
        "A girl": ["runs", "jumps", "plays", "reads", "writes"],
        "The teacher": ["teaches", "writes", "reads"],
        "The student": ["studies", "reads", "writes"],
        "My father": ["drives", "works", "cooks", "reads"],
        "His mother": ["cooks", "reads", "works", "writes"],
        "An artist": ["paints", "draws"],
        "The doctor": ["treats", "writes", "reads"],
        "A nurse": ["cares for", "helps"],
        "The farmer": ["plants", "harvests", "feeds"],
        "The musician": ["plays", "composes"],
        "An engineer": ["designs", "builds"],
        "The pilot": ["flies"],
        "The chef": ["cooks", "prepares"],
        "The policeman": ["protects", "patrols"],
        "The fireman": ["extinguishes", "rescues"]
    }

    animal_subjects = {
        "The cat": ["sleeps", "jumps", "runs", "eats", "plays"],
        "A dog": ["runs", "barks", "eats", "plays", "sleeps"],
        "The bird": ["flies", "sings", "nests"],
        "The fish": ["swims"],
        "The horse": ["runs", "gallops", "eats"],
        "A cow": ["eats", "walks", "lies down"]
    }

    # 合并所有主语
    subjects = {**human_subjects, **animal_subjects}

    # 定义动词和宾语
    verbs_transitive = {
        # 人类动作
        "eats": {
            "objects": ["the food", "an apple", "the cake", "breakfast", "lunch", "dinner"],
            "adverbs": ["quickly", "slowly", "happily", "hungrily"]
        },
        "plays": {
            "objects": ["the piano", "the guitar", "the game", "soccer", "basketball"],
            "adverbs": ["happily", "skillfully", "enthusiastically"]
        },
        "reads": {
            "objects": ["a book", "the newspaper", "a letter"],
            "adverbs": ["quietly", "attentively"]
        },
        "writes": {
            "objects": ["a letter", "a report", "a story"],
            "adverbs": ["carefully", "quickly"]
        },
        "paints": {
            "objects": ["a picture", "a portrait", "a landscape"],
            "adverbs": ["beautifully", "creatively"]
        },
        "drives": {
            "objects": ["a car", "the bus", "the truck"],
            "adverbs": ["carefully", "slowly", "quickly"]
        },
        "builds": {
            "objects": ["a house", "a bridge", "a tower"],
            "adverbs": ["skillfully", "carefully"]
        },
        "cooks": {
            "objects": ["dinner", "a meal", "breakfast"],
            "adverbs": ["deliciously", "quickly"]
        },
        "studies": {
            "objects": ["math", "science", "history"],
            "adverbs": ["diligently", "attentively"]
        },
        "teaches": {
            "objects": ["math", "science", "history"],
            "adverbs": ["patiently", "effectively"]
        },
        # 更多人类动作...
    }

    verbs_intransitive = {
        "sleeps": {
            "adverbs": ["peacefully", "quietly"],
            "prepositions": ["in", "on"],
            "places": ["the bed", "the sofa"]
        },
        "runs": {
            "adverbs": ["quickly", "fast", "energetically"],
            "prepositions": ["in", "through", "around"],
            "places": ["the park", "the field"]
        },
        "jumps": {
            "adverbs": ["high", "playfully"],
            "prepositions": ["over"],
            "places": ["the fence", "the hurdle"]
        },
        "flies": {
            "adverbs": ["gracefully", "swiftly"],
            "prepositions": ["over"],
            "places": ["the trees", "the mountains"]
        },
        "swims": {
            "adverbs": ["smoothly", "quickly"],
            "prepositions": ["in"],
            "places": ["the lake", "the river"]
        },
        # 更多不及物动词...
    }

    sentences = []
    attempts = 0  # 防止无限循环
    while len(sentences) < num_sentences and attempts < num_sentences * 10:
        attempts += 1
        subject = random.choice(list(subjects.keys()))
        subject_verbs = subjects[subject]
        verb = random.choice(subject_verbs)

        if subject in human_subjects:
            # 人类主语可以使用所有定义的动词
            if verb in verbs_transitive:
                verb_info = verbs_transitive[verb]
                obj = random.choice(verb_info["objects"])
                adverb = random.choice(verb_info["adverbs"] + [''])  # 可选副词
                components = [subject, verb, obj]
                if adverb:
                    components.append(adverb)
                sentence = ' '.join(components) + '.'

            elif verb in verbs_intransitive:
                verb_info = verbs_intransitive[verb]
                adverb = random.choice(verb_info["adverbs"] + [''])  # 可选副词
                preposition = random.choice(verb_info.get("prepositions", ['']) + [''])  # 可选介词
                place = random.choice(verb_info.get("places", ['']) + [''])  # 可选地点
                components = [subject, verb]
                if preposition and place:
                    components.extend([preposition, place])
                if adverb:
                    components.append(adverb)
                sentence = ' '.join(filter(None, components)) + '.'

            else:
                continue  # 如果动词不在已定义的动词列表中，跳过

        elif subject in animal_subjects:
            # 动物主语只能使用动物可以执行的动词
            if verb in ["sleeps", "runs", "jumps", "eats", "barks", "flies", "swims", "nests", "gallops", "walks", "lies down"]:
                if verb in verbs_intransitive:
                    verb_info = verbs_intransitive[verb]
                    adverb = random.choice(verb_info["adverbs"] + [''])  # 可选副词
                    preposition = random.choice(verb_info.get("prepositions", ['']) + [''])  # 可选介词
                    place = random.choice(verb_info.get("places", ['']) + [''])  # 可选地点
                    components = [subject, verb]
                    if preposition and place:
                        components.extend([preposition, place])
                    if adverb:
                        components.append(adverb)
                    sentence = ' '.join(filter(None, components)) + '.'

                elif verb == "eats":
                    verb_info = {
                        "objects": ["grass", "meat", "fish", "food"],
                        "adverbs": ["quickly", "hungrily"]
                    }
                    obj = random.choice(verb_info["objects"])
                    adverb = random.choice(verb_info["adverbs"] + [''])  # 可选副词
                    components = [subject, verb, obj]
                    if adverb:
                        components.append(adverb)
                    sentence = ' '.join(components) + '.'

                else:
                    continue  # 如果动词不在已定义的动词列表中，跳过

            else:
                continue  # 动物不能执行其他动词

        else:
            continue  # 未知的主语，跳过

        # 逻辑检查
        illogical = False

        # 动物不能拥有某些宾语
        if subject in animal_subjects and verb == "eats" and obj in ["breakfast", "lunch", "dinner"]:
            illogical = True

        if illogical:
            continue

        sentences.append(sentence)

    return sentences[:num_sentences]

if __name__ == "__main__":
    sentences = generate_sentences()
    with open("custom_dataset.txt", "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")
