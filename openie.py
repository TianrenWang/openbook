from pyopenie import OpenIE5
import csv
dir_path = "data/"
filename = 'openbook_facts.txt'

def get_relationship(extraction):
    arg1 = extraction['arg1']['text']
    rel = extraction['rel']['text']
    arg2s = extraction['arg2s']

    relationship = [arg1, rel]

    for arg in arg2s:
        relationship.append(arg['text'])
        relationship.append(None)

    relationship.pop()
    return relationship

def offset_complete(extraction):
    sentence = extraction['sentence']
    sen_len = len(sentence)
    arg1 = extraction['extraction']['arg1']
    arg2s = extraction['extraction']['arg2s']
    if len(arg2s) == 0:
        return False
    lastArgOffset = arg2s[len(arg2s) - 1]['offsets']

    if arg1['offsets'][0][0] != 0 or lastArgOffset[0][len(lastArgOffset[0]) - 1] != sen_len - 1:
        return False
    return True

def get_max_offset(extraction):
    max = 0
    arg2s = extraction['arg2s']

    for arg in arg2s:
        if max < len(arg['offsets'][0]):
            max = len(arg['offsets'][0])

    if max < len(extraction['arg1']['offsets'][0]):
        max = len(extraction['arg1']['offsets'][0])

    return max

def get_first_offset(extraction):
    arg1 = extraction['arg1']
    return arg1['offsets'][0][0]


if __name__ == '__main__':
    extractor = OpenIE5('http://localhost:8000')
    excepted_facts = []

    data = open(dir_path + filename, "r")
    line = data.readline().capitalize()

    counter = 0
    removed = 0

    relationships = []

    while line:
        counter += 1
        line = line[:-1]

        extractions = extractor.extract(line)
        # print(extractions)
        if len(extractions) == 0:
            pass
        elif len(extractions) == 1:
            extraction = extractions[0]['extraction']
            relationships.append(get_relationship(extraction))
        elif len(extractions) == 2:
            extraction1 = extractions[0]['extraction']
            extraction2 = extractions[1]['extraction']

            try:
                complete1 = offset_complete(extractions[0])
                complete2 = offset_complete(extractions[1])
                confidence1 = extractions[0]['confidence']
                confidence2 = extractions[1]['confidence']
                if complete1 and complete2:
                    max1 = get_max_offset(extraction1)
                    max2 = get_max_offset(extraction2)
                    if max1 < max2:
                        relationships.append(get_relationship(extraction1))
                    else:
                        relationships.append(get_relationship(extraction2))
                else:
                    extract1first = get_first_offset(extraction1)
                    extract2first = get_first_offset(extraction2)
                    if extract1first < extract2first:
                        first = extraction1
                        second = extraction2
                    else:
                        second = extraction1
                        first = extraction2
                    # if complete1:
                    #     complete_extra = extraction1
                    #     incomplete_extra = extraction2
                    # else:
                    #     complete_extra = extraction2
                    #     incomplete_extra = extraction1
                    firstRelation = get_relationship(first)
                    secondRelation = get_relationship(second)
                    firstRelation.pop()
                    finalRelation = firstRelation + secondRelation
                    relationships.append(finalRelation)
            except:
                print("An exception occured")
                excepted_facts.append(extractions[0]['sentence'])
        else:
            best = 0
            best_index = 0
            for i in range(len(extractions)):
                extraction = extractions[i]
                if best < extraction['confidence']:
                    best_index = i
            relationships.append(get_relationship(extractions[best_index]['extraction']))

        if extractions == []:
            removed += 1
            print(str(counter) + " : " + str(removed))

        line = data.readline().capitalize()
        # if counter == 50:
        #     break

    data.close()

    # Save the relationships as data

    with open('data/relationships.csv', mode='w') as relationship_data:
        writer = csv.writer(relationship_data, delimiter=',')

        for relationship in relationships:
            writer.writerow(relationship)

    with open('excepted_relationships.txt', 'w') as file:
        for fact in excepted_facts:
            file.write('%s\n' % fact)