# import nltk
# nltk.download('wordnet')  


from nltk.corpus import wordnet

dog = wordnet.synsets('dog')
dogset = dog[0]
weird = wordnet.synsets('dsf')
print(dog)
print(dogset)
print(dogset.path_similarity(dog[0]))
print(weird)