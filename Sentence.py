# Author: Deena Awny
# Version: 20.03.2020

# A sentence is some content/string.
# A sentence can be composed of one or more predicate argument structure
class Sentence:

  def __init__(self, content, position):
    self.content = content
    self.position = position

  def setContent(self, content):
    self.content = content

  def getContent(self):
    return self.content

  def setPredicates(self, predicates):
    self.predicates = predicates

  def getPredicates(self):
    return self.predicates

  def setPosition(self, position):
    self.position = position

  def getPosition(self):
    return self.position

  def lengthOfSentence(self):
    length = 0
    for p in self.getPredicates():
      length = length + p.lengthOfPredicate()
    return length