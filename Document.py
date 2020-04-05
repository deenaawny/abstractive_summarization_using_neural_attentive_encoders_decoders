# Author: Deena Awny
# Version: 03.04.2020

# A document has a title (include a variable title self.title)
# A document has a list of sentences
# A document has a list of predicates (from the sentences)
# A document has a list of clusters
import nltk
import re
import codecs
from Sentence import Sentence

class Document:

  def __init__(self, title, content):
    self.title = title
    self.content = content
    self.sentences = self.tokenizeDocument()
    self.setPredicatesFromSentences()

  def __init__(self, filedirectory):
    self.title = None
    self.summary = []
    f=codecs.open(filedirectory,"r",encoding='utf-8')
    #read the file to unicode string
    self.content= f.read()
    self.sentences = self.tokenizeDocument()

  def setTitle(self, title):
    self.title = title

  def getTitle(self):
    return self.title

  def setContent(self, content):
    self.content = content

  def getContent(self):
    return self.content

  def setSummary(self, summary):
    self.summary = summary

  def getSummary(self):
    return self.summary

  def setSentences(self, sentences):
    self.sentences = sentences

  def getSentences(self):
    return self.sentences

  def setPredicates(self, predicates):
    self.predicates = predicates

  def getPredicates(self):
    return self.predicates

  def lengthOfDocument(self):
    return len(self.getSentences())

  def cleanSentence(self,sentence):
    for x in nltk.sent_tokenize(sentence):
      return x
    return None

  def splitOnEmptyLines(self):
    blank_line_regex = r"(?:\r?\n){2,}"
    return re.split(blank_line_regex, self.content.strip())

  def tokenizeDocument(self):
    sentencesSplitted= self.splitOnEmptyLines()
    sentencePosition = 1
    self.sentences = []
    summaryflag= False
    for x in sentencesSplitted:
      if x == "@highlight":
        summaryflag= True
        continue
      if summaryflag == True:
        sentence= Sentence(x.replace('\n', ' '), sentencePosition)
        sentencePosition = sentencePosition + 1
        self.summary.append(sentence)
      else:
        sentence= Sentence(x.replace('\n', ' '), sentencePosition)
        sentencePosition = sentencePosition + 1
        self.sentences.append(sentence)
    return self.sentences

  def getSentencesAppended(self):
    text = ""
    for s in self.sentences:
      text = text + " " + s.content
    return text

  def getSummaryAppended(self):
    text = ""
    for s in self.summary:
      text = text + " " + s.content
    return text
  '''
 print("----------DOCUMENT EXAMPLE 1 ----------")
 d1 = Document("""A city trader who conned millions of pounds from wealthy investors was yesterday ordered to pay back £1.
 
 Nicholas Levene, 48, was jailed for 13 years last November after he admitted orchestrating a lucrative Ponzi scheme which raked in £316million.
 
 He used the money to finance his own lavish lifestyle with private jets, super yachts and round-the-world trips.
 """)
 print("----------DOCUMENT CONTENT----------")
 print(d1.content)
 print("----------DOCUMENT LENGTH----------")
 print(d1.lengthOfDocument())
 print("----------PREDICATES----------")
 for p in d1.getPredicates():
   print(p.getPredicateAsString())
 print("----------TAGGED PREDICATES----------")
 for p in d1.getPredicates():
   print(p.getPredicateAsStringWithTaggedPOS())
 
 print("----------DOCUMENT EXAMPLE 2 ----------")
 d1 = Document("""
     A city trader who conned millions of pounds from wealthy investors was yesterday ordered to pay back £1.
 
 Nicholas Levene, 48, was jailed for 13 years last November after he admitted orchestrating a lucrative Ponzi scheme which raked in £316million.
 
 He used the money to finance his own lavish lifestyle with private jets, super yachts and round-the-world trips.
 
 Must pay £1: Jailed city trader Nicholas Levene (pictured arriving at court in November last year), who conned wealthy investors out of £316million, was ordered to pay the nominal sum because he is bankrupt
 
 Now, because he is bankrupt, he has been given seven days to pay back a nominal sum of £1.
 
 The Serious Fraud Office found that Levene had conned £32,352,027 from some of Britain’s most successful businessmen.
 
   But with interest and lost profits, his clients are believed to be £101,685,406 out of pocket.
 
 It is unclear how much, if any of
 this, his victims have recouped. Justine  Davidge, representing the SFO,
 told Southwark Crown Court yesterday that Levene had been subject to a
 bankruptcy order since October 2009 and there were ongoing
 investigations into his assets.
 
 Jailed: Levene, nicknamed Beano because of his love of the comic (pictured left on the trading floor in 1990, and right with wife Tracy), was jailed for 13 years in November last year
 
 She added that anything seized would
 be dealt with by bankruptcy officials.
 
 ‘In respect of the realisable
 amount, we suggest the court make a nominal order of £1 to be paid in
 seven days,’ she said. ‘It may be in the future, Mr Levene could come
 into further realisable assets.’
 
 Levene admitted ripping off a series
 of high-fliers, including Sir Brian Souter and his sister Ann Gloag, the
 founders of the Stagecoach bus and rail group; Richard Caring, owner of
 The Ivy and Le Caprice restaurants in the West End; and Russell
 Bartlett, director of the R3 Investment Group and former owner of Hull
 City Football Club. Nicknamed Beano because of his childhood love of the
 comic book, Levene was a successful City worker with an estimated
 wealth of £15million to £20million in 2005.
 
 High-flyer: Levene conned some of Britain's most successful businessmen while owning this £2million eight-bedroom property in Barnet, North London
 
 Lavish lifestyle: Levene ran a multi-million pound illegal 'Ponzi' fraud scheme which he used to finance private jets (file picture), super yachts, a £150,000-a-year box at Ascot and on hosting £10,000-a-day pheasant shoots
 
 But he was addicted to gambling, spending fortunes on spread betting, and had an insatiable taste  for luxury.
 
                                                                                                        Levene, a former deputy chairman of
 Leyton Orient Football Club, admitted one count of false accounting, one
 of obtaining a money transfer by deception, and 12 of fraud.
 
 He would take from Peter to pay Paul
 and move the funds between accounts in the financial havens of Jersey,
 Switzerland and Israel.
 
 Seeing stars: The fraudster spent £588,000 on his second son's Bar Mitzvah celebration, which featured a performance by girl band The Saturdays (file picture)
 
 Spent big: His fraud scheme meant he could pay for a £150,000-a-year box at Ascot (file picture) but with interest and potential profits considered, clients are believed to have lost out by £101.6million
 
 Victim: Stagecoach Group's co-founders, brother and sister Sir Brian Souter and Ann Gloag (pictured) lost £10million
 
 With his network of contacts and
                     strong reputation, he won people’s faith with seemingly concrete
 investment deals from which he would take a commission or fee.
 
 The married father of three took
 millions of investors’ funds, promising to invest the money in lucrative
 rights-issue releases from  companies such as HSBC, Lloyds TSB and
 mining firms Xstrata and Rio Tinto.
 
 But he dug an ever-deepening financial
 hole for himself, having to fob off clients and make excuses about why
 he could not pay them.
 
 Living the high life, he had a chauffeur-driven Bentley and went on several holidays a year, each lasting several weeks.
 
 Investigators found evidence of round-the-world trips, yacht hire and top hotel stays in Australia, South Africa and Israel.
 
 The fraudster had a fleet of luxury
 cars and spent £588,000 on his second son’s Bar Mitzvah celebration,
 which featured a performance by girlband The Saturdays. Levene’s main
 house was a £2million eight-bedroom property in  Barnet, North London.
 
 His gambling was huge, with investigators finding evidence of him blowing £720,000 on a cricket match bet in 2007.
 
 Having been told about the seizure of
 Levene’s assets, Judge Martin Beddow said: ‘As there is nothing
 available, I direct the payable amount will be the nominal amount of £1
 to be paid in seven days.’
 
 Levene did not appear in court for the hearing.""")
 
 print("----------Document 2 Sentences----------")
 for s in d1.sentences:
   print(s)
   '''
  '''
 d3 = Document("C:/Users/admin/Documents/7lytixTest/dailymail_stories/dailymail/stories/file1.story")
 print("----------DOCUMENT CONTENT----------")
 print(d3.content)
 print("----------DOCUMENT LENGTH----------")
 print(d3.lengthOfDocument())
 print("----------PREDICATES----------")
 for p in d3.getPredicates():
   print(p.getPredicateAsString())
 print("----------TAGGED PREDICATES----------")
 for p in d3.getPredicates():
   print(p.getPredicateAsStringWithTaggedPOS())
 print("----------REFERENCE SUMMARY----------")
 summary = d3.summary
 for s in summary:
  print(s.content)
  '''