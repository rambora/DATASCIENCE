
############  Hillary Vs Trump Presidential debate  #############


setwd('G:/DATASCIENCE/DS-PRACTICE-PROJECTS/7_text_mining/Hillary_VS_Trump')

library(dplyr)
library(ggplot2)
library(wordcloud)
library(tm)
library(RSentiment)
library(data.table)
library(stringr)
library(RColorBrewer)
#library(tidytext)
#library(data.table)
library(gridExtra)

debate <- read.csv('debate.csv',h=T, stringsAsFactors = F)
colnames(debate)
str(debate)
nrow(debate)
#---------------------------------------------------------------
# Trump world cloud
Trump <- filter(debate, Speaker == 'Trump')
head(Trump$Speaker)
head(Trump$Text)
nrow(Trump)
colnames(Trump)
Trump_Corp <- Corpus(VectorSource(Trump$Text))
Trump_Corp
summary(Trump_Corp)
inspect(Trump_Corp[20])
writeLines(as.character(Trump_Corp[[20]]))
getTransformations()
toSpace <- content_transformer(function(x, pattern) gsub(pattern, ' ', x))
#docs <- tm_map(docs, content_transformer(gsub), pattern = '-|:|/@|\\||', replacement = ' ')
Trump_Corp <- tm_map(Trump_Corp, toSpace, '-:/@\\|')
Trump_Corp <- tm_map(Trump_Corp,content_transformer(tolower))
Trump_Corp <- tm_map(Trump_Corp, removeNumbers)
Trump_Corp <- tm_map(Trump_Corp, removePunctuation)
Trump_Corp <- tm_map(Trump_Corp, removeWords, stopwords("english"))
# removing own stop words
#Trump_Corp <- tm_map(Trump_Corp, removeWords, c('abc','xyz'))
Trump_Corp <- tm_map(Trump_Corp, stripWhitespace)
#Stemming
#Trump_Corp <- tm_map(Trump_Corp,stemDocument)
# Specific Transformation - usually after Stemming
#toString <- content_transformer(function(x, from, to) gsub(from,to,x))
#Trump_Corp <- tm_map(Trump_Corp, toString, 'abc', 'xyz')

# Document-Term Matrices / Term-Document Matrices 
Trump_dtm <- DocumentTermMatrix(Trump_Corp)
#Trump_dtmr <-DocumentTermMatrix(Trump_Corp, control=list(wordLengths=c(4, 20),bounds = list(global = c(3,27))))
Trump_dtm
class(Trump_dtm)
dim(Trump_dtm)
# Operations on term-document matrices
Trump_freq <- colSums(as.matrix(Trump_dtm))
head(Trump_freq)
#length should be total number of terms
length(Trump_freq)
#create sort order (descending)
Trump_ord <- order(Trump_freq)
head(Trump_ord)
#inspect least frequently occurring terms
Trump_freq[head(Trump_ord)]
#inspect most frequently occurring terms
Trump_freq[tail(Trump_ord)] 
# Distributon of term frequencies
head(table(Trump_freq),15)
tail(table(Trump_freq),15)
# Removing Sparse Terms
#dim(Trump_dtm)
#Trump_dtms <- removeSparseTerms(Trump_dtm, 0.001)
#dim(Trump_dtms)
# Identifying frequent Items and Associations
findFreqTerms(Trump_dtm, lowfreq=20)
findAssocs(Trump_dtm, 'country', corlimit=0.6)
# Correlation plots
#plot(Trump_dtm, terms=findFreqTerms(Trump_dtm, lowfreq=20))

# Quantitative analysis of the Text

#-----------------------------------------------------------------------------
# Hillary Word Clous
Clinton <- filter(debate, Speaker == 'Clinton')
head(Clinton$Speaker)
nrow(Clinton)
colnames(Clinton)
Clinton_Corp <- Corpus(VectorSource(Clinton$Text))
Clinton_Corp
summary(Clinton_Corp)
inspect(Clinton_Corp[20])
writeLines(as.character(Clinton_Corp[[20]]))
getTransformations()
toSpace <- content_transformer(function(x, pattern) gsub(pattern, ' ', x))
#docs <- tm_map(docs, content_transformer(gsub), pattern = '-|:|/@|\\||', replacement = ' ')
Clinton_Corp <- tm_map(Clinton_Corp, toSpace, '-:/@\\|')
Clinton_Corp <- tm_map(Clinton_Corp,content_transformer(tolower))
Clinton_Corp <- tm_map(Clinton_Corp, removeWords, stopwords("english"))
Clinton_Corp <- tm_map(Clinton_Corp, removeNumbers)
Clinton_Corp <- tm_map(Clinton_Corp, removePunctuation)
Clinton_Corp <- tm_map(Clinton_Corp, stripWhitespace)
#Stem document
#Clinton_Corp <- tm_map(Clinton_Corp,stemDocument)
#writeLines(as.character(Clinton_Corp[[30]]))

# Document-Term Matrices / Term-Document Matrices 
Clinton_dtm <- DocumentTermMatrix(Clinton_Corp)
Clinton_dtm
dim(Clinton_dtm)
class(Clinton_dtm)
dim(Clinton_dtm)
# Operations on term-document matrices
Clinton_freq <- colSums(as.matrix(Clinton_dtm))
head(Clinton_freq)
#length should be total number of terms
length(Clinton_freq)
#create sort order (descending)
Clinton_ord <- order(Clinton_freq)
head(Clinton_ord)
#inspect least frequently occurring terms
Clinton_freq[head(Clinton_ord)]
#inspect most frequently occurring terms
Clinton_freq[tail(Clinton_ord)] 
# Distributon of term frequencies
head(table(Clinton_freq),15)
tail(table(Clinton_freq),15)
# Removing Sparse Terms
#dim(Trump_dtm)
#Trump_dtms <- removeSparseTerms(Trump_dtm, 0.001)
#dim(Trump_dtms)
# Identifying frequent Items and Associations
findFreqTerms(Clinton_dtm, lowfreq=20)
findAssocs(Clinton_dtm, 'country', corlimit=0.6)
# Correlation plots
#plot(Trump_dtm, terms=findFreqTerms(Trump_dtm, lowfreq=20))

#plotting word frequencies
# --------------------------------------------------------------
# Who spoke much
Trump_total_words <- 0

for(line in seq(1,nrow(Trump))) {
  talk <- Trump[line, 'Text']
  words <- str_split(talk, ' ')
  Trump_total_words <- Trump_total_words + lengths(words)
#  return Trump_total_words
}

Clinton_total_words <- 0

for(line in seq(1,nrow(Clinton))) {
  talk <- Trump[line, 'Text']
  words <- str_split(talk, ' ')
  Clinton_total_words <- Clinton_total_words + lengths(words)
  #  return Trump_total_words
}

barplot(c(Trump_total_words,Clinton_total_words), xlab=c('candidates'), ylab=c('Word Count'),
        main = 'who spoke much ?',
        col = c('red','blue'), legend=c('Trump','Clinton'))
#---------------------------------------------------------------
# Word Frequency 
#plotting word frequencies
Trump_freq_sort <- sort(colSums(as.matrix(Trump_dtm)), decreasing=TRUE)
head(Trump_freq_sort)
Trump_wf=data.frame(word = names(Trump_freq_sort), freq=Trump_freq_sort) 
head(Trump_wf)

# Trump_wf <- transform(Trump_wf, word=reorder(word,freq)) # for reordering the plot
# Trump_word <- ggplot(subset(Trump_wf, Trump_freq_sort >15), aes(word, freq, fill= 'blue')) + 
#   geom_bar(stat='identity') + labs(title='Trump') + 
#   theme(axis.text.x=element_text(angle=45, hjust=1)) +
#   coord_flip()

Trump_word <- ggplot(subset(Trump_wf, Trump_freq_sort >15), aes(y=freq, x=reorder(word, freq))) + 
  geom_bar(stat='identity', fill='#CC6666') + labs(title='Trump') + 
  theme(axis.text.x=element_text(angle=45, hjust=1)) +
  coord_flip()
#--------------------------------------------------------------------
Clinton_freq_sort <- sort(colSums(as.matrix(Clinton_dtm)), decreasing=TRUE)
head(Clinton_freq_sort)
Clinton_wf=data.frame(word = names(Clinton_freq_sort), freq=Clinton_freq_sort) 
head(Clinton_wf)

# Clinton_wf <- transform(Clinton_wf, word=reorder(word,freq))
# Clinton_word <- ggplot(subset(Clinton_wf, Clinton_freq_sort >15), aes(word, freq)) +
#   geom_bar(stat='identity') + labs(title='Hillary') +
#   theme(axis.text.x=element_text(angle=45, hjust=1)) +
#   coord_flip()

Clinton_word <- ggplot(subset(Clinton_wf, Clinton_freq_sort >15), aes(y=freq, x=reorder(word, freq))) + 
  geom_bar(stat='identity', fill="#9999CC") + labs(title='Clinton') + 
  theme(axis.text.x=element_text(angle=45, hjust=1)) +
  coord_flip()

grid.arrange(Trump_word,Clinton_word,ncol=2)
#---------------------------------------------------------------
# Sentiment Analysis

Trump_score <- calculate_score(Trump$Text)
Trump_senti <- calculate_sentiment(Trump$Text)
Trump_senti$Speaker <- 'Trump'
class(Trump_senti$Speaker)
colnames(Trump_senti)
#Trump_total_senti <- calculate_total_presence_sentiment(Trump$Text)

Clinton_score <- calculate_score(Clinton$Text)
Clinton_senti <- calculate_sentiment(Clinton$Text)
Clinton_senti$Speaker <- 'Clinton'
class(Clinton_senti$Speaker)
#Clinton_total_senti <- calculate_total_presence_sentiment(Clinton$Text)

Total_senti <- rbind(Trump_senti,Clinton_senti)
colnames(Total_senti)
nrow(Total_senti)
Total_senti$Speaker <- as.factor(Total_senti$Speaker)
ggplot(Total_senti, aes(sentiment, fill=Speaker))+geom_bar(position='dodge')
#-------------------------------------------------------------------
# Word Cloud

#setting the same seed each time ensures consistent look across clouds
set.seed(210)
#limit words by specifying min frequency
#wordcloud(names(Trump_freq),Trump_freq, min.freq=10)
#.add color
wordcloud(names(Trump_freq),Trump_freq,min.freq=5,colors=brewer.pal(6,'Dark2'))
#wordcloud(names(Trump_freqr),Trump_freqr,max.words=100,colors=brewer.pal(6,'Dark2'))

#limit words by specifying min frequency
#wordcloud(names(Clinton_freq),Clinton_freq, min.freq=10)
#.add color
wordcloud(names(Clinton_freq),Clinton_freq,min.freq=5,colors=brewer.pal(6,'Dark2'))
#wordcloud(names(Clinton_freqr),Clinton_freqr,max.words=100,colors=brewer.pal(6,'Dark2'))
#------------------------------------------------------------------